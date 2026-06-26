"""
StreamMapNet - Inference Script (PyTorch)

Runs streaming inference on a sequence of frames, maintaining temporal state
across the sequence. Supports single-sequence inference, BEV visualization,
and FPS benchmarking.

Usage:
    # Basic inference on a sequence
    python inference.py --config configs/stream_mapnet_base.yaml \
                        --checkpoint work_dirs/stream_mapnet/checkpoints/epoch_24.pth \
                        --data_root data/nuscenes --sequence_id 0

    # With visualization
    python inference.py --config configs/stream_mapnet_base.yaml \
                        --checkpoint work_dirs/stream_mapnet/checkpoints/epoch_24.pth \
                        --visualize --output_dir work_dirs/inference_vis

    # Benchmark mode
    python inference.py --config configs/stream_mapnet_base.yaml \
                        --checkpoint work_dirs/stream_mapnet/checkpoints/epoch_24.pth \
                        --benchmark --num_warmup 10 --num_benchmark 100
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import yaml

from model import StreamMapNet, build_stream_mapnet


# =============================================================================
# Visualization Utilities
# =============================================================================


def create_bev_visualization(
    predictions: Dict[str, np.ndarray],
    bev_x_range: Tuple[float, float] = (-30.0, 30.0),
    bev_y_range: Tuple[float, float] = (-15.0, 15.0),
    img_size: Tuple[int, int] = (800, 400),
    score_threshold: float = 0.3,
    gt_labels: Optional[np.ndarray] = None,
    gt_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a BEV visualization of predicted map elements.

    Args:
        predictions: dict with 'scores', 'labels', 'points' arrays
        bev_x_range: BEV x range in meters
        bev_y_range: BEV y range in meters
        img_size: output image size (height, width)
        score_threshold: minimum score to display
        gt_labels: optional ground truth labels for comparison
        gt_points: optional ground truth points for comparison

    Returns:
        vis_img: (H, W, 3) uint8 BGR image
    """
    try:
        import cv2
    except ImportError:
        # Fallback: return blank image if cv2 not available
        return np.zeros((*img_size, 3), dtype=np.uint8)

    H, W = img_size
    vis_img = np.zeros((H, W, 3), dtype=np.uint8)

    # Background grid
    x_range = bev_x_range[1] - bev_x_range[0]
    y_range = bev_y_range[1] - bev_y_range[0]

    # Draw grid lines every 10 meters
    for x_m in np.arange(bev_x_range[0], bev_x_range[1] + 1, 10.0):
        x_px = int((x_m - bev_x_range[0]) / x_range * W)
        cv2.line(vis_img, (x_px, 0), (x_px, H - 1), (40, 40, 40), 1)
    for y_m in np.arange(bev_y_range[0], bev_y_range[1] + 1, 10.0):
        y_px = int((y_m - bev_y_range[0]) / y_range * H)
        cv2.line(vis_img, (0, y_px), (W - 1, y_px), (40, 40, 40), 1)

    # Draw ego vehicle position (center)
    ego_x = int((0 - bev_x_range[0]) / x_range * W)
    ego_y = int((0 - bev_y_range[0]) / y_range * H)
    cv2.circle(vis_img, (ego_x, ego_y), 5, (255, 255, 255), -1)
    cv2.putText(vis_img, "ego", (ego_x + 8, ego_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Class colors (BGR)
    class_colors = {
        0: (0, 255, 0),     # lane_divider: green
        1: (0, 165, 255),   # road_boundary: orange
        2: (255, 0, 0),     # ped_crossing: blue
    }
    class_names = {
        0: "lane_divider",
        1: "road_boundary",
        2: "ped_crossing",
    }

    # Draw ground truth (dashed, lighter colors)
    if gt_labels is not None and gt_points is not None:
        for i in range(len(gt_labels)):
            cls = int(gt_labels[i])
            pts = gt_points[i]  # (K, 2) normalized [0, 1]

            # Convert to pixel coordinates
            pts_px = np.zeros_like(pts, dtype=np.int32)
            pts_px[:, 0] = (pts[:, 0] * W).astype(np.int32)
            pts_px[:, 1] = (pts[:, 1] * H).astype(np.int32)

            color = tuple(c // 2 for c in class_colors.get(cls, (128, 128, 128)))
            for j in range(len(pts_px) - 1):
                cv2.line(vis_img, tuple(pts_px[j]), tuple(pts_px[j + 1]), color, 1,
                         cv2.LINE_AA)

    # Draw predictions (solid, brighter colors)
    scores = predictions.get("scores", np.array([]))
    labels = predictions.get("labels", np.array([]))
    points = predictions.get("points", np.zeros((0, 20, 2)))

    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue

        cls = int(labels[i])
        pts = points[i]  # (K, 2) normalized [0, 1]
        score = scores[i]

        # Convert to pixel coordinates
        pts_px = np.zeros_like(pts, dtype=np.int32)
        pts_px[:, 0] = (pts[:, 0] * W).astype(np.int32)
        pts_px[:, 1] = (pts[:, 1] * H).astype(np.int32)

        color = class_colors.get(cls, (128, 128, 128))

        # Draw polyline
        for j in range(len(pts_px) - 1):
            cv2.line(vis_img, tuple(pts_px[j]), tuple(pts_px[j + 1]), color, 2,
                     cv2.LINE_AA)

        # Draw start point marker
        cv2.circle(vis_img, tuple(pts_px[0]), 3, color, -1)

        # Score text at midpoint
        mid_idx = len(pts_px) // 2
        cv2.putText(
            vis_img, f"{score:.2f}",
            (pts_px[mid_idx, 0] + 5, pts_px[mid_idx, 1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1,
        )

    # Legend
    y_offset = 20
    for cls_idx, cls_name in class_names.items():
        color = class_colors[cls_idx]
        cv2.rectangle(vis_img, (W - 160, y_offset - 12), (W - 145, y_offset), color, -1)
        cv2.putText(vis_img, cls_name, (W - 140, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20

    return vis_img


def save_predictions_json(
    predictions: List[Dict],
    output_path: str,
    bev_x_range: Tuple[float, float] = (-30.0, 30.0),
    bev_y_range: Tuple[float, float] = (-15.0, 15.0),
):
    """
    Save predictions to JSON format.

    Args:
        predictions: list of per-frame prediction dicts
        output_path: path to save JSON file
        bev_x_range: BEV x range for coordinate conversion
        bev_y_range: BEV y range for coordinate conversion
    """
    x_range = bev_x_range[1] - bev_x_range[0]
    y_range = bev_y_range[1] - bev_y_range[0]

    class_names = ["lane_divider", "road_boundary", "ped_crossing"]

    output_data = {
        "metadata": {
            "bev_x_range": list(bev_x_range),
            "bev_y_range": list(bev_y_range),
            "classes": class_names,
        },
        "frames": [],
    }

    for frame_idx, pred in enumerate(predictions):
        frame_data = {
            "frame_idx": frame_idx,
            "elements": [],
        }

        scores = pred.get("scores", np.array([]))
        labels = pred.get("labels", np.array([]))
        points = pred.get("points", np.zeros((0, 20, 2)))

        for i in range(len(scores)):
            # Convert normalized points to meters
            pts_meters = points[i].copy()
            pts_meters[:, 0] = pts_meters[:, 0] * x_range + bev_x_range[0]
            pts_meters[:, 1] = pts_meters[:, 1] * y_range + bev_y_range[0]

            element = {
                "class": class_names[int(labels[i])] if int(labels[i]) < len(class_names) else "unknown",
                "score": float(scores[i]),
                "points_meters": pts_meters.tolist(),
            }
            frame_data["elements"].append(element)

        output_data["frames"].append(frame_data)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


# =============================================================================
# Inference Engine
# =============================================================================


class StreamMapNetInference:
    """
    Streaming inference engine for StreamMapNet.

    Processes frames sequentially while maintaining temporal state,
    mimicking online deployment behavior.
    """

    def __init__(
        self,
        model: StreamMapNet,
        config: dict,
        device: torch.device,
        score_threshold: float = 0.3,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.score_threshold = score_threshold

        data_cfg = config.get("data", {})
        self.bev_x_range = tuple(data_cfg.get("bev_range", {}).get("x", [-30.0, 30.0]))
        self.bev_y_range = tuple(data_cfg.get("bev_range", {}).get("y", [-15.0, 15.0]))
        self.num_cameras = data_cfg.get("num_cameras", 6)
        self.img_size = tuple(data_cfg.get("img_size", [256, 704]))
        self.num_points = config.get("model", {}).get("map_decoder", {}).get(
            "num_points_per_query", 20
        )

        # Image normalization
        self.img_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1) / 255.0
        self.img_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1) / 255.0

        self.model.eval()
        self.model.reset_temporal_state()
        self._frame_count = 0

    def reset(self):
        """Reset temporal state for a new sequence."""
        self.model.reset_temporal_state()
        self._frame_count = 0

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize images.

        Args:
            images: (B, N, 3, H, W) in [0, 1] range or (B, N, 3, H, W) raw

        Returns:
            normalized images
        """
        # Assume images are already in [0, 1] float range
        images = (images - self.img_mean.to(images.device)) / self.img_std.to(images.device)
        return images

    @torch.no_grad()
    def process_frame(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_motion: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Process a single frame in streaming mode.

        Args:
            images: (B, N_cams, 3, H, W) camera images
            intrinsics: (B, N_cams, 3, 3) camera intrinsics
            extrinsics: (B, N_cams, 4, 4) camera extrinsics
            ego_motion: (B, 4, 4) ego motion from previous frame.
                        None for first frame in sequence.

        Returns:
            dict with 'scores', 'labels', 'points' numpy arrays for each batch item
        """
        # Move to device
        images = images.to(self.device)
        intrinsics = intrinsics.to(self.device)
        extrinsics = extrinsics.to(self.device)
        if ego_motion is not None:
            ego_motion = ego_motion.to(self.device)

        # Use ego_motion=None for first frame
        if self._frame_count == 0:
            ego_motion = None

        # Forward pass
        outputs = self.model(images, intrinsics, extrinsics, ego_motion=ego_motion)

        self._frame_count += 1

        # Post-process
        return self._postprocess(outputs)

    def _postprocess(self, outputs: Dict[str, torch.Tensor]) -> List[Dict[str, np.ndarray]]:
        """
        Post-process model outputs into final predictions.

        Args:
            outputs: model output dict

        Returns:
            List of B prediction dicts with numpy arrays
        """
        logits = outputs["pred_logits"]  # (B, N_q, C+1)
        points = outputs["pred_points"]  # (B, N_q, K, 2)
        B = logits.shape[0]

        results = []
        probs = logits.softmax(dim=-1).cpu().numpy()
        pts_np = points.cpu().numpy()

        for b in range(B):
            # Get max class probability (excluding background)
            scores = probs[b, :, :-1].max(axis=-1)  # (N_q,)
            labels = probs[b, :, :-1].argmax(axis=-1)  # (N_q,)

            # Filter by threshold
            mask = scores > self.score_threshold
            results.append({
                "scores": scores[mask],
                "labels": labels[mask],
                "points": pts_np[b][mask],  # (N_keep, K, 2)
            })

        return results


# =============================================================================
# Benchmarking
# =============================================================================


def benchmark_model(
    model: StreamMapNet,
    config: dict,
    device: torch.device,
    num_warmup: int = 10,
    num_benchmark: int = 100,
) -> Dict[str, float]:
    """
    Benchmark model inference speed.

    Args:
        model: StreamMapNet model
        config: configuration dict
        device: compute device
        num_warmup: number of warmup iterations
        num_benchmark: number of benchmark iterations

    Returns:
        dict with timing statistics
    """
    data_cfg = config.get("data", {})
    N = data_cfg.get("num_cameras", 6)
    H, W = data_cfg.get("img_size", [256, 704])

    model.eval()
    model.reset_temporal_state()

    # Create dummy inputs
    images = torch.randn(1, N, 3, H, W, device=device)
    intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, N, -1, -1).clone()
    intrinsics[:, :, 0, 0] = 1260.0
    intrinsics[:, :, 1, 1] = 1260.0
    intrinsics[:, :, 0, 2] = W / 2.0
    intrinsics[:, :, 1, 2] = H / 2.0
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(1, N, -1, -1).clone()
    ego_motion = torch.eye(4, device=device).unsqueeze(0)
    ego_motion[:, 0, 3] = 0.5

    # Warmup
    print(f"  Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for i in range(num_warmup):
            ego = None if i == 0 else ego_motion
            _ = model(images, intrinsics, extrinsics, ego_motion=ego)
            if i == 0:
                model.reset_temporal_state()

    # Synchronize GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({num_benchmark} iterations)...")
    model.reset_temporal_state()
    times = []

    with torch.no_grad():
        for i in range(num_benchmark):
            ego = None if i == 0 else ego_motion

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(images, intrinsics, extrinsics, ego_motion=ego)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # ms

            # Reset state periodically to simulate sequence boundaries
            if (i + 1) % 50 == 0:
                model.reset_temporal_state()

    times = np.array(times)

    results = {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": float(1000.0 / np.mean(times)),
    }

    return results


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="StreamMapNet Inference")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_root", type=str, default="data/nuscenes",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--sequence_id", type=int, default=0,
        help="Sequence ID to process",
    )
    parser.add_argument(
        "--num_frames", type=int, default=40,
        help="Number of frames to process in the sequence",
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.3,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--output_dir", type=str, default="work_dirs/inference",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save BEV visualizations",
    )
    parser.add_argument(
        "--save_json", action="store_true",
        help="Save predictions as JSON",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run FPS benchmark",
    )
    parser.add_argument(
        "--num_warmup", type=int, default=10,
        help="Number of warmup iterations for benchmark",
    )
    parser.add_argument(
        "--num_benchmark", type=int, default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use FP16 inference",
    )
    return parser.parse_args()


def generate_synthetic_sequence(
    num_frames: int,
    num_cameras: int = 6,
    img_size: Tuple[int, int] = (256, 704),
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate a synthetic sequence for demonstration.

    In a real deployment, this would load actual camera data from disk
    or receive frames from a live camera stream.
    """
    H, W = img_size
    frames = []

    for t in range(num_frames):
        # Simulated camera images
        images = torch.randn(1, num_cameras, 3, H, W) * 0.2 + 0.5
        images = images.clamp(0, 1)

        # Camera intrinsics (fixed across frames)
        intrinsics = torch.zeros(1, num_cameras, 3, 3)
        intrinsics[:, :, 0, 0] = 1260.0
        intrinsics[:, :, 1, 1] = 1260.0
        intrinsics[:, :, 0, 2] = W / 2.0
        intrinsics[:, :, 1, 2] = H / 2.0
        intrinsics[:, :, 2, 2] = 1.0

        # Camera extrinsics (fixed across frames)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, num_cameras, -1, -1).clone()
        for cam in range(num_cameras):
            angle = cam * (2.0 * math.pi / num_cameras)
            extrinsics[0, cam, 0, 3] = 1.5 * math.cos(angle)
            extrinsics[0, cam, 1, 3] = 1.5 * math.sin(angle)
            extrinsics[0, cam, 2, 3] = 1.6

        # Ego motion: simulate driving forward at ~10 m/s at 10 Hz
        ego_motion = torch.eye(4).unsqueeze(0)
        if t > 0:
            ego_motion[0, 0, 3] = 1.0  # 1m forward per frame
            # Small random yaw
            theta = np.random.uniform(-0.01, 0.01)
            ego_motion[0, 0, 0] = math.cos(theta)
            ego_motion[0, 0, 1] = -math.sin(theta)
            ego_motion[0, 1, 0] = math.sin(theta)
            ego_motion[0, 1, 1] = math.cos(theta)
        else:
            ego_motion = None

        frames.append({
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "ego_motion": ego_motion,
        })

    return frames


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"StreamMapNet Inference")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  FP16: {args.fp16}")

    # Build model
    model = build_stream_mapnet(config).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded model from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded model weights")

    # Optional FP16
    if args.fp16 and device.type == "cuda":
        model = model.half()
        print("  Using FP16 inference")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # ---- Benchmark Mode ----
    if args.benchmark:
        print(f"\n{'='*60}")
        print("FPS Benchmark")
        print(f"{'='*60}")

        bench_results = benchmark_model(
            model, config, device,
            num_warmup=args.num_warmup,
            num_benchmark=args.num_benchmark,
        )

        print(f"\n  Results:")
        print(f"    Mean latency:   {bench_results['mean_ms']:.2f} ms")
        print(f"    Median latency: {bench_results['median_ms']:.2f} ms")
        print(f"    Std latency:    {bench_results['std_ms']:.2f} ms")
        print(f"    Min latency:    {bench_results['min_ms']:.2f} ms")
        print(f"    Max latency:    {bench_results['max_ms']:.2f} ms")
        print(f"    P95 latency:    {bench_results['p95_ms']:.2f} ms")
        print(f"    P99 latency:    {bench_results['p99_ms']:.2f} ms")
        print(f"    FPS:            {bench_results['fps']:.1f}")

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "benchmark_results.json", "w") as f:
                json.dump(bench_results, f, indent=2)
            print(f"\n  Results saved to: {output_dir / 'benchmark_results.json'}")

        return

    # ---- Sequence Inference Mode ----
    print(f"\n{'='*60}")
    print(f"Streaming Inference - Sequence {args.sequence_id}")
    print(f"{'='*60}")

    # Create output directory
    output_dir = Path(args.output_dir)
    if args.visualize or args.save_json:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create inference engine
    engine = StreamMapNetInference(
        model=model,
        config=config,
        device=device,
        score_threshold=args.score_threshold,
    )

    # Generate or load sequence
    data_cfg = config.get("data", {})
    frames = generate_synthetic_sequence(
        num_frames=args.num_frames,
        num_cameras=data_cfg.get("num_cameras", 6),
        img_size=tuple(data_cfg.get("img_size", [256, 704])),
        device=device,
    )

    # Process sequence frame by frame
    all_predictions = []
    frame_times = []

    engine.reset()
    print(f"\n  Processing {len(frames)} frames...")

    for t, frame in enumerate(frames):
        # Convert to appropriate dtype
        images = frame["images"]
        if args.fp16 and device.type == "cuda":
            images = images.half()

        start = time.perf_counter()
        results = engine.process_frame(
            images=images,
            intrinsics=frame["intrinsics"],
            extrinsics=frame["extrinsics"],
            ego_motion=frame["ego_motion"],
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms

        frame_times.append(elapsed)
        pred = results[0]  # First (only) batch item
        all_predictions.append(pred)

        num_elements = len(pred["scores"])
        if (t + 1) % 10 == 0 or t == 0:
            print(
                f"    Frame {t + 1:>3}/{len(frames)} | "
                f"Detections: {num_elements:>3} | "
                f"Time: {elapsed:.1f} ms"
            )

    # Summary statistics
    frame_times = np.array(frame_times)
    print(f"\n  Inference complete.")
    print(f"    Total frames:   {len(frames)}")
    print(f"    Mean time:      {frame_times.mean():.1f} ms/frame")
    print(f"    Median time:    {np.median(frame_times):.1f} ms/frame")
    print(f"    FPS:            {1000.0 / frame_times.mean():.1f}")
    print(f"    Avg detections: {np.mean([len(p['scores']) for p in all_predictions]):.1f}")

    # Save visualizations
    if args.visualize:
        print(f"\n  Saving visualizations to: {output_dir}")
        vis_dir = output_dir / "bev_vis"
        vis_dir.mkdir(exist_ok=True)

        for t, pred in enumerate(all_predictions):
            vis_img = create_bev_visualization(
                predictions=pred,
                bev_x_range=engine.bev_x_range,
                bev_y_range=engine.bev_y_range,
                score_threshold=args.score_threshold,
            )

            try:
                import cv2
                output_path = str(vis_dir / f"frame_{t:04d}.png")
                cv2.imwrite(output_path, vis_img)
            except ImportError:
                # Save as numpy if cv2 not available
                output_path = str(vis_dir / f"frame_{t:04d}.npy")
                np.save(output_path, vis_img)

        print(f"    Saved {len(all_predictions)} BEV visualization frames.")

    # Save JSON predictions
    if args.save_json:
        json_path = str(output_dir / "predictions.json")
        save_predictions_json(
            all_predictions,
            json_path,
            bev_x_range=engine.bev_x_range,
            bev_y_range=engine.bev_y_range,
        )
        print(f"    Predictions saved to: {json_path}")

    # Save timing summary
    timing_summary = {
        "num_frames": len(frames),
        "mean_ms": float(frame_times.mean()),
        "median_ms": float(np.median(frame_times)),
        "std_ms": float(frame_times.std()),
        "fps": float(1000.0 / frame_times.mean()),
        "total_time_s": float(frame_times.sum() / 1000.0),
    }
    with open(output_dir / "timing.json", "w") as f:
        json.dump(timing_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Inference complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
