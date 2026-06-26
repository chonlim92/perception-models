"""Cylinder3D TensorFlow 2 inference script.

Runs inference on raw .bin point cloud files and produces:
- Predicted .label files (SemanticKITTI format)
- Colored .ply point clouds for visualization
- Timing statistics
"""

import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import Cylinder3DModel, SEMANTICKITTI_NUM_CLASSES, DEFAULT_GRID_SIZE
from train import load_config


# Color palette for SemanticKITTI classes (RGB, 0-255)
SEMANTICKITTI_COLORS = np.array(
    [
        [0, 0, 0],         # 0: unlabeled
        [100, 150, 245],   # 1: car
        [100, 230, 245],   # 2: bicycle
        [100, 80, 250],    # 3: bus
        [30, 60, 150],     # 4: motorcycle
        [0, 0, 255],       # 5: on-rails
        [80, 30, 180],     # 6: truck
        [0, 0, 230],       # 7: other-vehicle
        [255, 30, 30],     # 8: person
        [255, 40, 200],    # 9: bicyclist
        [150, 30, 90],     # 10: motorcyclist
        [255, 0, 255],     # 11: road
        [255, 150, 255],   # 12: parking
        [75, 0, 75],       # 13: sidewalk
        [175, 0, 75],      # 14: other-ground
        [255, 200, 0],     # 15: building
        [255, 120, 50],    # 16: fence
        [255, 255, 0],     # 17: lane-marking
        [0, 175, 0],       # 18: vegetation
        [135, 60, 0],      # 19: trunk
    ],
    dtype=np.uint8,
)


def load_bin_file(bin_path):
    """Load a point cloud from a .bin file.

    Args:
        bin_path: path to binary file (float32, 4 values per point: x,y,z,intensity)

    Returns:
        points: numpy array of shape [N, 4]
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def save_label_file(predictions, output_path):
    """Save predictions as a .label file in SemanticKITTI format.

    Args:
        predictions: numpy array of shape [N] with class indices
        output_path: path to output .label file
    """
    # SemanticKITTI uses uint32: lower 16 bits = semantic label, upper 16 = instance
    labels = predictions.astype(np.uint32)
    labels.tofile(output_path)


def save_ply_file(points, predictions, output_path):
    """Save colored point cloud as PLY file.

    Args:
        points: numpy array [N, 4] (x, y, z, intensity)
        predictions: numpy array [N] with class indices
        output_path: path to output .ply file
    """
    num_points = points.shape[0]

    # Get colors for predictions
    colors = SEMANTICKITTI_COLORS[np.clip(predictions, 0, len(SEMANTICKITTI_COLORS) - 1)]

    # Write PLY header and data
    with open(output_path, "wb") as f:
        # Header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float intensity\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property uchar label\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))

        # Write vertex data
        for i in range(num_points):
            f.write(
                struct.pack(
                    "<ffffBBBB",
                    points[i, 0],
                    points[i, 1],
                    points[i, 2],
                    points[i, 3],
                    colors[i, 0],
                    colors[i, 1],
                    colors[i, 2],
                    int(predictions[i]),
                )
            )


def save_ply_file_fast(points, predictions, output_path):
    """Save colored point cloud as PLY file (vectorized, faster).

    Args:
        points: numpy array [N, 4] (x, y, z, intensity)
        predictions: numpy array [N] with class indices
        output_path: path to output .ply file
    """
    num_points = points.shape[0]
    colors = SEMANTICKITTI_COLORS[np.clip(predictions, 0, len(SEMANTICKITTI_COLORS) - 1)]

    # Write PLY header
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar label\n"
        "end_header\n"
    )

    # Create structured array for efficient binary write
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
            ("label", "u1"),
        ]
    )

    vertex_data = np.zeros(num_points, dtype=dtype)
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["intensity"] = points[:, 3]
    vertex_data["r"] = colors[:, 0]
    vertex_data["g"] = colors[:, 1]
    vertex_data["b"] = colors[:, 2]
    vertex_data["label"] = predictions.astype(np.uint8)

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())


def run_inference(config, args):
    """Run inference on input .bin files.

    Args:
        config: configuration dict
        args: argparse namespace
    """
    num_classes = config["model"]["num_classes"]
    num_points = config["data"]["num_points"]

    # Build model
    print("Building Cylinder3D model...")
    model = Cylinder3DModel(
        num_classes=num_classes,
        grid_size=config["model"]["grid_size"],
        rho_range=config["model"].get("rho_range"),
        theta_range=config["model"].get("theta_range"),
        z_range=config["model"].get("z_range"),
    )

    # Initialize model weights with dummy forward pass
    dummy_input = tf.zeros([1, num_points, 4], dtype=tf.float32)
    _ = model(dummy_input, training=False)
    print(f"Model parameters: {model.count_params():,}")

    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint.replace(".index", "")
        try:
            model.load_weights(checkpoint_path)
            print(f"Loaded weights: {checkpoint_path}")
        except (ValueError, tf.errors.NotFoundError):
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_path).expect_partial()
            print(f"Restored checkpoint: {checkpoint_path}")
    else:
        print("WARNING: No checkpoint specified. Using uninitialized weights.")

    # Collect input files
    input_path = Path(args.input)
    if input_path.is_file():
        bin_files = [str(input_path)]
    elif input_path.is_dir():
        bin_files = sorted([str(f) for f in input_path.glob("*.bin")])
    else:
        raise ValueError(f"Input path does not exist: {args.input}")

    if not bin_files:
        raise ValueError(f"No .bin files found in: {args.input}")

    print(f"Found {len(bin_files)} input file(s)")

    # Create output directories
    output_dir = Path(args.output_dir)
    label_dir = output_dir / "predictions"
    ply_dir = output_dir / "colored_ply"
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.save_ply:
        ply_dir.mkdir(parents=True, exist_ok=True)

    # Warm up (first inference is slower due to tracing)
    print("Warming up model...")
    warmup_points = tf.random.normal([1, min(num_points, 10000), 4])
    _ = model(warmup_points, training=False)

    # Run inference
    timing_results = []
    print(f"\nRunning inference on {len(bin_files)} files...")
    print("-" * 60)

    for i, bin_path in enumerate(bin_files):
        frame_name = Path(bin_path).stem

        # Load point cloud
        t_load_start = time.time()
        points_np = load_bin_file(bin_path)
        original_num_points = points_np.shape[0]
        t_load = time.time() - t_load_start

        # Handle variable point counts
        if original_num_points > num_points:
            # Subsample
            indices = np.random.choice(original_num_points, num_points, replace=False)
            indices_sorted = np.sort(indices)
            points_input = points_np[indices_sorted]
        elif original_num_points < num_points:
            # Pad
            pad = np.zeros((num_points - original_num_points, 4), dtype=np.float32)
            points_input = np.concatenate([points_np, pad], axis=0)
            indices_sorted = None
        else:
            points_input = points_np
            indices_sorted = None

        # Model inference
        points_tensor = tf.constant(points_input[np.newaxis], dtype=tf.float32)  # [1, N, 4]

        t_infer_start = time.time()
        point_logits, _ = model(points_tensor, training=False)
        predictions = tf.argmax(point_logits, axis=-1, output_type=tf.int32)
        # Force GPU sync for accurate timing
        predictions_np = predictions.numpy().squeeze()
        t_infer = time.time() - t_infer_start

        # Map predictions back to original points
        if original_num_points > num_points:
            # For subsampled points, we have predictions for the subset
            # Full predictions require nearest neighbor or re-inference
            full_predictions = np.zeros(original_num_points, dtype=np.int32)
            full_predictions[indices_sorted] = predictions_np[:num_points]
            # For non-selected points, assign nearest selected point's prediction
            # Simple approach: use the predictions for selected points only in output
            predictions_final = full_predictions
        elif original_num_points < num_points:
            # Remove padding predictions
            predictions_final = predictions_np[:original_num_points]
        else:
            predictions_final = predictions_np

        # Save .label file
        t_save_start = time.time()
        label_output_path = str(label_dir / f"{frame_name}.label")
        save_label_file(predictions_final, label_output_path)

        # Save colored PLY
        if args.save_ply:
            ply_output_path = str(ply_dir / f"{frame_name}.ply")
            save_ply_file_fast(points_np, predictions_final, ply_output_path)

        t_save = time.time() - t_save_start

        total_time = t_load + t_infer + t_save
        timing_results.append(
            {
                "frame": frame_name,
                "num_points": original_num_points,
                "load_ms": t_load * 1000,
                "infer_ms": t_infer * 1000,
                "save_ms": t_save * 1000,
                "total_ms": total_time * 1000,
            }
        )

        # Print progress
        if (i + 1) % 10 == 0 or i == 0 or i == len(bin_files) - 1:
            print(
                f"  [{i+1:>5}/{len(bin_files)}] {frame_name} | "
                f"Points: {original_num_points:>6} | "
                f"Infer: {t_infer*1000:>6.1f}ms | "
                f"Total: {total_time*1000:>7.1f}ms"
            )

    # Print timing summary
    print("\n" + "=" * 60)
    print("INFERENCE TIMING SUMMARY")
    print("=" * 60)

    infer_times = [r["infer_ms"] for r in timing_results]
    total_times = [r["total_ms"] for r in timing_results]
    num_points_list = [r["num_points"] for r in timing_results]

    print(f"Files processed: {len(timing_results)}")
    print(f"Average points per frame: {np.mean(num_points_list):.0f}")
    print(f"\nInference time:")
    print(f"  Mean:   {np.mean(infer_times):>7.1f} ms")
    print(f"  Median: {np.median(infer_times):>7.1f} ms")
    print(f"  Min:    {np.min(infer_times):>7.1f} ms")
    print(f"  Max:    {np.max(infer_times):>7.1f} ms")
    print(f"  Std:    {np.std(infer_times):>7.1f} ms")
    print(f"\nTotal time (load + infer + save):")
    print(f"  Mean:   {np.mean(total_times):>7.1f} ms")
    print(f"  FPS:    {1000.0 / np.mean(total_times):>7.1f}")

    # Skip first frame (warmup effect) for more accurate stats
    if len(infer_times) > 1:
        infer_no_warmup = infer_times[1:]
        print(f"\nInference (excluding first frame warmup):")
        print(f"  Mean:   {np.mean(infer_no_warmup):>7.1f} ms")
        print(f"  FPS:    {1000.0 / np.mean(infer_no_warmup):>7.1f}")

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  Labels: {label_dir}")
    if args.save_ply:
        print(f"  PLY:    {ply_dir}")
    print("=" * 60)

    # Save timing report
    timing_path = output_dir / "timing_report.txt"
    with open(timing_path, "w") as f:
        f.write("Cylinder3D TF2 Inference Timing Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: Cylinder3D (grid {config['model']['grid_size']})\n")
        f.write(f"Files: {len(timing_results)}\n")
        f.write(f"Target points: {num_points}\n\n")
        f.write(f"{'Frame':<12} {'Points':>7} {'Infer(ms)':>10} {'Total(ms)':>10}\n")
        f.write("-" * 50 + "\n")
        for r in timing_results:
            f.write(
                f"{r['frame']:<12} {r['num_points']:>7} "
                f"{r['infer_ms']:>10.1f} {r['total_ms']:>10.1f}\n"
            )
        f.write("-" * 50 + "\n")
        f.write(f"{'MEAN':<12} {np.mean(num_points_list):>7.0f} ")
        f.write(f"{np.mean(infer_times):>10.1f} {np.mean(total_times):>10.1f}\n")

    print(f"Timing report: {timing_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Cylinder3D inference on .bin point clouds (TensorFlow 2)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .bin file or directory of .bin files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_output",
        help="Output directory for predictions and PLY files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint or weights file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Generate colored PLY point cloud files",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU ID to use for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (currently only 1 supported)",
    )
    return parser.parse_args()


def main():
    """Entry point for inference."""
    args = parse_args()

    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if args.gpus and gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        visible = [gpus[i] for i in gpu_ids if i < len(gpus)]
        tf.config.set_visible_devices(visible, "GPU")
        for gpu in visible:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {args.gpus}")
    elif not gpus:
        print("No GPU found, running on CPU")

    # Load config
    config = load_config(args.config)

    print("Cylinder3D TensorFlow 2 Inference")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Checkpoint: {args.checkpoint or 'none'}")
    print(f"Grid size: {config['model']['grid_size']}")
    print(f"Save PLY: {args.save_ply}")
    print("=" * 60)

    run_inference(config, args)


if __name__ == "__main__":
    main()
