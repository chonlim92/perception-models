"""
RadarPillarNet evaluation script.

Computes nuScenes Detection Score (NDS) metrics:
- mAP (mean Average Precision) at multiple distance thresholds
- mATE (mean Average Translation Error)
- mASE (mean Average Scale Error)
- mAOE (mean Average Orientation Error)
- mAVE (mean Average Velocity Error)
- NDS (nuScenes Detection Score)

Supports:
- Per-class metrics
- Distance-stratified evaluation
- Loading from checkpoint or SavedModel
- Formatted results output
- JSON export
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from model import RadarPillarNet, DEFAULT_CONFIG, build_radar_pillarnet


# ---------------------------------------------------------------------------
# nuScenes class definitions
# ---------------------------------------------------------------------------

NUSCENES_CLASSES: List[str] = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Maximum detection range per class (meters)
NUSCENES_DETECTION_RANGES: Dict[str, float] = {
    "car": 50.0,
    "truck": 50.0,
    "construction_vehicle": 50.0,
    "bus": 50.0,
    "trailer": 50.0,
    "barrier": 30.0,
    "motorcycle": 40.0,
    "bicycle": 40.0,
    "pedestrian": 40.0,
    "traffic_cone": 30.0,
}

# Distance stratification ranges for evaluation
DISTANCE_RANGES: List[Tuple[float, float]] = [
    (0.0, 20.0),
    (20.0, 40.0),
    (40.0, 60.0),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Detection3D:
    """A single 3D detection."""

    center: np.ndarray  # (3,) x, y, z
    size: np.ndarray    # (3,) w, l, h
    yaw: float          # rotation in radians
    velocity: np.ndarray  # (2,) vx, vy
    score: float
    class_id: int
    class_name: str


@dataclass
class GroundTruth3D:
    """A single ground truth 3D bounding box."""

    center: np.ndarray  # (3,) x, y, z
    size: np.ndarray    # (3,) w, l, h
    yaw: float
    velocity: np.ndarray  # (2,) vx, vy
    class_id: int
    class_name: str


@dataclass
class EvalResults:
    """Aggregated evaluation results."""

    mAP: float = 0.0
    mATE: float = 0.0
    mASE: float = 0.0
    mAOE: float = 0.0
    mAVE: float = 0.0
    NDS: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    per_class_ate: Dict[str, float] = field(default_factory=dict)
    per_class_ase: Dict[str, float] = field(default_factory=dict)
    per_class_aoe: Dict[str, float] = field(default_factory=dict)
    per_class_ave: Dict[str, float] = field(default_factory=dict)
    per_distance_ap: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "NDS": self.NDS,
            "mAP": self.mAP,
            "mATE": self.mATE,
            "mASE": self.mASE,
            "mAOE": self.mAOE,
            "mAVE": self.mAVE,
            "per_class_ap": self.per_class_ap,
            "per_class_ate": self.per_class_ate,
            "per_class_ase": self.per_class_ase,
            "per_class_aoe": self.per_class_aoe,
            "per_class_ave": self.per_class_ave,
            "per_distance_ap": self.per_distance_ap,
        }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def center_distance(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute 2D center distance in BEV (x-y plane)."""
    return float(np.linalg.norm(det.center[:2] - gt.center[:2]))


def scale_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute 1 - IOU_3D (approximated via volume comparison)."""
    det_vol = float(np.prod(np.maximum(det.size, 1e-4)))
    gt_vol = float(np.prod(np.maximum(gt.size, 1e-4)))

    # Approximate IoU via min-volumes
    min_size = np.minimum(np.abs(det.size), np.abs(gt.size))
    intersection_vol = float(np.prod(min_size))
    union_vol = det_vol + gt_vol - intersection_vol
    iou = intersection_vol / max(union_vol, 1e-8)
    return 1.0 - iou


def orientation_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute absolute angular difference in yaw, mapped to [0, pi]."""
    diff = abs(det.yaw - gt.yaw)
    diff = diff % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return float(diff)


def velocity_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute L2 velocity error in m/s."""
    return float(np.linalg.norm(det.velocity - gt.velocity))


def compute_ap_for_class(
    detections: List[Detection3D],
    ground_truths: List[GroundTruth3D],
    class_name: str,
    distance_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
    max_range: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Compute Average Precision and true-positive metrics for a single class.

    Args:
        detections: all detections (will be filtered by class)
        ground_truths: all GT boxes (will be filtered by class)
        class_name: target class
        distance_thresholds: BEV center distance thresholds for TP matching
        max_range: optional maximum range filter
    Returns:
        (ap, ate, ase, aoe, ave)
    """
    # Filter by class
    class_dets = [d for d in detections if d.class_name == class_name]
    class_gts = [g for g in ground_truths if g.class_name == class_name]

    # Optional range filter
    if max_range is not None:
        class_dets = [d for d in class_dets if np.linalg.norm(d.center[:2]) <= max_range]
        class_gts = [g for g in class_gts if np.linalg.norm(g.center[:2]) <= max_range]

    if not class_gts:
        return 0.0, 1.0, 1.0, np.pi, 1.0

    # Sort detections by score descending
    class_dets.sort(key=lambda x: x.score, reverse=True)

    # Compute AP at each distance threshold
    aps_per_threshold: List[float] = []
    all_ate: List[float] = []
    all_ase: List[float] = []
    all_aoe: List[float] = []
    all_ave: List[float] = []

    for dist_thresh in distance_thresholds:
        matched_gt = set()
        tp_list: List[int] = []
        fp_list: List[int] = []

        for det in class_dets:
            best_dist = float("inf")
            best_gt_idx = -1

            for gt_idx, gt in enumerate(class_gts):
                if gt_idx in matched_gt:
                    continue
                dist = center_distance(det, gt)
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = gt_idx

            if best_dist <= dist_thresh and best_gt_idx >= 0:
                tp_list.append(1)
                fp_list.append(0)
                matched_gt.add(best_gt_idx)

                # Record TP error metrics
                gt = class_gts[best_gt_idx]
                all_ate.append(center_distance(det, gt))
                all_ase.append(scale_error(det, gt))
                all_aoe.append(orientation_error(det, gt))
                all_ave.append(velocity_error(det, gt))
            else:
                tp_list.append(0)
                fp_list.append(1)

        if not tp_list:
            aps_per_threshold.append(0.0)
            continue

        # Precision-recall curve
        tp_cumsum = np.cumsum(tp_list).astype(np.float64)
        fp_cumsum = np.cumsum(fp_list).astype(np.float64)
        recall = tp_cumsum / len(class_gts)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # 11-point interpolated AP
        ap = 0.0
        for r_thresh in np.linspace(0, 1, 11):
            prec_at_recall = precision[recall >= r_thresh]
            if len(prec_at_recall) > 0:
                ap += float(np.max(prec_at_recall))
        ap /= 11.0
        aps_per_threshold.append(ap)

    mean_ap = float(np.mean(aps_per_threshold))
    ate = float(np.mean(all_ate)) if all_ate else 1.0
    ase = float(np.mean(all_ase)) if all_ase else 1.0
    aoe = float(np.mean(all_aoe)) if all_aoe else float(np.pi)
    ave = float(np.mean(all_ave)) if all_ave else 1.0

    return mean_ap, ate, ase, aoe, ave


def compute_nds(results: EvalResults) -> float:
    """
    Compute nuScenes Detection Score.

    NDS = (5 * mAP + sum(max(1 - metric, 0) for TP metrics)) / 10
    """
    tp_scores = [
        max(1.0 - results.mATE, 0.0),
        max(1.0 - results.mASE, 0.0),
        max(1.0 - results.mAOE / np.pi, 0.0),
        max(1.0 - results.mAVE, 0.0),
        1.0,  # mAAE placeholder
    ]
    nds = (5.0 * results.mAP + sum(tp_scores)) / 10.0
    return float(nds)


# ---------------------------------------------------------------------------
# Decode model predictions to Detection3D
# ---------------------------------------------------------------------------


def decode_predictions_to_detections(
    model: RadarPillarNet,
    inputs: Dict[str, tf.Tensor],
    config: Dict[str, Any],
    score_threshold: float = 0.1,
    max_detections: int = 500,
) -> List[Detection3D]:
    """
    Run model inference and decode outputs to Detection3D list.

    Uses the model's predict_with_nms method for efficient post-processing.
    """
    # Run inference with NMS
    results = model.predict_with_nms(inputs)

    boxes = results["boxes"].numpy()         # (B, max_det, 7)
    scores = results["scores"].numpy()       # (B, max_det)
    labels = results["labels"].numpy()       # (B, max_det)
    velocities = results["velocities"].numpy()  # (B, max_det, 2)
    num_det = results["num_detections"].numpy()  # (B,)

    all_detections: List[Detection3D] = []

    for b in range(boxes.shape[0]):
        n = int(num_det[b])
        for i in range(n):
            score = float(scores[b, i])
            if score < score_threshold:
                continue

            box = boxes[b, i]  # [x, y, z, w, l, h, yaw]
            label = int(labels[b, i])
            vel = velocities[b, i]

            if 0 <= label < len(NUSCENES_CLASSES):
                class_name = NUSCENES_CLASSES[label]
            else:
                class_name = "unknown"

            det = Detection3D(
                center=np.array([box[0], box[1], box[2]], dtype=np.float64),
                size=np.array([box[3], box[4], box[5]], dtype=np.float64),
                yaw=float(box[6]),
                velocity=vel.astype(np.float64),
                score=score,
                class_id=label,
                class_name=class_name,
            )
            all_detections.append(det)

    return all_detections


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


class RadarPillarNetEvaluator:
    """Full evaluation pipeline for the RadarPillarNet model."""

    def __init__(
        self,
        model: RadarPillarNet,
        config: Dict[str, Any],
        data_root: str,
        split: str = "val",
        score_threshold: float = 0.1,
        max_detections: int = 500,
    ) -> None:
        self.model = model
        self.config = config
        self.data_root = Path(data_root)
        self.split = split
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    def _load_sample_list(self) -> List[Dict[str, Any]]:
        """Load validation sample list."""
        info_path = self.data_root / f"nuscenes_infos_{self.split}.json"
        if info_path.exists():
            with open(str(info_path), "r") as f:
                data = json.load(f)
            return data.get("infos", [])
        return []

    def _load_ground_truth(self, sample_token: str) -> List[GroundTruth3D]:
        """Load ground truth annotations for a sample."""
        anno_path = self.data_root / "annotations" / f"{sample_token}.json"
        gts: List[GroundTruth3D] = []

        if anno_path.exists():
            with open(str(anno_path), "r") as f:
                annotations = json.load(f)
            for ann in annotations:
                gt = GroundTruth3D(
                    center=np.array(ann["center"], dtype=np.float64),
                    size=np.array(ann["size"], dtype=np.float64),
                    yaw=float(ann["yaw"]),
                    velocity=np.array(ann.get("velocity", [0.0, 0.0]), dtype=np.float64),
                    class_id=int(ann["class_id"]),
                    class_name=ann["class_name"],
                )
                gts.append(gt)
        return gts

    def _prepare_input(self, sample_info: Dict[str, Any]) -> Dict[str, tf.Tensor]:
        """Prepare model input tensors for a single sample."""
        max_pillars = self.config["max_pillars"]
        max_pts = self.config["max_points_per_pillar"]
        grid_x = self.config["grid_x"]
        grid_y = self.config["grid_y"]

        token = sample_info.get("token", "dummy")

        # Load radar pillar data
        radar_path = self.data_root / "radar_pillars" / f"{token}.npz"
        if radar_path.exists():
            radar_data = np.load(str(radar_path))
            pillar_features = np.zeros((1, max_pillars, max_pts, 9), dtype=np.float32)
            pillar_mask = np.zeros((1, max_pillars, max_pts), dtype=np.float32)
            pillar_coords = np.zeros((1, max_pillars, 2), dtype=np.int32)
            n = min(radar_data["features"].shape[0], max_pillars)
            pillar_features[0, :n] = radar_data["features"][:n]
            pillar_mask[0, :n] = radar_data["mask"][:n].astype(np.float32)
            pillar_coords[0, :n] = radar_data["coords"][:n]
        else:
            pillar_features = np.zeros((1, max_pillars, max_pts, 9), dtype=np.float32)
            pillar_mask = np.zeros((1, max_pillars, max_pts), dtype=np.float32)
            pillar_coords = np.zeros((1, max_pillars, 2), dtype=np.int32)

        return {
            "pillar_features": tf.constant(pillar_features, dtype=tf.float32),
            "pillar_mask": tf.constant(pillar_mask, dtype=tf.float32),
            "pillar_coords": tf.constant(pillar_coords, dtype=tf.int32),
        }

    def evaluate(self) -> EvalResults:
        """Run full evaluation on the validation set."""
        samples = self._load_sample_list()
        if not samples:
            print("[WARN] No validation samples found. Returning empty results.")
            return EvalResults()

        print(f"[INFO] Evaluating {len(samples)} samples...")

        all_detections: List[Detection3D] = []
        all_ground_truths: List[GroundTruth3D] = []

        for i, sample_info in enumerate(samples):
            token = sample_info.get("token", f"sample_{i}")

            # Prepare input
            inputs = self._prepare_input(sample_info)

            # Run inference and decode
            detections = decode_predictions_to_detections(
                self.model, inputs, self.config,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
            )
            all_detections.extend(detections)

            # Load ground truth
            gts = self._load_ground_truth(token)
            all_ground_truths.extend(gts)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(samples)} samples "
                      f"({len(all_detections)} dets, {len(all_ground_truths)} GTs)")

        # Compute per-class metrics
        results = EvalResults()
        all_aps: List[float] = []
        all_ates: List[float] = []
        all_ases: List[float] = []
        all_aoes: List[float] = []
        all_aves: List[float] = []

        print("\n[INFO] Computing per-class metrics...")
        print(f"{'Class':<25s} | {'AP':>6s} | {'ATE':>6s} | {'ASE':>6s} | {'AOE':>6s} | {'AVE':>6s}")
        print("-" * 75)

        for cls_name in NUSCENES_CLASSES:
            max_range = NUSCENES_DETECTION_RANGES.get(cls_name)
            ap, ate, ase, aoe, ave = compute_ap_for_class(
                all_detections, all_ground_truths, cls_name,
                max_range=max_range,
            )
            results.per_class_ap[cls_name] = ap
            results.per_class_ate[cls_name] = ate
            results.per_class_ase[cls_name] = ase
            results.per_class_aoe[cls_name] = aoe
            results.per_class_ave[cls_name] = ave

            all_aps.append(ap)
            all_ates.append(ate)
            all_ases.append(ase)
            all_aoes.append(aoe)
            all_aves.append(ave)

            print(f"  {cls_name:<25s} | {ap:6.4f} | {ate:6.4f} | {ase:6.4f} | {aoe:6.4f} | {ave:6.4f}")

        # Aggregate
        results.mAP = float(np.mean(all_aps))
        results.mATE = float(np.mean(all_ates))
        results.mASE = float(np.mean(all_ases))
        results.mAOE = float(np.mean(all_aoes))
        results.mAVE = float(np.mean(all_aves))
        results.NDS = compute_nds(results)

        # Distance-stratified evaluation
        print("\n[INFO] Distance-stratified evaluation...")
        print(f"{'Range (m)':<15s} | {'mAP':>6s}")
        print("-" * 30)

        for d_min, d_max in DISTANCE_RANGES:
            range_dets = [
                d for d in all_detections
                if d_min <= np.linalg.norm(d.center[:2]) < d_max
            ]
            range_gts = [
                g for g in all_ground_truths
                if d_min <= np.linalg.norm(g.center[:2]) < d_max
            ]

            if range_gts:
                range_aps = []
                for cls_name in NUSCENES_CLASSES:
                    ap, _, _, _, _ = compute_ap_for_class(
                        range_dets, range_gts, cls_name
                    )
                    range_aps.append(ap)
                range_map = float(np.mean(range_aps))
            else:
                range_map = 0.0

            range_key = f"{d_min:.0f}-{d_max:.0f}m"
            results.per_distance_ap[range_key] = range_map
            print(f"  {range_key:<15s} | {range_map:6.4f}")

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"{'RadarPillarNet Evaluation Results':^60}")
        print(f"{'=' * 60}")
        print(f"  NDS:  {results.NDS:.4f}")
        print(f"  mAP:  {results.mAP:.4f}")
        print(f"  mATE: {results.mATE:.4f} m")
        print(f"  mASE: {results.mASE:.4f}")
        print(f"  mAOE: {results.mAOE:.4f} rad")
        print(f"  mAVE: {results.mAVE:.4f} m/s")
        print(f"{'=' * 60}")

        return results


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def run_evaluation(
    checkpoint_path: str,
    data_root: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    split: str = "val",
    score_threshold: float = 0.1,
    max_detections: int = 500,
) -> EvalResults:
    """
    Run model evaluation.

    Args:
        checkpoint_path: path to model weights or checkpoint dir
        data_root: nuScenes data root
        output_path: path to save evaluation results JSON
        config: model config overrides
        split: data split to evaluate on
        score_threshold: detection confidence threshold
        max_detections: max detections per sample
    Returns:
        EvalResults with all metrics
    """
    model_config = {**DEFAULT_CONFIG, **(config or {})}

    # Build model
    print("[INFO] Building RadarPillarNet model...")
    model = build_radar_pillarnet(config=model_config)

    # Build model with dummy forward pass
    dummy_inputs = {
        "pillar_features": tf.zeros([1, model_config["max_pillars"], model_config["max_points_per_pillar"], 9]),
        "pillar_mask": tf.zeros([1, model_config["max_pillars"], model_config["max_points_per_pillar"]]),
        "pillar_coords": tf.zeros([1, model_config["max_pillars"], 2], dtype=tf.int32),
    }
    _ = model(dummy_inputs, training=False)
    print(f"[INFO] Model built: {model.count_params():,} parameters")

    # Load weights
    print(f"[INFO] Loading weights from: {checkpoint_path}")
    if checkpoint_path.endswith(".h5") or checkpoint_path.endswith(".weights.h5"):
        model.load_weights(checkpoint_path)
    elif os.path.isdir(checkpoint_path):
        checkpoint = tf.train.Checkpoint(model=model)
        latest = tf.train.latest_checkpoint(checkpoint_path)
        if latest:
            checkpoint.restore(latest).expect_partial()
            print(f"[INFO] Restored checkpoint: {latest}")
        elif os.path.exists(os.path.join(checkpoint_path, "saved_model.pb")):
            print("[WARN] SavedModel detected; loading via checkpoint is preferred.")
            model.load_weights(os.path.join(checkpoint_path, "variables", "variables"))
        else:
            raise FileNotFoundError(f"No valid checkpoint found at: {checkpoint_path}")
    else:
        model.load_weights(checkpoint_path)

    # Run evaluation
    evaluator = RadarPillarNetEvaluator(
        model=model,
        config=model_config,
        data_root=data_root,
        split=split,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    start_time = time.time()
    results = evaluator.evaluate()
    eval_time = time.time() - start_time
    print(f"\n[INFO] Evaluation completed in {eval_time:.1f}s")

    # Save results
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results_dict = results.to_dict()
    results_dict["evaluation_time_seconds"] = eval_time
    results_dict["checkpoint"] = checkpoint_path
    results_dict["split"] = split
    results_dict["score_threshold"] = score_threshold

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[INFO] Results saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RadarPillarNet on nuScenes")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights or checkpoint dir")
    parser.add_argument("--data-root", type=str, default="/data/nuscenes", help="nuScenes data root")
    parser.add_argument("--output", type=str, default="./eval_results/radar_pillarnet_results.json",
                        help="Output results path")
    parser.add_argument("--split", type=str, default="val", help="Evaluation split")
    parser.add_argument("--score-threshold", type=float, default=0.1, help="Detection score threshold")
    parser.add_argument("--max-detections", type=int, default=500, help="Max detections per sample")
    parser.add_argument("--config", type=str, default=None, help="Path to model config JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load external config
    model_config = None
    if args.config:
        with open(args.config, "r") as f:
            ext = json.load(f)
        model_config = ext.get("model", ext)

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_path=args.output,
        config=model_config,
        split=args.split,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
    )
