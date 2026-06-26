"""
CRAFT model evaluation script.

Computes nuScenes Detection Score (NDS) metrics:
- mAP (mean Average Precision)
- mATE (mean Average Translation Error)
- mASE (mean Average Scale Error)
- mAOE (mean Average Orientation Error)
- mAVE (mean Average Velocity Error)

Supports:
- Per-class results
- Modality ablation (camera-only, radar-only, fused)
- Saving results to JSON
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

from model import CRAFTModel, DEFAULT_CONFIG, build_craft_model


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


# ---------------------------------------------------------------------------
# Data structures for evaluation
# ---------------------------------------------------------------------------


@dataclass
class Detection3D:
    """A single 3D detection."""

    center: np.ndarray  # (3,) x, y, z in global frame
    size: np.ndarray  # (3,) w, l, h
    yaw: float  # rotation angle in radians
    velocity: np.ndarray  # (2,) vx, vy
    score: float  # confidence score
    class_id: int  # class index
    class_name: str  # class string


@dataclass
class GroundTruth3D:
    """A single ground truth 3D box."""

    center: np.ndarray
    size: np.ndarray
    yaw: float
    velocity: np.ndarray
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
        }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def center_distance(det: Detection3D, gt: GroundTruth3D) -> float:
    """2D center distance in BEV."""
    return float(np.linalg.norm(det.center[:2] - gt.center[:2]))


def scale_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """1 - IoU of 3D aligned boxes (approximated as 1 - vol_intersection/vol_union)."""
    # Simplified: use ratio of volumes
    det_vol = float(np.prod(det.size))
    gt_vol = float(np.prod(gt.size))
    if det_vol <= 0 or gt_vol <= 0:
        return 1.0
    # Approximate IoU by comparing sizes directly
    min_size = np.minimum(det.size, gt.size)
    intersection_vol = float(np.prod(min_size))
    union_vol = det_vol + gt_vol - intersection_vol
    iou = intersection_vol / max(union_vol, 1e-8)
    return 1.0 - iou


def orientation_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """Absolute angular difference in yaw, mapped to [0, pi]."""
    diff = abs(det.yaw - gt.yaw)
    diff = diff % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return float(diff)


def velocity_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """L2 velocity error in m/s."""
    return float(np.linalg.norm(det.velocity - gt.velocity))


def compute_ap(
    detections: List[Detection3D],
    ground_truths: List[GroundTruth3D],
    class_name: str,
    distance_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
) -> Tuple[float, float, float, float, float]:
    """
    Compute Average Precision and true-positive metrics for a single class.

    Args:
        detections: sorted by score (descending)
        ground_truths: all GT boxes of this class
        class_name: class name for filtering
        distance_thresholds: BEV center distance thresholds for TP matching
    Returns:
        (ap, ate, ase, aoe, ave) for this class
    """
    # Filter detections and GT by class
    class_dets = [d for d in detections if d.class_name == class_name]
    class_gts = [g for g in ground_truths if g.class_name == class_name]

    if not class_gts:
        return 0.0, 1.0, 1.0, np.pi, 1.0

    # Sort detections by confidence
    class_dets.sort(key=lambda x: x.score, reverse=True)

    # Compute AP at each distance threshold
    aps_per_threshold = []
    all_ate = []
    all_ase = []
    all_aoe = []
    all_ave = []

    for dist_thresh in distance_thresholds:
        matched_gt = set()
        tp_list = []
        fp_list = []

        for det in class_dets:
            # Find best matching GT
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

                # Compute TP metrics
                gt = class_gts[best_gt_idx]
                all_ate.append(center_distance(det, gt))
                all_ase.append(scale_error(det, gt))
                all_aoe.append(orientation_error(det, gt))
                all_ave.append(velocity_error(det, gt))
            else:
                tp_list.append(0)
                fp_list.append(1)

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        recall = tp_cumsum / len(class_gts)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for r_thresh in np.linspace(0, 1, 11):
            prec_at_recall = precision[recall >= r_thresh]
            if len(prec_at_recall) > 0:
                ap += np.max(prec_at_recall)
        ap /= 11.0
        aps_per_threshold.append(ap)

    # Average AP across distance thresholds
    mean_ap = float(np.mean(aps_per_threshold))

    # Average TP metrics
    ate = float(np.mean(all_ate)) if all_ate else 1.0
    ase = float(np.mean(all_ase)) if all_ase else 1.0
    aoe = float(np.mean(all_aoe)) if all_aoe else float(np.pi)
    ave = float(np.mean(all_ave)) if all_ave else 1.0

    return mean_ap, ate, ase, aoe, ave


def compute_nds(results: EvalResults) -> float:
    """
    Compute nuScenes Detection Score.
    NDS = 1/10 * [5*mAP + sum(1 - min(1, metric)) for metric in [mATE, mASE, mAOE, mAVE, mAAE]]
    Simplified: NDS = (5*mAP + TP_score) / 10
    """
    tp_scores = [
        max(1.0 - results.mATE, 0.0),
        max(1.0 - results.mASE, 0.0),
        max(1.0 - results.mAOE / np.pi, 0.0),
        max(1.0 - results.mAVE, 0.0),
        1.0,  # mAAE placeholder (attribute error)
    ]
    nds = (5.0 * results.mAP + sum(tp_scores)) / 10.0
    return float(nds)


# ---------------------------------------------------------------------------
# Post-processing: decode heatmap to detections
# ---------------------------------------------------------------------------


def decode_predictions(
    predictions: Dict[str, tf.Tensor],
    config: Dict[str, Any],
    score_threshold: float = 0.1,
    max_detections: int = 500,
) -> List[List[Detection3D]]:
    """
    Decode network outputs to 3D detections.

    Args:
        predictions: model output dict (heatmap, regression, velocity, height)
        config: model config
        score_threshold: minimum confidence
        max_detections: maximum detections per sample
    Returns:
        List of detection lists, one per batch element
    """
    heatmap = predictions["heatmap"].numpy()  # (B, H, W, C)
    regression = predictions["regression"].numpy()
    velocity = predictions["velocity"].numpy()
    height = predictions["height"].numpy()

    batch_size = heatmap.shape[0]
    h, w = heatmap.shape[1], heatmap.shape[2]
    num_classes = heatmap.shape[3]

    # BEV grid parameters
    x_min = config["x_min"]
    x_max = config["x_max"]
    y_min = config["y_min"]
    y_max = config["y_max"]
    x_res = (x_max - x_min) / w
    y_res = (y_max - y_min) / h

    all_detections: List[List[Detection3D]] = []

    for b in range(batch_size):
        detections: List[Detection3D] = []

        for cls_id in range(num_classes):
            cls_heatmap = heatmap[b, :, :, cls_id]

            # Simple NMS: 3x3 max pooling
            cls_heatmap_tensor = tf.constant(cls_heatmap[np.newaxis, :, :, np.newaxis])
            pooled = tf.nn.max_pool2d(cls_heatmap_tensor, ksize=3, strides=1, padding="SAME")
            nms_mask = (cls_heatmap_tensor == pooled).numpy()[0, :, :, 0]
            cls_heatmap = cls_heatmap * nms_mask

            # Get top-k scores
            flat_scores = cls_heatmap.flatten()
            top_k = min(max_detections, (flat_scores > score_threshold).sum())
            if top_k == 0:
                continue

            top_indices = np.argsort(flat_scores)[-top_k:][::-1]

            for idx in top_indices:
                score = flat_scores[idx]
                if score < score_threshold:
                    break

                yi, xi = np.unravel_index(idx, (h, w))

                # Decode center position
                reg = regression[b, yi, xi]  # (num_reg_attrs,)
                dx, dy = reg[0], reg[1]
                dz = reg[2]
                bw, bl, bh = reg[3], reg[4], reg[5]
                sin_yaw, cos_yaw = reg[6], reg[7]

                # Convert grid position to world coordinates
                cx = x_min + (xi + dx) * x_res
                cy = y_min + (yi + dy) * y_res
                cz = height[b, yi, xi, 0]  # z_center from height head
                box_h = height[b, yi, xi, 1]

                yaw = np.arctan2(sin_yaw, cos_yaw)

                vel = velocity[b, yi, xi]  # (2,)

                det = Detection3D(
                    center=np.array([cx, cy, cz], dtype=np.float64),
                    size=np.array([bw, bl, box_h], dtype=np.float64),
                    yaw=float(yaw),
                    velocity=vel.astype(np.float64),
                    score=float(score),
                    class_id=cls_id,
                    class_name=NUSCENES_CLASSES[cls_id],
                )
                detections.append(det)

        # Sort by score and limit
        detections.sort(key=lambda d: d.score, reverse=True)
        detections = detections[:max_detections]
        all_detections.append(detections)

    return all_detections


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


class CRAFTEvaluator:
    """Full evaluation pipeline for the CRAFT model."""

    def __init__(
        self,
        model: CRAFTModel,
        config: Dict[str, Any],
        data_root: str,
        split: str = "val",
        score_threshold: float = 0.1,
        max_detections: int = 500,
        modality: str = "fused",  # "camera", "radar", "fused"
    ) -> None:
        self.model = model
        self.config = config
        self.data_root = Path(data_root)
        self.split = split
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.modality = modality

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

    def _load_sample_list(self) -> List[Dict[str, Any]]:
        """Load validation sample list."""
        info_path = self.data_root / f"nuscenes_infos_{self.split}.json"
        if info_path.exists():
            with open(str(info_path), "r") as f:
                data = json.load(f)
            return data.get("infos", [])
        return []

    def _prepare_input(self, sample_info: Dict[str, Any]) -> Dict[str, tf.Tensor]:
        """Prepare model input tensors for a single sample."""
        num_cameras = self.config["num_cameras"]
        img_h = self.config["image_height"]
        img_w = self.config["image_width"]
        max_pillars = self.config["max_pillars"]
        max_pts = self.config["max_points_per_pillar"]

        # Load images
        images = np.zeros((1, num_cameras, img_h, img_w, 3), dtype=np.float32)
        cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                     "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

        for cam_idx, cam_name in enumerate(cam_names):
            img_path = self.data_root / "samples" / cam_name / f"{sample_info.get('token', '')}.jpg"
            if img_path.exists():
                img_raw = tf.io.read_file(str(img_path))
                img = tf.image.decode_jpeg(img_raw, channels=3)
                img = tf.image.resize(img, [img_h, img_w])
                img = tf.cast(img, tf.float32) / 255.0
                mean = tf.constant([0.485, 0.456, 0.406])
                std = tf.constant([0.229, 0.224, 0.225])
                img = (img - mean) / std
                images[0, cam_idx] = img.numpy()

        # Load radar
        radar_path = self.data_root / "radar_pillars" / f"{sample_info.get('token', '')}.npz"
        if radar_path.exists():
            radar_data = np.load(str(radar_path))
            pillar_features = np.zeros((1, max_pillars, max_pts, 9), dtype=np.float32)
            pillar_mask = np.zeros((1, max_pillars, max_pts), dtype=np.float32)
            pillar_coords = np.zeros((1, max_pillars, 2), dtype=np.int32)
            n = min(radar_data["features"].shape[0], max_pillars)
            pillar_features[0, :n] = radar_data["features"][:n]
            pillar_mask[0, :n] = radar_data["mask"][:n]
            pillar_coords[0, :n] = radar_data["coords"][:n]
        else:
            pillar_features = np.zeros((1, max_pillars, max_pts, 9), dtype=np.float32)
            pillar_mask = np.zeros((1, max_pillars, max_pts), dtype=np.float32)
            pillar_coords = np.zeros((1, max_pillars, 2), dtype=np.int32)

        # Calibration
        lidar_to_cam = np.eye(4, dtype=np.float32)[np.newaxis, np.newaxis].repeat(num_cameras, axis=1)
        cam_intrinsics = np.eye(3, dtype=np.float32)[np.newaxis, np.newaxis].repeat(num_cameras, axis=1)
        cam_intrinsics[0, :, 0, 0] = 1266.0
        cam_intrinsics[0, :, 1, 1] = 1266.0
        cam_intrinsics[0, :, 0, 2] = img_w / 2.0
        cam_intrinsics[0, :, 1, 2] = img_h / 2.0

        # Load calibration from file if available
        calib_path = self.data_root / "calibration" / f"{sample_info.get('token', '')}.npz"
        if calib_path.exists():
            calib = np.load(str(calib_path))
            lidar_to_cam = calib["lidar_to_cam"][np.newaxis]
            cam_intrinsics = calib["cam_intrinsics"][np.newaxis]

        # Modality ablation: zero out unused modality
        if self.modality == "camera":
            pillar_features = np.zeros_like(pillar_features)
            pillar_mask = np.zeros_like(pillar_mask)
        elif self.modality == "radar":
            images = np.zeros_like(images)

        return {
            "images": tf.constant(images, dtype=tf.float32),
            "radar_pillars": tf.constant(pillar_features, dtype=tf.float32),
            "radar_pillar_mask": tf.constant(pillar_mask, dtype=tf.float32),
            "radar_pillar_coords": tf.constant(pillar_coords, dtype=tf.int32),
            "lidar_to_cam": tf.constant(lidar_to_cam, dtype=tf.float32),
            "cam_intrinsics": tf.constant(cam_intrinsics, dtype=tf.float32),
        }

    def evaluate(self) -> EvalResults:
        """Run full evaluation on the validation set."""
        samples = self._load_sample_list()
        if not samples:
            print("[WARN] No validation samples found. Returning empty results.")
            return EvalResults()

        print(f"[INFO] Evaluating {len(samples)} samples (modality: {self.modality})...")

        all_detections: List[Detection3D] = []
        all_ground_truths: List[GroundTruth3D] = []

        for i, sample_info in enumerate(samples):
            token = sample_info.get("token", f"sample_{i}")

            # Prepare input
            inputs = self._prepare_input(sample_info)

            # Run inference
            predictions = self.model(inputs, training=False)

            # Decode predictions
            batch_dets = decode_predictions(
                predictions, self.config,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
            )
            all_detections.extend(batch_dets[0])

            # Load ground truth
            gts = self._load_ground_truth(token)
            all_ground_truths.extend(gts)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(samples)} samples")

        # Compute per-class metrics
        results = EvalResults()
        all_aps = []
        all_ates = []
        all_ases = []
        all_aoes = []
        all_aves = []

        print("\n[INFO] Computing metrics per class...")
        for cls_name in NUSCENES_CLASSES:
            ap, ate, ase, aoe, ave = compute_ap(
                all_detections, all_ground_truths, cls_name
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

            print(f"  {cls_name:25s} | AP: {ap:.4f} | ATE: {ate:.4f} | ASE: {ase:.4f} | AOE: {aoe:.4f} | AVE: {ave:.4f}")

        # Aggregate
        results.mAP = float(np.mean(all_aps))
        results.mATE = float(np.mean(all_ates))
        results.mASE = float(np.mean(all_ases))
        results.mAOE = float(np.mean(all_aoes))
        results.mAVE = float(np.mean(all_aves))
        results.NDS = compute_nds(results)

        print(f"\n{'='*60}")
        print(f"  NDS:  {results.NDS:.4f}")
        print(f"  mAP:  {results.mAP:.4f}")
        print(f"  mATE: {results.mATE:.4f}")
        print(f"  mASE: {results.mASE:.4f}")
        print(f"  mAOE: {results.mAOE:.4f}")
        print(f"  mAVE: {results.mAVE:.4f}")
        print(f"{'='*60}")

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
    modality: str = "fused",
    use_ema: bool = True,
) -> EvalResults:
    """
    Run model evaluation.

    Args:
        checkpoint_path: path to model weights (.h5 or SavedModel dir)
        data_root: nuScenes data root
        output_path: path to save evaluation results JSON
        config: model config overrides
        split: data split to evaluate on
        score_threshold: detection confidence threshold
        max_detections: max detections per sample
        modality: "fused", "camera", or "radar"
        use_ema: whether to load EMA weights
    Returns:
        EvalResults with all metrics
    """
    model_config = {**DEFAULT_CONFIG, **(config or {})}

    # Build model
    print("[INFO] Building CRAFT model...")
    model = build_craft_model(config=model_config)

    # Build model by running a dummy forward pass
    dummy_inputs = {
        "images": tf.zeros([1, model_config["num_cameras"], model_config["image_height"], model_config["image_width"], 3]),
        "radar_pillars": tf.zeros([1, model_config["max_pillars"], model_config["max_points_per_pillar"], 9]),
        "radar_pillar_mask": tf.zeros([1, model_config["max_pillars"], model_config["max_points_per_pillar"]]),
        "radar_pillar_coords": tf.zeros([1, model_config["max_pillars"], 2], dtype=tf.int32),
        "lidar_to_cam": tf.zeros([1, model_config["num_cameras"], 4, 4]),
        "cam_intrinsics": tf.zeros([1, model_config["num_cameras"], 3, 3]),
    }
    _ = model(dummy_inputs, training=False)

    # Load weights
    print(f"[INFO] Loading weights from: {checkpoint_path}")
    if checkpoint_path.endswith(".h5") or checkpoint_path.endswith(".weights.h5"):
        model.load_weights(checkpoint_path)
    elif os.path.isdir(checkpoint_path):
        # Try loading as checkpoint directory
        checkpoint = tf.train.Checkpoint(model=model)
        latest = tf.train.latest_checkpoint(checkpoint_path)
        if latest:
            checkpoint.restore(latest).expect_partial()
            print(f"[INFO] Restored checkpoint: {latest}")
        else:
            # Try as SavedModel
            loaded = tf.saved_model.load(checkpoint_path)
            print("[INFO] Loaded as SavedModel")
    else:
        model.load_weights(checkpoint_path)

    print(f"[INFO] Model loaded with {model.count_params():,} parameters")

    # Run evaluation
    evaluator = CRAFTEvaluator(
        model=model,
        config=model_config,
        data_root=data_root,
        split=split,
        score_threshold=score_threshold,
        max_detections=max_detections,
        modality=modality,
    )

    start_time = time.time()
    results = evaluator.evaluate()
    eval_time = time.time() - start_time
    print(f"\n[INFO] Evaluation completed in {eval_time:.1f}s")

    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    results_dict = results.to_dict()
    results_dict["evaluation_time_seconds"] = eval_time
    results_dict["modality"] = modality
    results_dict["checkpoint"] = checkpoint_path
    results_dict["split"] = split

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[INFO] Results saved to: {output_path}")

    return results


def run_modality_ablation(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    split: str = "val",
) -> Dict[str, EvalResults]:
    """
    Run evaluation with modality ablation: camera-only, radar-only, and fused.

    Returns:
        Dict mapping modality name to results
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results: Dict[str, EvalResults] = {}

    for modality in ["fused", "camera", "radar"]:
        print(f"\n{'#'*60}")
        print(f"# Modality ablation: {modality}")
        print(f"{'#'*60}\n")

        output_path = os.path.join(output_dir, f"results_{modality}.json")
        results = run_evaluation(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            output_path=output_path,
            config=config,
            split=split,
            modality=modality,
        )
        all_results[modality] = results

    # Print comparison
    print(f"\n{'='*70}")
    print(f"{'Modality Ablation Comparison':^70}")
    print(f"{'='*70}")
    print(f"{'Modality':<12} {'NDS':<8} {'mAP':<8} {'mATE':<8} {'mASE':<8} {'mAOE':<8} {'mAVE':<8}")
    print(f"{'-'*70}")
    for mod, res in all_results.items():
        print(f"{mod:<12} {res.NDS:<8.4f} {res.mAP:<8.4f} {res.mATE:<8.4f} {res.mASE:<8.4f} {res.mAOE:<8.4f} {res.mAVE:<8.4f}")
    print(f"{'='*70}")

    # Save summary
    summary_path = os.path.join(output_dir, "ablation_summary.json")
    summary = {mod: res.to_dict() for mod, res in all_results.items()}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Ablation summary saved to: {summary_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CRAFT model on nuScenes")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="/data/nuscenes", help="nuScenes data root")
    parser.add_argument("--output", type=str, default="./eval_results/results.json", help="Output results path")
    parser.add_argument("--split", type=str, default="val", help="Evaluation split")
    parser.add_argument("--score-threshold", type=float, default=0.1, help="Detection score threshold")
    parser.add_argument("--max-detections", type=int, default=500, help="Max detections per sample")
    parser.add_argument("--modality", type=str, default="fused", choices=["fused", "camera", "radar"],
                        help="Modality to evaluate")
    parser.add_argument("--ablation", action="store_true", help="Run full modality ablation study")
    parser.add_argument("--config", type=str, default=None, help="Path to model config JSON")
    parser.add_argument("--no-ema", action="store_true", help="Do not use EMA weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load external config
    model_config = None
    if args.config:
        with open(args.config, "r") as f:
            ext = json.load(f)
        model_config = ext.get("model", ext)

    if args.ablation:
        output_dir = os.path.dirname(args.output) or "./eval_results"
        run_modality_ablation(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            output_dir=output_dir,
            config=model_config,
            split=args.split,
        )
    else:
        run_evaluation(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            output_path=args.output,
            config=model_config,
            split=args.split,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            modality=args.modality,
            use_ema=not args.no_ema,
        )
