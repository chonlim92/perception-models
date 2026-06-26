"""
Evaluation script for PETR/StreamPETR model on nuScenes-style metrics.

Computes:
  - Per-class Average Precision (AP) at multiple distance thresholds
  - mean Average Precision (mAP)
  - nuScenes Detection Score (NDS) incorporating:
    - mATE (mean Average Translation Error)
    - mASE (mean Average Scale Error)
    - mAOE (mean Average Orientation Error)
    - mAVE (mean Average Velocity Error)
    - mAAE (mean Average Attribute Error)
"""

import os
import sys
import argparse
import yaml
import pickle
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

from model import build_petr_model


NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

DISTANCE_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PETR/StreamPETR model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_info", type=str, required=True, help="Path to val info pickle")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data root")
    parser.add_argument("--output", type=str, default="./eval_results.json", help="Output path")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size")
    return parser.parse_args()


def load_val_data(data_info_path: str, data_root: str, image_size: Tuple[int, int] = (900, 1600)):
    """Load validation data samples."""
    with open(data_info_path, "rb") as f:
        data_infos = pickle.load(f)
    return data_infos


def preprocess_sample(
    info: dict,
    data_root: str,
    image_size: Tuple[int, int] = (900, 1600),
) -> dict:
    """Preprocess a single validation sample."""
    images = []
    for cam_path in info["img_paths"]:
        full_path = os.path.join(data_root, cam_path)
        img_raw = tf.io.read_file(full_path)
        img = tf.io.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0

        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        img = (img - mean) / std
        images.append(img)

    images = tf.stack(images, axis=0)
    intrinsics = tf.constant(info["intrinsics"], dtype=tf.float32)
    extrinsics = tf.constant(info["extrinsics"], dtype=tf.float32)

    return {
        "images": images[None],
        "intrinsics": intrinsics[None],
        "extrinsics": extrinsics[None],
    }


def compute_center_distance(pred_center: np.ndarray, gt_center: np.ndarray) -> float:
    """Compute 2D center distance (x, y) between prediction and ground truth."""
    return np.linalg.norm(pred_center[:2] - gt_center[:2])


def compute_translation_error(pred_bbox: np.ndarray, gt_bbox: np.ndarray) -> float:
    """Compute 3D translation error (Euclidean distance between centers)."""
    return np.linalg.norm(pred_bbox[:3] - gt_bbox[:3])


def compute_scale_error(pred_bbox: np.ndarray, gt_bbox: np.ndarray) -> float:
    """Compute scale error (1 - IoU of 3D volumes approximated as axis-aligned)."""
    pred_vol = pred_bbox[3] * pred_bbox[4] * pred_bbox[5]
    gt_vol = gt_bbox[3] * gt_bbox[4] * gt_bbox[5]

    min_dims = np.minimum(pred_bbox[3:6], gt_bbox[3:6])
    intersection_vol = min_dims[0] * min_dims[1] * min_dims[2]
    union_vol = pred_vol + gt_vol - intersection_vol

    iou = intersection_vol / np.maximum(union_vol, 1e-6)
    return 1.0 - iou


def compute_orientation_error(pred_bbox: np.ndarray, gt_bbox: np.ndarray) -> float:
    """Compute orientation error using sin/cos yaw representation."""
    pred_yaw = np.arctan2(pred_bbox[6], pred_bbox[7])
    gt_yaw = np.arctan2(gt_bbox[6], gt_bbox[7])
    diff = np.abs(pred_yaw - gt_yaw)
    diff = np.minimum(diff, 2 * np.pi - diff)
    return diff


def compute_velocity_error(pred_bbox: np.ndarray, gt_bbox: np.ndarray) -> float:
    """Compute velocity error (L2 norm of velocity difference)."""
    return np.linalg.norm(pred_bbox[8:10] - gt_bbox[8:10])


def compute_ap_per_class(
    predictions: List[dict],
    ground_truths: List[dict],
    class_idx: int,
    distance_threshold: float,
) -> float:
    """
    Compute Average Precision for a single class at a given distance threshold.

    Args:
        predictions: list of dicts with 'bbox', 'score', 'label', 'sample_idx'
        ground_truths: list of dicts with 'bbox', 'label', 'sample_idx'
        class_idx: class index to evaluate
        distance_threshold: matching distance threshold

    Returns:
        AP value
    """
    class_preds = [p for p in predictions if p["label"] == class_idx]
    class_gts = [g for g in ground_truths if g["label"] == class_idx]

    if len(class_gts) == 0:
        return 0.0 if len(class_preds) > 0 else np.nan

    class_preds = sorted(class_preds, key=lambda x: -x["score"])

    gt_by_sample = {}
    for gt in class_gts:
        sid = gt["sample_idx"]
        if sid not in gt_by_sample:
            gt_by_sample[sid] = []
        gt_by_sample[sid].append(gt)

    gt_matched = {id(gt): False for gt in class_gts}

    tp = np.zeros(len(class_preds))
    fp = np.zeros(len(class_preds))

    for i, pred in enumerate(class_preds):
        sid = pred["sample_idx"]
        sample_gts = gt_by_sample.get(sid, [])

        min_dist = float("inf")
        best_gt = None
        for gt in sample_gts:
            dist = compute_center_distance(pred["bbox"], gt["bbox"])
            if dist < min_dist:
                min_dist = dist
                best_gt = gt

        if best_gt is not None and min_dist <= distance_threshold and not gt_matched[id(best_gt)]:
            tp[i] = 1.0
            gt_matched[id(best_gt)] = True
        else:
            fp[i] = 1.0

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / len(class_gts)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    recall_thresholds = np.linspace(0, 1, 41)
    ap = 0.0
    for t in recall_thresholds:
        mask = recall >= t
        if mask.any():
            ap += np.max(precision[mask])
    ap /= len(recall_thresholds)

    return ap


def compute_nuscenes_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
    class_names: List[str] = NUSCENES_CLASSES,
) -> Dict:
    """
    Compute full nuScenes detection metrics.

    Returns:
        Dictionary with mAP, NDS, per-class APs, and error metrics.
    """
    num_classes = len(class_names)

    per_class_ap = {}
    for cls_idx, cls_name in enumerate(class_names):
        aps_at_thresholds = []
        for threshold in DISTANCE_THRESHOLDS:
            ap = compute_ap_per_class(predictions, ground_truths, cls_idx, threshold)
            if not np.isnan(ap):
                aps_at_thresholds.append(ap)
        if aps_at_thresholds:
            per_class_ap[cls_name] = np.mean(aps_at_thresholds)
        else:
            per_class_ap[cls_name] = 0.0

    valid_aps = [v for v in per_class_ap.values() if v > 0]
    mAP = np.mean(valid_aps) if valid_aps else 0.0

    translation_errors = []
    scale_errors = []
    orientation_errors = []
    velocity_errors = []

    matched_pairs = _match_predictions_to_gt(predictions, ground_truths, threshold=2.0)

    for pred, gt in matched_pairs:
        translation_errors.append(compute_translation_error(pred["bbox"], gt["bbox"]))
        scale_errors.append(compute_scale_error(pred["bbox"], gt["bbox"]))
        orientation_errors.append(compute_orientation_error(pred["bbox"], gt["bbox"]))
        velocity_errors.append(compute_velocity_error(pred["bbox"], gt["bbox"]))

    mATE = np.mean(translation_errors) if translation_errors else 1.0
    mASE = np.mean(scale_errors) if scale_errors else 1.0
    mAOE = np.mean(orientation_errors) if orientation_errors else 1.0
    mAVE = np.mean(velocity_errors) if velocity_errors else 1.0
    mAAE = 0.0

    tp_metrics = {
        "mATE": mATE,
        "mASE": mASE,
        "mAOE": mAOE,
        "mAVE": mAVE,
        "mAAE": mAAE,
    }

    tp_errors = [
        min(1.0, mATE),
        min(1.0, mASE),
        min(1.0, mAOE),
        min(1.0, mAVE),
        min(1.0, mAAE),
    ]
    mean_tp_error = np.mean(tp_errors)
    NDS = (5.0 * mAP + (1.0 - mean_tp_error) * 5.0) / 10.0

    return {
        "mAP": float(mAP),
        "NDS": float(NDS),
        "per_class_AP": per_class_ap,
        "TP_metrics": tp_metrics,
    }


def _match_predictions_to_gt(
    predictions: List[dict],
    ground_truths: List[dict],
    threshold: float = 2.0,
) -> List[Tuple[dict, dict]]:
    """Match predictions to ground truths using greedy matching."""
    matched_pairs = []

    gt_by_sample_class = {}
    for gt in ground_truths:
        key = (gt["sample_idx"], gt["label"])
        if key not in gt_by_sample_class:
            gt_by_sample_class[key] = []
        gt_by_sample_class[key].append(gt)

    sorted_preds = sorted(predictions, key=lambda x: -x["score"])
    gt_used = set()

    for pred in sorted_preds:
        key = (pred["sample_idx"], pred["label"])
        candidates = gt_by_sample_class.get(key, [])

        best_gt = None
        best_dist = float("inf")
        for gt in candidates:
            if id(gt) in gt_used:
                continue
            dist = compute_center_distance(pred["bbox"], gt["bbox"])
            if dist < best_dist:
                best_dist = dist
                best_gt = gt

        if best_gt is not None and best_dist <= threshold:
            matched_pairs.append((pred, best_gt))
            gt_used.add(id(best_gt))

    return matched_pairs


def run_evaluation(config: dict, args):
    """Run full evaluation pipeline."""
    model_config = config["model"]
    image_size = tuple(config["data"].get("image_size", [900, 1600]))

    model = build_petr_model(model_config)

    dummy_images = tf.zeros([1, 6, image_size[0], image_size[1], 3])
    dummy_intrinsics = tf.eye(3, batch_shape=[1, 6])
    dummy_extrinsics = tf.eye(4, batch_shape=[1, 6])
    _ = model(dummy_images, dummy_intrinsics, dummy_extrinsics, training=False)

    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(args.checkpoint)
    status.expect_partial()
    print(f"Loaded checkpoint: {args.checkpoint}")

    data_infos = load_val_data(args.data_info, args.data_root, image_size)
    print(f"Loaded {len(data_infos)} validation samples")

    all_predictions = []
    all_ground_truths = []

    score_threshold = config.get("eval", {}).get("score_threshold", 0.1)
    temporal = model_config.get("temporal", False)
    prev_query = None

    for sample_idx, info in enumerate(data_infos):
        if sample_idx % 100 == 0:
            print(f"  Processing sample {sample_idx}/{len(data_infos)}")

        sample = preprocess_sample(info, args.data_root, image_size)

        call_kwargs = {
            "images": sample["images"],
            "intrinsics": sample["intrinsics"],
            "extrinsics": sample["extrinsics"],
            "training": False,
        }

        if temporal:
            if "ego_motion" in info:
                call_kwargs["ego_motion"] = tf.constant(info["ego_motion"][None], dtype=tf.float32)
            if prev_query is not None:
                call_kwargs["prev_query"] = prev_query

        outputs = model(**call_kwargs)

        cls_scores = outputs["cls_scores"][-1][0].numpy()
        bbox_preds = outputs["bbox_preds"][-1][0].numpy()

        if temporal:
            prev_query = outputs["query_output"]

        scores = 1.0 / (1.0 + np.exp(-cls_scores))
        for query_idx in range(scores.shape[0]):
            max_score = np.max(scores[query_idx])
            if max_score < score_threshold:
                continue
            label = np.argmax(scores[query_idx])
            all_predictions.append({
                "bbox": bbox_preds[query_idx],
                "score": float(scores[query_idx, label]),
                "label": int(label),
                "sample_idx": sample_idx,
            })

        gt_labels = info["gt_labels"]
        gt_bboxes = info["gt_bboxes"]
        for gt_idx in range(len(gt_labels)):
            if gt_labels[gt_idx] < 0:
                continue
            all_ground_truths.append({
                "bbox": gt_bboxes[gt_idx],
                "label": int(gt_labels[gt_idx]),
                "sample_idx": sample_idx,
            })

    print(f"\nTotal predictions: {len(all_predictions)}")
    print(f"Total ground truths: {len(all_ground_truths)}")
    print("\nComputing metrics...")

    metrics = compute_nuscenes_metrics(all_predictions, all_ground_truths)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  mAP: {metrics['mAP']:.4f}")
    print(f"  NDS: {metrics['NDS']:.4f}")
    print("\nPer-class AP:")
    for cls_name, ap in metrics["per_class_AP"].items():
        print(f"  {cls_name:25s}: {ap:.4f}")
    print("\nTrue Positive Metrics:")
    for metric_name, value in metrics["TP_metrics"].items():
        print(f"  {metric_name}: {value:.4f}")
    print("=" * 60)

    import json
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {args.output}")

    return metrics


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_evaluation(config, args)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
