"""
Inference script for PETR/StreamPETR model.

Supports:
  - Loading from SavedModel or checkpoint
  - Processing multi-view images
  - NMS post-processing
  - Batch inference
  - Outputting 3D bounding boxes with class labels and confidence scores
"""

import os
import argparse
import yaml
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple

from model import build_petr_model


NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with PETR/StreamPETR")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SavedModel dir or checkpoint prefix")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input data (pickle info file or directory of images)")
    parser.add_argument("--output", type=str, default="./inference_results.pkl",
                        help="Path to save results")
    parser.add_argument("--score_threshold", type=float, default=0.3,
                        help="Minimum confidence score threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="NMS IoU threshold for BEV")
    parser.add_argument("--max_detections", type=int, default=300,
                        help="Maximum number of detections per frame")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(
    config: dict,
    model_path: str,
    image_size: Tuple[int, int] = (900, 1600),
) -> tf.keras.Model:
    """
    Load model from SavedModel directory or checkpoint.

    Args:
        config: model configuration dict
        model_path: path to SavedModel dir or checkpoint prefix
        image_size: expected input image dimensions (H, W)

    Returns:
        Loaded model ready for inference
    """
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
        print(f"Loading SavedModel from: {model_path}")
        model = tf.saved_model.load(model_path)
        return model

    print(f"Loading from checkpoint: {model_path}")
    model_config = config["model"]
    model = build_petr_model(model_config)

    dummy_images = tf.zeros([1, 6, image_size[0], image_size[1], 3])
    dummy_intrinsics = tf.eye(3, batch_shape=[1, 6])
    dummy_extrinsics = tf.eye(4, batch_shape=[1, 6])
    _ = model(dummy_images, dummy_intrinsics, dummy_extrinsics, training=False)

    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(model_path)
    status.expect_partial()
    print("Checkpoint restored successfully")

    return model


def preprocess_images(
    image_paths: List[str],
    image_size: Tuple[int, int] = (900, 1600),
) -> tf.Tensor:
    """
    Load and preprocess multi-view images.

    Args:
        image_paths: list of 6 image file paths (one per camera)
        image_size: target (H, W)

    Returns:
        images: (1, 6, H, W, 3) normalized tensor
    """
    images = []
    for path in image_paths:
        img_raw = tf.io.read_file(path)
        if path.lower().endswith(".png"):
            img = tf.io.decode_png(img_raw, channels=3)
        else:
            img = tf.io.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0

        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        img = (img - mean) / std
        images.append(img)

    images = tf.stack(images, axis=0)
    return images[None]


def bev_nms(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    nms_threshold: float = 0.5,
) -> np.ndarray:
    """
    Perform BEV (Bird's Eye View) NMS on 3D detections.

    NMS is applied per class using the 2D BEV (x, y, w, l, yaw) overlap.

    Args:
        bboxes: (N, 10) predicted boxes [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        scores: (N,) confidence scores
        labels: (N,) class labels
        nms_threshold: IoU threshold for suppression

    Returns:
        keep: indices of boxes to keep
    """
    if len(bboxes) == 0:
        return np.array([], dtype=np.int32)

    unique_labels = np.unique(labels)
    keep_indices = []

    for cls in unique_labels:
        cls_mask = labels == cls
        cls_indices = np.where(cls_mask)[0]
        cls_bboxes = bboxes[cls_mask]
        cls_scores = scores[cls_mask]

        cx = cls_bboxes[:, 0]
        cy = cls_bboxes[:, 1]
        w = cls_bboxes[:, 3]
        l = cls_bboxes[:, 4]

        x1 = cx - w / 2
        y1 = cy - l / 2
        x2 = cx + w / 2
        y2 = cy + l / 2

        areas = w * l
        order = np.argsort(-cls_scores)

        cls_keep = []
        while len(order) > 0:
            i = order[0]
            cls_keep.append(cls_indices[i])

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / np.maximum(union, 1e-6)

            remaining = np.where(iou <= nms_threshold)[0]
            order = order[remaining + 1]

        keep_indices.extend(cls_keep)

    return np.array(keep_indices, dtype=np.int32)


def postprocess_predictions(
    cls_scores: np.ndarray,
    bbox_preds: np.ndarray,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.5,
    max_detections: int = 300,
) -> Dict[str, np.ndarray]:
    """
    Post-process raw model outputs into final detections.

    Args:
        cls_scores: (Q, num_classes) classification logits
        bbox_preds: (Q, 10) regression predictions
        score_threshold: minimum confidence to keep
        nms_threshold: BEV NMS IoU threshold
        max_detections: maximum number of output detections

    Returns:
        Dict with 'bboxes', 'scores', 'labels' arrays
    """
    probs = 1.0 / (1.0 + np.exp(-cls_scores))
    max_scores = np.max(probs, axis=-1)
    pred_labels = np.argmax(probs, axis=-1)

    valid_mask = max_scores >= score_threshold
    valid_scores = max_scores[valid_mask]
    valid_labels = pred_labels[valid_mask]
    valid_bboxes = bbox_preds[valid_mask]

    if len(valid_scores) == 0:
        return {
            "bboxes": np.zeros((0, 10), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int32),
        }

    keep = bev_nms(valid_bboxes, valid_scores, valid_labels, nms_threshold)

    if len(keep) > max_detections:
        top_k = np.argsort(-valid_scores[keep])[:max_detections]
        keep = keep[top_k]

    return {
        "bboxes": valid_bboxes[keep],
        "scores": valid_scores[keep],
        "labels": valid_labels[keep],
    }


def decode_bbox(bbox_pred: np.ndarray) -> Dict:
    """
    Decode a single bbox prediction into human-readable format.

    Args:
        bbox_pred: (10,) [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]

    Returns:
        Dict with center, size, yaw, velocity
    """
    cx, cy, cz = bbox_pred[0], bbox_pred[1], bbox_pred[2]
    w, l, h = bbox_pred[3], bbox_pred[4], bbox_pred[5]
    sin_yaw, cos_yaw = bbox_pred[6], bbox_pred[7]
    vx, vy = bbox_pred[8], bbox_pred[9]

    yaw = np.arctan2(sin_yaw, cos_yaw)

    return {
        "center": [float(cx), float(cy), float(cz)],
        "size": [float(w), float(l), float(h)],
        "yaw": float(yaw),
        "velocity": [float(vx), float(vy)],
    }


def run_inference(config: dict, args):
    """Run inference on input data."""
    import pickle

    model_config = config["model"]
    image_size = tuple(config["data"].get("image_size", [900, 1600]))
    temporal = model_config.get("temporal", False)

    model = load_model(config, args.model_path, image_size)

    with open(args.input, "rb") as f:
        data_infos = pickle.load(f)

    print(f"Running inference on {len(data_infos)} samples...")

    all_results = []
    prev_query = None

    num_batches = (len(data_infos) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(data_infos))
        batch_infos = data_infos[start_idx:end_idx]

        batch_images = []
        batch_intrinsics = []
        batch_extrinsics = []

        for info in batch_infos:
            images = []
            for cam_path in info["img_paths"]:
                data_root = config["data"].get("data_root", "")
                full_path = os.path.join(data_root, cam_path)
                img_raw = tf.io.read_file(full_path)
                img = tf.io.decode_jpeg(img_raw, channels=3)
                img = tf.image.resize(img, image_size)
                img = tf.cast(img, tf.float32) / 255.0
                mean = tf.constant([0.485, 0.456, 0.406])
                std = tf.constant([0.229, 0.224, 0.225])
                img = (img - mean) / std
                images.append(img)
            batch_images.append(tf.stack(images, axis=0))
            batch_intrinsics.append(tf.constant(info["intrinsics"], dtype=tf.float32))
            batch_extrinsics.append(tf.constant(info["extrinsics"], dtype=tf.float32))

        images_tensor = tf.stack(batch_images, axis=0)
        intrinsics_tensor = tf.stack(batch_intrinsics, axis=0)
        extrinsics_tensor = tf.stack(batch_extrinsics, axis=0)

        call_kwargs = {
            "images": images_tensor,
            "intrinsics": intrinsics_tensor,
            "extrinsics": extrinsics_tensor,
            "training": False,
        }

        if temporal and prev_query is not None:
            ego_motions = []
            for info in batch_infos:
                if "ego_motion" in info:
                    ego_motions.append(info["ego_motion"])
                else:
                    ego_motions.append(np.eye(4, dtype=np.float32))
            call_kwargs["ego_motion"] = tf.constant(np.stack(ego_motions), dtype=tf.float32)
            call_kwargs["prev_query"] = prev_query

        if isinstance(model, tf.keras.Model):
            outputs = model(**call_kwargs)
        else:
            infer_fn = model.signatures["serving_default"]
            outputs = infer_fn(**call_kwargs)

        cls_scores_batch = outputs["cls_scores"][-1].numpy()
        bbox_preds_batch = outputs["bbox_preds"][-1].numpy()

        if temporal:
            prev_query = outputs["query_output"]

        for i in range(end_idx - start_idx):
            detections = postprocess_predictions(
                cls_scores=cls_scores_batch[i],
                bbox_preds=bbox_preds_batch[i],
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                max_detections=args.max_detections,
            )

            frame_results = {
                "sample_idx": start_idx + i,
                "detections": [],
            }

            for det_idx in range(len(detections["scores"])):
                bbox_decoded = decode_bbox(detections["bboxes"][det_idx])
                frame_results["detections"].append({
                    "class": NUSCENES_CLASSES[detections["labels"][det_idx]],
                    "class_id": int(detections["labels"][det_idx]),
                    "score": float(detections["scores"][det_idx]),
                    **bbox_decoded,
                })

            all_results.append(frame_results)

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"  Processed {end_idx}/{len(data_infos)} samples")

    with open(args.output, "wb") as f:
        pickle.dump(all_results, f)

    total_detections = sum(len(r["detections"]) for r in all_results)
    print(f"\nInference complete:")
    print(f"  Total samples: {len(all_results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections/frame: {total_detections / max(len(all_results), 1):.1f}")
    print(f"  Results saved to: {args.output}")

    return all_results


def run_single_frame(
    model: tf.keras.Model,
    image_paths: List[str],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_size: Tuple[int, int] = (900, 1600),
    score_threshold: float = 0.3,
    nms_threshold: float = 0.5,
    max_detections: int = 300,
    ego_motion: Optional[np.ndarray] = None,
    prev_query: Optional[tf.Tensor] = None,
) -> Tuple[Dict, Optional[tf.Tensor]]:
    """
    Run inference on a single frame (convenience function).

    Args:
        model: loaded PETR model
        image_paths: list of 6 camera image paths
        intrinsics: (6, 3, 3) camera intrinsics
        extrinsics: (6, 4, 4) camera extrinsics
        image_size: target image size
        score_threshold: min detection confidence
        nms_threshold: BEV NMS threshold
        max_detections: max output boxes
        ego_motion: (4, 4) ego-motion matrix (for temporal mode)
        prev_query: previous frame query (for temporal mode)

    Returns:
        (detections_dict, query_output_for_next_frame)
    """
    images = preprocess_images(image_paths, image_size)
    intrinsics_t = tf.constant(intrinsics[None], dtype=tf.float32)
    extrinsics_t = tf.constant(extrinsics[None], dtype=tf.float32)

    call_kwargs = {
        "images": images,
        "intrinsics": intrinsics_t,
        "extrinsics": extrinsics_t,
        "training": False,
    }

    if ego_motion is not None:
        call_kwargs["ego_motion"] = tf.constant(ego_motion[None], dtype=tf.float32)
    if prev_query is not None:
        call_kwargs["prev_query"] = prev_query

    outputs = model(**call_kwargs)

    cls_scores = outputs["cls_scores"][-1][0].numpy()
    bbox_preds = outputs["bbox_preds"][-1][0].numpy()
    query_out = outputs["query_output"]

    detections = postprocess_predictions(
        cls_scores, bbox_preds, score_threshold, nms_threshold, max_detections
    )

    results = []
    for i in range(len(detections["scores"])):
        bbox_decoded = decode_bbox(detections["bboxes"][i])
        results.append({
            "class": NUSCENES_CLASSES[detections["labels"][i]],
            "class_id": int(detections["labels"][i]),
            "score": float(detections["scores"][i]),
            **bbox_decoded,
        })

    return {"detections": results}, query_out


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_inference(config, args)
