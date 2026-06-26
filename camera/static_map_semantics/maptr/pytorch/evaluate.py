"""
Evaluation script for MapTR: Chamfer-distance based Average Precision.

Evaluates map element predictions against ground truth using the Chamfer distance
metric at multiple thresholds, computing per-category and mean Average Precision.

Usage:
    python evaluate.py --checkpoint path/to/model.pth --data_root path/to/nuscenes \
        --output results.json --batch_size 4
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# MapTR category definitions
CATEGORY_NAMES = ["ped_crossing", "divider", "boundary"]
CATEGORY_IDS = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}
NUM_CATEGORIES = len(CATEGORY_NAMES)

# Default evaluation thresholds in meters
DEFAULT_THRESHOLDS = [0.5, 1.0, 1.5]


class ChamferDistance:
    """
    Computes the Chamfer distance between two ordered point sets (polylines).

    The Chamfer distance measures the similarity between two point sets by computing
    the average nearest-neighbor distance in both directions.
    """

    def __init__(self, mode: str = "max"):
        """
        Args:
            mode: How to combine the two directed distances.
                  'max' - maximum of both directed distances (stricter)
                  'mean' - average of both directed distances
                  'pred_to_gt' - only distance from predictions to ground truth
                  'gt_to_pred' - only distance from ground truth to predictions
        """
        if mode not in ("max", "mean", "pred_to_gt", "gt_to_pred"):
            raise ValueError(f"Invalid mode: {mode}. Choose from 'max', 'mean', 'pred_to_gt', 'gt_to_pred'.")
        self.mode = mode

    def __call__(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
    ) -> float:
        """
        Compute Chamfer distance between predicted and ground truth point sets.

        Args:
            pred_points: Predicted polyline points, shape [N_pred, 2] (x, y in meters)
            gt_points: Ground truth polyline points, shape [N_gt, 2] (x, y in meters)

        Returns:
            Chamfer distance value in meters.
        """
        if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
            return float("inf")

        # Compute pairwise distance matrix: [N_pred, N_gt]
        # Using broadcasting: (N_pred, 1, 2) - (1, N_gt, 2) -> (N_pred, N_gt, 2)
        diff = pred_points[:, None, :] - gt_points[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))  # [N_pred, N_gt]

        # Directed distance: pred -> gt
        # For each point in pred, find minimum distance to any GT point
        min_pred_to_gt = np.min(dist_matrix, axis=1)  # [N_pred]
        d_pred_to_gt = np.mean(min_pred_to_gt)

        # Directed distance: gt -> pred
        # For each point in gt, find minimum distance to any pred point
        min_gt_to_pred = np.min(dist_matrix, axis=0)  # [N_gt]
        d_gt_to_pred = np.mean(min_gt_to_pred)

        if self.mode == "max":
            return float(max(d_pred_to_gt, d_gt_to_pred))
        elif self.mode == "mean":
            return float((d_pred_to_gt + d_gt_to_pred) / 2.0)
        elif self.mode == "pred_to_gt":
            return float(d_pred_to_gt)
        else:  # gt_to_pred
            return float(d_gt_to_pred)

    def batch_compute(
        self,
        pred_list: List[np.ndarray],
        gt_list: List[np.ndarray],
    ) -> List[float]:
        """
        Compute Chamfer distances for a batch of polyline pairs.

        Args:
            pred_list: List of predicted polyline arrays, each [N_i, 2]
            gt_list: List of GT polyline arrays, each [M_i, 2]

        Returns:
            List of Chamfer distances.
        """
        distances = []
        for pred, gt in zip(pred_list, gt_list):
            distances.append(self(pred, gt))
        return distances


class MapEvaluator:
    """
    Evaluator for vectorized map element predictions.

    Computes Average Precision (AP) based on Chamfer distance matching between
    predicted and ground truth map elements at multiple distance thresholds.
    """

    def __init__(
        self,
        thresholds: Optional[List[float]] = None,
        categories: Optional[List[str]] = None,
        chamfer_mode: str = "mean",
        interpolation: str = "all_point",
        num_points_per_polyline: int = 20,
    ):
        """
        Args:
            thresholds: List of Chamfer distance thresholds (meters) for TP/FP determination.
            categories: List of category names to evaluate.
            chamfer_mode: Mode for Chamfer distance computation ('max' or 'mean').
            interpolation: AP interpolation method - '11_point' or 'all_point'.
            num_points_per_polyline: Number of points to resample each polyline to
                                     before computing Chamfer distance (for fair comparison).
        """
        self.thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
        self.categories = categories if categories is not None else CATEGORY_NAMES
        self.chamfer_dist = ChamferDistance(mode=chamfer_mode)
        self.interpolation = interpolation
        self.num_points_per_polyline = num_points_per_polyline

    def _resample_polyline(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """
        Resample a polyline to a fixed number of equally-spaced points.

        Args:
            points: Original polyline points [N, 2]
            num_points: Target number of points

        Returns:
            Resampled points [num_points, 2]
        """
        if points.shape[0] < 2:
            # Single point or empty: replicate to fill
            return np.tile(points[0:1], (num_points, 1)) if points.shape[0] == 1 else np.zeros((num_points, 2))

        # Compute cumulative arc length along the polyline
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]

        if total_length < 1e-8:
            # Degenerate polyline (all points coincide)
            return np.tile(points[0:1], (num_points, 1))

        # Create uniformly spaced parameter values
        target_lengths = np.linspace(0, total_length, num_points)

        # Interpolate x and y independently along arc length
        resampled = np.zeros((num_points, 2))
        resampled[:, 0] = np.interp(target_lengths, cumulative_lengths, points[:, 0])
        resampled[:, 1] = np.interp(target_lengths, cumulative_lengths, points[:, 1])

        return resampled

    def _match_predictions_to_gt(
        self,
        pred_polylines: List[np.ndarray],
        pred_scores: List[float],
        pred_categories: List[int],
        gt_polylines: List[np.ndarray],
        gt_categories: List[int],
        threshold: float,
        category_id: int,
    ) -> Tuple[List[bool], int]:
        """
        Match predictions to ground truth for a single sample and category.

        Predictions are processed in descending confidence order. Each GT can only
        be matched once (greedy matching).

        Args:
            pred_polylines: List of predicted polyline point arrays
            pred_scores: Confidence scores for each prediction
            pred_categories: Category ID for each prediction
            gt_polylines: List of GT polyline point arrays
            gt_categories: Category ID for each GT element
            threshold: Chamfer distance threshold for TP determination
            category_id: Category to evaluate

        Returns:
            tp_flags: Boolean list (True = TP, False = FP) for each filtered prediction
            num_gt: Number of GT elements for this category
        """
        # Filter predictions and GT by category
        pred_indices = [i for i, c in enumerate(pred_categories) if c == category_id]
        gt_indices = [i for i, c in enumerate(gt_categories) if c == category_id]

        num_gt = len(gt_indices)

        if len(pred_indices) == 0:
            return [], num_gt

        # Sort predictions by confidence (descending)
        pred_sorted = sorted(pred_indices, key=lambda i: pred_scores[i], reverse=True)

        # Resample GT polylines for fair comparison
        gt_resampled = []
        for gi in gt_indices:
            gt_resampled.append(self._resample_polyline(gt_polylines[gi], self.num_points_per_polyline))

        # Track which GTs have been matched
        gt_matched = [False] * len(gt_indices)

        tp_flags = []
        for pi in pred_sorted:
            pred_resampled = self._resample_polyline(pred_polylines[pi], self.num_points_per_polyline)

            # Find best matching GT (minimum Chamfer distance)
            best_dist = float("inf")
            best_gt_idx = -1

            for gi_local, gt_pts in enumerate(gt_resampled):
                if gt_matched[gi_local]:
                    continue
                dist = self.chamfer_dist(pred_resampled, gt_pts)
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = gi_local

            # Determine TP or FP
            if best_gt_idx >= 0 and best_dist < threshold:
                tp_flags.append(True)
                gt_matched[best_gt_idx] = True
            else:
                tp_flags.append(False)

        return tp_flags, num_gt

    def _compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        Compute Average Precision from recall-precision arrays.

        Args:
            recalls: Recall values (sorted ascending)
            precisions: Precision values corresponding to each recall

        Returns:
            Average Precision value
        """
        if len(recalls) == 0:
            return 0.0

        if self.interpolation == "11_point":
            # 11-point interpolation (PASCAL VOC style)
            ap = 0.0
            for t in np.linspace(0, 1, 11):
                # Precision at recall >= t
                mask = recalls >= t
                if mask.any():
                    ap += np.max(precisions[mask])
            ap /= 11.0
            return float(ap)
        else:
            # All-point interpolation (COCO style)
            # Prepend sentinel values
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([1.0], precisions, [0.0]))

            # Make precision monotonically decreasing (from right to left)
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])

            # Find points where recall changes
            change_indices = np.where(mrec[1:] != mrec[:-1])[0]

            # Sum area under the precision-recall curve
            ap = np.sum((mrec[change_indices + 1] - mrec[change_indices]) * mpre[change_indices + 1])
            return float(ap)

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth across all samples.

        Args:
            predictions: List of prediction dicts per sample, each containing:
                - 'polylines': List[np.ndarray], each [N_points, 2] in BEV meters
                - 'scores': List[float], confidence scores
                - 'categories': List[int], category IDs
            ground_truths: List of GT dicts per sample, each containing:
                - 'polylines': List[np.ndarray], each [N_points, 2] in BEV meters
                - 'categories': List[int], category IDs

        Returns:
            Dictionary with evaluation results:
                - 'mAP': float, mean AP across all categories and thresholds
                - 'per_category': dict mapping category name to per-threshold APs
                - 'per_threshold': dict mapping threshold to mean AP across categories
                - 'detailed': full AP matrix [categories x thresholds]
        """
        assert len(predictions) == len(ground_truths), (
            f"Number of predictions ({len(predictions)}) must match "
            f"ground truths ({len(ground_truths)})"
        )

        num_samples = len(predictions)

        # Collect TP/FP flags and GT counts across all samples per (category, threshold)
        # Structure: {(cat_id, threshold): {'tp_flags': [...], 'scores': [...], 'num_gt': int}}
        eval_data = {}
        for cat_id in range(NUM_CATEGORIES):
            for thresh in self.thresholds:
                eval_data[(cat_id, thresh)] = {
                    "tp_flags": [],
                    "scores": [],
                    "num_gt": 0,
                }

        for sample_idx in range(num_samples):
            pred = predictions[sample_idx]
            gt = ground_truths[sample_idx]

            pred_polylines = pred["polylines"]
            pred_scores = pred["scores"]
            pred_categories = pred["categories"]
            gt_polylines = gt["polylines"]
            gt_categories = gt["categories"]

            for cat_id in range(NUM_CATEGORIES):
                for thresh in self.thresholds:
                    tp_flags, num_gt = self._match_predictions_to_gt(
                        pred_polylines, pred_scores, pred_categories,
                        gt_polylines, gt_categories,
                        threshold=thresh,
                        category_id=cat_id,
                    )

                    # Collect scores for the category-filtered predictions (sorted by confidence)
                    cat_pred_indices = [
                        i for i, c in enumerate(pred_categories) if c == cat_id
                    ]
                    cat_scores = sorted(
                        [pred_scores[i] for i in cat_pred_indices], reverse=True
                    )

                    eval_data[(cat_id, thresh)]["tp_flags"].extend(tp_flags)
                    eval_data[(cat_id, thresh)]["scores"].extend(cat_scores)
                    eval_data[(cat_id, thresh)]["num_gt"] += num_gt

        # Compute AP for each (category, threshold) pair
        ap_matrix = np.zeros((NUM_CATEGORIES, len(self.thresholds)))

        for cat_idx, cat_id in enumerate(range(NUM_CATEGORIES)):
            for thresh_idx, thresh in enumerate(self.thresholds):
                data = eval_data[(cat_id, thresh)]
                tp_flags = np.array(data["tp_flags"], dtype=bool)
                scores = np.array(data["scores"])
                num_gt = data["num_gt"]

                if num_gt == 0:
                    # No GT for this category - AP is 0 (or could be skipped)
                    ap_matrix[cat_idx, thresh_idx] = 0.0
                    continue

                if len(tp_flags) == 0:
                    # No predictions for this category
                    ap_matrix[cat_idx, thresh_idx] = 0.0
                    continue

                # Sort by confidence (already sorted, but ensure consistency)
                sort_order = np.argsort(-scores)
                tp_flags = tp_flags[sort_order]

                # Compute cumulative TP and FP
                cum_tp = np.cumsum(tp_flags).astype(float)
                cum_fp = np.cumsum(~tp_flags).astype(float)

                # Precision and recall at each detection
                precisions = cum_tp / (cum_tp + cum_fp)
                recalls = cum_tp / num_gt

                ap_matrix[cat_idx, thresh_idx] = self._compute_ap(recalls, precisions)

        # Aggregate results
        per_category = {}
        for cat_idx, cat_name in enumerate(self.categories):
            per_category[cat_name] = {
                f"AP@{thresh:.1f}": float(ap_matrix[cat_idx, thresh_idx])
                for thresh_idx, thresh in enumerate(self.thresholds)
            }
            per_category[cat_name]["AP_mean"] = float(np.mean(ap_matrix[cat_idx, :]))

        per_threshold = {}
        for thresh_idx, thresh in enumerate(self.thresholds):
            per_threshold[f"mAP@{thresh:.1f}"] = float(np.mean(ap_matrix[:, thresh_idx]))

        mAP = float(np.mean(ap_matrix))

        results = {
            "mAP": mAP,
            "per_category": per_category,
            "per_threshold": per_threshold,
            "ap_matrix": ap_matrix.tolist(),
            "num_samples": num_samples,
            "thresholds": self.thresholds,
            "categories": self.categories,
        }

        return results

    def format_results(self, results: Dict[str, Any]) -> str:
        """
        Format evaluation results as a human-readable table.

        Args:
            results: Dictionary returned by evaluate()

        Returns:
            Formatted string table
        """
        lines = []
        lines.append("=" * 70)
        lines.append("MapTR Evaluation Results")
        lines.append("=" * 70)
        lines.append(f"Number of samples: {results['num_samples']}")
        lines.append(f"Thresholds (meters): {results['thresholds']}")
        lines.append("")

        # Header row
        header = f"{'Category':<16}"
        for thresh in results["thresholds"]:
            header += f"{'AP@' + f'{thresh:.1f}m':<12}"
        header += f"{'Mean':<12}"
        lines.append(header)
        lines.append("-" * 70)

        # Per-category rows
        for cat_name in results["categories"]:
            row = f"{cat_name:<16}"
            cat_data = results["per_category"][cat_name]
            for thresh in results["thresholds"]:
                ap_val = cat_data[f"AP@{thresh:.1f}"]
                row += f"{ap_val:.4f}      "
            row += f"{cat_data['AP_mean']:.4f}"
            lines.append(row)

        lines.append("-" * 70)

        # Mean row
        row = f"{'mAP':<16}"
        for thresh_key, thresh_val in results["per_threshold"].items():
            row += f"{thresh_val:.4f}      "
        row += f"{results['mAP']:.4f}"
        lines.append(row)

        lines.append("=" * 70)

        return "\n".join(lines)


def build_model(
    checkpoint_path: str,
    device: torch.device,
    bev_h: int = 200,
    bev_w: int = 100,
    num_queries: int = 50,
    num_points_per_query: int = 20,
) -> nn.Module:
    """
    Build and load a MapTR model from checkpoint.

    Args:
        checkpoint_path: Path to saved model weights.
        device: Target device for the model.
        bev_h: BEV grid height.
        bev_w: BEV grid width.
        num_queries: Number of map element queries.
        num_points_per_query: Number of points per polyline query.

    Returns:
        Loaded model in eval mode.
    """
    from backbone import ResNet50FPN
    from gkt import GKT

    class MapTRModel(nn.Module):
        """Minimal MapTR model assembling backbone, BEV encoder, and decoder head."""

        def __init__(self, bev_h, bev_w, num_queries, num_points_per_query, embed_dim=256):
            super().__init__()
            self.bev_h = bev_h
            self.bev_w = bev_w
            self.num_queries = num_queries
            self.num_points_per_query = num_points_per_query
            self.embed_dim = embed_dim

            # Backbone
            self.backbone = ResNet50FPN(pretrained=False, fpn_out_channels=embed_dim)

            # BEV encoder
            self.bev_encoder = GKT(
                embed_dim=embed_dim,
                bev_h=bev_h,
                bev_w=bev_w,
                num_layers=3,
            )

            # Decoder queries
            self.query_embed = nn.Embedding(num_queries, embed_dim)
            self.query_pos = nn.Embedding(num_queries, embed_dim)

            # Transformer decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=1024,
                dropout=0.1, batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

            # Prediction heads
            self.class_head = nn.Linear(embed_dim, NUM_CATEGORIES + 1)  # +1 for no-object
            self.points_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, num_points_per_query * 2),  # (x, y) per point
            )

        def forward(self, images, intrinsics, extrinsics):
            """
            Args:
                images: [B, N_cams, 3, H, W]
                intrinsics: [B, N_cams, 3, 3]
                extrinsics: [B, N_cams, 4, 4]

            Returns:
                class_logits: [B, num_queries, NUM_CATEGORIES + 1]
                point_coords: [B, num_queries, num_points_per_query, 2] normalized [0, 1]
            """
            B = images.shape[0]

            # Backbone features
            fpn_feats = self.backbone(images)

            # BEV encoding
            bev_feat = self.bev_encoder(fpn_feats, intrinsics, extrinsics)
            # bev_feat: [B, C, bev_h, bev_w]

            # Flatten BEV features as memory for decoder
            memory = bev_feat.flatten(2).permute(0, 2, 1)  # [B, bev_h*bev_w, C]

            # Decode queries
            query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
            query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
            tgt = query_embed + query_pos

            decoded = self.decoder(tgt, memory)  # [B, num_queries, C]

            # Predict classes and points
            class_logits = self.class_head(decoded)  # [B, num_queries, NUM_CATEGORIES+1]
            points_raw = self.points_head(decoded)  # [B, num_queries, num_points*2]
            point_coords = points_raw.reshape(B, self.num_queries, self.num_points_per_query, 2)
            point_coords = torch.sigmoid(point_coords)  # Normalize to [0, 1]

            return class_logits, point_coords

    # Build model
    model = MapTRModel(bev_h, bev_w, num_queries, num_points_per_query)

    # Load checkpoint
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Handle potential key prefix mismatches
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # Remove DataParallel wrapper prefix
            cleaned_state_dict[new_key] = v

        model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}, using random weights.")

    model = model.to(device)
    model.eval()
    return model


def create_validation_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    bev_x_range: Tuple[float, float] = (-50.0, 50.0),
    bev_y_range: Tuple[float, float] = (-50.0, 50.0),
):
    """
    Create a validation dataloader for nuScenes-style map data.

    This function creates a simple dataset that loads pre-processed validation
    data from disk. The expected directory structure:
        data_root/
            samples/        - camera images
            maps/           - map annotations
            val_infos.pkl   - validation sample metadata

    Args:
        data_root: Root directory of the dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        bev_x_range: BEV x-axis range.
        bev_y_range: BEV y-axis range.

    Returns:
        DataLoader and metadata dict.
    """
    import pickle
    from torch.utils.data import Dataset, DataLoader

    class MapTRValDataset(Dataset):
        """Validation dataset for MapTR evaluation."""

        def __init__(self, data_root, bev_x_range, bev_y_range):
            self.data_root = data_root
            self.bev_x_range = bev_x_range
            self.bev_y_range = bev_y_range

            # Load validation info
            info_path = os.path.join(data_root, "val_infos.pkl")
            if os.path.exists(info_path):
                with open(info_path, "rb") as f:
                    self.infos = pickle.load(f)
            else:
                # Attempt JSON fallback
                json_path = os.path.join(data_root, "val_infos.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        self.infos = json.load(f)
                else:
                    raise FileNotFoundError(
                        f"No validation info found. Expected {info_path} or {json_path}"
                    )

        def __len__(self):
            return len(self.infos)

        def __getitem__(self, idx):
            info = self.infos[idx]

            # Load camera images
            images = []
            for cam_path in info["cam_paths"]:
                full_path = os.path.join(self.data_root, cam_path)
                img = self._load_image(full_path)
                images.append(img)
            images = torch.stack(images, dim=0)  # [N_cams, 3, H, W]

            # Camera parameters
            intrinsics = torch.tensor(info["intrinsics"], dtype=torch.float32)  # [N_cams, 3, 3]
            extrinsics = torch.tensor(info["extrinsics"], dtype=torch.float32)  # [N_cams, 4, 4]

            # Ground truth map elements
            gt_polylines = []
            gt_categories = []
            for ann in info.get("annotations", []):
                pts = np.array(ann["points"], dtype=np.float32)  # [N, 2] in BEV meters
                gt_polylines.append(pts)
                gt_categories.append(ann["category_id"])

            sample = {
                "images": images,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "gt_polylines": gt_polylines,
                "gt_categories": gt_categories,
                "sample_token": info.get("token", f"sample_{idx}"),
            }

            return sample

        def _load_image(self, path: str) -> torch.Tensor:
            """Load and preprocess a single camera image."""
            try:
                from PIL import Image
                from torchvision import transforms

                transform = transforms.Compose([
                    transforms.Resize((480, 800)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                img = Image.open(path).convert("RGB")
                return transform(img)
            except Exception:
                # Return a dummy tensor if image loading fails
                return torch.zeros(3, 480, 800)

    def collate_fn(batch):
        """Custom collate to handle variable-length GT annotations."""
        images = torch.stack([s["images"] for s in batch])
        intrinsics = torch.stack([s["intrinsics"] for s in batch])
        extrinsics = torch.stack([s["extrinsics"] for s in batch])

        gt_polylines = [s["gt_polylines"] for s in batch]
        gt_categories = [s["gt_categories"] for s in batch]
        sample_tokens = [s["sample_token"] for s in batch]

        return {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "gt_polylines": gt_polylines,
            "gt_categories": gt_categories,
            "sample_tokens": sample_tokens,
        }

    dataset = MapTRValDataset(data_root, bev_x_range, bev_y_range)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    metadata = {
        "num_samples": len(dataset),
        "bev_x_range": bev_x_range,
        "bev_y_range": bev_y_range,
    }

    return dataloader, metadata


def decode_model_outputs(
    class_logits: torch.Tensor,
    point_coords: torch.Tensor,
    bev_x_range: Tuple[float, float] = (-50.0, 50.0),
    bev_y_range: Tuple[float, float] = (-50.0, 50.0),
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Decode raw model outputs into evaluation-ready predictions.

    Args:
        class_logits: [B, num_queries, NUM_CATEGORIES + 1]
        point_coords: [B, num_queries, num_points, 2] normalized [0, 1]
        bev_x_range: BEV x range for denormalization
        bev_y_range: BEV y range for denormalization
        score_threshold: Minimum confidence to keep a prediction

    Returns:
        List of prediction dicts (one per sample in batch).
    """
    B = class_logits.shape[0]

    # Apply softmax to get class probabilities
    class_probs = torch.softmax(class_logits, dim=-1)  # [B, Q, NUM_CATEGORIES + 1]

    # Last class is "no-object" - exclude it for scoring
    fg_probs = class_probs[:, :, :NUM_CATEGORIES]  # [B, Q, NUM_CATEGORIES]

    # Best category and score per query
    max_scores, max_categories = fg_probs.max(dim=-1)  # [B, Q], [B, Q]

    # Denormalize point coordinates from [0, 1] to BEV meters
    x_min, x_max = bev_x_range
    y_min, y_max = bev_y_range

    points_denorm = point_coords.clone()
    points_denorm[..., 0] = points_denorm[..., 0] * (x_max - x_min) + x_min
    points_denorm[..., 1] = points_denorm[..., 1] * (y_max - y_min) + y_min

    predictions = []
    for b in range(B):
        sample_polylines = []
        sample_scores = []
        sample_categories = []

        for q in range(max_scores.shape[1]):
            score = max_scores[b, q].item()
            if score < score_threshold:
                continue
            cat_id = max_categories[b, q].item()
            pts = points_denorm[b, q].detach().cpu().numpy()  # [num_points, 2]

            sample_polylines.append(pts)
            sample_scores.append(score)
            sample_categories.append(cat_id)

        predictions.append({
            "polylines": sample_polylines,
            "scores": sample_scores,
            "categories": sample_categories,
        })

    return predictions


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    dataloader,
    evaluator: MapEvaluator,
    device: torch.device,
    bev_x_range: Tuple[float, float] = (-50.0, 50.0),
    bev_y_range: Tuple[float, float] = (-50.0, 50.0),
    score_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Run full evaluation loop: inference on all samples then compute metrics.

    Args:
        model: MapTR model in eval mode.
        dataloader: Validation data loader.
        evaluator: MapEvaluator instance.
        device: Compute device.
        bev_x_range: BEV x range for coordinate denormalization.
        bev_y_range: BEV y range for coordinate denormalization.
        score_threshold: Minimum score to include predictions.

    Returns:
        Evaluation results dictionary.
    """
    all_predictions = []
    all_ground_truths = []

    total_time = 0.0
    num_batches = 0

    print("Running inference on validation set...")
    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        gt_polylines_batch = batch["gt_polylines"]
        gt_categories_batch = batch["gt_categories"]

        # Model inference
        start_time = time.time()
        class_logits, point_coords = model(images, intrinsics, extrinsics)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - start_time
        total_time += elapsed
        num_batches += 1

        # Decode predictions
        batch_preds = decode_model_outputs(
            class_logits, point_coords,
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            score_threshold=score_threshold,
        )
        all_predictions.extend(batch_preds)

        # Format ground truths
        B = images.shape[0]
        for b in range(B):
            all_ground_truths.append({
                "polylines": gt_polylines_batch[b],
                "categories": gt_categories_batch[b],
            })

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"  Processed batch {batch_idx + 1}/{len(dataloader)} "
                  f"({elapsed * 1000:.1f} ms/batch)")

    # Print timing summary
    avg_time = total_time / max(num_batches, 1)
    num_samples = len(all_predictions)
    print(f"\nInference complete: {num_samples} samples in {total_time:.2f}s "
          f"({avg_time * 1000:.1f} ms/batch avg)")

    # Compute metrics
    print("\nComputing evaluation metrics...")
    results = evaluator.evaluate(all_predictions, all_ground_truths)

    # Add timing info
    results["inference_time_total_s"] = total_time
    results["inference_time_per_sample_ms"] = (total_time / max(num_samples, 1)) * 1000

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MapTR model using Chamfer-distance AP metrics."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint file (.pth)"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory of the validation dataset"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json",
        help="Path to save evaluation results JSON"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use: 'cuda', 'cpu', or 'auto'"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.5, 1.0, 1.5],
        help="Chamfer distance thresholds in meters for AP computation"
    )
    parser.add_argument(
        "--chamfer_mode", type=str, default="mean",
        choices=["max", "mean", "pred_to_gt", "gt_to_pred"],
        help="Chamfer distance aggregation mode"
    )
    parser.add_argument(
        "--interpolation", type=str, default="all_point",
        choices=["11_point", "all_point"],
        help="AP interpolation method"
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.0,
        help="Minimum confidence score for predictions (0.0 to include all)"
    )
    parser.add_argument(
        "--bev_x_range", type=float, nargs=2, default=[-50.0, 50.0],
        help="BEV x-axis range in meters"
    )
    parser.add_argument(
        "--bev_y_range", type=float, nargs=2, default=[-50.0, 50.0],
        help="BEV y-axis range in meters"
    )
    parser.add_argument(
        "--num_queries", type=int, default=50,
        help="Number of map element queries in the model"
    )
    parser.add_argument(
        "--num_points", type=int, default=20,
        help="Number of points per polyline query"
    )

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build model
    print("Loading model...")
    model = build_model(
        checkpoint_path=args.checkpoint,
        device=device,
        num_queries=args.num_queries,
        num_points_per_query=args.num_points,
    )

    # Create dataloader
    print("Preparing validation data...")
    bev_x_range = tuple(args.bev_x_range)
    bev_y_range = tuple(args.bev_y_range)
    dataloader, metadata = create_validation_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bev_x_range=bev_x_range,
        bev_y_range=bev_y_range,
    )
    print(f"Validation set: {metadata['num_samples']} samples")

    # Create evaluator
    evaluator = MapEvaluator(
        thresholds=args.thresholds,
        chamfer_mode=args.chamfer_mode,
        interpolation=args.interpolation,
        num_points_per_polyline=args.num_points,
    )

    # Run evaluation
    results = run_evaluation(
        model=model,
        dataloader=dataloader,
        evaluator=evaluator,
        device=device,
        bev_x_range=bev_x_range,
        bev_y_range=bev_y_range,
        score_threshold=args.score_threshold,
    )

    # Print formatted results
    print("\n")
    formatted = evaluator.format_results(results)
    print(formatted)

    # Save results to JSON
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
