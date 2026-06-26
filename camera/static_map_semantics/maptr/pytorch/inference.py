"""
Inference and visualization script for MapTR.

Runs model inference on individual samples or batches and produces BEV
visualizations of predicted map elements with optional ground truth overlay.

Usage:
    python inference.py --checkpoint model.pth --data_root /path/to/data \
        --sample_idx 0 --output_dir ./vis_output --confidence_threshold 0.3
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# MapTR category definitions and colors
CATEGORY_NAMES = ["ped_crossing", "divider", "boundary"]
CATEGORY_IDS = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}
NUM_CATEGORIES = len(CATEGORY_NAMES)

# Color scheme for visualization (RGB tuples, 0-1 scale)
CATEGORY_COLORS = {
    "ped_crossing": (0.2, 0.4, 0.9),   # Blue
    "divider": (1.0, 0.6, 0.1),         # Orange
    "boundary": (0.2, 0.8, 0.3),        # Green
}
CATEGORY_COLORS_BY_ID = {
    0: CATEGORY_COLORS["ped_crossing"],
    1: CATEGORY_COLORS["divider"],
    2: CATEGORY_COLORS["boundary"],
}

# GT overlay uses slightly different shades
GT_COLORS_BY_ID = {
    0: (0.4, 0.5, 0.95),   # Light blue
    1: (0.95, 0.75, 0.3),  # Light orange
    2: (0.4, 0.9, 0.5),    # Light green
}


class PostProcessor:
    """
    Post-processing for MapTR model outputs.

    Handles confidence filtering, coordinate denormalization, and NMS-like
    filtering of overlapping predictions.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        bev_x_range: Tuple[float, float] = (-50.0, 50.0),
        bev_y_range: Tuple[float, float] = (-50.0, 50.0),
        nms_threshold: float = 0.0,
        num_points_per_polyline: int = 20,
    ):
        """
        Args:
            confidence_threshold: Minimum confidence to retain a prediction.
            bev_x_range: BEV x-axis range in meters.
            bev_y_range: BEV y-axis range in meters.
            nms_threshold: Chamfer distance threshold for NMS-like suppression.
                          Set to 0 to disable NMS.
            num_points_per_polyline: Expected number of points per polyline.
        """
        self.confidence_threshold = confidence_threshold
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.nms_threshold = nms_threshold
        self.num_points_per_polyline = num_points_per_polyline

    def process(
        self,
        class_logits: torch.Tensor,
        point_coords: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Full post-processing pipeline for a batch of predictions.

        Args:
            class_logits: [B, num_queries, NUM_CATEGORIES + 1]
            point_coords: [B, num_queries, num_points, 2] normalized [0, 1]

        Returns:
            List of processed prediction dicts per sample.
        """
        B = class_logits.shape[0]

        # Softmax over classes
        class_probs = torch.softmax(class_logits, dim=-1)
        fg_probs = class_probs[:, :, :NUM_CATEGORIES]

        # Best class and score per query
        max_scores, max_categories = fg_probs.max(dim=-1)

        # Denormalize coordinates
        x_min, x_max = self.bev_x_range
        y_min, y_max = self.bev_y_range

        points_denorm = point_coords.clone().detach().cpu()
        points_denorm[..., 0] = points_denorm[..., 0] * (x_max - x_min) + x_min
        points_denorm[..., 1] = points_denorm[..., 1] * (y_max - y_min) + y_min

        results = []
        for b in range(B):
            sample_result = self._process_sample(
                scores=max_scores[b].detach().cpu().numpy(),
                categories=max_categories[b].detach().cpu().numpy(),
                points=points_denorm[b].numpy(),
            )
            results.append(sample_result)

        return results

    def _process_sample(
        self,
        scores: np.ndarray,
        categories: np.ndarray,
        points: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Process predictions for a single sample.

        Args:
            scores: [num_queries] confidence scores
            categories: [num_queries] category IDs
            points: [num_queries, num_points, 2] denormalized coordinates

        Returns:
            Dict with filtered polylines, scores, and categories.
        """
        # Filter by confidence
        keep_mask = scores >= self.confidence_threshold
        keep_indices = np.where(keep_mask)[0]

        if len(keep_indices) == 0:
            return {"polylines": [], "scores": [], "categories": []}

        # Sort by confidence (descending)
        sorted_indices = keep_indices[np.argsort(-scores[keep_indices])]

        filtered_polylines = [points[i] for i in sorted_indices]
        filtered_scores = [float(scores[i]) for i in sorted_indices]
        filtered_categories = [int(categories[i]) for i in sorted_indices]

        # Apply NMS-like filtering if enabled
        if self.nms_threshold > 0:
            filtered_polylines, filtered_scores, filtered_categories = self._nms_filter(
                filtered_polylines, filtered_scores, filtered_categories
            )

        return {
            "polylines": filtered_polylines,
            "scores": filtered_scores,
            "categories": filtered_categories,
        }

    def _nms_filter(
        self,
        polylines: List[np.ndarray],
        scores: List[float],
        categories: List[int],
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        NMS-like filtering: suppress predictions that are too close to a
        higher-confidence prediction of the same category.

        Uses mean Chamfer distance between polylines as the overlap criterion.

        Args:
            polylines: List of polyline arrays (already sorted by confidence desc)
            scores: Confidence scores (sorted desc)
            categories: Category IDs

        Returns:
            Filtered (polylines, scores, categories) tuples.
        """
        if len(polylines) <= 1:
            return polylines, scores, categories

        keep = [True] * len(polylines)

        for i in range(len(polylines)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(polylines)):
                if not keep[j]:
                    continue
                # Only suppress within same category
                if categories[i] != categories[j]:
                    continue

                # Compute Chamfer distance between polylines i and j
                dist = self._chamfer_distance(polylines[i], polylines[j])
                if dist < self.nms_threshold:
                    # Suppress j (lower confidence)
                    keep[j] = False

        kept_polylines = [p for p, k in zip(polylines, keep) if k]
        kept_scores = [s for s, k in zip(scores, keep) if k]
        kept_categories = [c for c, k in zip(categories, keep) if k]

        return kept_polylines, kept_scores, kept_categories

    @staticmethod
    def _chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        """Compute mean Chamfer distance between two point sets."""
        if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
            return float("inf")

        diff = pts_a[:, None, :] - pts_b[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        d_a_to_b = np.mean(np.min(dist_matrix, axis=1))
        d_b_to_a = np.mean(np.min(dist_matrix, axis=0))

        return float((d_a_to_b + d_b_to_a) / 2.0)


class MapVisualizer:
    """
    Visualization utilities for MapTR predictions and ground truth.

    Produces BEV (Bird's Eye View) plots, camera image grids, and combined
    visualizations for qualitative evaluation.
    """

    def __init__(
        self,
        bev_x_range: Tuple[float, float] = (-50.0, 50.0),
        bev_y_range: Tuple[float, float] = (-50.0, 50.0),
        figsize_bev: Tuple[float, float] = (10, 10),
        figsize_cameras: Tuple[float, float] = (16, 6),
        figsize_combined: Tuple[float, float] = (20, 12),
        point_size: float = 12.0,
        line_width: float = 2.0,
        gt_line_width: float = 2.5,
        dpi: int = 150,
    ):
        """
        Args:
            bev_x_range: X-axis range for BEV canvas (meters).
            bev_y_range: Y-axis range for BEV canvas (meters).
            figsize_bev: Figure size for BEV-only plots.
            figsize_cameras: Figure size for camera grid plots.
            figsize_combined: Figure size for combined plots.
            point_size: Size of polyline vertex dots.
            line_width: Line width for predicted polylines.
            gt_line_width: Line width for GT polylines.
            dpi: Resolution for saved images.
        """
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.figsize_bev = figsize_bev
        self.figsize_cameras = figsize_cameras
        self.figsize_combined = figsize_combined
        self.point_size = point_size
        self.line_width = line_width
        self.gt_line_width = gt_line_width
        self.dpi = dpi

    def visualize_bev(
        self,
        predictions: Dict[str, Any],
        gt: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Draw predicted map elements on a BEV canvas.

        Args:
            predictions: Dict with 'polylines', 'scores', 'categories'.
            gt: Optional GT dict with 'polylines', 'categories'.
            save_path: Path to save the figure. If None, returns without saving.
            title: Optional title for the plot.

        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize_bev)
        ax.set_xlim(self.bev_x_range)
        ax.set_ylim(self.bev_y_range)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        ax.grid(True, alpha=0.15, color="white", linewidth=0.5)
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)

        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        else:
            ax.set_title("MapTR BEV Predictions", fontsize=12, fontweight="bold")

        # Draw ego vehicle indicator at origin
        ego_rect = mpatches.FancyBboxPatch(
            (-1.0, -2.0), 2.0, 4.0,
            boxstyle="round,pad=0.1",
            facecolor="white", edgecolor="gray", alpha=0.7, linewidth=1.5,
        )
        ax.add_patch(ego_rect)
        ax.annotate("EGO", (0, 0), ha="center", va="center", fontsize=7, fontweight="bold")

        # Draw ground truth (if provided) as dashed lines
        if gt is not None:
            self._draw_polylines(
                ax, gt["polylines"], gt["categories"],
                scores=None, is_gt=True,
            )

        # Draw predictions as solid lines
        if predictions["polylines"]:
            self._draw_polylines(
                ax, predictions["polylines"], predictions["categories"],
                scores=predictions["scores"], is_gt=False,
            )

        # Legend
        legend_handles = self._build_legend(has_gt=(gt is not None))
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.8)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"BEV visualization saved to: {save_path}")

        return fig

    def _draw_polylines(
        self,
        ax: plt.Axes,
        polylines: List[np.ndarray],
        categories: List[int],
        scores: Optional[List[float]] = None,
        is_gt: bool = False,
    ):
        """Draw polylines on the BEV axes."""
        for idx, (pts, cat_id) in enumerate(zip(polylines, categories)):
            if isinstance(pts, list):
                pts = np.array(pts)

            if pts.ndim != 2 or pts.shape[0] < 2:
                continue

            cat_id = int(cat_id)
            if is_gt:
                color = GT_COLORS_BY_ID.get(cat_id, (0.7, 0.7, 0.7))
                linestyle = "--"
                lw = self.gt_line_width
                alpha = 0.7
                marker_alpha = 0.5
            else:
                color = CATEGORY_COLORS_BY_ID.get(cat_id, (0.7, 0.7, 0.7))
                linestyle = "-"
                lw = self.line_width
                alpha = 0.9
                marker_alpha = 0.9

            # Draw connecting lines
            ax.plot(
                pts[:, 0], pts[:, 1],
                color=color, linestyle=linestyle, linewidth=lw,
                alpha=alpha, zorder=3 if not is_gt else 2,
            )

            # Draw vertex points
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=[color], s=self.point_size,
                alpha=marker_alpha, edgecolors="none",
                zorder=4 if not is_gt else 2,
            )

            # Show confidence score near the midpoint of the polyline
            if scores is not None and not is_gt:
                mid_idx = len(pts) // 2
                score = scores[idx]
                ax.annotate(
                    f"{score:.2f}",
                    (pts[mid_idx, 0], pts[mid_idx, 1]),
                    fontsize=6, color="white", alpha=0.8,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.5, edgecolor="none"),
                    zorder=5,
                )

    def _build_legend(self, has_gt: bool = False) -> List[mpatches.Patch]:
        """Build legend handles for the visualization."""
        handles = []
        for cat_name, color in CATEGORY_COLORS.items():
            handles.append(mpatches.Patch(color=color, label=f"Pred: {cat_name}"))

        if has_gt:
            handles.append(plt.Line2D(
                [0], [0], color="white", linestyle="--", linewidth=2,
                label="Ground Truth",
            ))

        return handles

    def visualize_cameras(
        self,
        images: np.ndarray,
        save_path: Optional[str] = None,
        camera_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Display camera images in a grid layout.

        Args:
            images: Camera images array [N_cams, H, W, 3] in RGB uint8 or float [0, 1].
            save_path: Path to save the figure.
            camera_names: Optional list of camera names for subplot titles.

        Returns:
            matplotlib Figure object.
        """
        n_cams = images.shape[0]

        if camera_names is None:
            camera_names = [
                "FRONT_LEFT", "FRONT", "FRONT_RIGHT",
                "BACK_LEFT", "BACK", "BACK_RIGHT",
            ][:n_cams]

        # Determine grid layout
        if n_cams <= 3:
            nrows, ncols = 1, n_cams
        elif n_cams <= 6:
            nrows, ncols = 2, 3
        else:
            ncols = 4
            nrows = (n_cams + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=self.figsize_cameras)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)

        for cam_idx in range(n_cams):
            row = cam_idx // ncols
            col = cam_idx % ncols
            ax = axes[row, col]

            img = images[cam_idx]
            # Normalize if needed
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = np.clip(img, 0, 1)
            elif img.max() > 1:
                img = img.astype(np.float32) / 255.0

            ax.imshow(img)
            ax.set_title(camera_names[cam_idx] if cam_idx < len(camera_names) else f"Cam {cam_idx}",
                        fontsize=9)
            ax.axis("off")

        # Hide empty subplots
        for idx in range(n_cams, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis("off")

        plt.suptitle("Multi-Camera Images", fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Camera visualization saved to: {save_path}")

        return fig

    def visualize_combined(
        self,
        images: np.ndarray,
        predictions: Dict[str, Any],
        gt: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        camera_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Combined visualization with camera images on top and BEV map below.

        Args:
            images: Camera images [N_cams, H, W, 3].
            predictions: Prediction dict with 'polylines', 'scores', 'categories'.
            gt: Optional GT dict.
            save_path: Path to save.
            title: Optional overall title.
            camera_names: Optional camera names.

        Returns:
            matplotlib Figure object.
        """
        n_cams = images.shape[0]

        if camera_names is None:
            camera_names = [
                "FRONT_LEFT", "FRONT", "FRONT_RIGHT",
                "BACK_LEFT", "BACK", "BACK_RIGHT",
            ][:n_cams]

        # Layout: cameras on top rows, BEV on bottom spanning full width
        cam_cols = min(n_cams, 3)
        cam_rows = (n_cams + cam_cols - 1) // cam_cols

        fig = plt.figure(figsize=self.figsize_combined)

        # Use GridSpec for flexible layout
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(cam_rows + 2, cam_cols, figure=fig, hspace=0.3, wspace=0.2)

        # Camera images in top row(s)
        for cam_idx in range(n_cams):
            row = cam_idx // cam_cols
            col = cam_idx % cam_cols
            ax_cam = fig.add_subplot(gs[row, col])

            img = images[cam_idx]
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = np.clip(img, 0, 1)
            elif img.max() > 1:
                img = img.astype(np.float32) / 255.0

            ax_cam.imshow(img)
            ax_cam.set_title(
                camera_names[cam_idx] if cam_idx < len(camera_names) else f"Cam {cam_idx}",
                fontsize=8,
            )
            ax_cam.axis("off")

        # BEV visualization in bottom rows
        ax_bev = fig.add_subplot(gs[cam_rows:, :])
        ax_bev.set_xlim(self.bev_x_range)
        ax_bev.set_ylim(self.bev_y_range)
        ax_bev.set_aspect("equal")
        ax_bev.set_facecolor("#1a1a2e")
        ax_bev.grid(True, alpha=0.15, color="white", linewidth=0.5)
        ax_bev.set_xlabel("X (meters)", fontsize=9)
        ax_bev.set_ylabel("Y (meters)", fontsize=9)
        ax_bev.set_title("BEV Map Prediction", fontsize=10, fontweight="bold")

        # Ego vehicle
        ego_rect = mpatches.FancyBboxPatch(
            (-1.0, -2.0), 2.0, 4.0,
            boxstyle="round,pad=0.1",
            facecolor="white", edgecolor="gray", alpha=0.7, linewidth=1.5,
        )
        ax_bev.add_patch(ego_rect)
        ax_bev.annotate("EGO", (0, 0), ha="center", va="center", fontsize=7, fontweight="bold")

        # Draw GT
        if gt is not None:
            self._draw_polylines(
                ax_bev, gt["polylines"], gt["categories"],
                scores=None, is_gt=True,
            )

        # Draw predictions
        if predictions["polylines"]:
            self._draw_polylines(
                ax_bev, predictions["polylines"], predictions["categories"],
                scores=predictions["scores"], is_gt=False,
            )

        # Legend
        legend_handles = self._build_legend(has_gt=(gt is not None))
        ax_bev.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.8)

        if title:
            plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Combined visualization saved to: {save_path}")

        return fig


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
        device: Target device.
        bev_h: BEV grid height.
        bev_w: BEV grid width.
        num_queries: Number of map element queries.
        num_points_per_query: Number of points per polyline.

    Returns:
        Loaded model in eval mode.
    """
    from backbone import ResNet50FPN
    from gkt import GKT

    class MapTRModel(nn.Module):
        """Minimal MapTR model for inference."""

        def __init__(self, bev_h, bev_w, num_queries, num_points_per_query, embed_dim=256):
            super().__init__()
            self.bev_h = bev_h
            self.bev_w = bev_w
            self.num_queries = num_queries
            self.num_points_per_query = num_points_per_query
            self.embed_dim = embed_dim

            self.backbone = ResNet50FPN(pretrained=False, fpn_out_channels=embed_dim)
            self.bev_encoder = GKT(
                embed_dim=embed_dim, bev_h=bev_h, bev_w=bev_w, num_layers=3,
            )

            self.query_embed = nn.Embedding(num_queries, embed_dim)
            self.query_pos = nn.Embedding(num_queries, embed_dim)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=1024,
                dropout=0.1, batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

            self.class_head = nn.Linear(embed_dim, NUM_CATEGORIES + 1)
            self.points_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, num_points_per_query * 2),
            )

        def forward(self, images, intrinsics, extrinsics):
            B = images.shape[0]
            fpn_feats = self.backbone(images)
            bev_feat = self.bev_encoder(fpn_feats, intrinsics, extrinsics)
            memory = bev_feat.flatten(2).permute(0, 2, 1)

            query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
            query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
            tgt = query_embed + query_pos

            decoded = self.decoder(tgt, memory)

            class_logits = self.class_head(decoded)
            points_raw = self.points_head(decoded)
            point_coords = points_raw.reshape(B, self.num_queries, self.num_points_per_query, 2)
            point_coords = torch.sigmoid(point_coords)

            return class_logits, point_coords

    model = MapTRModel(bev_h, bev_w, num_queries, num_points_per_query)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            cleaned_state_dict[new_key] = v

        model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}, using random weights.")

    model = model.to(device)
    model.eval()
    return model


def load_sample(
    data_root: str,
    sample_idx: int,
    bev_x_range: Tuple[float, float] = (-50.0, 50.0),
    bev_y_range: Tuple[float, float] = (-50.0, 50.0),
) -> Dict[str, Any]:
    """
    Load a single sample from the dataset for inference.

    Args:
        data_root: Root directory of the dataset.
        sample_idx: Index of the sample to load.
        bev_x_range: BEV x range.
        bev_y_range: BEV y range.

    Returns:
        Sample dict with images, camera params, and optional GT.
    """
    import pickle
    from PIL import Image
    from torchvision import transforms

    # Load sample info
    info_path = os.path.join(data_root, "val_infos.pkl")
    if os.path.exists(info_path):
        with open(info_path, "rb") as f:
            infos = pickle.load(f)
    else:
        json_path = os.path.join(data_root, "val_infos.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                infos = json.load(f)
        else:
            raise FileNotFoundError(
                f"No data info found. Expected {info_path} or {json_path}"
            )

    if sample_idx >= len(infos):
        raise IndexError(f"sample_idx {sample_idx} exceeds dataset size {len(infos)}")

    info = infos[sample_idx]

    # Image preprocessing
    img_transform = transforms.Compose([
        transforms.Resize((480, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # For visualization (un-normalized)
    vis_transform = transforms.Compose([
        transforms.Resize((480, 800)),
        transforms.ToTensor(),
    ])

    images_tensor = []
    images_vis = []
    for cam_path in info["cam_paths"]:
        full_path = os.path.join(data_root, cam_path)
        try:
            img = Image.open(full_path).convert("RGB")
            images_tensor.append(img_transform(img))
            images_vis.append(vis_transform(img).permute(1, 2, 0).numpy())  # [H, W, 3]
        except Exception as e:
            print(f"Warning: Could not load image {full_path}: {e}")
            images_tensor.append(torch.zeros(3, 480, 800))
            images_vis.append(np.zeros((480, 800, 3), dtype=np.float32))

    images_tensor = torch.stack(images_tensor, dim=0)  # [N_cams, 3, H, W]
    images_vis = np.stack(images_vis, axis=0)  # [N_cams, H, W, 3]

    # Camera parameters
    intrinsics = torch.tensor(info["intrinsics"], dtype=torch.float32)
    extrinsics = torch.tensor(info["extrinsics"], dtype=torch.float32)

    # Ground truth (if available)
    gt = None
    if "annotations" in info and info["annotations"]:
        gt_polylines = []
        gt_categories = []
        for ann in info["annotations"]:
            pts = np.array(ann["points"], dtype=np.float32)
            gt_polylines.append(pts)
            gt_categories.append(ann["category_id"])
        gt = {"polylines": gt_polylines, "categories": gt_categories}

    sample = {
        "images": images_tensor,           # [N_cams, 3, H, W] preprocessed
        "images_vis": images_vis,           # [N_cams, H, W, 3] for visualization
        "intrinsics": intrinsics,           # [N_cams, 3, 3]
        "extrinsics": extrinsics,           # [N_cams, 4, 4]
        "gt": gt,
        "sample_token": info.get("token", f"sample_{sample_idx}"),
    }

    return sample


@torch.no_grad()
def run_inference(
    model: nn.Module,
    sample: Dict[str, Any],
    post_processor: PostProcessor,
    device: torch.device,
) -> Tuple[Dict[str, Any], float]:
    """
    Run inference on a single sample.

    Args:
        model: MapTR model in eval mode.
        sample: Sample dict from load_sample.
        post_processor: PostProcessor instance.
        device: Compute device.

    Returns:
        Tuple of (predictions dict, inference time in ms).
    """
    # Prepare inputs: add batch dimension
    images = sample["images"].unsqueeze(0).to(device)         # [1, N_cams, 3, H, W]
    intrinsics = sample["intrinsics"].unsqueeze(0).to(device)  # [1, N_cams, 3, 3]
    extrinsics = sample["extrinsics"].unsqueeze(0).to(device)  # [1, N_cams, 4, 4]

    # Warm-up (first run can be slow due to CUDA compilation)
    start_time = time.time()
    class_logits, point_coords = model(images, intrinsics, extrinsics)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_ms = (time.time() - start_time) * 1000

    # Post-process
    results = post_processor.process(class_logits, point_coords)
    predictions = results[0]  # Single sample

    return predictions, inference_time_ms


@torch.no_grad()
def run_batch_inference(
    model: nn.Module,
    images_batch: torch.Tensor,
    intrinsics_batch: torch.Tensor,
    extrinsics_batch: torch.Tensor,
    post_processor: PostProcessor,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run inference on a batch of samples.

    Args:
        model: MapTR model in eval mode.
        images_batch: [B, N_cams, 3, H, W]
        intrinsics_batch: [B, N_cams, 3, 3]
        extrinsics_batch: [B, N_cams, 4, 4]
        post_processor: PostProcessor instance.
        device: Compute device.

    Returns:
        Tuple of (list of prediction dicts, inference time in ms).
    """
    images_batch = images_batch.to(device)
    intrinsics_batch = intrinsics_batch.to(device)
    extrinsics_batch = extrinsics_batch.to(device)

    start_time = time.time()
    class_logits, point_coords = model(images_batch, intrinsics_batch, extrinsics_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_ms = (time.time() - start_time) * 1000

    predictions = post_processor.process(class_logits, point_coords)

    return predictions, inference_time_ms


def main():
    parser = argparse.ArgumentParser(
        description="MapTR inference and visualization."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--sample_idx", type=int, default=0,
        help="Index of the sample to run inference on"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./inference_output",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.3,
        help="Minimum confidence score to visualize predictions"
    )
    parser.add_argument(
        "--nms_threshold", type=float, default=0.0,
        help="Chamfer distance threshold for NMS (0 = disabled)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'"
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
        help="Number of map element queries"
    )
    parser.add_argument(
        "--num_points", type=int, default=20,
        help="Number of points per polyline"
    )
    parser.add_argument(
        "--show_gt", action="store_true", default=True,
        help="Overlay ground truth on BEV visualization"
    )
    parser.add_argument(
        "--no_gt", action="store_true",
        help="Do not show ground truth overlay"
    )
    parser.add_argument(
        "--save_predictions", action="store_true",
        help="Save raw predictions to JSON"
    )
    parser.add_argument(
        "--batch_mode", action="store_true",
        help="Run inference on multiple consecutive samples"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of samples in batch mode"
    )

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    bev_x_range = tuple(args.bev_x_range)
    bev_y_range = tuple(args.bev_y_range)

    # Build model
    print("Loading model...")
    model = build_model(
        checkpoint_path=args.checkpoint,
        device=device,
        num_queries=args.num_queries,
        num_points_per_query=args.num_points,
    )

    # Create post-processor
    post_processor = PostProcessor(
        confidence_threshold=args.confidence_threshold,
        bev_x_range=bev_x_range,
        bev_y_range=bev_y_range,
        nms_threshold=args.nms_threshold,
    )

    # Create visualizer
    visualizer = MapVisualizer(
        bev_x_range=bev_x_range,
        bev_y_range=bev_y_range,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    show_gt = args.show_gt and not args.no_gt

    if args.batch_mode:
        # Batch mode: process multiple samples
        print(f"\nRunning batch inference on samples {args.sample_idx} to "
              f"{args.sample_idx + args.num_samples - 1}...")

        for i in range(args.num_samples):
            idx = args.sample_idx + i
            try:
                sample = load_sample(data_root=args.data_root, sample_idx=idx,
                                    bev_x_range=bev_x_range, bev_y_range=bev_y_range)
            except (IndexError, FileNotFoundError) as e:
                print(f"  Skipping sample {idx}: {e}")
                continue

            predictions, time_ms = run_inference(model, sample, post_processor, device)

            n_preds = len(predictions["polylines"])
            print(f"  Sample {idx}: {n_preds} predictions, {time_ms:.1f} ms")

            # Save BEV visualization
            gt = sample["gt"] if show_gt else None
            save_path = os.path.join(args.output_dir, f"sample_{idx:04d}_bev.png")
            visualizer.visualize_bev(
                predictions, gt=gt, save_path=save_path,
                title=f"Sample {idx} ({n_preds} predictions, {time_ms:.0f}ms)",
            )
            plt.close()

            # Save combined visualization
            save_path_combined = os.path.join(args.output_dir, f"sample_{idx:04d}_combined.png")
            visualizer.visualize_combined(
                images=sample["images_vis"],
                predictions=predictions,
                gt=gt,
                save_path=save_path_combined,
                title=f"MapTR Inference - Sample {idx}",
            )
            plt.close()

            # Save predictions JSON if requested
            if args.save_predictions:
                pred_json_path = os.path.join(args.output_dir, f"sample_{idx:04d}_pred.json")
                pred_serializable = {
                    "polylines": [p.tolist() if isinstance(p, np.ndarray) else p
                                 for p in predictions["polylines"]],
                    "scores": predictions["scores"],
                    "categories": predictions["categories"],
                    "inference_time_ms": time_ms,
                }
                with open(pred_json_path, "w") as f:
                    json.dump(pred_serializable, f, indent=2)

    else:
        # Single sample mode
        print(f"\nLoading sample {args.sample_idx}...")
        sample = load_sample(
            data_root=args.data_root,
            sample_idx=args.sample_idx,
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
        )

        print("Running inference...")
        predictions, time_ms = run_inference(model, sample, post_processor, device)

        n_preds = len(predictions["polylines"])
        print(f"  Predictions: {n_preds} map elements")
        print(f"  Inference time: {time_ms:.1f} ms")

        # Print per-category counts
        cat_counts = {name: 0 for name in CATEGORY_NAMES}
        for cat_id in predictions["categories"]:
            if cat_id < NUM_CATEGORIES:
                cat_counts[CATEGORY_NAMES[cat_id]] += 1
        for name, count in cat_counts.items():
            print(f"    {name}: {count}")

        gt = sample["gt"] if show_gt else None

        # BEV visualization
        bev_path = os.path.join(args.output_dir, f"sample_{args.sample_idx:04d}_bev.png")
        visualizer.visualize_bev(
            predictions, gt=gt, save_path=bev_path,
            title=f"MapTR - Sample {args.sample_idx} ({n_preds} predictions)",
        )
        plt.close()

        # Camera visualization
        cam_path = os.path.join(args.output_dir, f"sample_{args.sample_idx:04d}_cameras.png")
        visualizer.visualize_cameras(
            images=sample["images_vis"],
            save_path=cam_path,
        )
        plt.close()

        # Combined visualization
        combined_path = os.path.join(args.output_dir, f"sample_{args.sample_idx:04d}_combined.png")
        visualizer.visualize_combined(
            images=sample["images_vis"],
            predictions=predictions,
            gt=gt,
            save_path=combined_path,
            title=f"MapTR Inference - Sample {args.sample_idx}",
        )
        plt.close()

        # Save predictions JSON if requested
        if args.save_predictions:
            pred_json_path = os.path.join(args.output_dir, f"sample_{args.sample_idx:04d}_pred.json")
            pred_serializable = {
                "sample_idx": args.sample_idx,
                "sample_token": sample["sample_token"],
                "num_predictions": n_preds,
                "inference_time_ms": time_ms,
                "confidence_threshold": args.confidence_threshold,
                "polylines": [p.tolist() if isinstance(p, np.ndarray) else p
                             for p in predictions["polylines"]],
                "scores": predictions["scores"],
                "categories": predictions["categories"],
                "category_names": [CATEGORY_NAMES[c] for c in predictions["categories"]],
            }
            with open(pred_json_path, "w") as f:
                json.dump(pred_serializable, f, indent=2)
            print(f"  Predictions saved to: {pred_json_path}")

        print(f"\nVisualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
