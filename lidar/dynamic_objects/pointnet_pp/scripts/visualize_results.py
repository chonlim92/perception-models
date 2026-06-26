#!/usr/bin/env python3
"""
Visualization script for PointNet++ 3D object detection results.

Supports 3D interactive visualization (Open3D) and Bird's Eye View (matplotlib).
Loads point clouds in KITTI .bin or .npy format and overlays predicted/ground-truth
3D bounding boxes with per-class coloring and confidence scores.

Usage:
    python visualize_results.py --point_cloud cloud.bin --predictions preds.npy --mode 3d
    python visualize_results.py --point_cloud cloud.npy --predictions preds.json --mode bev --save output.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "Car": [1.0, 0.0, 0.0],         # Red
    "Pedestrian": [0.0, 0.0, 1.0],  # Blue
    "Cyclist": [1.0, 1.0, 0.0],     # Yellow
    "Truck": [1.0, 0.5, 0.0],       # Orange
    "Van": [0.5, 0.0, 0.5],         # Purple
    "Tram": [0.0, 1.0, 1.0],        # Cyan
    "Misc": [0.5, 0.5, 0.5],        # Gray
}

GT_COLOR = [0.0, 1.0, 0.0]  # Green for ground truth

# Matplotlib equivalents (0-1 RGBA)
CLASS_COLORS_MPL = {
    "Car": (1.0, 0.0, 0.0, 0.8),
    "Pedestrian": (0.0, 0.0, 1.0, 0.8),
    "Cyclist": (1.0, 1.0, 0.0, 0.8),
    "Truck": (1.0, 0.5, 0.0, 0.8),
    "Van": (0.5, 0.0, 0.5, 0.8),
    "Tram": (0.0, 1.0, 1.0, 0.8),
    "Misc": (0.5, 0.5, 0.5, 0.8),
}

GT_COLOR_MPL = (0.0, 1.0, 0.0, 0.9)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize PointNet++ 3D object detection results."
    )
    parser.add_argument(
        "--point_cloud", type=str, required=True,
        help="Path to point cloud file (.bin KITTI format or .npy)"
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions file (.npy or .json)"
    )
    parser.add_argument(
        "--ground_truth", type=str, default=None,
        help="Path to ground truth labels (.npy or .json, optional)"
    )
    parser.add_argument(
        "--mode", type=str, default="3d", choices=["3d", "bev"],
        help="Visualization mode: '3d' for interactive Open3D, 'bev' for bird's eye view"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save screenshot (PNG)"
    )
    parser.add_argument(
        "--classes", type=str, default="Car,Pedestrian,Cyclist",
        help="Comma-separated class names (default: Car,Pedestrian,Cyclist)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Point cloud loading
# ---------------------------------------------------------------------------

def load_point_cloud(filepath: str) -> np.ndarray:
    """
    Load point cloud from .bin (KITTI format: N x 4 float32) or .npy file.
    Returns array of shape (N, 4) with columns [x, y, z, reflectance].
    """
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"[ERROR] Point cloud file not found: {filepath}")

    if path.suffix == ".bin":
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    elif path.suffix == ".npy":
        points = np.load(str(path))
        if points.ndim == 1:
            points = points.reshape(-1, 4)
        if points.shape[1] < 4:
            # Pad with zeros for reflectance if missing
            pad = np.zeros((points.shape[0], 4 - points.shape[1]), dtype=np.float32)
            points = np.hstack([points, pad])
    else:
        sys.exit(f"[ERROR] Unsupported point cloud format: {path.suffix}")

    return points


# ---------------------------------------------------------------------------
# Predictions / ground truth loading
# ---------------------------------------------------------------------------

def load_detections(filepath: str, class_names: list) -> list:
    """
    Load detection results from .npy or .json.

    Expected format (per detection):
        {
            "class_id": int or "class_name": str,
            "bbox": [x, y, z, w, h, l, yaw],
            "score": float  (confidence, optional for GT)
        }

    For .npy: array of shape (N, 8) or (N, 9) where columns are
        [class_id, x, y, z, w, h, l, yaw] or
        [class_id, x, y, z, w, h, l, yaw, score]

    Returns list of dicts with keys: class_name, bbox (7,), score.
    """
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"[ERROR] Detections file not found: {filepath}")

    detections = []

    if path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "detections" in data:
            data = data["detections"]
        for det in data:
            class_name = det.get("class_name", None)
            if class_name is None:
                cid = int(det.get("class_id", 0))
                class_name = class_names[cid] if cid < len(class_names) else "Misc"
            bbox = np.array(det["bbox"], dtype=np.float64)
            score = float(det.get("score", 1.0))
            detections.append({
                "class_name": class_name,
                "bbox": bbox,
                "score": score,
            })

    elif path.suffix == ".npy":
        arr = np.load(str(path))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for row in arr:
            cid = int(row[0])
            class_name = class_names[cid] if cid < len(class_names) else "Misc"
            bbox = row[1:8].astype(np.float64)
            score = float(row[8]) if row.shape[0] > 8 else 1.0
            detections.append({
                "class_name": class_name,
                "bbox": bbox,
                "score": score,
            })
    else:
        sys.exit(f"[ERROR] Unsupported detections format: {path.suffix}")

    return detections


# ---------------------------------------------------------------------------
# Coloring helpers
# ---------------------------------------------------------------------------

def color_by_height(points: np.ndarray) -> np.ndarray:
    """
    Map point height (z) to a blue-to-red colormap.
    Returns (N, 3) array of RGB colors in [0, 1].
    """
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-6:
        z_norm = np.zeros_like(z)
    else:
        z_norm = (z - z_min) / (z_max - z_min)

    colors = np.zeros((len(z), 3), dtype=np.float64)
    colors[:, 0] = z_norm          # Red channel increases with height
    colors[:, 2] = 1.0 - z_norm   # Blue channel decreases with height
    colors[:, 1] = 0.3            # Slight green tint for visibility
    return colors


def color_by_reflectance(points: np.ndarray) -> np.ndarray:
    """
    Map reflectance (4th column) to grayscale with warm tint.
    Returns (N, 3) array of RGB colors in [0, 1].
    """
    r = points[:, 3]
    r_min, r_max = r.min(), r.max()
    if r_max - r_min < 1e-6:
        r_norm = np.ones_like(r) * 0.5
    else:
        r_norm = (r - r_min) / (r_max - r_min)

    colors = np.zeros((len(r), 3), dtype=np.float64)
    colors[:, 0] = r_norm * 0.9 + 0.1
    colors[:, 1] = r_norm * 0.7 + 0.1
    colors[:, 2] = r_norm * 0.4 + 0.1
    return colors


# ---------------------------------------------------------------------------
# 3D bounding box creation (Open3D)
# ---------------------------------------------------------------------------

def create_bbox_lineset(bbox: np.ndarray, color: list, line_width: float = 1.0):
    """
    Create an Open3D LineSet representing an oriented 3D bounding box.

    Parameters
    ----------
    bbox : array of shape (7,) -> [x, y, z, w, h, l, yaw]
        x, y, z: center position
        w: width (along x-axis before rotation)
        h: height (along z-axis)
        l: length (along y-axis before rotation)
        yaw: rotation around z-axis (radians)
    color : list of [R, G, B] in [0, 1]
    line_width : not directly supported in Open3D LineSet but kept for API consistency

    Returns
    -------
    open3d.geometry.LineSet
    """
    import open3d as o3d

    x, y, z, w, h, l, yaw = bbox
    # Half dimensions
    hw, hh, hl = w / 2.0, h / 2.0, l / 2.0

    # 8 corners in local frame (before rotation)
    # Bottom 4 corners, then top 4 corners
    corners_local = np.array([
        [-hw, -hl, -hh],
        [+hw, -hl, -hh],
        [+hw, +hl, -hh],
        [-hw, +hl, -hh],
        [-hw, -hl, +hh],
        [+hw, -hl, +hh],
        [+hw, +hl, +hh],
        [-hw, +hl, +hh],
    ])

    # Rotation matrix around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1],
    ])

    # Rotate and translate
    corners_world = (R @ corners_local.T).T + np.array([x, y, z])

    # 12 edges of the box
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners_world)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return lineset


def create_label_3d(text: str, position: np.ndarray, color: list):
    """
    Create a small sphere at the label position as a marker.
    Open3D does not natively render text in 3D; we use a small sphere
    and print the label info to console.
    """
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    return sphere


# ---------------------------------------------------------------------------
# 3D Visualization (Open3D)
# ---------------------------------------------------------------------------

def visualize_3d(points: np.ndarray, predictions: list, ground_truth: list,
                 class_names: list, save_path: str = None):
    """
    Interactive 3D visualization using Open3D.
    """
    import open3d as o3d

    print("\n" + "=" * 60)
    print("  PointNet++ 3D Visualization - Controls")
    print("=" * 60)
    print("  Mouse Left   : Rotate")
    print("  Mouse Right  : Pan")
    print("  Scroll       : Zoom")
    print("  [P]          : Toggle predictions visibility")
    print("  [G]          : Toggle ground truth visibility")
    print("  [H]          : Toggle height / reflectance coloring")
    print("  [R]          : Reset view")
    print("  [S]          : Save screenshot")
    print("  [Q] / Esc    : Quit")
    print("=" * 60 + "\n")

    # Create point cloud geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(color_by_height(points))

    # Create prediction bounding boxes
    pred_geometries = []
    for det in predictions:
        cls_name = det["class_name"]
        color = CLASS_COLORS.get(cls_name, [0.8, 0.8, 0.8])
        bbox = det["bbox"]
        score = det["score"]

        lineset = create_bbox_lineset(bbox, color)
        pred_geometries.append(lineset)

        # Label marker at top of box
        label_pos = np.array([bbox[0], bbox[1], bbox[2] + bbox[4] / 2.0 + 0.3])
        marker = create_label_3d(f"{cls_name} {score:.2f}", label_pos, color)
        pred_geometries.append(marker)

        print(f"  [PRED] {cls_name}: score={score:.3f}, "
              f"pos=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}), "
              f"dim=({bbox[3]:.1f}, {bbox[4]:.1f}, {bbox[5]:.1f}), "
              f"yaw={bbox[6]:.2f}")

    # Create ground truth bounding boxes
    gt_geometries = []
    for det in ground_truth:
        cls_name = det["class_name"]
        bbox = det["bbox"]

        lineset = create_bbox_lineset(bbox, GT_COLOR)
        gt_geometries.append(lineset)

        label_pos = np.array([bbox[0], bbox[1], bbox[2] + bbox[4] / 2.0 + 0.5])
        marker = create_label_3d(f"GT:{cls_name}", label_pos, GT_COLOR)
        gt_geometries.append(marker)

        print(f"  [GT]   {cls_name}: "
              f"pos=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}), "
              f"dim=({bbox[3]:.1f}, {bbox[4]:.1f}, {bbox[5]:.1f}), "
              f"yaw={bbox[6]:.2f}")

    # Coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0, origin=[0, 0, 0]
    )

    # Collect all geometries
    all_geometries = [pcd, coord_frame] + pred_geometries + gt_geometries

    # State for toggling
    vis_state = {
        "show_preds": True,
        "show_gt": True,
        "color_mode": "height",  # "height" or "reflectance"
    }

    def toggle_predictions(vis):
        vis_state["show_preds"] = not vis_state["show_preds"]
        status = "ON" if vis_state["show_preds"] else "OFF"
        print(f"  Predictions: {status}")
        # Rebuild scene
        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        if vis_state["show_preds"]:
            for g in pred_geometries:
                vis.add_geometry(g)
        if vis_state["show_gt"]:
            for g in gt_geometries:
                vis.add_geometry(g)
        return False

    def toggle_ground_truth(vis):
        vis_state["show_gt"] = not vis_state["show_gt"]
        status = "ON" if vis_state["show_gt"] else "OFF"
        print(f"  Ground Truth: {status}")
        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        if vis_state["show_preds"]:
            for g in pred_geometries:
                vis.add_geometry(g)
        if vis_state["show_gt"]:
            for g in gt_geometries:
                vis.add_geometry(g)
        return False

    def toggle_color_mode(vis):
        if vis_state["color_mode"] == "height":
            vis_state["color_mode"] = "reflectance"
            pcd.colors = o3d.utility.Vector3dVector(color_by_reflectance(points))
        else:
            vis_state["color_mode"] = "height"
            pcd.colors = o3d.utility.Vector3dVector(color_by_height(points))
        print(f"  Color mode: {vis_state['color_mode']}")
        vis.update_geometry(pcd)
        return False

    def save_screenshot(vis):
        out_path = save_path if save_path else "screenshot_3d.png"
        vis.capture_screen_image(out_path)
        print(f"  Screenshot saved to: {out_path}")
        return False

    # Setup visualizer with key callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PointNet++ Results - 3D View", width=1280, height=720)

    # Register key callbacks
    vis.register_key_callback(ord("P"), toggle_predictions)
    vis.register_key_callback(ord("G"), toggle_ground_truth)
    vis.register_key_callback(ord("H"), toggle_color_mode)
    vis.register_key_callback(ord("S"), save_screenshot)

    # Add geometries
    for geom in all_geometries:
        vis.add_geometry(geom)

    # Set rendering options
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.05, 0.05, 0.1])
    render_opt.line_width = 3.0

    # Set initial viewpoint (looking down slightly)
    view_ctrl = vis.get_view_control()
    view_ctrl.set_zoom(0.3)
    view_ctrl.set_front([0, -1, -0.3])
    view_ctrl.set_up([0, 0, 1])
    view_ctrl.set_lookat([0, 20, 0])

    # If save path specified, capture and close
    if save_path:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        print(f"\n  Screenshot saved to: {save_path}")
        vis.destroy_window()
    else:
        vis.run()
        vis.destroy_window()


# ---------------------------------------------------------------------------
# BEV (Bird's Eye View) Visualization (matplotlib)
# ---------------------------------------------------------------------------

def visualize_bev(points: np.ndarray, predictions: list, ground_truth: list,
                  class_names: list, save_path: str = None):
    """
    Bird's Eye View visualization using matplotlib.
    Projects points onto XY plane and draws rotated rectangles for boxes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_facecolor("#0a0a14")
    fig.patch.set_facecolor("#1a1a2e")

    # ---- Plot point cloud (BEV projection) ----
    # Color by height
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-6:
        z_norm = np.zeros_like(z)
    else:
        z_norm = (z - z_min) / (z_max - z_min)

    scatter = ax.scatter(
        points[:, 0], points[:, 1],
        c=z_norm, cmap="plasma", s=0.3, alpha=0.6,
        rasterized=True
    )

    # ---- Draw ground truth boxes ----
    gt_handles = []
    for det in ground_truth:
        cls_name = det["class_name"]
        bbox = det["bbox"]
        x, y, z_c, w, h, l, yaw = bbox

        # Rotated rectangle (width along x, length along y before rotation)
        rect = patches.Rectangle(
            (-w / 2, -l / 2), w, l,
            linewidth=1.5, edgecolor=GT_COLOR_MPL[:3], facecolor="none",
            linestyle="--", alpha=GT_COLOR_MPL[3]
        )
        # Apply rotation and translation
        t = Affine2D().rotate(yaw).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        # Direction indicator (front of box)
        front_x = x + (l / 2) * np.cos(yaw + np.pi / 2)
        front_y = y + (l / 2) * np.sin(yaw + np.pi / 2)
        ax.plot([x, front_x], [y, front_y], color=GT_COLOR_MPL[:3],
                linewidth=1.0, linestyle="--", alpha=0.7)

    # ---- Draw prediction boxes ----
    for det in predictions:
        cls_name = det["class_name"]
        bbox = det["bbox"]
        score = det["score"]
        x, y, z_c, w, h, l, yaw = bbox

        color = CLASS_COLORS_MPL.get(cls_name, (0.8, 0.8, 0.8, 0.8))

        rect = patches.Rectangle(
            (-w / 2, -l / 2), w, l,
            linewidth=2.0, edgecolor=color[:3], facecolor=color[:3],
            alpha=0.15, linestyle="-"
        )
        t = Affine2D().rotate(yaw).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        # Solid border
        rect_border = patches.Rectangle(
            (-w / 2, -l / 2), w, l,
            linewidth=2.0, edgecolor=color[:3], facecolor="none",
            alpha=color[3], linestyle="-"
        )
        t2 = Affine2D().rotate(yaw).translate(x, y) + ax.transData
        rect_border.set_transform(t2)
        ax.add_patch(rect_border)

        # Direction indicator (front of box)
        front_x = x + (l / 2) * np.cos(yaw + np.pi / 2)
        front_y = y + (l / 2) * np.sin(yaw + np.pi / 2)
        ax.plot([x, front_x], [y, front_y], color=color[:3], linewidth=1.5, alpha=0.9)

        # Label with score
        ax.text(
            x, y + l / 2 + 0.8,
            f"{cls_name} {score:.2f}",
            color=color[:3], fontsize=7, ha="center", va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none")
        )

    # ---- Grid overlay ----
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.3, color="gray")

    # Major grid every 10m
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    # Minor grid every 5m
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.grid(which="minor", linestyle=":", linewidth=0.2, alpha=0.2, color="gray")

    # ---- Labels and formatting ----
    ax.set_xlabel("X (m)", color="white", fontsize=11)
    ax.set_ylabel("Y (m)", color="white", fontsize=11)
    ax.set_title("PointNet++ Detection Results - Bird's Eye View",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white", labelsize=9)
    ax.set_aspect("equal")

    # Auto-range with padding
    x_range = points[:, 0]
    y_range = points[:, 1]
    pad = 5.0
    ax.set_xlim(x_range.min() - pad, x_range.max() + pad)
    ax.set_ylim(y_range.min() - pad, y_range.max() + pad)

    # Spine colors
    for spine in ax.spines.values():
        spine.set_color("gray")
        spine.set_linewidth(0.5)

    # ---- Legend ----
    legend_elements = []
    for cls_name in class_names:
        color = CLASS_COLORS_MPL.get(cls_name, (0.8, 0.8, 0.8, 0.8))
        legend_elements.append(
            Line2D([0], [0], color=color[:3], linewidth=2, label=f"Pred: {cls_name}")
        )
    if ground_truth:
        legend_elements.append(
            Line2D([0], [0], color=GT_COLOR_MPL[:3], linewidth=1.5,
                   linestyle="--", label="Ground Truth")
        )

    legend = ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=9, framealpha=0.7,
        facecolor="#2a2a3e", edgecolor="gray",
        labelcolor="white"
    )

    # Colorbar for height
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Height (normalized)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.set_ticklabels(
        [f"{t:.1f}" for t in cbar.get_ticks()], color="white", fontsize=8
    )

    plt.tight_layout()

    # ---- Save or show ----
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n  BEV screenshot saved to: {save_path}")
    else:
        print("\n  Displaying BEV plot. Close the window to exit.")
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    class_names = [c.strip() for c in args.classes.split(",")]

    print("\n" + "-" * 60)
    print("  PointNet++ Results Visualization")
    print("-" * 60)
    print(f"  Point cloud : {args.point_cloud}")
    print(f"  Predictions : {args.predictions}")
    print(f"  Ground truth: {args.ground_truth or 'None'}")
    print(f"  Mode        : {args.mode}")
    print(f"  Classes     : {class_names}")
    print(f"  Save path   : {args.save or 'None (interactive)'}")
    print("-" * 60)

    # Load data
    print("\n  Loading point cloud...")
    points = load_point_cloud(args.point_cloud)
    print(f"  Loaded {points.shape[0]:,} points, shape: {points.shape}")

    print("  Loading predictions...")
    predictions = load_detections(args.predictions, class_names)
    print(f"  Loaded {len(predictions)} predictions")

    ground_truth = []
    if args.ground_truth:
        print("  Loading ground truth...")
        ground_truth = load_detections(args.ground_truth, class_names)
        print(f"  Loaded {len(ground_truth)} ground truth boxes")

    # Summary
    print(f"\n  Detection summary:")
    for cls_name in class_names:
        pred_count = sum(1 for d in predictions if d["class_name"] == cls_name)
        gt_count = sum(1 for d in ground_truth if d["class_name"] == cls_name)
        if pred_count > 0 or gt_count > 0:
            print(f"    {cls_name:12s}: {pred_count} predictions, {gt_count} GT")

    # Visualize
    if args.mode == "3d":
        visualize_3d(points, predictions, ground_truth, class_names, args.save)
    elif args.mode == "bev":
        visualize_bev(points, predictions, ground_truth, class_names, args.save)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
