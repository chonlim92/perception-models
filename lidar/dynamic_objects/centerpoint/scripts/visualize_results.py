"""
CenterPoint Detection and Tracking Results Visualization Script.

Provides BEV (Bird's Eye View) and 3D visualization of CenterPoint model
predictions including bounding boxes, velocity arrows, and tracking IDs.
Supports single-frame visualization and sequence animation with video export.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Constants
# ============================================================================

NUSCENES_CLASSES = [
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

CLASS_COLORS: Dict[str, Tuple[float, float, float]] = {
    "car": (0.0, 0.8, 0.0),             # green
    "truck": (0.0, 0.0, 1.0),           # blue
    "bus": (0.0, 1.0, 1.0),             # cyan
    "pedestrian": (1.0, 0.0, 0.0),      # red
    "bicycle": (1.0, 0.647, 0.0),       # orange
    "motorcycle": (0.502, 0.0, 0.502),  # purple
    "barrier": (0.5, 0.5, 0.5),         # gray
    "trailer": (0.545, 0.271, 0.075),   # brown
    "construction_vehicle": (0.0, 0.0, 0.545),  # darkblue
    "traffic_cone": (1.0, 1.0, 0.0),    # yellow
}

# Matplotlib named colors for BEV plot
CLASS_COLORS_NAMED: Dict[str, str] = {
    "car": "green",
    "truck": "blue",
    "bus": "cyan",
    "pedestrian": "red",
    "bicycle": "orange",
    "motorcycle": "purple",
    "barrier": "gray",
    "trailer": "brown",
    "construction_vehicle": "darkblue",
    "traffic_cone": "yellow",
}


# ============================================================================
# Data Loading
# ============================================================================


def load_point_cloud(bin_path: str) -> np.ndarray:
    """
    Load a point cloud from a .bin file.

    Expected format: Nx5 float32 array with columns (x, y, z, intensity, time_lag).

    Args:
        bin_path: Path to the .bin point cloud file.

    Returns:
        Numpy array of shape (N, 5) with columns [x, y, z, intensity, time_lag].
    """
    points = np.fromfile(bin_path, dtype=np.float32)
    points = points.reshape(-1, 5)
    return points


def load_predictions(json_path: str) -> List[Dict]:
    """
    Load prediction results from a JSON file.

    Expected JSON structure:
    {
        "frame_id": "...",
        "timestamp": ...,
        "predictions": [
            {
                "box": [x, y, z, w, l, h, yaw],
                "score": 0.95,
                "class_name": "car",
                "velocity": [vx, vy],
                "track_id": 1
            },
            ...
        ]
    }

    Args:
        json_path: Path to the predictions JSON file.

    Returns:
        List of prediction dictionaries.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "predictions" in data:
            return data["predictions"]
        return [data]
    return data


def load_ground_truth(json_path: str) -> List[Dict]:
    """
    Load ground truth annotations from a JSON file.

    Expected format is similar to predictions but may lack score/track_id.

    Args:
        json_path: Path to the ground truth JSON file.

    Returns:
        List of ground truth annotation dictionaries.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "annotations" in data:
            return data["annotations"]
        if "ground_truth" in data:
            return data["ground_truth"]
        if "predictions" in data:
            return data["predictions"]
        return [data]
    return data


def filter_predictions(
    predictions: List[Dict], score_threshold: float
) -> List[Dict]:
    """
    Filter predictions by confidence score.

    Args:
        predictions: List of prediction dictionaries.
        score_threshold: Minimum score to keep.

    Returns:
        Filtered list of predictions.
    """
    filtered = []
    for pred in predictions:
        score = pred.get("score", 1.0)
        if score >= score_threshold:
            filtered.append(pred)
    return filtered


# ============================================================================
# Geometry Utilities
# ============================================================================


def get_box_corners_2d(
    x: float, y: float, w: float, l: float, yaw: float
) -> np.ndarray:
    """
    Compute the 4 corners of a 2D oriented bounding box.

    Args:
        x: Center x coordinate.
        y: Center y coordinate.
        w: Width of the box (along local y-axis).
        l: Length of the box (along local x-axis).
        yaw: Rotation angle around Z-axis in radians.

    Returns:
        Array of shape (4, 2) with corner coordinates in order:
        front-left, front-right, rear-right, rear-left.
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Half dimensions
    hl = l / 2.0
    hw = w / 2.0

    # Corners in local frame (length along x, width along y)
    corners_local = np.array(
        [
            [hl, hw],    # front-left
            [hl, -hw],   # front-right
            [-hl, -hw],  # rear-right
            [-hl, hw],   # rear-left
        ]
    )

    # Rotation matrix
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    # Rotate and translate
    corners_world = corners_local @ rotation.T + np.array([x, y])
    return corners_world


def get_box_corners_3d(
    x: float, y: float, z: float, w: float, l: float, h: float, yaw: float
) -> np.ndarray:
    """
    Compute the 8 corners of a 3D oriented bounding box.

    Args:
        x, y, z: Center coordinates of the box.
        w: Width (local y-axis).
        l: Length (local x-axis).
        h: Height (local z-axis).
        yaw: Rotation around Z-axis in radians.

    Returns:
        Array of shape (8, 3) with corner coordinates.
        Order: bottom 4 corners then top 4 corners,
        each group in front-left, front-right, rear-right, rear-left order.
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    hl = l / 2.0
    hw = w / 2.0
    hh = h / 2.0

    # 8 corners in local frame
    corners_local = np.array(
        [
            [hl, hw, -hh],    # bottom front-left
            [hl, -hw, -hh],   # bottom front-right
            [-hl, -hw, -hh],  # bottom rear-right
            [-hl, hw, -hh],   # bottom rear-left
            [hl, hw, hh],     # top front-left
            [hl, -hw, hh],    # top front-right
            [-hl, -hw, hh],   # top rear-right
            [-hl, hw, hh],    # top rear-left
        ]
    )

    # Rotation matrix (around Z-axis)
    rotation = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    corners_world = corners_local @ rotation.T + np.array([x, y, z])
    return corners_world


# ============================================================================
# BEV Visualization (Matplotlib)
# ============================================================================


def draw_bev_frame(
    ax,
    points: np.ndarray,
    predictions: List[Dict],
    ground_truth: Optional[List[Dict]] = None,
    vis_range: float = 60.0,
    show_velocity: bool = True,
    show_track_ids: bool = True,
    color_by: str = "height",
    frame_info: str = "",
):
    """
    Draw a single BEV frame on a matplotlib axes.

    Args:
        ax: Matplotlib axes object.
        points: Point cloud array of shape (N, 5).
        predictions: List of prediction dicts.
        ground_truth: Optional list of ground truth dicts.
        vis_range: Visualization range in meters.
        show_velocity: Whether to draw velocity arrows.
        show_track_ids: Whether to show track ID labels.
        color_by: Color points by 'height' or 'intensity'.
        frame_info: String with frame info to display in title.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, Polygon

    ax.clear()

    # Filter points within range
    mask = (
        (np.abs(points[:, 0]) <= vis_range)
        & (np.abs(points[:, 1]) <= vis_range)
    )
    pts = points[mask]

    # Color points
    if color_by == "intensity":
        colors = pts[:, 3]
        cmap = "viridis"
    else:
        colors = pts[:, 2]  # z-height
        cmap = "coolwarm"

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=colors,
        s=0.3,
        cmap=cmap,
        alpha=0.5,
        rasterized=True,
    )

    # Draw ground truth boxes (if provided) with dashed lines
    if ground_truth is not None:
        for gt in ground_truth:
            box = gt.get("box", None)
            if box is None:
                continue
            bx, by, bz, bw, bl, bh, byaw = box
            class_name = gt.get("class_name", "car")
            color = CLASS_COLORS_NAMED.get(class_name, "white")

            corners = get_box_corners_2d(bx, by, bw, bl, byaw)
            polygon = Polygon(
                corners, closed=True, fill=False,
                edgecolor=color, linewidth=1.0, linestyle="--", alpha=0.7
            )
            ax.add_patch(polygon)

    # Draw prediction boxes
    legend_entries = {}
    for pred in predictions:
        box = pred.get("box", None)
        if box is None:
            continue
        bx, by, bz, bw, bl, bh, byaw = box
        class_name = pred.get("class_name", "car")
        color = CLASS_COLORS_NAMED.get(class_name, "white")
        score = pred.get("score", 0.0)
        track_id = pred.get("track_id", None)
        velocity = pred.get("velocity", [0.0, 0.0])

        # Draw oriented rectangle
        corners = get_box_corners_2d(bx, by, bw, bl, byaw)
        polygon = Polygon(
            corners, closed=True, fill=False,
            edgecolor=color, linewidth=1.5, linestyle="-"
        )
        ax.add_patch(polygon)

        # Draw heading arrow (from center toward front)
        arrow_len = bl * 0.4
        head_x = bx + arrow_len * np.cos(byaw)
        head_y = by + arrow_len * np.sin(byaw)
        ax.annotate(
            "",
            xy=(head_x, head_y),
            xytext=(bx, by),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

        # Draw velocity arrow
        if show_velocity and velocity is not None:
            vx, vy = velocity[0], velocity[1]
            speed = np.sqrt(vx ** 2 + vy ** 2)
            if speed > 0.5:  # Only show if speed > 0.5 m/s
                vel_scale = 1.0  # 1 m/s = 1 meter in plot
                ax.annotate(
                    "",
                    xy=(bx + vx * vel_scale, by + vy * vel_scale),
                    xytext=(bx, by),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="magenta",
                        lw=2.0,
                        mutation_scale=10,
                    ),
                )

        # Draw track ID label
        if show_track_ids and track_id is not None:
            ax.text(
                bx + bl * 0.3,
                by + bw * 0.3,
                str(track_id),
                fontsize=7,
                color=color,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        # Collect legend entries
        if class_name not in legend_entries:
            legend_entries[class_name] = color

    # Build legend
    legend_patches = []
    for cls_name, cls_color in sorted(legend_entries.items()):
        patch = mpatches.Patch(color=cls_color, label=cls_name)
        legend_patches.append(patch)
    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7)

    # Axes formatting
    ax.set_xlim(-vis_range, vis_range)
    ax.set_ylim(-vis_range, vis_range)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    title = "CenterPoint BEV Visualization"
    if frame_info:
        title += f" | {frame_info}"
    ax.set_title(title, fontsize=10)


def visualize_bev(
    points: np.ndarray,
    predictions: List[Dict],
    ground_truth: Optional[List[Dict]] = None,
    vis_range: float = 60.0,
    show_velocity: bool = True,
    show_track_ids: bool = True,
    output_path: Optional[str] = None,
    frame_info: str = "",
):
    """
    Visualize a single frame in Bird's Eye View.

    Args:
        points: Point cloud array (N, 5).
        predictions: Filtered prediction list.
        ground_truth: Optional ground truth list.
        vis_range: Range in meters.
        show_velocity: Show velocity arrows.
        show_track_ids: Show track ID labels.
        output_path: If provided, save figure to this path.
        frame_info: Frame information string for title.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    draw_bev_frame(
        ax,
        points,
        predictions,
        ground_truth=ground_truth,
        vis_range=vis_range,
        show_velocity=show_velocity,
        show_track_ids=show_track_ids,
        frame_info=frame_info,
    )

    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved BEV image to: {output_path}")

    plt.show()
    plt.close(fig)


# ============================================================================
# 3D Visualization (Open3D)
# ============================================================================


def create_colormap_by_height(points: np.ndarray) -> np.ndarray:
    """
    Create RGB colors for points based on Z-height using a colormap.

    Args:
        points: Point cloud array (N, 5).

    Returns:
        Array of shape (N, 3) with RGB values in [0, 1].
    """
    import matplotlib.cm as cm

    z_values = points[:, 2]
    z_min = np.percentile(z_values, 1)
    z_max = np.percentile(z_values, 99)

    if z_max - z_min < 1e-6:
        z_max = z_min + 1.0

    z_normalized = np.clip((z_values - z_min) / (z_max - z_min), 0.0, 1.0)
    colormap = cm.get_cmap("jet")
    colors = colormap(z_normalized)[:, :3]  # Drop alpha channel
    return colors


def create_3d_box_lineset(
    box: List[float], color: Tuple[float, float, float]
):
    """
    Create an Open3D LineSet for a 3D bounding box.

    Args:
        box: [x, y, z, w, l, h, yaw] box parameters.
        color: RGB tuple for box color.

    Returns:
        Open3D LineSet geometry.
    """
    import open3d as o3d

    x, y, z, w, l, h, yaw = box
    corners = get_box_corners_3d(x, y, z, w, l, h, yaw)

    # Define 12 edges of the box
    lines = [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    colors_list = [color for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors_list)

    return line_set


def create_heading_arrow_lineset(
    box: List[float], color: Tuple[float, float, float]
):
    """
    Create a line from the box center toward the front to indicate heading.

    Args:
        box: [x, y, z, w, l, h, yaw] box parameters.
        color: RGB tuple.

    Returns:
        Open3D LineSet geometry.
    """
    import open3d as o3d

    x, y, z, w, l, h, yaw = box
    arrow_len = l * 0.6
    end_x = x + arrow_len * np.cos(yaw)
    end_y = y + arrow_len * np.sin(yaw)

    points = np.array([[x, y, z], [end_x, end_y, z]])
    lines = [[0, 1]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])

    return line_set


def visualize_3d(
    points: np.ndarray,
    predictions: List[Dict],
    ground_truth: Optional[List[Dict]] = None,
    show_track_ids: bool = True,
    show_velocity: bool = True,
):
    """
    Visualize a single frame in 3D using Open3D.

    Args:
        points: Point cloud array (N, 5).
        predictions: Filtered prediction list.
        ground_truth: Optional ground truth list.
        show_track_ids: Show track IDs (limited support in Open3D).
        show_velocity: Show velocity as lines.
    """
    import open3d as o3d

    geometries = []

    # Create point cloud geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = create_colormap_by_height(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)

    # Draw ground truth boxes (thinner lines, we simulate via slightly different shade)
    if ground_truth is not None:
        for gt in ground_truth:
            box = gt.get("box", None)
            if box is None:
                continue
            class_name = gt.get("class_name", "car")
            color = CLASS_COLORS.get(class_name, (1.0, 1.0, 1.0))
            # Dim the color for GT to distinguish from predictions
            gt_color = tuple(c * 0.5 for c in color)
            lineset = create_3d_box_lineset(box, gt_color)
            geometries.append(lineset)

    # Draw prediction boxes
    for pred in predictions:
        box = pred.get("box", None)
        if box is None:
            continue
        class_name = pred.get("class_name", "car")
        color = CLASS_COLORS.get(class_name, (1.0, 1.0, 1.0))

        # Box
        lineset = create_3d_box_lineset(box, color)
        geometries.append(lineset)

        # Heading arrow
        heading_lineset = create_heading_arrow_lineset(box, color)
        geometries.append(heading_lineset)

        # Velocity line
        if show_velocity:
            velocity = pred.get("velocity", [0.0, 0.0])
            if velocity is not None:
                vx, vy = velocity[0], velocity[1]
                speed = np.sqrt(vx ** 2 + vy ** 2)
                if speed > 0.5:
                    x, y, z = box[0], box[1], box[2]
                    vel_end = np.array([x + vx, y + vy, z])
                    vel_points = np.array([[x, y, z], vel_end])
                    vel_lines = [[0, 1]]
                    vel_lineset = o3d.geometry.LineSet()
                    vel_lineset.points = o3d.utility.Vector3dVector(vel_points)
                    vel_lineset.lines = o3d.utility.Vector2iVector(vel_lines)
                    vel_lineset.colors = o3d.utility.Vector3dVector(
                        [(1.0, 0.0, 1.0)]  # magenta
                    )
                    geometries.append(vel_lineset)

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0, origin=[0, 0, 0]
    )
    geometries.append(coord_frame)

    # Print track ID information to console since Open3D text support is limited
    if show_track_ids:
        print("\n--- Track IDs in current frame ---")
        for pred in predictions:
            track_id = pred.get("track_id", None)
            if track_id is not None:
                box = pred.get("box", [0, 0, 0, 0, 0, 0, 0])
                class_name = pred.get("class_name", "unknown")
                print(
                    f"  Track {track_id}: {class_name} at "
                    f"({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f})"
                )
        print("-----------------------------------\n")

    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="CenterPoint 3D Visualization", width=1600, height=900
    )

    for geom in geometries:
        vis.add_geometry(geom)

    # Set camera to top-down view initially
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0.0, 0.0, 1.0])
    view_ctl.set_lookat([0.0, 0.0, 0.0])
    view_ctl.set_up([0.0, 1.0, 0.0])
    view_ctl.set_zoom(0.15)

    # Render options
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.05, 0.05, 0.05])
    render_opt.point_size = 2.0

    vis.run()
    vis.destroy_window()


# ============================================================================
# Sequence Animation
# ============================================================================


def get_sorted_files(folder: str, extension: str) -> List[str]:
    """
    Get sorted list of files with a given extension from a folder.

    Args:
        folder: Directory path.
        extension: File extension (e.g., '.bin', '.json').

    Returns:
        Sorted list of full file paths.
    """
    folder_path = Path(folder)
    files = sorted(folder_path.glob(f"*{extension}"))
    return [str(f) for f in files]


def animate_bev_sequence(
    pc_folder: str,
    pred_folder: str,
    gt_folder: Optional[str] = None,
    vis_range: float = 60.0,
    show_velocity: bool = True,
    show_track_ids: bool = True,
    score_threshold: float = 0.3,
    output_path: Optional[str] = None,
    fps: int = 10,
):
    """
    Animate a sequence of frames in BEV mode and optionally save as video.

    Args:
        pc_folder: Folder containing .bin point cloud files.
        pred_folder: Folder containing prediction .json files.
        gt_folder: Optional folder containing ground truth .json files.
        vis_range: Visualization range in meters.
        show_velocity: Show velocity arrows.
        show_track_ids: Show track IDs.
        score_threshold: Minimum score to display.
        output_path: Path to save output video (.mp4).
        fps: Frames per second for animation/video.
    """
    import matplotlib.pyplot as plt

    pc_files = get_sorted_files(pc_folder, ".bin")
    pred_files = get_sorted_files(pred_folder, ".json")

    if len(pc_files) == 0:
        print(f"Error: No .bin files found in {pc_folder}")
        return
    if len(pred_files) == 0:
        print(f"Error: No .json files found in {pred_folder}")
        return

    num_frames = min(len(pc_files), len(pred_files))
    print(f"Animating {num_frames} frames at {fps} FPS...")

    gt_files = None
    if gt_folder is not None:
        gt_files = get_sorted_files(gt_folder, ".json")

    # Determine if we save to video
    save_video = output_path is not None and output_path.endswith(".mp4")
    save_frames = output_path is not None and not output_path.endswith(".mp4")

    frames_for_video = []

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    for frame_idx in range(num_frames):
        # Load data
        points = load_point_cloud(pc_files[frame_idx])
        predictions = load_predictions(pred_files[frame_idx])
        predictions = filter_predictions(predictions, score_threshold)

        ground_truth = None
        if gt_files is not None and frame_idx < len(gt_files):
            ground_truth = load_ground_truth(gt_files[frame_idx])

        frame_info = (
            f"Frame {frame_idx + 1}/{num_frames} | "
            f"File: {Path(pc_files[frame_idx]).stem}"
        )

        draw_bev_frame(
            ax,
            points,
            predictions,
            ground_truth=ground_truth,
            vis_range=vis_range,
            show_velocity=show_velocity,
            show_track_ids=show_track_ids,
            frame_info=frame_info,
        )

        if save_video:
            # Render figure to numpy array
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(height, width, 3)
            frames_for_video.append(image)
        elif save_frames:
            frame_output = os.path.join(
                output_path, f"frame_{frame_idx:06d}.png"
            )
            os.makedirs(output_path, exist_ok=True)
            fig.savefig(frame_output, dpi=100, bbox_inches="tight")
        else:
            plt.pause(1.0 / fps)

        if frame_idx % 10 == 0:
            print(f"  Processed frame {frame_idx + 1}/{num_frames}")

    plt.close(fig)

    # Save video if requested
    if save_video and frames_for_video:
        try:
            import imageio

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            writer = imageio.get_writer(output_path, fps=fps)
            for frame_image in frames_for_video:
                writer.append_data(frame_image)
            writer.close()
            print(f"Video saved to: {output_path}")
        except ImportError:
            # Fallback: use matplotlib animation
            print(
                "imageio not available. Attempting matplotlib animation..."
            )
            try:
                from matplotlib.animation import FFMpegWriter

                fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
                writer = FFMpegWriter(fps=fps)
                os.makedirs(
                    os.path.dirname(output_path) or ".", exist_ok=True
                )
                with writer.saving(fig2, output_path, dpi=100):
                    for fidx in range(num_frames):
                        pts = load_point_cloud(pc_files[fidx])
                        preds = load_predictions(pred_files[fidx])
                        preds = filter_predictions(preds, score_threshold)
                        gt = None
                        if gt_files and fidx < len(gt_files):
                            gt = load_ground_truth(gt_files[fidx])
                        info = (
                            f"Frame {fidx + 1}/{num_frames} | "
                            f"File: {Path(pc_files[fidx]).stem}"
                        )
                        draw_bev_frame(
                            ax2, pts, preds,
                            ground_truth=gt,
                            vis_range=vis_range,
                            show_velocity=show_velocity,
                            show_track_ids=show_track_ids,
                            frame_info=info,
                        )
                        writer.grab_frame()
                plt.close(fig2)
                print(f"Video saved to: {output_path}")
            except Exception as e:
                print(f"Failed to save video: {e}")
                print("Install imageio or ffmpeg for video export.")

    if save_frames:
        print(f"Frames saved to folder: {output_path}")

    print("Sequence animation complete.")


def animate_3d_sequence(
    pc_folder: str,
    pred_folder: str,
    gt_folder: Optional[str] = None,
    show_velocity: bool = True,
    show_track_ids: bool = True,
    score_threshold: float = 0.3,
    fps: int = 10,
):
    """
    Animate a sequence of frames in 3D using Open3D non-blocking visualizer.

    Args:
        pc_folder: Folder containing .bin point cloud files.
        pred_folder: Folder containing prediction .json files.
        gt_folder: Optional folder containing ground truth .json files.
        show_velocity: Show velocity lines.
        show_track_ids: Print track IDs to console.
        score_threshold: Minimum score threshold.
        fps: Target frames per second.
    """
    import time

    import open3d as o3d

    pc_files = get_sorted_files(pc_folder, ".bin")
    pred_files = get_sorted_files(pred_folder, ".json")

    if len(pc_files) == 0:
        print(f"Error: No .bin files found in {pc_folder}")
        return
    if len(pred_files) == 0:
        print(f"Error: No .json files found in {pred_folder}")
        return

    num_frames = min(len(pc_files), len(pred_files))
    gt_files = None
    if gt_folder is not None:
        gt_files = get_sorted_files(gt_folder, ".json")

    print(f"3D sequence animation: {num_frames} frames at {fps} FPS")
    print("Press Q or close window to stop.")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="CenterPoint 3D Sequence", width=1600, height=900
    )

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.05, 0.05, 0.05])
    render_opt.point_size = 2.0

    # Initialize with first frame
    first_points = load_point_cloud(pc_files[0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(first_points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(
        create_colormap_by_height(first_points)
    )
    vis.add_geometry(pcd)

    # Set initial camera
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0.0, -0.3, 1.0])
    view_ctl.set_lookat([0.0, 0.0, 0.0])
    view_ctl.set_up([0.0, 0.0, 1.0])
    view_ctl.set_zoom(0.12)

    # Keep track of added geometries for removal
    current_geometries = []

    frame_delay = 1.0 / fps

    for frame_idx in range(num_frames):
        frame_start = time.time()

        # Remove old geometries (except point cloud which we update in-place)
        for geom in current_geometries:
            vis.remove_geometry(geom, reset_bounding_box=False)
        current_geometries.clear()

        # Load frame data
        points = load_point_cloud(pc_files[frame_idx])
        predictions = load_predictions(pred_files[frame_idx])
        predictions = filter_predictions(predictions, score_threshold)

        ground_truth = None
        if gt_files is not None and frame_idx < len(gt_files):
            ground_truth = load_ground_truth(gt_files[frame_idx])

        # Update point cloud
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(
            create_colormap_by_height(points)
        )
        vis.update_geometry(pcd)

        # Draw GT boxes
        if ground_truth is not None:
            for gt in ground_truth:
                box = gt.get("box", None)
                if box is None:
                    continue
                class_name = gt.get("class_name", "car")
                color = CLASS_COLORS.get(class_name, (1.0, 1.0, 1.0))
                gt_color = tuple(c * 0.5 for c in color)
                lineset = create_3d_box_lineset(box, gt_color)
                vis.add_geometry(lineset, reset_bounding_box=False)
                current_geometries.append(lineset)

        # Draw prediction boxes
        for pred in predictions:
            box = pred.get("box", None)
            if box is None:
                continue
            class_name = pred.get("class_name", "car")
            color = CLASS_COLORS.get(class_name, (1.0, 1.0, 1.0))

            # Box lineset
            lineset = create_3d_box_lineset(box, color)
            vis.add_geometry(lineset, reset_bounding_box=False)
            current_geometries.append(lineset)

            # Heading arrow
            heading_ls = create_heading_arrow_lineset(box, color)
            vis.add_geometry(heading_ls, reset_bounding_box=False)
            current_geometries.append(heading_ls)

            # Velocity line
            if show_velocity:
                velocity = pred.get("velocity", [0.0, 0.0])
                if velocity is not None:
                    vx, vy = velocity[0], velocity[1]
                    speed = np.sqrt(vx ** 2 + vy ** 2)
                    if speed > 0.5:
                        bx, by, bz = box[0], box[1], box[2]
                        vel_pts = np.array(
                            [[bx, by, bz], [bx + vx, by + vy, bz]]
                        )
                        vel_ls = o3d.geometry.LineSet()
                        vel_ls.points = o3d.utility.Vector3dVector(vel_pts)
                        vel_ls.lines = o3d.utility.Vector2iVector([[0, 1]])
                        vel_ls.colors = o3d.utility.Vector3dVector(
                            [(1.0, 0.0, 1.0)]
                        )
                        vis.add_geometry(vel_ls, reset_bounding_box=False)
                        current_geometries.append(vel_ls)

        # Print track info
        if show_track_ids and frame_idx % max(1, num_frames // 10) == 0:
            tracked = [
                p for p in predictions if p.get("track_id") is not None
            ]
            if tracked:
                print(
                    f"  Frame {frame_idx + 1}: "
                    f"{len(tracked)} tracked objects"
                )

        vis.poll_events()
        vis.update_renderer()

        # Frame rate control
        elapsed = time.time() - frame_start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

        # Check if window was closed
        if not vis.poll_events():
            break

    vis.destroy_window()
    print("3D sequence animation complete.")


# ============================================================================
# Main Entry Point
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CenterPoint Detection & Tracking Results Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single frame BEV visualization
  python visualize_results.py --point-cloud frame.bin --predictions preds.json --mode bev

  # Single frame 3D visualization
  python visualize_results.py --point-cloud frame.bin --predictions preds.json --mode 3d

  # Sequence animation (BEV) saved as video
  python visualize_results.py --point-cloud ./pc_folder --predictions ./pred_folder \\
      --mode sequence --output output.mp4 --fps 10

  # With ground truth comparison
  python visualize_results.py --point-cloud frame.bin --predictions preds.json \\
      --ground-truth gt.json --mode bev --show-velocity --show-track-ids
""",
    )

    parser.add_argument(
        "--point-cloud",
        type=str,
        required=True,
        help="Path to a .bin file or a folder of .bin files.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to a predictions JSON file or a folder of JSON files.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Optional path to ground truth JSON file or folder.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bev", "3d", "sequence"],
        default="bev",
        help="Visualization mode: bev, 3d, or sequence (default: bev).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path. For single frame: PNG file path. "
            "For sequence: MP4 file path or folder for PNG frames."
        ),
    )
    parser.add_argument(
        "--range",
        type=float,
        default=60.0,
        help="Visualization range in meters (default: 60).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence score to display (default: 0.3).",
    )
    parser.add_argument(
        "--show-velocity",
        action="store_true",
        default=False,
        help="Show velocity arrows on detections.",
    )
    parser.add_argument(
        "--show-track-ids",
        action="store_true",
        default=False,
        help="Show track ID labels on detections.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for sequence animation (default: 10).",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        choices=["height", "intensity"],
        default="height",
        help="Color points by height or intensity (default: height).",
    )

    return parser.parse_args()


def main():
    """Main entry point for the visualization script."""
    args = parse_args()

    pc_path = args.point_cloud
    pred_path = args.predictions
    gt_path = args.ground_truth
    mode = args.mode
    output_path = args.output
    vis_range = args.range
    score_threshold = args.score_threshold
    show_velocity = args.show_velocity
    show_track_ids = args.show_track_ids
    fps = args.fps

    # Determine if inputs are files or folders
    pc_is_folder = os.path.isdir(pc_path)
    pred_is_folder = os.path.isdir(pred_path)

    if mode == "sequence":
        # Sequence mode requires folders
        if not pc_is_folder:
            print(
                "Error: --point-cloud must be a folder in sequence mode."
            )
            sys.exit(1)
        if not pred_is_folder:
            print(
                "Error: --predictions must be a folder in sequence mode."
            )
            sys.exit(1)

        gt_folder = None
        if gt_path is not None:
            if os.path.isdir(gt_path):
                gt_folder = gt_path
            else:
                print(
                    "Warning: --ground-truth is not a folder, ignoring "
                    "in sequence mode."
                )

        # Determine if 3D or BEV sequence
        # Default is BEV; if output is None and we have open3d, ask or use BEV
        use_3d = False
        if output_path is None:
            # Check if open3d is available for 3D sequence
            try:
                import open3d  # noqa: F401

                # If both are available, prefer BEV unless user wants 3D
                # For sequence + no output, we use BEV with plt.pause
                use_3d = False
            except ImportError:
                use_3d = False

        if use_3d:
            animate_3d_sequence(
                pc_folder=pc_path,
                pred_folder=pred_path,
                gt_folder=gt_folder,
                show_velocity=show_velocity,
                show_track_ids=show_track_ids,
                score_threshold=score_threshold,
                fps=fps,
            )
        else:
            animate_bev_sequence(
                pc_folder=pc_path,
                pred_folder=pred_path,
                gt_folder=gt_folder,
                vis_range=vis_range,
                show_velocity=show_velocity,
                show_track_ids=show_track_ids,
                score_threshold=score_threshold,
                output_path=output_path,
                fps=fps,
            )

    elif mode == "3d":
        # 3D single frame
        try:
            import open3d  # noqa: F401
        except ImportError:
            print(
                "Error: open3d is required for 3D visualization. "
                "Install with: pip install open3d"
            )
            sys.exit(1)

        if pc_is_folder:
            # Use first file from folder
            pc_files = get_sorted_files(pc_path, ".bin")
            if not pc_files:
                print(f"Error: No .bin files found in {pc_path}")
                sys.exit(1)
            pc_file = pc_files[0]
        else:
            pc_file = pc_path

        if pred_is_folder:
            pred_files = get_sorted_files(pred_path, ".json")
            if not pred_files:
                print(f"Error: No .json files found in {pred_path}")
                sys.exit(1)
            pred_file = pred_files[0]
        else:
            pred_file = pred_path

        points = load_point_cloud(pc_file)
        predictions = load_predictions(pred_file)
        predictions = filter_predictions(predictions, score_threshold)

        ground_truth = None
        if gt_path is not None:
            if os.path.isdir(gt_path):
                gt_files = get_sorted_files(gt_path, ".json")
                if gt_files:
                    ground_truth = load_ground_truth(gt_files[0])
            else:
                ground_truth = load_ground_truth(gt_path)

        print(
            f"Loaded {len(points)} points, "
            f"{len(predictions)} predictions (score >= {score_threshold})"
        )

        visualize_3d(
            points,
            predictions,
            ground_truth=ground_truth,
            show_track_ids=show_track_ids,
            show_velocity=show_velocity,
        )

    else:
        # BEV single frame
        if pc_is_folder:
            pc_files = get_sorted_files(pc_path, ".bin")
            if not pc_files:
                print(f"Error: No .bin files found in {pc_path}")
                sys.exit(1)
            pc_file = pc_files[0]
        else:
            pc_file = pc_path

        if pred_is_folder:
            pred_files = get_sorted_files(pred_path, ".json")
            if not pred_files:
                print(f"Error: No .json files found in {pred_path}")
                sys.exit(1)
            pred_file = pred_files[0]
        else:
            pred_file = pred_path

        points = load_point_cloud(pc_file)
        predictions = load_predictions(pred_file)
        predictions = filter_predictions(predictions, score_threshold)

        ground_truth = None
        if gt_path is not None:
            if os.path.isdir(gt_path):
                gt_files = get_sorted_files(gt_path, ".json")
                if gt_files:
                    ground_truth = load_ground_truth(gt_files[0])
            else:
                ground_truth = load_ground_truth(gt_path)

        frame_info = f"File: {Path(pc_file).stem}"
        print(
            f"Loaded {len(points)} points, "
            f"{len(predictions)} predictions (score >= {score_threshold})"
        )

        visualize_bev(
            points,
            predictions,
            ground_truth=ground_truth,
            vis_range=vis_range,
            show_velocity=show_velocity,
            show_track_ids=show_track_ids,
            output_path=output_path,
            frame_info=frame_info,
        )


if __name__ == "__main__":
    main()
