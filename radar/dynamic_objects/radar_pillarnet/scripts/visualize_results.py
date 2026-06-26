"""
Visualization script for radar-based 3D object detection results.

Provides BEV (Bird's Eye View) visualization of radar point clouds,
predicted bounding boxes, ground truth annotations, and velocity vectors
for nuScenes-style radar detection pipelines.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation


def get_class_colors() -> Dict[str, str]:
    """Return a dictionary mapping nuScenes 10 detection class names to distinct colors."""
    return {
        'car': '#1f77b4',               # blue
        'truck': '#ff7f0e',             # orange
        'construction_vehicle': '#8c564b',  # brown
        'bus': '#9467bd',               # purple
        'trailer': '#7f7f7f',           # gray
        'barrier': '#bcbd22',           # olive
        'motorcycle': '#17becf',        # cyan
        'bicycle': '#2ca02c',           # green
        'pedestrian': '#d62728',        # red
        'traffic_cone': '#e377c2',      # pink
    }


class BEVVisualizer:
    """Bird's Eye View visualizer for radar detection results.

    Renders radar point clouds, 3D bounding boxes projected to BEV,
    velocity arrows, and classification legends on a 2D top-down plot.
    """

    def __init__(
        self,
        point_range: Tuple[float, float, float, float] = (-51.2, -51.2, 51.2, 51.2),
        figsize: Tuple[float, float] = (12, 12),
        dpi: int = 150,
    ):
        """Initialize the BEV visualizer.

        Args:
            point_range: (x_min, y_min, x_max, y_max) in meters for the BEV display area.
                         nuScenes convention: x=forward, y=left.
            figsize: Matplotlib figure size in inches.
            dpi: Dots per inch for rendering.
        """
        self.point_range = point_range
        self.figsize = figsize
        self.dpi = dpi
        self.class_colors = get_class_colors()

        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        self.ax.set_xlim(point_range[0], point_range[2])
        self.ax.set_ylim(point_range[1], point_range[3])
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m) - Forward')
        self.ax.set_ylabel('Y (m) - Left')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_facecolor('#f0f0f0')

        self._colorbar = None

    def draw_points(
        self,
        points: np.ndarray,
        color_by: str = 'rcs',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Draw radar points in BEV, colored by a scalar attribute.

        Args:
            points: Array of shape (N, C) where columns are at minimum
                    [x, y, z, rcs, vr_compensated, ...]. Expected column layout:
                    0=x, 1=y, 2=z, 3=rcs, 4=vr_compensated (radial velocity).
            color_by: 'rcs' to color by radar cross section, 'velocity' for radial velocity.
            vmin: Minimum value for colormap normalization.
            vmax: Maximum value for colormap normalization.
        """
        if points is None or len(points) == 0:
            return

        x = points[:, 0]
        y = points[:, 1]

        if color_by == 'rcs':
            if points.shape[1] > 3:
                values = points[:, 3]
            else:
                values = np.zeros(len(points))
            cmap = 'viridis'
            label = 'RCS (dBsm)'
            if vmin is None:
                vmin = -10.0
            if vmax is None:
                vmax = 30.0
        elif color_by == 'velocity':
            if points.shape[1] > 4:
                values = points[:, 4]
            else:
                values = np.zeros(len(points))
            cmap = 'coolwarm'
            label = 'Radial Velocity (m/s)'
            if vmin is None:
                vmin = -15.0
            if vmax is None:
                vmax = 15.0
        else:
            raise ValueError(f"color_by must be 'rcs' or 'velocity', got '{color_by}'")

        scatter = self.ax.scatter(
            x, y,
            c=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=3,
            alpha=0.7,
            zorder=1,
        )

        if self._colorbar is not None:
            self._colorbar.remove()
        self._colorbar = self.fig.colorbar(scatter, ax=self.ax, shrink=0.7, pad=0.02)
        self._colorbar.set_label(label)

    def draw_box(
        self,
        center: Tuple[float, float],
        size: Tuple[float, float],
        yaw: float,
        color: str = 'blue',
        label: Optional[str] = None,
        linewidth: float = 1.5,
        linestyle: str = '-',
    ):
        """Draw a single rotated 2D rectangle in BEV representing a 3D bounding box.

        Args:
            center: (x, y) center of the box in meters.
            size: (length, width) of the box in meters. Length is along the heading direction.
            yaw: Heading angle in radians (counter-clockwise from x-axis).
            color: Color string or hex for the box outline.
            label: Optional text label to display near the box.
            linewidth: Line width of the box outline.
            linestyle: Line style ('-' for solid, '--' for dashed, etc.).
        """
        cx, cy = center
        length, width = size

        # Compute the four corners of the box (unrotated, centered at origin)
        # length is along x (forward), width is along y (left)
        corners = np.array([
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ])

        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw],
        ])

        # Rotate and translate
        rotated_corners = corners @ rotation.T + np.array([cx, cy])

        # Draw as polygon
        polygon = patches.Polygon(
            rotated_corners,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=3,
        )
        self.ax.add_patch(polygon)

        # Draw heading indicator (front of box)
        front_center = np.array([length / 2, 0.0]) @ rotation.T + np.array([cx, cy])
        self.ax.plot(
            [cx, front_center[0]],
            [cy, front_center[1]],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=3,
        )

        # Add label if provided
        if label is not None:
            self.ax.text(
                cx, cy + width / 2 + 0.5,
                label,
                fontsize=6,
                color=color,
                ha='center',
                va='bottom',
                zorder=4,
            )

    def draw_boxes(
        self,
        boxes: np.ndarray,
        colors: Union[List[str], str] = 'blue',
        labels: Optional[List[str]] = None,
        linestyle: str = '-',
        linewidth: float = 1.5,
    ):
        """Draw multiple bounding boxes in BEV.

        Args:
            boxes: Array of shape (N, 7+) where columns are
                   [x, y, z, length, width, height, yaw, ...].
            colors: Single color or list of colors per box.
            labels: Optional list of text labels per box.
            linestyle: Line style for all boxes.
            linewidth: Line width for all boxes.
        """
        if boxes is None or len(boxes) == 0:
            return

        if isinstance(colors, str):
            colors = [colors] * len(boxes)

        for i, box in enumerate(boxes):
            cx, cy = box[0], box[1]
            length, width = box[3], box[4]
            yaw = box[6]
            color = colors[i] if i < len(colors) else colors[-1]
            label = labels[i] if labels is not None and i < len(labels) else None

            self.draw_box(
                center=(cx, cy),
                size=(length, width),
                yaw=yaw,
                color=color,
                label=label,
                linewidth=linewidth,
                linestyle=linestyle,
            )

    def draw_velocity_arrows(
        self,
        centers: np.ndarray,
        velocities: np.ndarray,
        color: str = 'green',
        scale: float = 1.0,
    ):
        """Draw velocity vectors as arrows using quiver.

        Args:
            centers: Array of shape (N, 2) with (x, y) positions.
            velocities: Array of shape (N, 2) with (vx, vy) velocity components.
            color: Arrow color.
            scale: Scaling factor for arrow length (1.0 = real scale in m/s).
        """
        if centers is None or velocities is None or len(centers) == 0:
            return

        # Filter out zero-velocity entries for cleaner visualization
        speed = np.linalg.norm(velocities, axis=1)
        mask = speed > 0.1  # Only show arrows for objects with meaningful velocity

        if not np.any(mask):
            return

        self.ax.quiver(
            centers[mask, 0],
            centers[mask, 1],
            velocities[mask, 0] * scale,
            velocities[mask, 1] * scale,
            color=color,
            angles='xy',
            scale_units='xy',
            scale=1.0,
            width=0.15,
            headwidth=3,
            headlength=3,
            alpha=0.8,
            zorder=5,
        )

    def draw_legend(self):
        """Add a legend showing class-to-color mapping."""
        legend_elements = []
        for class_name, color in self.class_colors.items():
            legend_elements.append(
                patches.Patch(facecolor='none', edgecolor=color, linewidth=2,
                              label=class_name.replace('_', ' ').title())
            )
        self.ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=7,
            framealpha=0.8,
            ncol=2,
        )

    def set_title(self, title: str):
        """Set the plot title.

        Args:
            title: Title string to display above the plot.
        """
        self.ax.set_title(title, fontsize=12, fontweight='bold')

    def save(self, filepath: str):
        """Save the current figure to a file.

        Args:
            filepath: Output path (supports .png, .pdf, .svg, .jpg).
        """
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1)

    def show(self):
        """Display the figure interactively."""
        plt.show()

    def clear(self):
        """Clear the axes for the next frame while preserving axis configuration."""
        self.ax.cla()
        self.ax.set_xlim(self.point_range[0], self.point_range[2])
        self.ax.set_ylim(self.point_range[1], self.point_range[3])
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m) - Forward')
        self.ax.set_ylabel('Y (m) - Left')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_facecolor('#f0f0f0')
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None


def load_predictions(filepath: str) -> Dict:
    """Load prediction results from a pickle or JSON file.

    Expected format (pickle): A dictionary with keys:
        - 'boxes_3d': np.ndarray of shape (N, 9) [x, y, z, l, w, h, yaw, vx, vy]
        - 'scores_3d': np.ndarray of shape (N,)
        - 'labels_3d': np.ndarray of shape (N,) with integer class indices

    Or a list of per-frame prediction dictionaries.

    Args:
        filepath: Path to the prediction file (.pkl or .json).

    Returns:
        Loaded prediction data.
    """
    filepath = str(filepath)
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    return data


def load_ground_truth(filepath: str) -> Dict:
    """Load ground truth annotations from an info pickle file.

    Expected format: A dictionary or list of dictionaries containing:
        - 'gt_boxes': np.ndarray of shape (M, 9) [x, y, z, l, w, h, yaw, vx, vy]
        - 'gt_names': list of class name strings
        - 'gt_velocity': (optional) np.ndarray of shape (M, 2)

    Args:
        filepath: Path to the ground truth info file (.pkl).

    Returns:
        Loaded ground truth data.
    """
    filepath = str(filepath)
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    return data


def load_radar_points(filepath: str) -> np.ndarray:
    """Load radar point cloud from a binary or PCD file.

    Supports:
        - .bin files: Raw float32 binary, reshaped to (N, num_features).
          Default assumes 18 features per point (nuScenes radar format:
          x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, ...).
          Falls back to 5 features [x, y, z, rcs, vr] if 18 does not divide evenly.
        - .pcd files: ASCII or binary PCD format (reads ASCII variant).
        - .npy files: NumPy saved arrays.

    Args:
        filepath: Path to the radar point cloud file.

    Returns:
        np.ndarray of shape (N, C) with point features.
    """
    filepath = str(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.npy':
        points = np.load(filepath)
    elif ext == '.bin':
        raw = np.fromfile(filepath, dtype=np.float32)
        # nuScenes radar .bin has 18 features per point
        if raw.size % 18 == 0:
            points = raw.reshape(-1, 18)
        elif raw.size % 5 == 0:
            points = raw.reshape(-1, 5)
        else:
            # Try common radar feature counts
            for n_feat in [7, 6, 4, 3]:
                if raw.size % n_feat == 0:
                    points = raw.reshape(-1, n_feat)
                    break
            else:
                raise ValueError(
                    f"Cannot determine feature count for binary file with {raw.size} floats"
                )
    elif ext == '.pcd':
        points = _read_pcd_file(filepath)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

    return points


def _read_pcd_file(filepath: str) -> np.ndarray:
    """Parse a PCD file (ASCII or binary) and return points as numpy array.

    Args:
        filepath: Path to the .pcd file.

    Returns:
        np.ndarray of shape (N, num_fields).
    """
    with open(filepath, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line.startswith('DATA'):
                break

        # Parse header
        num_points = 0
        fields = []
        data_format = 'ascii'
        for line in header_lines:
            if line.startswith('POINTS'):
                num_points = int(line.split()[1])
            elif line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('DATA'):
                data_format = line.split()[1].lower()

        num_fields = len(fields) if fields else 3

        if data_format == 'ascii':
            points = []
            for _ in range(num_points):
                line = f.readline().decode('ascii', errors='ignore').strip()
                if line:
                    values = [float(v) for v in line.split()]
                    points.append(values)
            points = np.array(points, dtype=np.float32)
        else:
            # Binary format
            raw = np.frombuffer(f.read(), dtype=np.float32)
            if raw.size >= num_points * num_fields:
                points = raw[:num_points * num_fields].reshape(num_points, num_fields)
            else:
                points = raw.reshape(-1, num_fields)

    return points


def create_animation(
    visualizer: BEVVisualizer,
    frames_data: List[Dict],
    output_path: str,
    fps: int = 5,
):
    """Create a multi-frame animation from a sequence of detection results.

    Args:
        visualizer: BEVVisualizer instance to use for rendering.
        frames_data: List of dicts, each containing:
            - 'points': np.ndarray radar points (optional)
            - 'pred_boxes': np.ndarray predicted boxes (optional)
            - 'pred_scores': np.ndarray scores (optional)
            - 'pred_labels': np.ndarray or list of label indices/names (optional)
            - 'gt_boxes': np.ndarray ground truth boxes (optional)
            - 'gt_labels': list of class name strings (optional)
            - 'velocities': np.ndarray velocity vectors (optional)
            - 'title': str frame title (optional)
        output_path: Output file path (.gif or .mp4).
        fps: Frames per second for the animation.
    """
    class_colors = get_class_colors()
    class_names = list(class_colors.keys())

    def update_frame(frame_idx):
        visualizer.clear()
        frame = frames_data[frame_idx]

        # Draw points
        points = frame.get('points')
        if points is not None and len(points) > 0:
            visualizer.draw_points(points, color_by='rcs')

        # Draw ground truth boxes (dashed)
        gt_boxes = frame.get('gt_boxes')
        gt_labels = frame.get('gt_labels')
        if gt_boxes is not None and len(gt_boxes) > 0:
            gt_colors = []
            gt_label_texts = []
            for i, label in enumerate(gt_labels if gt_labels is not None else []):
                if isinstance(label, str):
                    name = label
                else:
                    name = class_names[int(label)] if int(label) < len(class_names) else 'unknown'
                gt_colors.append(class_colors.get(name, '#000000'))
                gt_label_texts.append(f'GT: {name}')
            if not gt_colors:
                gt_colors = ['#000000'] * len(gt_boxes)
                gt_label_texts = None
            visualizer.draw_boxes(
                gt_boxes, colors=gt_colors, labels=gt_label_texts,
                linestyle='--', linewidth=1.0,
            )

        # Draw predicted boxes (solid)
        pred_boxes = frame.get('pred_boxes')
        pred_scores = frame.get('pred_scores')
        pred_labels = frame.get('pred_labels')
        if pred_boxes is not None and len(pred_boxes) > 0:
            pred_colors = []
            pred_label_texts = []
            for i in range(len(pred_boxes)):
                if pred_labels is not None and i < len(pred_labels):
                    label = pred_labels[i]
                    if isinstance(label, str):
                        name = label
                    else:
                        name = class_names[int(label)] if int(label) < len(class_names) else 'unknown'
                else:
                    name = 'unknown'
                pred_colors.append(class_colors.get(name, '#000000'))
                score = pred_scores[i] if pred_scores is not None and i < len(pred_scores) else 0.0
                pred_label_texts.append(f'{name} {score:.2f}')
            visualizer.draw_boxes(
                pred_boxes, colors=pred_colors, labels=pred_label_texts,
                linestyle='-', linewidth=1.5,
            )

        # Draw velocity arrows
        velocities = frame.get('velocities')
        if velocities is not None and pred_boxes is not None and len(pred_boxes) > 0:
            centers = pred_boxes[:, :2]
            visualizer.draw_velocity_arrows(centers, velocities, color='green', scale=1.0)

        # Title
        title = frame.get('title', f'Frame {frame_idx}')
        visualizer.set_title(title)
        visualizer.draw_legend()

    anim = FuncAnimation(
        visualizer.fig,
        update_frame,
        frames=len(frames_data),
        interval=1000 // fps,
        repeat=True,
    )

    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    if output_path.endswith('.gif'):
        anim.save(output_path, writer='pillow', fps=fps)
    elif output_path.endswith('.mp4'):
        anim.save(output_path, writer='ffmpeg', fps=fps)
    else:
        anim.save(output_path, fps=fps)

    print(f"Animation saved to: {output_path}")


def visualize_single_frame(
    points: Optional[np.ndarray] = None,
    pred_boxes: Optional[np.ndarray] = None,
    gt_boxes: Optional[np.ndarray] = None,
    pred_scores: Optional[np.ndarray] = None,
    pred_labels: Optional[Union[np.ndarray, List]] = None,
    gt_labels: Optional[Union[np.ndarray, List]] = None,
    velocities: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    show: bool = False,
    point_range: Tuple[float, float, float, float] = (-51.2, -51.2, 51.2, 51.2),
    color_by: str = 'rcs',
    title: str = 'Radar BEV Detection',
):
    """Convenience function to visualize a single detection frame.

    Draws radar points, predicted boxes (solid), ground truth boxes (dashed),
    and velocity arrows on a BEV plot.

    Args:
        points: Radar point cloud array of shape (N, C).
        pred_boxes: Predicted bounding boxes (M, 7+) [x, y, z, l, w, h, yaw, ...].
        gt_boxes: Ground truth boxes (K, 7+) [x, y, z, l, w, h, yaw, ...].
        pred_scores: Prediction confidence scores of shape (M,).
        pred_labels: Prediction class labels (int indices or string names).
        gt_labels: Ground truth class labels (string names or int indices).
        velocities: Velocity vectors (M, 2) for predicted boxes [vx, vy].
        output_path: If set, save the figure to this path.
        show: If True, display the figure interactively.
        point_range: BEV display range (x_min, y_min, x_max, y_max).
        color_by: Attribute to color points by ('rcs' or 'velocity').
        title: Title for the plot.
    """
    class_colors = get_class_colors()
    class_names = list(class_colors.keys())

    vis = BEVVisualizer(point_range=point_range)

    # Draw radar points
    if points is not None and len(points) > 0:
        vis.draw_points(points, color_by=color_by)

    # Draw ground truth boxes (dashed lines)
    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_colors = []
        gt_label_texts = []
        for i in range(len(gt_boxes)):
            if gt_labels is not None and i < len(gt_labels):
                label = gt_labels[i]
                if isinstance(label, str):
                    name = label
                else:
                    name = class_names[int(label)] if int(label) < len(class_names) else 'unknown'
            else:
                name = 'unknown'
            gt_colors.append(class_colors.get(name, '#000000'))
            gt_label_texts.append(f'GT: {name}')
        vis.draw_boxes(
            gt_boxes, colors=gt_colors, labels=gt_label_texts,
            linestyle='--', linewidth=1.0,
        )

    # Draw predicted boxes (solid lines)
    if pred_boxes is not None and len(pred_boxes) > 0:
        pred_colors = []
        pred_label_texts = []
        for i in range(len(pred_boxes)):
            if pred_labels is not None and i < len(pred_labels):
                label = pred_labels[i]
                if isinstance(label, str):
                    name = label
                else:
                    name = class_names[int(label)] if int(label) < len(class_names) else 'unknown'
            else:
                name = 'unknown'
            pred_colors.append(class_colors.get(name, '#000000'))
            score = pred_scores[i] if pred_scores is not None and i < len(pred_scores) else 0.0
            pred_label_texts.append(f'{name} {score:.2f}')
        vis.draw_boxes(
            pred_boxes, colors=pred_colors, labels=pred_label_texts,
            linestyle='-', linewidth=1.5,
        )

    # Draw velocity arrows
    if velocities is not None and pred_boxes is not None and len(pred_boxes) > 0:
        centers = pred_boxes[:len(velocities), :2]
        vis.draw_velocity_arrows(centers, velocities, color='green', scale=1.0)

    vis.set_title(title)
    vis.draw_legend()

    if output_path is not None:
        vis.save(output_path)
        print(f"Figure saved to: {output_path}")

    if show:
        vis.show()

    plt.close(vis.fig)


def _resolve_class_label(label, class_names: List[str]) -> str:
    """Convert a label (int index or string) to a class name string."""
    if isinstance(label, str):
        return label
    idx = int(label)
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return 'unknown'


def main():
    """Main CLI entry point for radar detection visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize radar-based 3D object detection results in BEV.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--predictions', type=str, default=None,
        help='Path to prediction results file (.pkl or .json).',
    )
    parser.add_argument(
        '--ground-truth', type=str, default=None,
        help='Path to ground truth info file (.pkl or .json).',
    )
    parser.add_argument(
        '--radar-data', type=str, default=None,
        help='Path to radar point cloud file (.bin/.pcd/.npy) or directory of files.',
    )
    parser.add_argument(
        '--output-dir', type=str, default='./vis_output',
        help='Directory to save output images/animations.',
    )
    parser.add_argument(
        '--color-by', type=str, default='rcs', choices=['rcs', 'velocity'],
        help='Attribute to color radar points by.',
    )
    parser.add_argument(
        '--show-gt', action='store_true', default=True,
        help='Show ground truth boxes.',
    )
    parser.add_argument(
        '--no-show-gt', action='store_true', default=False,
        help='Do not show ground truth boxes.',
    )
    parser.add_argument(
        '--show-pred', action='store_true', default=True,
        help='Show predicted boxes.',
    )
    parser.add_argument(
        '--no-show-pred', action='store_true', default=False,
        help='Do not show predicted boxes.',
    )
    parser.add_argument(
        '--show-velocity', action='store_true', default=True,
        help='Show velocity arrows on predicted boxes.',
    )
    parser.add_argument(
        '--no-show-velocity', action='store_true', default=False,
        help='Do not show velocity arrows.',
    )
    parser.add_argument(
        '--animate', action='store_true', default=False,
        help='Create animation from a sequence of frames.',
    )
    parser.add_argument(
        '--fps', type=int, default=5,
        help='Frames per second for animation output.',
    )
    parser.add_argument(
        '--score-threshold', type=float, default=0.3,
        help='Minimum confidence score to display a prediction.',
    )
    parser.add_argument(
        '--point-range', type=float, nargs=4, default=[-51.2, -51.2, 51.2, 51.2],
        metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
        help='BEV display range in meters [x_min, y_min, x_max, y_max].',
    )
    parser.add_argument(
        '--sample-indices', type=int, nargs='+', default=None,
        help='Specific sample indices to visualize (0-based). If not set, visualize all.',
    )

    args = parser.parse_args()

    # Resolve flags
    show_gt = args.show_gt and not args.no_show_gt
    show_pred = args.show_pred and not args.no_show_pred
    show_velocity = args.show_velocity and not args.no_show_velocity

    point_range = tuple(args.point_range)
    os.makedirs(args.output_dir, exist_ok=True)

    class_colors = get_class_colors()
    class_names = list(class_colors.keys())

    # Load predictions
    predictions = None
    if args.predictions is not None:
        predictions = load_predictions(args.predictions)
        print(f"Loaded predictions from: {args.predictions}")
        if isinstance(predictions, dict) and 'results' in predictions:
            predictions = predictions['results']

    # Load ground truth
    ground_truth = None
    if args.ground_truth is not None:
        ground_truth = load_ground_truth(args.ground_truth)
        print(f"Loaded ground truth from: {args.ground_truth}")
        if isinstance(ground_truth, dict) and 'infos' in ground_truth:
            ground_truth = ground_truth['infos']

    # Load radar data
    radar_files = []
    if args.radar_data is not None:
        radar_path = Path(args.radar_data)
        if radar_path.is_dir():
            # Collect all radar files sorted by name
            extensions = ['.bin', '.pcd', '.npy']
            for ext in extensions:
                radar_files.extend(sorted(radar_path.glob(f'*{ext}')))
            radar_files = sorted(radar_files, key=lambda p: p.stem)
        elif radar_path.is_file():
            radar_files = [radar_path]
        print(f"Found {len(radar_files)} radar point cloud file(s)")

    # Determine number of frames
    num_frames = 0
    if predictions is not None and isinstance(predictions, list):
        num_frames = max(num_frames, len(predictions))
    if ground_truth is not None and isinstance(ground_truth, list):
        num_frames = max(num_frames, len(ground_truth))
    if radar_files:
        num_frames = max(num_frames, len(radar_files))
    if num_frames == 0:
        # Single frame mode from dict-style predictions
        num_frames = 1

    # Filter to requested sample indices
    if args.sample_indices is not None:
        sample_indices = [i for i in args.sample_indices if 0 <= i < num_frames]
    else:
        sample_indices = list(range(num_frames))

    print(f"Visualizing {len(sample_indices)} frame(s)...")

    # Build frame data
    frames_data = []
    for idx in sample_indices:
        frame = {}

        # Radar points
        if idx < len(radar_files):
            try:
                frame['points'] = load_radar_points(str(radar_files[idx]))
            except Exception as e:
                print(f"Warning: Could not load radar file {radar_files[idx]}: {e}")
                frame['points'] = None
        else:
            frame['points'] = None

        # Predictions
        if show_pred and predictions is not None:
            if isinstance(predictions, list) and idx < len(predictions):
                pred_frame = predictions[idx]
            elif isinstance(predictions, dict):
                pred_frame = predictions
            else:
                pred_frame = {}

            pred_boxes = pred_frame.get('boxes_3d', pred_frame.get('boxes', None))
            pred_scores = pred_frame.get('scores_3d', pred_frame.get('scores', None))
            pred_labels = pred_frame.get('labels_3d', pred_frame.get('labels', None))

            if pred_boxes is not None:
                pred_boxes = np.array(pred_boxes)
                pred_scores = np.array(pred_scores) if pred_scores is not None else np.ones(len(pred_boxes))
                pred_labels = np.array(pred_labels) if pred_labels is not None else np.zeros(len(pred_boxes), dtype=int)

                # Apply score threshold
                score_mask = pred_scores >= args.score_threshold
                pred_boxes = pred_boxes[score_mask]
                pred_scores = pred_scores[score_mask]
                pred_labels = pred_labels[score_mask]

                frame['pred_boxes'] = pred_boxes
                frame['pred_scores'] = pred_scores
                frame['pred_labels'] = pred_labels

                # Extract velocities from boxes if available (columns 7, 8 = vx, vy)
                if show_velocity and pred_boxes.shape[1] >= 9:
                    frame['velocities'] = pred_boxes[:, 7:9]
                elif show_velocity:
                    frame['velocities'] = None
            else:
                frame['pred_boxes'] = None
                frame['pred_scores'] = None
                frame['pred_labels'] = None
                frame['velocities'] = None
        else:
            frame['pred_boxes'] = None
            frame['pred_scores'] = None
            frame['pred_labels'] = None
            frame['velocities'] = None

        # Ground truth
        if show_gt and ground_truth is not None:
            if isinstance(ground_truth, list) and idx < len(ground_truth):
                gt_frame = ground_truth[idx]
            elif isinstance(ground_truth, dict):
                gt_frame = ground_truth
            else:
                gt_frame = {}

            gt_boxes = gt_frame.get('gt_boxes', gt_frame.get('boxes', None))
            gt_labels = gt_frame.get('gt_names', gt_frame.get('labels', None))

            if gt_boxes is not None:
                frame['gt_boxes'] = np.array(gt_boxes)
                frame['gt_labels'] = gt_labels if gt_labels is not None else []
            else:
                frame['gt_boxes'] = None
                frame['gt_labels'] = None
        else:
            frame['gt_boxes'] = None
            frame['gt_labels'] = None

        frame['title'] = f'Radar BEV Detection - Frame {idx}'
        frames_data.append(frame)

    # Render
    if args.animate and len(frames_data) > 1:
        # Animation mode
        vis = BEVVisualizer(point_range=point_range)
        output_path = os.path.join(args.output_dir, 'detection_animation.gif')
        create_animation(vis, frames_data, output_path, fps=args.fps)
        plt.close(vis.fig)
    else:
        # Single-frame or multi-frame image mode
        for i, frame in enumerate(frames_data):
            frame_idx = sample_indices[i]
            output_path = os.path.join(args.output_dir, f'frame_{frame_idx:06d}.png')
            visualize_single_frame(
                points=frame['points'],
                pred_boxes=frame.get('pred_boxes'),
                gt_boxes=frame.get('gt_boxes'),
                pred_scores=frame.get('pred_scores'),
                pred_labels=frame.get('pred_labels'),
                gt_labels=frame.get('gt_labels'),
                velocities=frame.get('velocities'),
                output_path=output_path,
                show=False,
                point_range=point_range,
                color_by=args.color_by,
                title=frame.get('title', f'Frame {frame_idx}'),
            )

    print(f"Visualization complete. Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
