"""Bird's Eye View (BEV) visualization for autonomous driving perception.

Provides functions to render 3D bounding boxes, HD map elements, ego vehicle,
velocity arrows, and occupancy grids from a top-down perspective.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

# ---------------------------------------------------------------------------
# Default color palette for autonomous driving classes
# ---------------------------------------------------------------------------
DEFAULT_CLASS_COLORS: Dict[str, str] = {
    "car": "#1f77b4",
    "truck": "#ff7f0e",
    "bus": "#2ca02c",
    "trailer": "#d62728",
    "construction_vehicle": "#9467bd",
    "pedestrian": "#e377c2",
    "motorcycle": "#7f7f7f",
    "bicycle": "#bcbd22",
    "traffic_cone": "#17becf",
    "barrier": "#aec7e8",
}

GT_COLOR = "#00ff00"  # green for ground truth
PRED_COLOR = "#ff4444"  # red for predictions
EGO_COLOR = "#ffcc00"  # yellow for ego vehicle


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rotation_matrix_2d(yaw: float) -> np.ndarray:
    """Return a 2x2 rotation matrix for the given yaw angle (radians)."""
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.array([[cos, -sin], [sin, cos]])


def _box_corners_bev(
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    yaw: float,
) -> np.ndarray:
    """Compute the 4 BEV corners of a rotated bounding box.

    Parameters
    ----------
    center_x, center_y : float
        Center of the box in BEV coordinates (meters).
    length, width : float
        Dimensions of the box (meters).
    yaw : float
        Heading angle in radians (counter-clockwise from +x axis).

    Returns
    -------
    np.ndarray
        Shape (4, 2) array of corner coordinates.
    """
    half_l, half_w = length / 2.0, width / 2.0
    # Corners relative to center: front-left, front-right, rear-right, rear-left
    corners = np.array([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w],
    ])
    rot = _rotation_matrix_2d(yaw)
    rotated = corners @ rot.T
    rotated[:, 0] += center_x
    rotated[:, 1] += center_y
    return rotated


# ---------------------------------------------------------------------------
# Core BEV Visualizer
# ---------------------------------------------------------------------------


class BEVVisualizer:
    """Bird's Eye View visualization engine.

    Parameters
    ----------
    bev_range : tuple of float
        (x_min, x_max, y_min, y_max) in meters defining the BEV extent.
        Default is (-50, 50, -50, 50).
    figsize : tuple of float
        Matplotlib figure size in inches. Default is (10, 10).
    dpi : int
        Dots per inch for saved figures. Default is 150.
    class_colors : dict or None
        Mapping from class name to hex color string. Uses DEFAULT_CLASS_COLORS
        if not provided.
    background_color : str
        Background color of the BEV canvas. Default is "#1a1a1a" (dark gray).
    """

    def __init__(
        self,
        bev_range: Tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0),
        figsize: Tuple[float, float] = (10, 10),
        dpi: int = 150,
        class_colors: Optional[Dict[str, str]] = None,
        background_color: str = "#1a1a1a",
    ) -> None:
        self.bev_range = bev_range
        self.figsize = figsize
        self.dpi = dpi
        self.class_colors = class_colors or DEFAULT_CLASS_COLORS
        self.background_color = background_color

        self._fig: Optional[Figure] = None
        self._ax: Optional[plt.Axes] = None

    # ------------------------------------------------------------------
    # Figure management
    # ------------------------------------------------------------------

    def _init_figure(self) -> Tuple[Figure, plt.Axes]:
        """Create a fresh figure and axes for BEV plotting."""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        ax.set_facecolor(self.background_color)
        fig.patch.set_facecolor(self.background_color)
        x_min, x_max, y_min, y_max = self.bev_range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.grid(True, alpha=0.15, color="white")
        self._fig = fig
        self._ax = ax
        return fig, ax

    def get_figure(self) -> Tuple[Figure, plt.Axes]:
        """Return the current figure and axes, creating them if needed."""
        if self._fig is None or self._ax is None:
            return self._init_figure()
        return self._fig, self._ax

    def reset(self) -> None:
        """Close the current figure and reset internal state."""
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------
    # Plotting methods
    # ------------------------------------------------------------------

    def plot_boxes(
        self,
        boxes: np.ndarray,
        classes: Optional[Sequence[str]] = None,
        confidences: Optional[Sequence[float]] = None,
        color_override: Optional[str] = None,
        linewidth: float = 1.5,
        label_prefix: str = "",
    ) -> None:
        """Plot 3D bounding boxes as rotated rectangles in BEV.

        Parameters
        ----------
        boxes : np.ndarray
            Shape (N, 7+) array where columns are
            [center_x, center_y, center_z, length, width, height, yaw, ...].
            Only x, y, length, width, yaw are used for BEV.
        classes : sequence of str, optional
            Class name for each box (used for color coding).
        confidences : sequence of float, optional
            Confidence scores in [0, 1] for each box (affects alpha).
        color_override : str, optional
            If provided, use this color for all boxes (ignores class colors).
        linewidth : float
            Line width for box edges.
        label_prefix : str
            Prefix to add to legend labels (e.g., "GT " or "Pred ").
        """
        _, ax = self.get_figure()

        n_boxes = len(boxes)
        if classes is None:
            classes = ["car"] * n_boxes
        if confidences is None:
            confidences = [1.0] * n_boxes

        labeled_classes: set = set()

        for i in range(n_boxes):
            cx, cy = boxes[i, 0], boxes[i, 1]
            length, width = boxes[i, 3], boxes[i, 4]
            yaw = boxes[i, 6]

            corners = _box_corners_bev(cx, cy, length, width, yaw)

            if color_override:
                color = color_override
            else:
                color = self.class_colors.get(classes[i], "#ffffff")

            alpha = 0.4 + 0.6 * confidences[i]  # map confidence to alpha range

            cls_label = f"{label_prefix}{classes[i]}"
            label = cls_label if cls_label not in labeled_classes else None
            labeled_classes.add(cls_label)

            polygon = Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )
            ax.add_patch(polygon)

            # Draw heading indicator (front edge thicker)
            front_mid = (corners[0] + corners[1]) / 2.0
            ax.plot(
                [cx, front_mid[0]],
                [cy, front_mid[1]],
                color=color,
                linewidth=linewidth * 1.5,
                alpha=alpha,
            )

    def plot_velocity_arrows(
        self,
        boxes: np.ndarray,
        velocities: np.ndarray,
        color: str = "#00ffff",
        scale: float = 1.0,
    ) -> None:
        """Plot velocity arrows for each bounding box.

        Parameters
        ----------
        boxes : np.ndarray
            Shape (N, 7+) with box parameters (only x, y used as arrow origin).
        velocities : np.ndarray
            Shape (N, 2) with [vx, vy] in m/s for each box.
        color : str
            Arrow color.
        scale : float
            Multiplier for arrow length visualization.
        """
        _, ax = self.get_figure()

        for i in range(len(boxes)):
            cx, cy = boxes[i, 0], boxes[i, 1]
            vx, vy = velocities[i, 0] * scale, velocities[i, 1] * scale
            speed = np.sqrt(vx**2 + vy**2)
            if speed < 0.1:
                continue
            ax.annotate(
                "",
                xy=(cx + vx, cy + vy),
                xytext=(cx, cy),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.5,
                ),
            )

    def plot_map_elements(
        self,
        polylines: List[np.ndarray],
        element_type: str = "lane_divider",
        color: Optional[str] = None,
        linewidth: float = 1.0,
    ) -> None:
        """Plot HD map elements as polylines in BEV.

        Parameters
        ----------
        polylines : list of np.ndarray
            Each array has shape (M, 2) or (M, 3); only x, y are used.
        element_type : str
            Type of map element (for color selection). Supported types:
            'lane_divider', 'road_boundary', 'crosswalk', 'stop_line'.
        color : str or None
            Override color. If None, uses default by element type.
        linewidth : float
            Line width.
        """
        _, ax = self.get_figure()

        type_colors = {
            "lane_divider": "#ffff00",
            "road_boundary": "#ff8800",
            "crosswalk": "#ffffff",
            "stop_line": "#ff0000",
        }

        line_color = color or type_colors.get(element_type, "#cccccc")

        linestyle = "--" if element_type == "lane_divider" else "-"

        for polyline in polylines:
            pts = np.asarray(polyline)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=line_color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.8,
            )

    def plot_ego_vehicle(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        length: float = 4.5,
        width: float = 2.0,
    ) -> None:
        """Plot the ego vehicle as a filled rectangle with heading indicator.

        Parameters
        ----------
        x, y : float
            Ego position in BEV coordinates.
        yaw : float
            Ego heading (radians).
        length, width : float
            Ego vehicle dimensions.
        """
        _, ax = self.get_figure()

        corners = _box_corners_bev(x, y, length, width, yaw)

        polygon = Polygon(
            corners,
            closed=True,
            fill=True,
            facecolor=EGO_COLOR,
            edgecolor="white",
            linewidth=2.0,
            alpha=0.7,
            label="Ego",
            zorder=10,
        )
        ax.add_patch(polygon)

        # Heading arrow
        front_mid = (corners[0] + corners[1]) / 2.0
        ax.annotate(
            "",
            xy=(front_mid[0], front_mid[1]),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="white", lw=2),
            zorder=11,
        )

    def plot_predictions_vs_gt(
        self,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        gt_classes: Optional[Sequence[str]] = None,
        pred_classes: Optional[Sequence[str]] = None,
        pred_confidences: Optional[Sequence[float]] = None,
    ) -> None:
        """Overlay predictions vs ground truth with distinct colors.

        Parameters
        ----------
        gt_boxes : np.ndarray
            Ground truth boxes, shape (N, 7+).
        pred_boxes : np.ndarray
            Predicted boxes, shape (M, 7+).
        gt_classes : sequence of str, optional
            Classes for GT boxes.
        pred_classes : sequence of str, optional
            Classes for predicted boxes.
        pred_confidences : sequence of float, optional
            Confidence scores for predicted boxes.
        """
        self.plot_boxes(
            gt_boxes,
            classes=gt_classes,
            color_override=GT_COLOR,
            linewidth=2.0,
            label_prefix="GT ",
        )
        self.plot_boxes(
            pred_boxes,
            classes=pred_classes,
            confidences=pred_confidences,
            color_override=PRED_COLOR,
            linewidth=1.5,
            label_prefix="Pred ",
        )

    def plot_occupancy_grid(
        self,
        grid: np.ndarray,
        extent: Optional[Tuple[float, float, float, float]] = None,
        cmap: str = "viridis",
        alpha: float = 0.5,
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> None:
        """Plot an occupancy grid as a semi-transparent heatmap.

        Parameters
        ----------
        grid : np.ndarray
            2D array of occupancy values (e.g., probabilities in [0, 1]).
        extent : tuple of float, optional
            (x_min, x_max, y_min, y_max) for the grid. Defaults to bev_range.
        cmap : str
            Matplotlib colormap name.
        alpha : float
            Transparency of the grid overlay.
        vmin, vmax : float
            Value range for colormap normalization.
        """
        _, ax = self.get_figure()

        if extent is None:
            extent = self.bev_range

        ax.imshow(
            grid,
            extent=extent,
            origin="lower",
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            zorder=0,
        )

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def add_legend(self, loc: str = "upper right", fontsize: int = 8) -> None:
        """Add a legend to the current plot."""
        _, ax = self.get_figure()
        ax.legend(
            loc=loc,
            fontsize=fontsize,
            facecolor="#333333",
            edgecolor="white",
            labelcolor="white",
        )

    def set_title(self, title: str) -> None:
        """Set the figure title."""
        _, ax = self.get_figure()
        ax.set_title(title, color="white", fontsize=14)

    def save(self, filepath: str, tight: bool = True) -> None:
        """Save the current figure to a file.

        Parameters
        ----------
        filepath : str
            Output file path (e.g., 'bev_output.png').
        tight : bool
            Whether to use tight bounding box.
        """
        fig, _ = self.get_figure()
        bbox = "tight" if tight else None
        fig.savefig(filepath, bbox_inches=bbox, facecolor=fig.get_facecolor())

    def show(self) -> None:
        """Display the figure interactively."""
        plt.show()

    def render(self) -> Figure:
        """Return the matplotlib Figure object for further manipulation."""
        fig, _ = self.get_figure()
        return fig


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def plot_bev_frame(
    boxes: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    velocities: Optional[np.ndarray] = None,
    map_polylines: Optional[Dict[str, List[np.ndarray]]] = None,
    occupancy_grid: Optional[np.ndarray] = None,
    gt_boxes: Optional[np.ndarray] = None,
    gt_classes: Optional[Sequence[str]] = None,
    bev_range: Tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0),
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """High-level convenience function to create a complete BEV visualization.

    Parameters
    ----------
    boxes : np.ndarray, optional
        Predicted boxes, shape (N, 7+).
    classes : sequence of str, optional
        Class labels for predicted boxes.
    confidences : sequence of float, optional
        Confidence scores for predicted boxes.
    velocities : np.ndarray, optional
        Velocities shape (N, 2) for predicted boxes.
    map_polylines : dict, optional
        Mapping from element type to list of polyline arrays.
    occupancy_grid : np.ndarray, optional
        2D occupancy grid.
    gt_boxes : np.ndarray, optional
        Ground truth boxes for comparison overlay.
    gt_classes : sequence of str, optional
        Classes for GT boxes.
    bev_range : tuple
        BEV extent (x_min, x_max, y_min, y_max).
    title : str
        Figure title.
    save_path : str, optional
        If provided, save figure to this path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    viz = BEVVisualizer(bev_range=bev_range)

    # Occupancy grid (background layer)
    if occupancy_grid is not None:
        viz.plot_occupancy_grid(occupancy_grid)

    # Map elements
    if map_polylines is not None:
        for element_type, polylines in map_polylines.items():
            viz.plot_map_elements(polylines, element_type=element_type)

    # Ego vehicle at origin
    viz.plot_ego_vehicle()

    # Ground truth vs predictions overlay
    if gt_boxes is not None and boxes is not None:
        viz.plot_predictions_vs_gt(
            gt_boxes=gt_boxes,
            pred_boxes=boxes,
            gt_classes=gt_classes,
            pred_classes=classes,
            pred_confidences=confidences,
        )
    elif boxes is not None:
        viz.plot_boxes(boxes, classes=classes, confidences=confidences)

    # Velocity arrows
    if velocities is not None and boxes is not None:
        viz.plot_velocity_arrows(boxes, velocities)

    if title:
        viz.set_title(title)

    viz.add_legend()

    if save_path:
        viz.save(save_path)

    if show:
        viz.show()

    return viz.render()
