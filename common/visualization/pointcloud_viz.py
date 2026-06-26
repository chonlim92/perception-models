"""3D point cloud visualization for autonomous driving perception.

Supports rendering point clouds colored by height, intensity, or semantic class,
with optional 3D bounding box wireframes. Uses Open3D when available, with a
matplotlib 3D fallback.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional Open3D import
# ---------------------------------------------------------------------------
try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# ---------------------------------------------------------------------------
# Default color palette
# ---------------------------------------------------------------------------
DEFAULT_CLASS_COLORS: Dict[str, Tuple[float, float, float]] = {
    "car": (0.12, 0.47, 0.71),
    "truck": (1.0, 0.50, 0.05),
    "bus": (0.17, 0.63, 0.17),
    "trailer": (0.84, 0.15, 0.16),
    "construction_vehicle": (0.58, 0.40, 0.74),
    "pedestrian": (0.89, 0.47, 0.76),
    "motorcycle": (0.50, 0.50, 0.50),
    "bicycle": (0.74, 0.74, 0.13),
    "traffic_cone": (0.09, 0.75, 0.81),
    "barrier": (0.68, 0.78, 0.91),
    "vegetation": (0.0, 0.60, 0.0),
    "ground": (0.40, 0.26, 0.13),
    "building": (0.55, 0.55, 0.55),
    "unknown": (0.80, 0.80, 0.80),
}

# Height-based colormap range (meters relative to sensor)
DEFAULT_HEIGHT_RANGE: Tuple[float, float] = (-3.0, 3.0)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _height_to_color(
    heights: np.ndarray,
    height_range: Tuple[float, float] = DEFAULT_HEIGHT_RANGE,
    cmap_name: str = "jet",
) -> np.ndarray:
    """Map height values to RGB colors using a matplotlib colormap.

    Parameters
    ----------
    heights : np.ndarray
        1D array of height (z) values.
    height_range : tuple
        (min_height, max_height) for normalization.
    cmap_name : str
        Name of matplotlib colormap.

    Returns
    -------
    np.ndarray
        Shape (N, 3) RGB colors in [0, 1].
    """
    cmap = plt.get_cmap(cmap_name)
    z_min, z_max = height_range
    normalized = np.clip((heights - z_min) / (z_max - z_min + 1e-8), 0.0, 1.0)
    colors = cmap(normalized)[:, :3]  # drop alpha
    return colors


def _intensity_to_color(
    intensities: np.ndarray,
    cmap_name: str = "hot",
) -> np.ndarray:
    """Map intensity values to RGB colors.

    Parameters
    ----------
    intensities : np.ndarray
        1D array of intensity values (assumed [0, 255] or [0, 1]).
    cmap_name : str
        Colormap name.

    Returns
    -------
    np.ndarray
        Shape (N, 3) RGB colors.
    """
    cmap = plt.get_cmap(cmap_name)
    max_val = intensities.max() if intensities.max() > 0 else 1.0
    normalized = np.clip(intensities / max_val, 0.0, 1.0)
    return cmap(normalized)[:, :3]


def _class_to_color(
    class_ids: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Map semantic class IDs to RGB colors.

    Parameters
    ----------
    class_ids : np.ndarray
        1D integer array of class labels.
    class_names : dict, optional
        Mapping from class ID to class name (for color lookup).

    Returns
    -------
    np.ndarray
        Shape (N, 3) RGB colors.
    """
    colors = np.zeros((len(class_ids), 3), dtype=np.float64)
    unique_ids = np.unique(class_ids)

    # Generate colors for IDs not in the palette
    cmap = plt.get_cmap("tab20")

    for uid in unique_ids:
        mask = class_ids == uid
        name = class_names.get(int(uid), "unknown") if class_names else "unknown"
        if name in DEFAULT_CLASS_COLORS:
            colors[mask] = DEFAULT_CLASS_COLORS[name]
        else:
            colors[mask] = cmap(int(uid) % 20)[:3]

    return colors


def _box_wireframe_lines(
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """Compute 12 line segments forming a 3D wireframe box.

    Parameters
    ----------
    center : np.ndarray
        [x, y, z] center of the box.
    size : np.ndarray
        [length, width, height] of the box.
    yaw : float
        Heading angle (radians) around the z-axis.

    Returns
    -------
    np.ndarray
        Shape (12, 2, 3) array of line segment endpoints.
    """
    l, w, h = size[0] / 2, size[1] / 2, size[2] / 2
    # 8 corners in local frame
    corners_local = np.array([
        [l, w, h], [l, -w, h], [-l, -w, h], [-l, w, h],
        [l, w, -h], [l, -w, -h], [-l, -w, -h], [-l, w, -h],
    ])

    # Rotation matrix (yaw around z-axis)
    cos, sin = np.cos(yaw), np.sin(yaw)
    rot = np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1],
    ])

    corners = corners_local @ rot.T + center

    # 12 edges connecting corners
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # top face
        (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    lines = np.array([[corners[e[0]], corners[e[1]]] for e in edges])
    return lines


def filter_points(
    points: np.ndarray,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    class_ids: Optional[np.ndarray] = None,
    keep_classes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Filter point cloud by spatial range and/or class.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 3+) point cloud.
    x_range, y_range, z_range : tuple, optional
        (min, max) spatial filters.
    class_ids : np.ndarray, optional
        Per-point class labels.
    keep_classes : list of int, optional
        Class IDs to keep.

    Returns
    -------
    filtered_points : np.ndarray
        Filtered point array.
    filtered_classes : np.ndarray or None
        Filtered class IDs if provided.
    """
    mask = np.ones(len(points), dtype=bool)

    if x_range is not None:
        mask &= (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
    if y_range is not None:
        mask &= (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    if z_range is not None:
        mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])

    if class_ids is not None and keep_classes is not None:
        class_mask = np.isin(class_ids, keep_classes)
        mask &= class_mask

    filtered_classes = class_ids[mask] if class_ids is not None else None
    return points[mask], filtered_classes


# ---------------------------------------------------------------------------
# Open3D visualization
# ---------------------------------------------------------------------------


class Open3DPointCloudVisualizer:
    """Point cloud visualization using Open3D (interactive 3D viewer).

    Falls back to matplotlib if Open3D is unavailable.
    """

    def __init__(self, window_name: str = "Point Cloud Viewer", width: int = 1280, height: int = 720):
        if not HAS_OPEN3D:
            raise ImportError(
                "Open3D is not installed. Install with: pip install open3d"
            )
        self.window_name = window_name
        self.width = width
        self.height = height
        self._geometries: List = []

    def add_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        color_mode: str = "height",
        intensities: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None,
        class_names: Optional[Dict[int, str]] = None,
        point_size: float = 1.0,
    ) -> None:
        """Add a point cloud to the scene.

        Parameters
        ----------
        points : np.ndarray
            Shape (N, 3) point positions [x, y, z].
        colors : np.ndarray, optional
            Shape (N, 3) pre-computed RGB colors in [0, 1].
        color_mode : str
            How to color points if `colors` is None:
            'height', 'intensity', or 'class'.
        intensities : np.ndarray, optional
            Intensity values (required if color_mode='intensity').
        class_ids : np.ndarray, optional
            Semantic class IDs (required if color_mode='class').
        class_names : dict, optional
            Mapping from class ID to name.
        point_size : float
            Rendering point size (stored for visualization settings).
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3].astype(np.float64))
        elif color_mode == "height":
            c = _height_to_color(points[:, 2])
            pcd.colors = o3d.utility.Vector3dVector(c)
        elif color_mode == "intensity" and intensities is not None:
            c = _intensity_to_color(intensities)
            pcd.colors = o3d.utility.Vector3dVector(c)
        elif color_mode == "class" and class_ids is not None:
            c = _class_to_color(class_ids, class_names)
            pcd.colors = o3d.utility.Vector3dVector(c)
        else:
            c = _height_to_color(points[:, 2])
            pcd.colors = o3d.utility.Vector3dVector(c)

        self._geometries.append(pcd)

    def add_boxes(
        self,
        boxes: np.ndarray,
        colors: Optional[Union[np.ndarray, List[Tuple[float, float, float]]]] = None,
        classes: Optional[Sequence[str]] = None,
        linewidth: float = 2.0,
    ) -> None:
        """Add 3D bounding box wireframes.

        Parameters
        ----------
        boxes : np.ndarray
            Shape (N, 7+): [x, y, z, length, width, height, yaw, ...].
        colors : array-like, optional
            Per-box RGB colors. If None, uses class colors.
        classes : sequence of str, optional
            Class name per box (for color lookup).
        linewidth : float
            Line width for wireframes.
        """
        for i in range(len(boxes)):
            center = boxes[i, :3]
            size = boxes[i, 3:6]
            yaw = boxes[i, 6]

            lines_pts = _box_wireframe_lines(center, size, yaw)

            # Flatten to unique points and edges
            all_pts = lines_pts.reshape(-1, 3)
            # 12 edges, each defined by consecutive pairs
            line_indices = [[2 * j, 2 * j + 1] for j in range(12)]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(all_pts)
            line_set.lines = o3d.utility.Vector2iVector(line_indices)

            if colors is not None:
                color = colors[i] if hasattr(colors, "__len__") and len(colors) > i else (1, 0, 0)
            elif classes is not None and classes[i] in DEFAULT_CLASS_COLORS:
                color = DEFAULT_CLASS_COLORS[classes[i]]
            else:
                color = (1.0, 0.0, 0.0)

            line_set.colors = o3d.utility.Vector3dVector(
                [color for _ in range(12)]
            )
            self._geometries.append(line_set)

    def show(self) -> None:
        """Launch interactive Open3D viewer."""
        o3d.visualization.draw_geometries(
            self._geometries,
            window_name=self.window_name,
            width=self.width,
            height=self.height,
        )

    def reset(self) -> None:
        """Clear all geometries."""
        self._geometries.clear()


# ---------------------------------------------------------------------------
# Matplotlib 3D fallback
# ---------------------------------------------------------------------------


class MatplotlibPointCloudVisualizer:
    """Point cloud visualization using matplotlib (static 3D scatter).

    Use this when Open3D is not available or non-interactive rendering is needed.
    """

    def __init__(
        self,
        figsize: Tuple[float, float] = (12, 8),
        dpi: int = 100,
        elev: float = 30.0,
        azim: float = -60.0,
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.elev = elev
        self.azim = azim
        self._fig: Optional[Figure] = None
        self._ax = None
        self._point_clouds: List[dict] = []
        self._boxes: List[dict] = []

    def add_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        color_mode: str = "height",
        intensities: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None,
        class_names: Optional[Dict[int, str]] = None,
        point_size: float = 0.5,
        subsample: Optional[int] = 50000,
    ) -> None:
        """Add a point cloud for rendering.

        Parameters
        ----------
        points : np.ndarray
            Shape (N, 3+) point positions.
        colors : np.ndarray, optional
            Pre-computed RGB colors.
        color_mode : str
            Coloring strategy: 'height', 'intensity', or 'class'.
        intensities : np.ndarray, optional
            Intensity values.
        class_ids : np.ndarray, optional
            Class labels.
        class_names : dict, optional
            Class ID to name mapping.
        point_size : float
            Scatter point size.
        subsample : int or None
            Max points to render (random subsampling for performance).
        """
        pts = points[:, :3].copy()

        # Subsample for matplotlib performance
        if subsample is not None and len(pts) > subsample:
            idx = np.random.choice(len(pts), subsample, replace=False)
            pts = pts[idx]
            if colors is not None:
                colors = colors[idx]
            if intensities is not None:
                intensities = intensities[idx]
            if class_ids is not None:
                class_ids = class_ids[idx]

        if colors is None:
            if color_mode == "height":
                colors = _height_to_color(pts[:, 2])
            elif color_mode == "intensity" and intensities is not None:
                colors = _intensity_to_color(intensities)
            elif color_mode == "class" and class_ids is not None:
                colors = _class_to_color(class_ids, class_names)
            else:
                colors = _height_to_color(pts[:, 2])

        self._point_clouds.append({
            "points": pts,
            "colors": colors,
            "point_size": point_size,
        })

    def add_boxes(
        self,
        boxes: np.ndarray,
        classes: Optional[Sequence[str]] = None,
        color_override: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Add 3D bounding box wireframes.

        Parameters
        ----------
        boxes : np.ndarray
            Shape (N, 7+): [x, y, z, length, width, height, yaw, ...].
        classes : sequence of str, optional
            Class name per box.
        color_override : tuple, optional
            Single color for all boxes.
        """
        self._boxes.append({
            "boxes": boxes,
            "classes": classes,
            "color_override": color_override,
        })

    def render(
        self,
        title: str = "Point Cloud",
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
    ) -> Figure:
        """Render the scene and return the figure.

        Parameters
        ----------
        title : str
            Figure title.
        x_range, y_range, z_range : tuple, optional
            Axis limits.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=self.elev, azim=self.azim)

        # Plot point clouds
        for pc_data in self._point_clouds:
            pts = pc_data["points"]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=pc_data["colors"],
                s=pc_data["point_size"],
                alpha=0.6,
                depthshade=True,
            )

        # Plot boxes
        for box_data in self._boxes:
            boxes = box_data["boxes"]
            classes = box_data["classes"]
            color_override = box_data["color_override"]

            for i in range(len(boxes)):
                center = boxes[i, :3]
                size = boxes[i, 3:6]
                yaw = boxes[i, 6]
                lines = _box_wireframe_lines(center, size, yaw)

                if color_override:
                    color = color_override
                elif classes and classes[i] in DEFAULT_CLASS_COLORS:
                    color = DEFAULT_CLASS_COLORS[classes[i]]
                else:
                    color = (1.0, 0.0, 0.0)

                for line in lines:
                    ax.plot3D(
                        line[:, 0], line[:, 1], line[:, 2],
                        color=color, linewidth=1.0,
                    )

        # Set limits
        if x_range:
            ax.set_xlim3d(x_range)
        if y_range:
            ax.set_ylim3d(y_range)
        if z_range:
            ax.set_zlim3d(z_range)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)

        self._fig = fig
        self._ax = ax
        return fig

    def save(self, filepath: str, tight: bool = True) -> None:
        """Save the rendered figure."""
        if self._fig is None:
            self.render()
        bbox = "tight" if tight else None
        self._fig.savefig(filepath, bbox_inches=bbox, dpi=self.dpi)

    def show(self) -> None:
        """Display the figure."""
        if self._fig is None:
            self.render()
        plt.show()

    def reset(self) -> None:
        """Clear all stored data and close figure."""
        self._point_clouds.clear()
        self._boxes.clear()
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None


# ---------------------------------------------------------------------------
# Side-by-side multi-viewpoint rendering
# ---------------------------------------------------------------------------


def render_multi_view(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    color_mode: str = "height",
    boxes: Optional[np.ndarray] = None,
    box_classes: Optional[Sequence[str]] = None,
    viewpoints: Optional[List[Tuple[float, float]]] = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (18, 6),
    save_path: Optional[str] = None,
    subsample: int = 30000,
) -> Figure:
    """Render point cloud from multiple viewpoints side by side.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 3+) point cloud.
    colors : np.ndarray, optional
        Pre-computed colors.
    color_mode : str
        Coloring mode if colors is None.
    boxes : np.ndarray, optional
        Bounding boxes shape (M, 7+).
    box_classes : sequence of str, optional
        Class labels for boxes.
    viewpoints : list of (elev, azim), optional
        Matplotlib 3D view angles. Default: front, side, top.
    titles : list of str, optional
        Subplot titles.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    subsample : int
        Max points per view.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if viewpoints is None:
        viewpoints = [(30, -60), (0, 0), (90, -90)]
    if titles is None:
        titles = ["Perspective", "Front", "Top-down"]

    n_views = len(viewpoints)
    fig = plt.figure(figsize=figsize)

    # Subsample once
    pts = points[:, :3].copy()
    if len(pts) > subsample:
        idx = np.random.choice(len(pts), subsample, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]

    if colors is None:
        colors = _height_to_color(pts[:, 2])

    for i, (elev, azim) in enumerate(viewpoints):
        ax = fig.add_subplot(1, n_views, i + 1, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.3, alpha=0.5)

        # Draw boxes
        if boxes is not None:
            for j in range(len(boxes)):
                center = boxes[j, :3]
                size = boxes[j, 3:6]
                yaw = boxes[j, 6]
                lines = _box_wireframe_lines(center, size, yaw)

                if box_classes and box_classes[j] in DEFAULT_CLASS_COLORS:
                    color = DEFAULT_CLASS_COLORS[box_classes[j]]
                else:
                    color = (1.0, 0.0, 0.0)

                for line in lines:
                    ax.plot3D(line[:, 0], line[:, 1], line[:, 2], color=color, lw=1.0)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if i < len(titles):
            ax.set_title(titles[i])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ---------------------------------------------------------------------------
# Sequence animation (matplotlib)
# ---------------------------------------------------------------------------


def animate_sequence(
    point_clouds: List[np.ndarray],
    color_mode: str = "height",
    boxes_sequence: Optional[List[np.ndarray]] = None,
    interval_ms: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    subsample: int = 30000,
) -> None:
    """Animate a sequence of point cloud frames.

    Parameters
    ----------
    point_clouds : list of np.ndarray
        Each element is shape (N_i, 3+) point cloud for one frame.
    color_mode : str
        Coloring strategy.
    boxes_sequence : list of np.ndarray, optional
        Bounding boxes per frame, each shape (M_i, 7+).
    interval_ms : int
        Milliseconds between frames.
    save_path : str, optional
        Path to save animation (e.g., 'animation.gif' or 'animation.mp4').
    figsize : tuple
        Figure size.
    subsample : int
        Max points per frame.
    """
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx: int):
        ax.cla()
        pts = point_clouds[frame_idx][:, :3]

        if len(pts) > subsample:
            idx = np.random.choice(len(pts), subsample, replace=False)
            pts = pts[idx]

        colors = _height_to_color(pts[:, 2])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.3, alpha=0.5)

        if boxes_sequence is not None and frame_idx < len(boxes_sequence):
            boxes = boxes_sequence[frame_idx]
            if boxes is not None:
                for j in range(len(boxes)):
                    lines = _box_wireframe_lines(boxes[j, :3], boxes[j, 3:6], boxes[j, 6])
                    for line in lines:
                        ax.plot3D(line[:, 0], line[:, 1], line[:, 2], color="r", lw=1.0)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Frame {frame_idx}")
        ax.view_init(elev=30, azim=-60)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(point_clouds),
        interval=interval_ms,
        repeat=True,
    )

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval_ms)
        else:
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval_ms)
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience: auto-select backend
# ---------------------------------------------------------------------------


def visualize_point_cloud(
    points: np.ndarray,
    color_mode: str = "height",
    intensities: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    boxes: Optional[np.ndarray] = None,
    box_classes: Optional[Sequence[str]] = None,
    use_open3d: bool = True,
    save_path: Optional[str] = None,
    title: str = "Point Cloud",
) -> Optional[Figure]:
    """Convenience function to visualize a point cloud with automatic backend selection.

    Uses Open3D for interactive viewing if available and `use_open3d=True`,
    otherwise falls back to matplotlib.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 3+) point cloud.
    color_mode : str
        'height', 'intensity', or 'class'.
    intensities : np.ndarray, optional
        Point intensities (needed for color_mode='intensity').
    class_ids : np.ndarray, optional
        Semantic labels (needed for color_mode='class').
    class_names : dict, optional
        Class ID to name mapping.
    boxes : np.ndarray, optional
        3D bounding boxes, shape (M, 7+).
    box_classes : sequence of str, optional
        Class names for boxes.
    use_open3d : bool
        Prefer Open3D if available.
    save_path : str, optional
        Save figure to path (matplotlib only).
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if matplotlib is used, None for Open3D.
    """
    if use_open3d and HAS_OPEN3D and save_path is None:
        viz = Open3DPointCloudVisualizer(window_name=title)
        viz.add_point_cloud(
            points,
            color_mode=color_mode,
            intensities=intensities,
            class_ids=class_ids,
            class_names=class_names,
        )
        if boxes is not None:
            viz.add_boxes(boxes, classes=box_classes)
        viz.show()
        return None
    else:
        viz = MatplotlibPointCloudVisualizer()
        viz.add_point_cloud(
            points,
            color_mode=color_mode,
            intensities=intensities,
            class_ids=class_ids,
            class_names=class_names,
        )
        if boxes is not None:
            viz.add_boxes(boxes, classes=box_classes)
        fig = viz.render(title=title)
        if save_path:
            viz.save(save_path)
        return fig
