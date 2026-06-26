"""Visualization utilities for autonomous driving perception models.

This package provides three visualization modules:

- **bev_visualizer**: Bird's Eye View (BEV) rendering of bounding boxes,
  map elements, ego vehicle, and occupancy grids.
- **pointcloud_viz**: 3D point cloud visualization with Open3D or matplotlib.
- **image_viz**: 2D camera image overlays including projected 3D boxes,
  segmentation masks, depth maps, and multi-camera grid layouts.
"""

from common.visualization.bev_visualizer import (
    BEVVisualizer,
    plot_bev_frame,
    DEFAULT_CLASS_COLORS as BEV_CLASS_COLORS,
)
from common.visualization.pointcloud_viz import (
    MatplotlibPointCloudVisualizer,
    Open3DPointCloudVisualizer,
    animate_sequence,
    filter_points,
    render_multi_view,
    visualize_point_cloud,
    HAS_OPEN3D,
    DEFAULT_CLASS_COLORS as PC_CLASS_COLORS,
)
from common.visualization.image_viz import (
    create_multicamera_grid,
    draw_2d_boxes,
    draw_lane_markings,
    draw_projected_3d_boxes,
    overlay_depth_map,
    overlay_segmentation,
    plot_image_with_boxes,
    plot_multicamera_figure,
    project_points_to_image,
    write_annotated_video,
    write_annotated_video_from_generator,
    DEFAULT_CLASS_COLORS as IMG_CLASS_COLORS,
    SEGMENTATION_COLORS,
)

__all__ = [
    # BEV visualization
    "BEVVisualizer",
    "plot_bev_frame",
    "BEV_CLASS_COLORS",
    # Point cloud visualization
    "MatplotlibPointCloudVisualizer",
    "Open3DPointCloudVisualizer",
    "animate_sequence",
    "filter_points",
    "render_multi_view",
    "visualize_point_cloud",
    "HAS_OPEN3D",
    "PC_CLASS_COLORS",
    # Image visualization
    "create_multicamera_grid",
    "draw_2d_boxes",
    "draw_lane_markings",
    "draw_projected_3d_boxes",
    "overlay_depth_map",
    "overlay_segmentation",
    "plot_image_with_boxes",
    "plot_multicamera_figure",
    "project_points_to_image",
    "write_annotated_video",
    "write_annotated_video_from_generator",
    "IMG_CLASS_COLORS",
    "SEGMENTATION_COLORS",
]
