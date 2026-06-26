"""
Anchor generation and box encoding/decoding for PointPillars 3D object detection.

This module provides:
- AnchorGenerator: generates dense 3D anchors at every BEV grid position
- encode_boxes: encodes ground-truth boxes relative to anchors
- decode_boxes: decodes regression deltas back to absolute boxes
- iou_2d_bev: computes bird's-eye-view IoU between two sets of oriented boxes
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AnchorGenerator(nn.Module):
    """Generates 3D anchor boxes at every position of the BEV feature map.

    For each class, anchors are placed at every (x, y) grid cell center with
    the class-specific z-center height.  Two rotations (0 and pi/2) are used
    by default, yielding 2 * num_classes anchors per spatial location.

    Parameters
    ----------
    anchor_ranges : List[List[float]]
        Per-class spatial ranges [xmin, ymin, zmin, xmax, ymax, zmax].
    anchor_sizes : List[List[float]]
        Per-class anchor dimensions [w, l, h].
    anchor_rotations : List[float], optional
        Rotation angles in radians for each anchor.  Default is [0, pi/2].
    anchor_heights : List[float]
        Per-class z-center values for the anchors.
    feature_map_size : Tuple[int, int]
        Spatial size (H, W) of the backbone output feature map.
    """

    def __init__(
        self,
        anchor_ranges: List[List[float]],
        anchor_sizes: List[List[float]],
        anchor_rotations: Optional[List[float]] = None,
        anchor_heights: Optional[List[float]] = None,
        feature_map_size: Tuple[int, int] = (200, 176),
    ) -> None:
        super().__init__()

        if anchor_rotations is None:
            anchor_rotations = [0.0, math.pi / 2.0]
        if anchor_heights is None:
            anchor_heights = [-1.0] * len(anchor_sizes)

        self.anchor_ranges = anchor_ranges
        self.anchor_sizes = anchor_sizes
        self.anchor_rotations = anchor_rotations
        self.anchor_heights = anchor_heights
        self.feature_map_size = feature_map_size

        # Pre-generate and register as buffer so anchors move with model device
        anchors = self._generate_all_anchors()
        self.register_buffer("anchors", anchors)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_single_class_anchors(
        self,
        anchor_range: List[float],
        size: List[float],
        height: float,
    ) -> Tensor:
        """Generate anchors for a single class over the entire feature map.

        Returns
        -------
        Tensor of shape (H * W * num_rotations, 7)
            Each row is [x, y, z, w, l, h, theta].
        """
        xmin, ymin, _zmin, xmax, ymax, _zmax = anchor_range
        H, W = self.feature_map_size

        # Compute grid spacing
        x_stride = (xmax - xmin) / W
        y_stride = (ymax - ymin) / H

        # Grid center coordinates
        x_centers = torch.linspace(
            xmin + x_stride / 2.0, xmax - x_stride / 2.0, W
        )
        y_centers = torch.linspace(
            ymin + y_stride / 2.0, ymax - y_stride / 2.0, H
        )

        # Meshgrid: y along rows (H), x along columns (W)
        # Output shapes: (H, W)
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")

        # Flatten spatial dims
        xx_flat = xx.reshape(-1)  # (H*W,)
        yy_flat = yy.reshape(-1)  # (H*W,)

        num_locations = H * W
        num_rotations = len(self.anchor_rotations)

        # Repeat for each rotation
        xx_rep = xx_flat.unsqueeze(1).expand(-1, num_rotations).reshape(-1)
        yy_rep = yy_flat.unsqueeze(1).expand(-1, num_rotations).reshape(-1)

        total = num_locations * num_rotations

        zz = torch.full((total,), height, dtype=torch.float32)
        ww = torch.full((total,), size[0], dtype=torch.float32)
        ll = torch.full((total,), size[1], dtype=torch.float32)
        hh = torch.full((total,), size[2], dtype=torch.float32)

        # Rotations tiled across all locations
        rotations = torch.tensor(self.anchor_rotations, dtype=torch.float32)
        theta = rotations.unsqueeze(0).expand(num_locations, -1).reshape(-1)

        # Stack into (N, 7)
        anchors = torch.stack([xx_rep, yy_rep, zz, ww, ll, hh, theta], dim=1)
        return anchors

    def _generate_all_anchors(self) -> Tensor:
        """Generate anchors for all classes and concatenate.

        Returns
        -------
        Tensor of shape (num_anchors, 7)
        """
        all_anchors = []
        for i, (rng, sz) in enumerate(
            zip(self.anchor_ranges, self.anchor_sizes)
        ):
            height = self.anchor_heights[i]
            class_anchors = self._generate_single_class_anchors(rng, sz, height)
            all_anchors.append(class_anchors)
        return torch.cat(all_anchors, dim=0)

    def forward(self) -> Tensor:
        """Return the pre-generated anchors.

        Returns
        -------
        Tensor of shape (num_anchors, 7)
            Columns are [x, y, z, w, l, h, theta].
        """
        return self.anchors

    @property
    def num_anchors(self) -> int:
        """Total number of anchors across all classes."""
        return self.anchors.shape[0]

    @property
    def num_anchors_per_location(self) -> int:
        """Number of anchors at each spatial location."""
        return len(self.anchor_sizes) * len(self.anchor_rotations)


# --------------------------------------------------------------------------
# Box Encoding / Decoding
# --------------------------------------------------------------------------


def encode_boxes(anchors: Tensor, gt_boxes: Tensor) -> Tensor:
    """Encode ground-truth boxes relative to anchors.

    Uses the encoding from the PointPillars / SECOND papers:
        delta_x = (gt_x - a_x) / d
        delta_y = (gt_y - a_y) / d
        delta_z = (gt_z - a_z) / a_h
        delta_w = log(gt_w / a_w)
        delta_l = log(gt_l / a_l)
        delta_h = log(gt_h / a_h)
        delta_theta = gt_theta - a_theta

    where d = sqrt(a_w^2 + a_l^2) is the diagonal of the anchor base.

    Parameters
    ----------
    anchors : Tensor, shape (N, 7)
        Anchor boxes [x, y, z, w, l, h, theta].
    gt_boxes : Tensor, shape (N, 7)
        Ground-truth boxes [x, y, z, w, l, h, theta].

    Returns
    -------
    Tensor, shape (N, 7)
        Encoded regression targets.
    """
    a_x, a_y, a_z = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    a_w, a_l, a_h = anchors[:, 3], anchors[:, 4], anchors[:, 5]
    a_theta = anchors[:, 6]

    g_x, g_y, g_z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
    g_w, g_l, g_h = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
    g_theta = gt_boxes[:, 6]

    # Diagonal of anchor base (used to normalise x/y deltas)
    diagonal = torch.sqrt(a_w ** 2 + a_l ** 2)

    delta_x = (g_x - a_x) / diagonal
    delta_y = (g_y - a_y) / diagonal
    delta_z = (g_z - a_z) / a_h
    delta_w = torch.log(g_w / a_w)
    delta_l = torch.log(g_l / a_l)
    delta_h = torch.log(g_h / a_h)
    delta_theta = g_theta - a_theta

    return torch.stack(
        [delta_x, delta_y, delta_z, delta_w, delta_l, delta_h, delta_theta],
        dim=1,
    )


def decode_boxes(anchors: Tensor, deltas: Tensor) -> Tensor:
    """Decode regression deltas to absolute box parameters.

    Inverse of :func:`encode_boxes`.

    Parameters
    ----------
    anchors : Tensor, shape (N, 7)
        Anchor boxes [x, y, z, w, l, h, theta].
    deltas : Tensor, shape (N, 7)
        Predicted regression deltas.

    Returns
    -------
    Tensor, shape (N, 7)
        Decoded boxes [x, y, z, w, l, h, theta].
    """
    a_x, a_y, a_z = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    a_w, a_l, a_h = anchors[:, 3], anchors[:, 4], anchors[:, 5]
    a_theta = anchors[:, 6]

    d_x, d_y, d_z = deltas[:, 0], deltas[:, 1], deltas[:, 2]
    d_w, d_l, d_h = deltas[:, 3], deltas[:, 4], deltas[:, 5]
    d_theta = deltas[:, 6]

    diagonal = torch.sqrt(a_w ** 2 + a_l ** 2)

    g_x = a_x + d_x * diagonal
    g_y = a_y + d_y * diagonal
    g_z = a_z + d_z * a_h
    g_w = a_w * torch.exp(d_w)
    g_l = a_l * torch.exp(d_l)
    g_h = a_h * torch.exp(d_h)
    g_theta = a_theta + d_theta

    return torch.stack([g_x, g_y, g_z, g_w, g_l, g_h, g_theta], dim=1)


# --------------------------------------------------------------------------
# BEV IoU Computation
# --------------------------------------------------------------------------


def _corners_from_boxes_bev(boxes: Tensor) -> Tensor:
    """Compute the four BEV corners of oriented boxes.

    Parameters
    ----------
    boxes : Tensor, shape (N, 7)
        Boxes [x, y, z, w, l, h, theta].

    Returns
    -------
    Tensor, shape (N, 4, 2)
        Four corner points (x, y) for each box in counter-clockwise order.
    """
    x, y = boxes[:, 0], boxes[:, 1]
    w, l = boxes[:, 3], boxes[:, 4]
    theta = boxes[:, 6]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Half extents
    hw = w / 2.0
    hl = l / 2.0

    # Corner offsets before rotation (dx, dy) relative to center
    # Order: front-left, front-right, rear-right, rear-left
    dx = torch.stack([hl, hl, -hl, -hl], dim=1)  # (N, 4)
    dy = torch.stack([-hw, hw, hw, -hw], dim=1)  # (N, 4)

    # Rotate
    cos_t = cos_t.unsqueeze(1)  # (N, 1)
    sin_t = sin_t.unsqueeze(1)  # (N, 1)
    rot_x = dx * cos_t - dy * sin_t  # (N, 4)
    rot_y = dx * sin_t + dy * cos_t  # (N, 4)

    # Translate
    corners_x = rot_x + x.unsqueeze(1)
    corners_y = rot_y + y.unsqueeze(1)

    corners = torch.stack([corners_x, corners_y], dim=2)  # (N, 4, 2)
    return corners


def _polygon_area(vertices: Tensor) -> Tensor:
    """Compute the area of convex polygons using the shoelace formula.

    Parameters
    ----------
    vertices : Tensor, shape (N, K, 2)
        Ordered polygon vertices.

    Returns
    -------
    Tensor, shape (N,)
        Polygon areas (absolute value).
    """
    # Shoelace: sum of (x_i * y_{i+1} - x_{i+1} * y_i)
    x = vertices[:, :, 0]
    y = vertices[:, :, 1]
    x_next = torch.roll(x, shifts=-1, dims=1)
    y_next = torch.roll(y, shifts=-1, dims=1)
    area = 0.5 * torch.abs((x * y_next - x_next * y).sum(dim=1))
    return area


def _cross_2d(o: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Cross product of vectors (a - o) and (b - o).

    Parameters
    ----------
    o, a, b : Tensor, shape (..., 2)

    Returns
    -------
    Tensor, shape (...)
        Scalar cross product values.
    """
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (
        a[..., 1] - o[..., 1]
    ) * (b[..., 0] - o[..., 0])


def _line_segment_intersection(
    p1: Tensor, p2: Tensor, p3: Tensor, p4: Tensor
) -> Tuple[Tensor, Tensor]:
    """Find intersection points of line segments (p1-p2) and (p3-p4).

    Uses parametric form: intersection at p1 + t*(p2-p1) where 0<=t<=1
    and p3 + u*(p4-p3) where 0<=u<=1.

    Parameters
    ----------
    p1, p2, p3, p4 : Tensor, shape (..., 2)

    Returns
    -------
    points : Tensor, shape (..., 2)
        Intersection coordinates (valid only where mask is True).
    mask : Tensor, shape (...)
        Boolean mask indicating valid intersections.
    """
    d1 = p2 - p1  # (..., 2)
    d2 = p4 - p3  # (..., 2)

    denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]

    # Avoid division by zero for parallel segments
    parallel = denom.abs() < 1e-10

    # Safe denominator
    safe_denom = torch.where(parallel, torch.ones_like(denom), denom)

    d3 = p3 - p1
    t = (d3[..., 0] * d2[..., 1] - d3[..., 1] * d2[..., 0]) / safe_denom
    u = (d3[..., 0] * d1[..., 1] - d3[..., 1] * d1[..., 0]) / safe_denom

    valid = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

    # Compute intersection point
    ix = p1[..., 0] + t * d1[..., 0]
    iy = p1[..., 1] + t * d1[..., 1]
    points = torch.stack([ix, iy], dim=-1)

    return points, valid


def _point_in_quadrilateral(point: Tensor, quad: Tensor) -> Tensor:
    """Test whether points lie inside convex quadrilaterals.

    Uses the cross-product sign test: a point is inside a convex polygon
    if all cross products with consecutive edges have the same sign.

    Parameters
    ----------
    point : Tensor, shape (..., 2)
    quad : Tensor, shape (..., 4, 2)

    Returns
    -------
    Tensor, shape (...)
        Boolean mask.
    """
    num_vertices = 4
    inside = torch.ones(point.shape[:-1], dtype=torch.bool, device=point.device)

    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        edge_start = quad[..., i, :]  # (..., 2)
        edge_end = quad[..., j, :]  # (..., 2)
        cross = _cross_2d(edge_start, edge_end, point)
        if i == 0:
            sign = cross > 0
        else:
            inside = inside & ((cross > 0) == sign)

    return inside


def _convex_hull_intersection_area_single(
    corners_a: Tensor, corners_b: Tensor
) -> float:
    """Compute intersection area of two convex quadrilaterals (single pair).

    Gathers all intersection points (edge-edge intersections and
    vertices of each polygon inside the other), computes their convex
    hull, and returns its area via the shoelace formula.

    Parameters
    ----------
    corners_a : Tensor, shape (4, 2)
    corners_b : Tensor, shape (4, 2)

    Returns
    -------
    float
        Intersection area.
    """
    intersection_points: List[Tensor] = []

    # 1. Edge-edge intersections (4 edges x 4 edges = 16 checks)
    for i in range(4):
        i_next = (i + 1) % 4
        for j in range(4):
            j_next = (j + 1) % 4
            pt, valid = _line_segment_intersection(
                corners_a[i], corners_a[i_next],
                corners_b[j], corners_b[j_next],
            )
            if valid.item():
                intersection_points.append(pt)

    # 2. Vertices of A inside B
    for i in range(4):
        pt = corners_a[i]
        if _point_in_quadrilateral(
            pt.unsqueeze(0), corners_b.unsqueeze(0)
        ).item():
            intersection_points.append(pt)

    # 3. Vertices of B inside A
    for i in range(4):
        pt = corners_b[i]
        if _point_in_quadrilateral(
            pt.unsqueeze(0), corners_a.unsqueeze(0)
        ).item():
            intersection_points.append(pt)

    if len(intersection_points) < 3:
        return 0.0

    # Convex hull via angular sort around centroid
    pts = torch.stack(intersection_points, dim=0)  # (K, 2)
    centroid = pts.mean(dim=0)  # (2,)
    angles = torch.atan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = angles.argsort()
    pts_sorted = pts[order]

    # Shoelace area
    x = pts_sorted[:, 0]
    y = pts_sorted[:, 1]
    x_next = torch.roll(x, shifts=-1, dims=0)
    y_next = torch.roll(y, shifts=-1, dims=0)
    area = 0.5 * torch.abs((x * y_next - x_next * y).sum())
    return area.item()


def iou_2d_bev(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute pairwise BEV IoU between two sets of oriented 3D boxes.

    This implements proper rotated IoU by computing the intersection polygon
    of each pair of oriented rectangles in bird's-eye view.

    Parameters
    ----------
    boxes_a : Tensor, shape (M, 7)
        First set of boxes [x, y, z, w, l, h, theta].
    boxes_b : Tensor, shape (N, 7)
        Second set of boxes [x, y, z, w, l, h, theta].

    Returns
    -------
    Tensor, shape (M, N)
        Pairwise BEV IoU matrix.
    """
    M = boxes_a.shape[0]
    N = boxes_b.shape[0]

    corners_a = _corners_from_boxes_bev(boxes_a)  # (M, 4, 2)
    corners_b = _corners_from_boxes_bev(boxes_b)  # (N, 4, 2)

    # Areas of each box in BEV (w * l)
    area_a = boxes_a[:, 3] * boxes_a[:, 4]  # (M,)
    area_b = boxes_b[:, 3] * boxes_b[:, 4]  # (N,)

    iou_matrix = boxes_a.new_zeros((M, N))

    for i in range(M):
        for j in range(N):
            inter_area = _convex_hull_intersection_area_single(
                corners_a[i], corners_b[j]
            )
            union_area = area_a[i].item() + area_b[j].item() - inter_area
            if union_area > 1e-10:
                iou_matrix[i, j] = inter_area / union_area

    return iou_matrix


def iou_2d_bev_axis_aligned(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute pairwise BEV IoU using axis-aligned bounding box approximation.

    This is a faster alternative to :func:`iou_2d_bev` that ignores rotation
    and computes IoU based on the axis-aligned bounding boxes derived from
    (x, y, w, l). Suitable when boxes are roughly axis-aligned or when
    speed is critical (e.g., during training target assignment).

    Parameters
    ----------
    boxes_a : Tensor, shape (M, 7)
        First set of boxes [x, y, z, w, l, h, theta].
    boxes_b : Tensor, shape (N, 7)
        Second set of boxes [x, y, z, w, l, h, theta].

    Returns
    -------
    Tensor, shape (M, N)
        Pairwise axis-aligned BEV IoU matrix.
    """
    # Extract center and size
    xa, ya = boxes_a[:, 0], boxes_a[:, 1]
    wa, la = boxes_a[:, 3], boxes_a[:, 4]

    xb, yb = boxes_b[:, 0], boxes_b[:, 1]
    wb, lb = boxes_b[:, 3], boxes_b[:, 4]

    # Convert to min/max form
    a_x1 = xa - wa / 2.0
    a_x2 = xa + wa / 2.0
    a_y1 = ya - la / 2.0
    a_y2 = ya + la / 2.0

    b_x1 = xb - wb / 2.0
    b_x2 = xb + wb / 2.0
    b_y1 = yb - lb / 2.0
    b_y2 = yb + lb / 2.0

    # Pairwise intersection
    # (M, 1) vs (1, N) -> (M, N) via broadcasting
    inter_x1 = torch.max(a_x1.unsqueeze(1), b_x1.unsqueeze(0))
    inter_x2 = torch.min(a_x2.unsqueeze(1), b_x2.unsqueeze(0))
    inter_y1 = torch.max(a_y1.unsqueeze(1), b_y1.unsqueeze(0))
    inter_y2 = torch.min(a_y2.unsqueeze(1), b_y2.unsqueeze(0))

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    # Areas
    area_a = (wa * la).unsqueeze(1)  # (M, 1)
    area_b = (wb * lb).unsqueeze(0)  # (1, N)

    union_area = area_a + area_b - inter_area

    iou = inter_area / union_area.clamp(min=1e-10)
    return iou
