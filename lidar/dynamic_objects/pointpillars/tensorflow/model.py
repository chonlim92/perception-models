"""PointPillars: Fast Encoders for Object Detection from Point Clouds (TF2/Keras).

Complete implementation of the PointPillars architecture for 3D object detection
from LiDAR point clouds. Based on:
    Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds",
    CVPR 2019.

This module provides:
    - PillarFeatureNet: Learns per-pillar features via PointNet.
    - PointPillarsScatter: Scatters pillar features to a BEV pseudo-image.
    - Backbone2D: Multi-scale 2D convolutional feature extractor.
    - Neck: Feature Pyramid Network-style upsampling and concatenation.
    - AnchorHead: Dense prediction head for class, bbox, and direction.
    - PointPillarsModel: End-to-end model combining all components.
    - decode_predictions: Decodes network outputs to 3D bounding boxes.
    - apply_nms: Non-maximum suppression post-processing.

Default parameters match the KITTI configuration from the original paper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# Constants (KITTI configuration from the PointPillars paper)
# ---------------------------------------------------------------------------

MAX_POINTS_PER_PILLAR: int = 100
MAX_NUM_PILLARS: int = 12000
PILLAR_X_SIZE: float = 0.16  # meters
PILLAR_Y_SIZE: float = 0.16  # meters
PILLAR_Z_SIZE: float = 4.0   # meters
X_MIN: float = 0.0
X_MAX: float = 69.12
Y_MIN: float = -39.68
Y_MAX: float = 39.68
Z_MIN: float = -3.0
Z_MAX: float = 1.0
GRID_X_SIZE: int = 432  # (X_MAX - X_MIN) / PILLAR_X_SIZE
GRID_Y_SIZE: int = 496  # (Y_MAX - Y_MIN) / PILLAR_Y_SIZE
NUM_FEATURES: int = 9   # x, y, z, intensity, xc, yc, zc, xp, yp
PILLAR_FEAT_DIM: int = 64
NUM_CLASSES: int = 3     # Car, Pedestrian, Cyclist
NUM_ANCHORS_PER_CELL: int = 6  # 3 classes x 2 orientations (0 and 90 deg)
BOX_CODE_SIZE: int = 7   # x, y, z, w, l, h, theta
NUM_DIR_BINS: int = 2    # direction classification bins


# ---------------------------------------------------------------------------
# PillarFeatureNet
# ---------------------------------------------------------------------------

class PillarFeatureNet(keras.Model):
    """Learns a representation for each pillar using a simplified PointNet.

    The network augments each point with offsets from the arithmetic mean of
    all points in the pillar and offsets from the pillar center. A shared MLP
    (Dense + BatchNorm + ReLU) is applied, followed by channel-wise max pooling
    to produce a fixed-size feature per pillar.

    Args:
        num_input_features: Number of raw point features (default 4: x, y, z, intensity).
        num_filters: Output feature dimension per pillar.
        max_points_per_pillar: Maximum number of points sampled per pillar.
        pillar_x_size: Pillar extent in x (meters).
        pillar_y_size: Pillar extent in y (meters).
        pillar_z_size: Pillar extent in z (meters).
        x_offset: X coordinate of the lower-left corner of the grid.
        y_offset: Y coordinate of the lower-left corner of the grid.
        z_offset: Z coordinate of the lower-left corner of the grid.
    """

    def __init__(
        self,
        num_input_features: int = 4,
        num_filters: int = PILLAR_FEAT_DIM,
        max_points_per_pillar: int = MAX_POINTS_PER_PILLAR,
        pillar_x_size: float = PILLAR_X_SIZE,
        pillar_y_size: float = PILLAR_Y_SIZE,
        pillar_z_size: float = PILLAR_Z_SIZE,
        x_offset: float = X_MIN,
        y_offset: float = Y_MIN,
        z_offset: float = Z_MIN,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_input_features = num_input_features
        self.num_filters = num_filters
        self.max_points_per_pillar = max_points_per_pillar
        self.pillar_x_size = pillar_x_size
        self.pillar_y_size = pillar_y_size
        self.pillar_z_size = pillar_z_size
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset

        # Augmented feature count: raw features + 3 (offset from mean) + 2 (offset from pillar center)
        augmented_features = num_input_features + 5

        # Shared MLP: Linear -> BatchNorm -> ReLU
        self.linear = keras.layers.Dense(
            num_filters, use_bias=False, name="pillar_linear"
        )
        self.batch_norm = keras.layers.BatchNormalization(
            epsilon=1e-3, momentum=0.99, name="pillar_bn"
        )
        self.relu = keras.layers.ReLU(name="pillar_relu")

    def call(
        self,
        pillars: tf.Tensor,
        pillar_indices: tf.Tensor,
        num_points_per_pillar: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass through PillarFeatureNet.

        Args:
            pillars: Point data within each pillar.
                Shape: (batch_size, max_num_pillars, max_points_per_pillar, num_input_features).
            pillar_indices: Grid indices (x, y) for each pillar.
                Shape: (batch_size, max_num_pillars, 2).
            num_points_per_pillar: Number of valid points in each pillar.
                Shape: (batch_size, max_num_pillars).
            training: Whether in training mode.

        Returns:
            Pillar features of shape (batch_size, max_num_pillars, num_filters).
        """
        batch_size = tf.shape(pillars)[0]
        max_pillars = tf.shape(pillars)[1]
        max_points = tf.shape(pillars)[2]

        # Create a mask for valid points: (batch, pillars, points, 1)
        point_range = tf.range(max_points, dtype=tf.int32)  # (max_points,)
        point_range = tf.reshape(point_range, [1, 1, -1])  # (1, 1, max_points)
        num_pts_expanded = tf.expand_dims(num_points_per_pillar, axis=-1)  # (batch, pillars, 1)
        mask = tf.cast(point_range < num_pts_expanded, dtype=tf.float32)  # (batch, pillars, max_points)
        mask = tf.expand_dims(mask, axis=-1)  # (batch, pillars, max_points, 1)

        # Zero out padded points
        pillars_masked = pillars * mask

        # Compute arithmetic mean of points in each pillar
        num_pts_safe = tf.maximum(
            tf.cast(num_points_per_pillar, dtype=tf.float32), 1.0
        )  # avoid division by zero
        num_pts_for_mean = tf.reshape(num_pts_safe, [batch_size, max_pillars, 1, 1])
        points_sum = tf.reduce_sum(pillars_masked, axis=2, keepdims=True)  # (B, P, 1, F)
        points_mean = points_sum / num_pts_for_mean  # (B, P, 1, F)
        points_mean_broadcast = tf.broadcast_to(
            points_mean, tf.shape(pillars_masked)
        )  # (B, P, N, F)

        # Offset from pillar arithmetic mean (x_c, y_c, z_c)
        offset_from_mean = pillars_masked - points_mean_broadcast  # (B, P, N, F)
        # We only need offsets for x, y, z (first 3 features)
        offset_xyz = offset_from_mean[:, :, :, :3]  # (B, P, N, 3)

        # Compute pillar center coordinates from indices
        # pillar_indices: (B, P, 2) with (x_idx, y_idx)
        pillar_x_centers = (
            tf.cast(pillar_indices[:, :, 0], tf.float32) * self.pillar_x_size
            + self.x_offset
            + self.pillar_x_size / 2.0
        )  # (B, P)
        pillar_y_centers = (
            tf.cast(pillar_indices[:, :, 1], tf.float32) * self.pillar_y_size
            + self.y_offset
            + self.pillar_y_size / 2.0
        )  # (B, P)

        # Offset from pillar center (x_p, y_p): point_xy - pillar_center_xy
        # pillars_masked[:, :, :, 0] is x, [:, :, :, 1] is y
        pillar_x_centers_exp = tf.reshape(pillar_x_centers, [batch_size, max_pillars, 1])
        pillar_y_centers_exp = tf.reshape(pillar_y_centers, [batch_size, max_pillars, 1])

        offset_x_from_center = pillars_masked[:, :, :, 0] - pillar_x_centers_exp  # (B, P, N)
        offset_y_from_center = pillars_masked[:, :, :, 1] - pillar_y_centers_exp  # (B, P, N)

        offset_from_center = tf.stack(
            [offset_x_from_center, offset_y_from_center], axis=-1
        )  # (B, P, N, 2)

        # Concatenate: [raw_features, offset_from_mean_xyz, offset_from_center_xy]
        augmented = tf.concat(
            [pillars_masked, offset_xyz, offset_from_center], axis=-1
        )  # (B, P, N, num_input_features + 5)

        # Apply mask again to zero out padded positions after augmentation
        augmented = augmented * mask

        # Shared MLP: Dense -> BN -> ReLU
        # Reshape for Dense layer: merge batch, pillars, points
        shape = tf.shape(augmented)
        augmented_flat = tf.reshape(augmented, [-1, shape[-1]])
        features_flat = self.linear(augmented_flat)
        features = tf.reshape(features_flat, [batch_size, max_pillars, max_points, self.num_filters])
        # BatchNorm over the feature dimension
        features = self.batch_norm(features, training=training)
        features = self.relu(features)

        # Apply mask before max pooling
        features = features * mask  # zero out padded points

        # Max pooling across points dimension
        # Replace zeros with -inf for proper max pooling, then take max
        neg_inf_mask = (1.0 - mask) * (-1e9)
        features_for_pool = features + neg_inf_mask
        pillar_features = tf.reduce_max(features_for_pool, axis=2)  # (B, P, num_filters)

        # For completely empty pillars (all padded), set to zero
        has_points = tf.cast(
            tf.greater(num_points_per_pillar, 0), tf.float32
        )  # (B, P)
        pillar_features = pillar_features * tf.expand_dims(has_points, axis=-1)

        return pillar_features


# ---------------------------------------------------------------------------
# PointPillarsScatter
# ---------------------------------------------------------------------------

class PointPillarsScatter(keras.Model):
    """Scatters pillar features back to the BEV pseudo-image canvas.

    Converts the sparse pillar representation to a dense 2D pseudo-image
    by placing each pillar's feature vector at its corresponding grid location.

    Args:
        grid_x_size: Number of grid cells in x direction.
        grid_y_size: Number of grid cells in y direction.
        num_features: Feature dimension per pillar.
    """

    def __init__(
        self,
        grid_x_size: int = GRID_X_SIZE,
        grid_y_size: int = GRID_Y_SIZE,
        num_features: int = PILLAR_FEAT_DIM,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.grid_x_size = grid_x_size
        self.grid_y_size = grid_y_size
        self.num_features = num_features

    def call(
        self,
        pillar_features: tf.Tensor,
        pillar_indices: tf.Tensor,
    ) -> tf.Tensor:
        """Scatter pillar features to a dense BEV pseudo-image.

        Args:
            pillar_features: Features per pillar, shape (batch_size, max_pillars, C).
            pillar_indices: Grid positions (x_idx, y_idx), shape (batch_size, max_pillars, 2).

        Returns:
            BEV pseudo-image of shape (batch_size, grid_y_size, grid_x_size, C).
        """
        batch_size = tf.shape(pillar_features)[0]
        max_pillars = tf.shape(pillar_features)[1]
        num_channels = tf.shape(pillar_features)[2]

        # Initialize the canvas with zeros
        canvas = tf.zeros(
            [batch_size, self.grid_y_size, self.grid_x_size, num_channels],
            dtype=pillar_features.dtype,
        )

        # Build scatter indices: (batch_idx, y_idx, x_idx)
        batch_indices = tf.range(batch_size, dtype=tf.int32)  # (B,)
        batch_indices = tf.reshape(batch_indices, [batch_size, 1])  # (B, 1)
        batch_indices = tf.broadcast_to(
            batch_indices, [batch_size, max_pillars]
        )  # (B, P)
        batch_indices = tf.reshape(batch_indices, [-1, 1])  # (B*P, 1)

        x_indices = tf.reshape(pillar_indices[:, :, 0], [-1, 1])  # (B*P, 1)
        y_indices = tf.reshape(pillar_indices[:, :, 1], [-1, 1])  # (B*P, 1)

        # Clip indices to valid range to prevent out-of-bounds
        x_indices = tf.clip_by_value(x_indices, 0, self.grid_x_size - 1)
        y_indices = tf.clip_by_value(y_indices, 0, self.grid_y_size - 1)

        scatter_indices = tf.concat(
            [batch_indices, y_indices, x_indices], axis=-1
        )  # (B*P, 3)

        flat_features = tf.reshape(
            pillar_features, [-1, num_channels]
        )  # (B*P, C)

        canvas = tf.tensor_scatter_nd_update(
            canvas, scatter_indices, flat_features
        )

        return canvas


# ---------------------------------------------------------------------------
# Backbone2D
# ---------------------------------------------------------------------------

class _ConvBlock(keras.layers.Layer):
    """A single convolutional block: Conv2D -> BatchNorm -> ReLU.

    Args:
        filters: Number of output filters.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        stride: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=False,
        )
        self.bn = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.99)
        self.relu = keras.layers.ReLU()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class Backbone2D(keras.Model):
    """Multi-scale 2D convolutional backbone.

    Consists of multiple blocks, each starting with a stride-2 downsampling
    convolution followed by several stride-1 convolutions. Produces multi-scale
    feature maps for the neck.

    Default configuration matches the PointPillars paper:
        Block 1: S=2, 4 layers, 64 filters  -> (H/2, W/2, 64)
        Block 2: S=2, 6 layers, 128 filters -> (H/4, W/4, 128)
        Block 3: S=2, 6 layers, 256 filters -> (H/8, W/8, 256)

    Args:
        layer_nums: Number of convolution layers in each block.
        layer_strides: Stride of the first convolution in each block.
        num_filters: Number of filters for each block.
    """

    def __init__(
        self,
        layer_nums: List[int] = None,
        layer_strides: List[int] = None,
        num_filters: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if layer_nums is None:
            layer_nums = [4, 6, 6]
        if layer_strides is None:
            layer_strides = [2, 2, 2]
        if num_filters is None:
            num_filters = [64, 128, 256]

        assert len(layer_nums) == len(layer_strides) == len(num_filters)

        self.blocks: List[List[_ConvBlock]] = []
        for block_idx, (num_layers, stride, filters) in enumerate(
            zip(layer_nums, layer_strides, num_filters)
        ):
            block_layers: List[_ConvBlock] = []
            # First layer has the specified stride (downsampling)
            block_layers.append(
                _ConvBlock(
                    filters, kernel_size=3, stride=stride,
                    name=f"block{block_idx}_conv0",
                )
            )
            # Remaining layers have stride 1
            for layer_idx in range(1, num_layers):
                block_layers.append(
                    _ConvBlock(
                        filters, kernel_size=3, stride=1,
                        name=f"block{block_idx}_conv{layer_idx}",
                    )
                )
            self.blocks.append(block_layers)

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> List[tf.Tensor]:
        """Forward pass producing multi-scale features.

        Args:
            x: Input BEV pseudo-image, shape (B, H, W, C).
            training: Whether in training mode.

        Returns:
            List of feature maps at each scale.
        """
        outputs: List[tf.Tensor] = []
        for block in self.blocks:
            for layer in block:
                x = layer(x, training=training)
            outputs.append(x)
        return outputs


# ---------------------------------------------------------------------------
# Neck (FPN-style upsampling and concatenation)
# ---------------------------------------------------------------------------

class Neck(keras.Model):
    """Feature Pyramid Network-style neck for multi-scale feature fusion.

    Upsamples feature maps from each backbone stage to the same spatial
    resolution and concatenates them channel-wise.

    Default configuration:
        Stage 1: upsample x1 (stride 1), 128 filters
        Stage 2: upsample x2 (stride 2), 128 filters
        Stage 3: upsample x4 (stride 4), 128 filters
        Output: concatenation -> 384 channels

    Args:
        upsample_strides: Upsample factor for each backbone stage.
        num_upsample_filters: Number of filters after upsampling for each stage.
    """

    def __init__(
        self,
        upsample_strides: List[int] = None,
        num_upsample_filters: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if upsample_strides is None:
            upsample_strides = [1, 2, 4]
        if num_upsample_filters is None:
            num_upsample_filters = [128, 128, 128]

        assert len(upsample_strides) == len(num_upsample_filters)

        self.deconv_layers: List[keras.layers.Conv2DTranspose] = []
        self.bn_layers: List[keras.layers.BatchNormalization] = []
        self.relu_layers: List[keras.layers.ReLU] = []

        for idx, (stride, filters) in enumerate(
            zip(upsample_strides, num_upsample_filters)
        ):
            self.deconv_layers.append(
                keras.layers.Conv2DTranspose(
                    filters,
                    kernel_size=stride,
                    strides=stride,
                    padding="same",
                    use_bias=False,
                    name=f"upsample_{idx}",
                )
            )
            self.bn_layers.append(
                keras.layers.BatchNormalization(
                    epsilon=1e-3, momentum=0.99, name=f"upsample_bn_{idx}"
                )
            )
            self.relu_layers.append(keras.layers.ReLU(name=f"upsample_relu_{idx}"))

    def call(
        self, multi_scale_features: List[tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        """Upsample and concatenate multi-scale features.

        Args:
            multi_scale_features: List of feature maps from the backbone.
            training: Whether in training mode.

        Returns:
            Concatenated feature map, shape (B, H', W', sum(num_upsample_filters)).
        """
        upsampled: List[tf.Tensor] = []
        for feat, deconv, bn, relu in zip(
            multi_scale_features,
            self.deconv_layers,
            self.bn_layers,
            self.relu_layers,
        ):
            x = deconv(feat)
            x = bn(x, training=training)
            x = relu(x)
            upsampled.append(x)

        return tf.concat(upsampled, axis=-1)


# ---------------------------------------------------------------------------
# AnchorHead
# ---------------------------------------------------------------------------

class AnchorHead(keras.Model):
    """Dense prediction head for anchor-based 3D object detection.

    Produces per-anchor predictions for:
        - Classification (num_classes scores per anchor)
        - Bounding box regression (7 values: x, y, z, w, l, h, theta)
        - Direction classification (2 bins per anchor)

    Args:
        num_classes: Number of object classes.
        num_anchors_per_cell: Number of anchors per spatial cell.
        box_code_size: Number of regression targets per box.
        num_dir_bins: Number of direction classification bins.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_anchors_per_cell: int = NUM_ANCHORS_PER_CELL,
        box_code_size: int = BOX_CODE_SIZE,
        num_dir_bins: int = NUM_DIR_BINS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors_per_cell = num_anchors_per_cell
        self.box_code_size = box_code_size
        self.num_dir_bins = num_dir_bins

        # Classification head
        self.cls_conv = keras.layers.Conv2D(
            num_anchors_per_cell * num_classes,
            kernel_size=1,
            strides=1,
            padding="same",
            bias_initializer=keras.initializers.Constant(-np.log((1 - 0.01) / 0.01)),
            name="cls_head",
        )

        # Bounding box regression head
        self.box_conv = keras.layers.Conv2D(
            num_anchors_per_cell * box_code_size,
            kernel_size=1,
            strides=1,
            padding="same",
            name="box_head",
        )

        # Direction classification head
        self.dir_conv = keras.layers.Conv2D(
            num_anchors_per_cell * num_dir_bins,
            kernel_size=1,
            strides=1,
            padding="same",
            name="dir_head",
        )

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """Produce dense predictions over the BEV feature map.

        Args:
            x: Input feature map from neck, shape (B, H, W, C).
            training: Whether in training mode.

        Returns:
            Dictionary with keys:
                'cls_preds': (B, H, W, num_anchors * num_classes)
                'box_preds': (B, H, W, num_anchors * box_code_size)
                'dir_preds': (B, H, W, num_anchors * num_dir_bins)
        """
        cls_preds = self.cls_conv(x)
        box_preds = self.box_conv(x)
        dir_preds = self.dir_conv(x)

        batch_size = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]

        # Reshape to (B, H*W*num_anchors, num_classes/box_code_size/num_dir_bins)
        cls_preds = tf.reshape(
            cls_preds, [batch_size, h * w * self.num_anchors_per_cell, self.num_classes]
        )
        box_preds = tf.reshape(
            box_preds, [batch_size, h * w * self.num_anchors_per_cell, self.box_code_size]
        )
        dir_preds = tf.reshape(
            dir_preds, [batch_size, h * w * self.num_anchors_per_cell, self.num_dir_bins]
        )

        return {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds,
        }


# ---------------------------------------------------------------------------
# Full PointPillars Model
# ---------------------------------------------------------------------------

class PointPillarsModel(keras.Model):
    """End-to-end PointPillars model for 3D object detection from LiDAR.

    Combines PillarFeatureNet, PointPillarsScatter, Backbone2D, Neck, and
    AnchorHead into a single trainable model.

    Args:
        num_classes: Number of object classes.
        num_input_features: Number of raw point features (x, y, z, intensity).
        pillar_feat_dim: Pillar feature dimension.
        max_points_per_pillar: Max points per pillar for PillarFeatureNet.
        grid_x_size: Grid size along x.
        grid_y_size: Grid size along y.
        backbone_layer_nums: Conv layers per backbone block.
        backbone_layer_strides: Strides for backbone blocks.
        backbone_num_filters: Filters per backbone block.
        neck_upsample_strides: Upsample strides for neck.
        neck_num_filters: Filters per neck upsample stage.
        num_anchors_per_cell: Number of anchors per spatial cell.
        box_code_size: Box regression target size.
        num_dir_bins: Direction classification bins.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_input_features: int = 4,
        pillar_feat_dim: int = PILLAR_FEAT_DIM,
        max_points_per_pillar: int = MAX_POINTS_PER_PILLAR,
        grid_x_size: int = GRID_X_SIZE,
        grid_y_size: int = GRID_Y_SIZE,
        backbone_layer_nums: Optional[List[int]] = None,
        backbone_layer_strides: Optional[List[int]] = None,
        backbone_num_filters: Optional[List[int]] = None,
        neck_upsample_strides: Optional[List[int]] = None,
        neck_num_filters: Optional[List[int]] = None,
        num_anchors_per_cell: int = NUM_ANCHORS_PER_CELL,
        box_code_size: int = BOX_CODE_SIZE,
        num_dir_bins: int = NUM_DIR_BINS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.pillar_feature_net = PillarFeatureNet(
            num_input_features=num_input_features,
            num_filters=pillar_feat_dim,
            max_points_per_pillar=max_points_per_pillar,
            name="pillar_feature_net",
        )

        self.scatter = PointPillarsScatter(
            grid_x_size=grid_x_size,
            grid_y_size=grid_y_size,
            num_features=pillar_feat_dim,
            name="scatter",
        )

        self.backbone = Backbone2D(
            layer_nums=backbone_layer_nums,
            layer_strides=backbone_layer_strides,
            num_filters=backbone_num_filters,
            name="backbone",
        )

        self.neck = Neck(
            upsample_strides=neck_upsample_strides,
            num_upsample_filters=neck_num_filters,
            name="neck",
        )

        self.head = AnchorHead(
            num_classes=num_classes,
            num_anchors_per_cell=num_anchors_per_cell,
            box_code_size=box_code_size,
            num_dir_bins=num_dir_bins,
            name="head",
        )

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Forward pass of the full PointPillars model.

        Args:
            inputs: Dictionary with keys:
                'pillars': (B, max_pillars, max_points, num_features)
                'pillar_indices': (B, max_pillars, 2)  [x_idx, y_idx]
                'num_points_per_pillar': (B, max_pillars)
            training: Whether in training mode.

        Returns:
            Dictionary with keys:
                'cls_preds': (B, total_anchors, num_classes)
                'box_preds': (B, total_anchors, box_code_size)
                'dir_preds': (B, total_anchors, num_dir_bins)
        """
        pillars = inputs["pillars"]
        pillar_indices = inputs["pillar_indices"]
        num_points = inputs["num_points_per_pillar"]

        # Extract pillar features
        pillar_features = self.pillar_feature_net(
            pillars, pillar_indices, num_points, training=training
        )

        # Scatter to BEV pseudo-image
        bev_image = self.scatter(pillar_features, pillar_indices)

        # Backbone feature extraction
        multi_scale_features = self.backbone(bev_image, training=training)

        # Neck: upsample and concatenate
        fused_features = self.neck(multi_scale_features, training=training)

        # Detection head
        predictions = self.head(fused_features, training=training)

        return predictions


# ---------------------------------------------------------------------------
# Anchor Generation
# ---------------------------------------------------------------------------

def generate_anchors(
    grid_x_size: int = GRID_X_SIZE,
    grid_y_size: int = GRID_Y_SIZE,
    x_min: float = X_MIN,
    y_min: float = Y_MIN,
    z_min: float = Z_MIN,
    pillar_x_size: float = PILLAR_X_SIZE,
    pillar_y_size: float = PILLAR_Y_SIZE,
    anchor_sizes: Optional[List[List[float]]] = None,
    anchor_rotations: Optional[List[float]] = None,
    anchor_z_centers: Optional[List[float]] = None,
    feature_map_stride: int = 2,
) -> tf.Tensor:
    """Generate anchor boxes for the detection head.

    Anchors are generated at the resolution of the feature map output
    from the first backbone block (stride 2 relative to BEV pseudo-image).

    Args:
        grid_x_size: BEV grid size in x.
        grid_y_size: BEV grid size in y.
        x_min: Minimum x coordinate of the point cloud range.
        y_min: Minimum y coordinate of the point cloud range.
        z_min: Minimum z coordinate of the point cloud range.
        pillar_x_size: Pillar size in x (meters).
        pillar_y_size: Pillar size in y (meters).
        anchor_sizes: List of [width, length, height] per class.
            Default: Car=[1.6, 3.9, 1.56], Ped=[0.6, 0.8, 1.73], Cyc=[0.6, 1.76, 1.73]
        anchor_rotations: List of rotation angles in radians.
            Default: [0, pi/2]
        anchor_z_centers: Z center for each class anchor.
            Default: [-1.0, -0.6, -0.6] (Car, Ped, Cyclist)
        feature_map_stride: Stride of the feature map relative to BEV image.

    Returns:
        Anchors tensor of shape (num_y, num_x, num_anchors_per_cell, 7)
        where 7 = (x, y, z, w, l, h, rotation).
    """
    if anchor_sizes is None:
        anchor_sizes = [
            [1.6, 3.9, 1.56],   # Car
            [0.6, 0.8, 1.73],   # Pedestrian
            [0.6, 1.76, 1.73],  # Cyclist
        ]
    if anchor_rotations is None:
        anchor_rotations = [0.0, np.pi / 2.0]
    if anchor_z_centers is None:
        anchor_z_centers = [-1.0, -0.6, -0.6]

    # Feature map spatial dimensions
    fm_x = grid_x_size // feature_map_stride
    fm_y = grid_y_size // feature_map_stride

    # Anchor center positions in world coordinates
    x_stride = pillar_x_size * feature_map_stride
    y_stride = pillar_y_size * feature_map_stride

    x_centers = np.arange(fm_x, dtype=np.float32) * x_stride + x_min + x_stride / 2.0
    y_centers = np.arange(fm_y, dtype=np.float32) * y_stride + y_min + y_stride / 2.0

    # Create meshgrid: (fm_y, fm_x)
    xx, yy = np.meshgrid(x_centers, y_centers)  # both (fm_y, fm_x)

    num_anchor_types = len(anchor_sizes) * len(anchor_rotations)
    anchors = np.zeros((fm_y, fm_x, num_anchor_types, 7), dtype=np.float32)

    anchor_idx = 0
    for cls_idx, (size, z_center) in enumerate(zip(anchor_sizes, anchor_z_centers)):
        w, l, h = size
        for rot in anchor_rotations:
            anchors[:, :, anchor_idx, 0] = xx
            anchors[:, :, anchor_idx, 1] = yy
            anchors[:, :, anchor_idx, 2] = z_center
            anchors[:, :, anchor_idx, 3] = w
            anchors[:, :, anchor_idx, 4] = l
            anchors[:, :, anchor_idx, 5] = h
            anchors[:, :, anchor_idx, 6] = rot
            anchor_idx += 1

    return tf.constant(anchors, dtype=tf.float32)


# ---------------------------------------------------------------------------
# Post-processing: Decode Predictions
# ---------------------------------------------------------------------------

def decode_predictions(
    box_preds: tf.Tensor,
    anchors: tf.Tensor,
) -> tf.Tensor:
    """Decode bounding box predictions relative to anchors.

    Uses the encoding scheme from the PointPillars paper:
        x_t = (x_gt - x_a) / d_a
        y_t = (y_gt - y_a) / d_a
        z_t = (z_gt - z_a) / h_a
        w_t = log(w_gt / w_a)
        l_t = log(l_gt / l_a)
        h_t = log(h_gt / h_a)
        theta_t = sin(theta_gt - theta_a)

    This function inverts the encoding to recover world-frame boxes.

    Args:
        box_preds: Predicted box residuals, shape (B, N, 7) or (N, 7).
            Residuals: [dx, dy, dz, dw, dl, dh, dtheta]
        anchors: Anchor boxes, shape (N, 7) or broadcastable.
            Values: [x, y, z, w, l, h, theta]

    Returns:
        Decoded boxes in world frame, same shape as box_preds.
            Values: [x, y, z, w, l, h, theta]
    """
    # Extract anchor components
    xa = anchors[..., 0]
    ya = anchors[..., 1]
    za = anchors[..., 2]
    wa = anchors[..., 3]
    la = anchors[..., 4]
    ha = anchors[..., 5]
    theta_a = anchors[..., 6]

    # Diagonal of the anchor base (used for x, y normalization)
    da = tf.sqrt(wa ** 2 + la ** 2)

    # Extract prediction residuals
    dx = box_preds[..., 0]
    dy = box_preds[..., 1]
    dz = box_preds[..., 2]
    dw = box_preds[..., 3]
    dl = box_preds[..., 4]
    dh = box_preds[..., 5]
    dtheta = box_preds[..., 6]

    # Decode
    x = dx * da + xa
    y = dy * da + ya
    z = dz * ha + za
    w = tf.exp(dw) * wa
    l = tf.exp(dl) * la
    h = tf.exp(dh) * ha
    theta = dtheta + theta_a  # direct regression of angle residual

    decoded = tf.stack([x, y, z, w, l, h, theta], axis=-1)
    return decoded


# ---------------------------------------------------------------------------
# Post-processing: Non-Maximum Suppression
# ---------------------------------------------------------------------------

def apply_nms(
    cls_preds: tf.Tensor,
    decoded_boxes: tf.Tensor,
    dir_preds: tf.Tensor,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    max_detections_per_class: int = 100,
    max_total_detections: int = 300,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, tf.Tensor]:
    """Apply class-aware NMS to decoded predictions.

    Uses tf.image.combined_non_max_suppression for efficient multi-class NMS.
    Since 3D IoU is expensive, we project boxes to BEV (bird's eye view) for
    NMS, using the (x, y, w, l) components as a 2D bounding box approximation.

    Args:
        cls_preds: Classification logits, shape (B, N, num_classes).
        decoded_boxes: Decoded 3D boxes, shape (B, N, 7).
        dir_preds: Direction classification logits, shape (B, N, 2).
        score_threshold: Minimum score to keep a detection.
        nms_iou_threshold: IoU threshold for NMS.
        max_detections_per_class: Maximum detections per class after NMS.
        max_total_detections: Maximum total detections after NMS.
        num_classes: Number of classes.

    Returns:
        Dictionary with:
            'boxes': (B, max_total_detections, 7) - 3D boxes
            'scores': (B, max_total_detections) - detection scores
            'classes': (B, max_total_detections) - class indices (0-indexed)
            'num_detections': (B,) - number of valid detections per sample
            'dir_labels': (B, max_total_detections) - direction labels
    """
    batch_size = tf.shape(cls_preds)[0]
    num_boxes = tf.shape(cls_preds)[1]

    # Convert logits to scores
    cls_scores = tf.sigmoid(cls_preds)  # (B, N, num_classes)

    # Project 3D boxes to BEV 2D boxes for NMS
    # BEV box: center (x, y) with dimensions (w, l)
    # Convert to [y_min, x_min, y_max, x_max] format for tf NMS
    x_center = decoded_boxes[:, :, 0]
    y_center = decoded_boxes[:, :, 1]
    w = decoded_boxes[:, :, 3]
    l = decoded_boxes[:, :, 4]

    # Approximate axis-aligned BEV bounding box (ignoring rotation for NMS efficiency)
    half_diag = tf.sqrt(w ** 2 + l ** 2) / 2.0
    y_min = y_center - half_diag
    x_min = x_center - half_diag
    y_max = y_center + half_diag
    x_max = x_center + half_diag

    # Normalize to [0, 1] range for tf.image.combined_non_max_suppression
    # Use the point cloud range for normalization
    x_range = X_MAX - X_MIN
    y_range = Y_MAX - Y_MIN

    y_min_norm = (y_min - Y_MIN) / y_range
    x_min_norm = (x_min - X_MIN) / x_range
    y_max_norm = (y_max - Y_MIN) / y_range
    x_max_norm = (x_max - X_MIN) / x_range

    # Shape needed for combined_non_max_suppression:
    # boxes: (B, N, num_classes, 4) -- class-specific or shared
    # scores: (B, N, num_classes)
    bev_boxes_2d = tf.stack(
        [y_min_norm, x_min_norm, y_max_norm, x_max_norm], axis=-1
    )  # (B, N, 4)

    # Expand boxes for each class (shared boxes across classes)
    bev_boxes_multi_class = tf.expand_dims(bev_boxes_2d, axis=2)  # (B, N, 1, 4)
    bev_boxes_multi_class = tf.broadcast_to(
        bev_boxes_multi_class, [batch_size, num_boxes, num_classes, 4]
    )

    # Apply combined NMS
    nmsed_boxes, nmsed_scores, nmsed_classes, num_detections = (
        tf.image.combined_non_max_suppression(
            bev_boxes_multi_class,
            cls_scores,
            max_output_size_per_class=max_detections_per_class,
            max_total_size=max_total_detections,
            iou_threshold=nms_iou_threshold,
            score_threshold=score_threshold,
            pad_per_class=False,
            clip_boxes=False,
        )
    )

    # nmsed_boxes are 2D BEV boxes from NMS; we need to retrieve the original 3D boxes.
    # Use the nmsed indices to gather original 3D boxes.
    # Since combined_non_max_suppression doesn't return indices, we match by score.
    # A more robust approach: perform per-class NMS manually.

    # Manual per-class NMS for proper 3D box retrieval
    all_boxes_list: List[tf.Tensor] = []
    all_scores_list: List[tf.Tensor] = []
    all_classes_list: List[tf.Tensor] = []
    all_dirs_list: List[tf.Tensor] = []
    all_num_dets: List[tf.Tensor] = []

    dir_labels = tf.argmax(dir_preds, axis=-1)  # (B, N)

    for b in tf.range(batch_size):
        sample_boxes = decoded_boxes[b]  # (N, 7)
        sample_scores = cls_scores[b]  # (N, num_classes)
        sample_bev = bev_boxes_2d[b]  # (N, 4)
        sample_dirs = dir_labels[b]  # (N,)

        det_boxes = tf.zeros([0, 7], dtype=tf.float32)
        det_scores = tf.zeros([0], dtype=tf.float32)
        det_classes = tf.zeros([0], dtype=tf.int32)
        det_dirs = tf.zeros([0], dtype=tf.int64)

        for c in tf.range(num_classes):
            class_scores = sample_scores[:, c]  # (N,)
            score_mask = class_scores > score_threshold
            filtered_indices = tf.where(score_mask)[:, 0]  # (K,)

            if tf.shape(filtered_indices)[0] == 0:
                continue

            filtered_scores = tf.gather(class_scores, filtered_indices)
            filtered_bev = tf.gather(sample_bev, filtered_indices)
            filtered_3d = tf.gather(sample_boxes, filtered_indices)
            filtered_dirs = tf.gather(sample_dirs, filtered_indices)

            # Apply NMS on BEV boxes
            nms_indices = tf.image.non_max_suppression(
                filtered_bev,
                filtered_scores,
                max_output_size=max_detections_per_class,
                iou_threshold=nms_iou_threshold,
                score_threshold=score_threshold,
            )

            nms_boxes = tf.gather(filtered_3d, nms_indices)
            nms_scores = tf.gather(filtered_scores, nms_indices)
            nms_dirs = tf.gather(filtered_dirs, nms_indices)
            nms_classes = tf.fill([tf.shape(nms_indices)[0]], c)

            det_boxes = tf.concat([det_boxes, nms_boxes], axis=0)
            det_scores = tf.concat([det_scores, nms_scores], axis=0)
            det_classes = tf.concat([det_classes, nms_classes], axis=0)
            det_dirs = tf.concat([det_dirs, nms_dirs], axis=0)

        # Sort by score and keep top max_total_detections
        num_dets = tf.minimum(tf.shape(det_scores)[0], max_total_detections)
        top_k_values, top_k_indices = tf.math.top_k(det_scores, k=num_dets)
        det_boxes = tf.gather(det_boxes, top_k_indices)
        det_scores = top_k_values
        det_classes = tf.gather(det_classes, top_k_indices)
        det_dirs = tf.gather(det_dirs, top_k_indices)

        # Pad to max_total_detections
        pad_size = max_total_detections - num_dets
        det_boxes = tf.pad(det_boxes, [[0, pad_size], [0, 0]])
        det_scores = tf.pad(det_scores, [[0, pad_size]])
        det_classes = tf.pad(det_classes, [[0, pad_size]])
        det_dirs = tf.pad(det_dirs, [[0, pad_size]])

        all_boxes_list.append(det_boxes)
        all_scores_list.append(det_scores)
        all_classes_list.append(det_classes)
        all_dirs_list.append(det_dirs)
        all_num_dets.append(num_dets)

    result_boxes = tf.stack(all_boxes_list, axis=0)
    result_scores = tf.stack(all_scores_list, axis=0)
    result_classes = tf.stack(all_classes_list, axis=0)
    result_dirs = tf.stack(all_dirs_list, axis=0)
    result_num_dets = tf.stack(all_num_dets, axis=0)

    return {
        "boxes": result_boxes,
        "scores": result_scores,
        "classes": result_classes,
        "num_detections": result_num_dets,
        "dir_labels": result_dirs,
    }


# ---------------------------------------------------------------------------
# Convenience: Inference Function
# ---------------------------------------------------------------------------

def pointpillars_inference(
    model: PointPillarsModel,
    inputs: Dict[str, tf.Tensor],
    anchors: tf.Tensor,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    max_detections_per_class: int = 100,
    max_total_detections: int = 300,
) -> Dict[str, tf.Tensor]:
    """Run full PointPillars inference pipeline.

    Performs forward pass, decodes predictions, and applies NMS.

    Args:
        model: Trained PointPillarsModel instance.
        inputs: Input dictionary for the model.
        anchors: Pre-generated anchors, shape (H, W, num_anchors_per_cell, 7).
        score_threshold: Minimum detection score.
        nms_iou_threshold: IoU threshold for NMS.
        max_detections_per_class: Max detections per class.
        max_total_detections: Max total detections.

    Returns:
        Dictionary with detection results (boxes, scores, classes, etc.).
    """
    # Forward pass
    predictions = model(inputs, training=False)

    cls_preds = predictions["cls_preds"]
    box_preds = predictions["box_preds"]
    dir_preds = predictions["dir_preds"]

    # Flatten anchors to match prediction shape
    anchors_flat = tf.reshape(anchors, [1, -1, 7])  # (1, N, 7)
    anchors_flat = tf.broadcast_to(
        anchors_flat, [tf.shape(box_preds)[0], tf.shape(box_preds)[1], 7]
    )

    # Decode boxes
    decoded_boxes = decode_predictions(box_preds, anchors_flat)

    # Apply direction classification to refine heading
    dir_labels = tf.argmax(dir_preds, axis=-1)  # (B, N)
    # Flip heading by pi if direction label is 1
    dir_offset = tf.cast(dir_labels, tf.float32) * np.pi
    heading = decoded_boxes[..., 6] + dir_offset
    # Normalize heading to [-pi, pi]
    heading = tf.math.atan2(tf.sin(heading), tf.cos(heading))
    decoded_boxes = tf.concat(
        [decoded_boxes[..., :6], tf.expand_dims(heading, axis=-1)], axis=-1
    )

    # NMS
    detections = apply_nms(
        cls_preds=cls_preds,
        decoded_boxes=decoded_boxes,
        dir_preds=dir_preds,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections_per_class=max_detections_per_class,
        max_total_detections=max_total_detections,
    )

    return detections


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class PointPillarsLoss(keras.Model):
    """Combined loss for PointPillars training.

    Includes:
        - Focal loss for classification
        - Smooth L1 loss for box regression
        - Cross-entropy loss for direction classification

    Args:
        num_classes: Number of object classes.
        alpha: Focal loss alpha parameter.
        gamma: Focal loss gamma parameter.
        box_loss_weight: Weight for box regression loss.
        dir_loss_weight: Weight for direction classification loss.
        cls_loss_weight: Weight for classification loss.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        alpha: float = 0.25,
        gamma: float = 2.0,
        box_loss_weight: float = 2.0,
        dir_loss_weight: float = 0.2,
        cls_loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.box_loss_weight = box_loss_weight
        self.dir_loss_weight = dir_loss_weight
        self.cls_loss_weight = cls_loss_weight

    def focal_loss(
        self,
        cls_preds: tf.Tensor,
        cls_targets: tf.Tensor,
        weights: tf.Tensor,
    ) -> tf.Tensor:
        """Compute focal loss for classification.

        Args:
            cls_preds: Predicted logits, shape (B, N, num_classes).
            cls_targets: One-hot targets, shape (B, N, num_classes).
            weights: Per-anchor weights, shape (B, N).

        Returns:
            Scalar focal loss.
        """
        pred_sigmoid = tf.sigmoid(cls_preds)
        # Compute focal weight
        pt = cls_targets * pred_sigmoid + (1.0 - cls_targets) * (1.0 - pred_sigmoid)
        focal_weight = tf.pow(1.0 - pt, self.gamma)

        # Alpha weighting
        alpha_weight = cls_targets * self.alpha + (1.0 - cls_targets) * (1.0 - self.alpha)

        # Binary cross-entropy
        bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=cls_targets, logits=cls_preds
        )

        loss = focal_weight * alpha_weight * bce  # (B, N, num_classes)
        loss = tf.reduce_sum(loss, axis=-1)  # (B, N)
        loss = loss * weights
        normalizer = tf.maximum(tf.reduce_sum(weights), 1.0)
        return tf.reduce_sum(loss) / normalizer

    def smooth_l1_loss(
        self,
        box_preds: tf.Tensor,
        box_targets: tf.Tensor,
        weights: tf.Tensor,
        sigma: float = 3.0,
    ) -> tf.Tensor:
        """Compute smooth L1 loss for box regression.

        Args:
            box_preds: Predicted residuals, shape (B, N, 7).
            box_targets: Target residuals, shape (B, N, 7).
            weights: Per-anchor weights, shape (B, N).
            sigma: Smooth L1 transition point.

        Returns:
            Scalar smooth L1 loss.
        """
        diff = box_preds - box_targets
        abs_diff = tf.abs(diff)
        threshold = 1.0 / (sigma ** 2)
        smooth_l1 = tf.where(
            abs_diff < threshold,
            0.5 * (sigma ** 2) * (diff ** 2),
            abs_diff - 0.5 * threshold,
        )
        loss = tf.reduce_sum(smooth_l1, axis=-1)  # (B, N)
        loss = loss * weights
        normalizer = tf.maximum(tf.reduce_sum(weights), 1.0)
        return tf.reduce_sum(loss) / normalizer

    def direction_loss(
        self,
        dir_preds: tf.Tensor,
        dir_targets: tf.Tensor,
        weights: tf.Tensor,
    ) -> tf.Tensor:
        """Compute cross-entropy loss for direction classification.

        Args:
            dir_preds: Direction logits, shape (B, N, 2).
            dir_targets: Direction labels (0 or 1), shape (B, N).
            weights: Per-anchor weights, shape (B, N).

        Returns:
            Scalar direction classification loss.
        """
        dir_targets_onehot = tf.one_hot(
            tf.cast(dir_targets, tf.int32), depth=2
        )
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=dir_targets_onehot, logits=dir_preds
        )  # (B, N)
        loss = loss * weights
        normalizer = tf.maximum(tf.reduce_sum(weights), 1.0)
        return tf.reduce_sum(loss) / normalizer

    def call(
        self,
        predictions: Dict[str, tf.Tensor],
        targets: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Compute combined PointPillars loss.

        Args:
            predictions: Model output dictionary with cls_preds, box_preds, dir_preds.
            targets: Target dictionary with:
                'cls_targets': (B, N, num_classes) one-hot class labels
                'box_targets': (B, N, 7) box regression targets
                'dir_targets': (B, N) direction bin labels
                'positive_mask': (B, N) binary mask for positive anchors
                'negative_mask': (B, N) binary mask for negative anchors
            training: Whether in training mode (unused, kept for API consistency).

        Returns:
            Dictionary with 'total_loss', 'cls_loss', 'box_loss', 'dir_loss'.
        """
        cls_preds = predictions["cls_preds"]
        box_preds = predictions["box_preds"]
        dir_preds = predictions["dir_preds"]

        cls_targets = targets["cls_targets"]
        box_targets = targets["box_targets"]
        dir_targets = targets["dir_targets"]
        positive_mask = targets["positive_mask"]
        negative_mask = targets["negative_mask"]

        # Classification uses both positive and negative anchors
        cls_weights = positive_mask + negative_mask

        # Regression and direction only use positive anchors
        reg_weights = positive_mask

        cls_loss = self.focal_loss(cls_preds, cls_targets, cls_weights)
        box_loss = self.smooth_l1_loss(box_preds, box_targets, reg_weights)
        dir_loss = self.direction_loss(dir_preds, dir_targets, reg_weights)

        total_loss = (
            self.cls_loss_weight * cls_loss
            + self.box_loss_weight * box_loss
            + self.dir_loss_weight * dir_loss
        )

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "dir_loss": dir_loss,
        }


# ---------------------------------------------------------------------------
# Voxelization (Pillarization) Preprocessing
# ---------------------------------------------------------------------------

def create_pillars_from_points(
    points: tf.Tensor,
    max_points_per_pillar: int = MAX_POINTS_PER_PILLAR,
    max_num_pillars: int = MAX_NUM_PILLARS,
    x_min: float = X_MIN,
    x_max: float = X_MAX,
    y_min: float = Y_MIN,
    y_max: float = Y_MAX,
    z_min: float = Z_MIN,
    z_max: float = Z_MAX,
    pillar_x_size: float = PILLAR_X_SIZE,
    pillar_y_size: float = PILLAR_Y_SIZE,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert raw point cloud to pillar representation using TF operations.

    This function voxelizes a point cloud into pillars (vertical columns) and
    samples points within each pillar.

    Args:
        points: Raw point cloud, shape (N, 4+) with columns [x, y, z, intensity, ...].
        max_points_per_pillar: Maximum number of points to keep per pillar.
        max_num_pillars: Maximum number of non-empty pillars to keep.
        x_min: Minimum x for the point cloud range.
        x_max: Maximum x for the point cloud range.
        y_min: Minimum y for the point cloud range.
        y_max: Maximum y for the point cloud range.
        z_min: Minimum z for the point cloud range.
        z_max: Maximum z for the point cloud range.
        pillar_x_size: Pillar size along x in meters.
        pillar_y_size: Pillar size along y in meters.

    Returns:
        Tuple of:
            pillars: (max_num_pillars, max_points_per_pillar, num_features)
            pillar_indices: (max_num_pillars, 2)  -- (x_idx, y_idx) grid coordinates
            num_points_per_pillar: (max_num_pillars,) -- count of valid points
    """
    num_features = tf.shape(points)[1]

    # Filter points within the detection range
    mask_x = tf.logical_and(points[:, 0] >= x_min, points[:, 0] < x_max)
    mask_y = tf.logical_and(points[:, 1] >= y_min, points[:, 1] < y_max)
    mask_z = tf.logical_and(points[:, 2] >= z_min, points[:, 2] < z_max)
    range_mask = tf.logical_and(tf.logical_and(mask_x, mask_y), mask_z)
    points = tf.boolean_mask(points, range_mask)

    # Compute grid indices for each point
    x_indices = tf.cast((points[:, 0] - x_min) / pillar_x_size, tf.int32)
    y_indices = tf.cast((points[:, 1] - y_min) / pillar_y_size, tf.int32)

    grid_x_size = int((x_max - x_min) / pillar_x_size)
    grid_y_size = int((y_max - y_min) / pillar_y_size)

    # Clip indices to valid range
    x_indices = tf.clip_by_value(x_indices, 0, grid_x_size - 1)
    y_indices = tf.clip_by_value(y_indices, 0, grid_y_size - 1)

    # Create a unique pillar ID for each point: pillar_id = x_idx * grid_y_size + y_idx
    pillar_ids = x_indices * grid_y_size + y_indices

    # Find unique pillar IDs
    unique_ids, point_to_pillar_idx = tf.unique(pillar_ids)
    num_unique_pillars = tf.shape(unique_ids)[0]

    # Limit number of pillars
    num_pillars_to_use = tf.minimum(num_unique_pillars, max_num_pillars)

    # If we have more pillars than the limit, randomly sample
    if_needs_sampling = tf.greater(num_unique_pillars, max_num_pillars)
    selected_pillar_local_indices = tf.cond(
        if_needs_sampling,
        lambda: tf.random.shuffle(tf.range(num_unique_pillars))[:max_num_pillars],
        lambda: tf.range(num_unique_pillars),
    )
    selected_pillar_local_indices = tf.sort(selected_pillar_local_indices)

    # Initialize output tensors
    pillars = tf.zeros(
        [max_num_pillars, max_points_per_pillar, num_features], dtype=tf.float32
    )
    pillar_indices = tf.zeros([max_num_pillars, 2], dtype=tf.int32)
    num_points_per_pillar = tf.zeros([max_num_pillars], dtype=tf.int32)

    # For each selected pillar, gather its points
    pillars_ta = tf.TensorArray(
        dtype=tf.float32, size=max_num_pillars, dynamic_size=False,
        element_shape=[max_points_per_pillar, num_features],
    )
    indices_ta = tf.TensorArray(
        dtype=tf.int32, size=max_num_pillars, dynamic_size=False,
        element_shape=[2],
    )
    counts_ta = tf.TensorArray(
        dtype=tf.int32, size=max_num_pillars, dynamic_size=False,
        element_shape=[],
    )

    # Initialize with zeros
    zero_pillar = tf.zeros([max_points_per_pillar, num_features], dtype=tf.float32)
    zero_index = tf.zeros([2], dtype=tf.int32)

    for i in tf.range(max_num_pillars):
        pillars_ta = pillars_ta.write(i, zero_pillar)
        indices_ta = indices_ta.write(i, zero_index)
        counts_ta = counts_ta.write(i, 0)

    def fill_pillar(i, pillars_ta, indices_ta, counts_ta):
        """Fill one pillar's data into the tensor arrays."""
        pillar_local_idx = selected_pillar_local_indices[i]
        pillar_id = unique_ids[pillar_local_idx]

        # Get points belonging to this pillar
        point_mask = tf.equal(point_to_pillar_idx, pillar_local_idx)
        pillar_points = tf.boolean_mask(points, point_mask)
        n_points = tf.shape(pillar_points)[0]

        # Limit or pad points
        n_to_use = tf.minimum(n_points, max_points_per_pillar)

        # If more points than limit, randomly sample
        sampled_points = tf.cond(
            tf.greater(n_points, max_points_per_pillar),
            lambda: tf.gather(
                pillar_points,
                tf.random.shuffle(tf.range(n_points))[:max_points_per_pillar],
            ),
            lambda: pillar_points,
        )

        # Pad to max_points_per_pillar
        pad_size = max_points_per_pillar - tf.shape(sampled_points)[0]
        padded_points = tf.pad(sampled_points, [[0, pad_size], [0, 0]])

        # Recover grid indices from pillar_id
        x_idx = pillar_id // grid_y_size
        y_idx = pillar_id % grid_y_size
        grid_coords = tf.stack([x_idx, y_idx])

        pillars_ta = pillars_ta.write(i, padded_points)
        indices_ta = indices_ta.write(i, grid_coords)
        counts_ta = counts_ta.write(i, n_to_use)

        return i + 1, pillars_ta, indices_ta, counts_ta

    # Fill selected pillars using a while loop
    _, pillars_ta, indices_ta, counts_ta = tf.while_loop(
        cond=lambda i, *_: i < num_pillars_to_use,
        body=fill_pillar,
        loop_vars=[tf.constant(0), pillars_ta, indices_ta, counts_ta],
        parallel_iterations=1,
    )

    pillars = pillars_ta.stack()  # (max_num_pillars, max_points, features)
    pillar_indices = indices_ta.stack()  # (max_num_pillars, 2)
    num_points_per_pillar = counts_ta.stack()  # (max_num_pillars,)

    return pillars, pillar_indices, num_points_per_pillar


# ---------------------------------------------------------------------------
# Model Builder Utility
# ---------------------------------------------------------------------------

def build_pointpillars(
    num_classes: int = NUM_CLASSES,
    num_input_features: int = 4,
    pillar_feat_dim: int = PILLAR_FEAT_DIM,
    max_points_per_pillar: int = MAX_POINTS_PER_PILLAR,
    grid_x_size: int = GRID_X_SIZE,
    grid_y_size: int = GRID_Y_SIZE,
) -> Tuple[PointPillarsModel, PointPillarsLoss]:
    """Build a PointPillars model and loss function with default KITTI config.

    Args:
        num_classes: Number of detection classes.
        num_input_features: Point feature dimension (x, y, z, intensity).
        pillar_feat_dim: Output dimension of PillarFeatureNet.
        max_points_per_pillar: Maximum points per pillar.
        grid_x_size: Grid cells in x direction.
        grid_y_size: Grid cells in y direction.

    Returns:
        Tuple of (model, loss_fn).
    """
    model = PointPillarsModel(
        num_classes=num_classes,
        num_input_features=num_input_features,
        pillar_feat_dim=pillar_feat_dim,
        max_points_per_pillar=max_points_per_pillar,
        grid_x_size=grid_x_size,
        grid_y_size=grid_y_size,
    )

    loss_fn = PointPillarsLoss(num_classes=num_classes)

    return model, loss_fn
