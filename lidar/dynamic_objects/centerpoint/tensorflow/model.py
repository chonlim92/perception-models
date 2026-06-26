"""
CenterPoint: Center-based 3D Object Detection from LiDAR Point Clouds.

TensorFlow 2 / Keras implementation for nuScenes dataset.
Architecture: Points -> Voxelization -> 3D Backbone -> BEV collapse -> 2D Backbone -> Center Heads -> Decode
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Optional

# =============================================================================
# Configuration
# =============================================================================

VOXEL_SIZE = [0.075, 0.075, 0.2]
POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
GRID_SIZE = [1440, 1440, 40]  # (X, Y, Z) voxel grid dimensions

NUSCENES_TASK_GROUPS = [
    ['car'],
    ['truck', 'construction_vehicle'],
    ['bus', 'trailer'],
    ['barrier'],
    ['motorcycle', 'bicycle'],
    ['pedestrian', 'traffic_cone'],
]


# =============================================================================
# Dynamic Voxelization
# =============================================================================

@tf.function
def dynamic_voxelization(
    points: tf.Tensor,
    voxel_size: List[float] = VOXEL_SIZE,
    point_cloud_range: List[float] = POINT_CLOUD_RANGE,
    grid_size: List[int] = GRID_SIZE,
    max_voxels: int = 60000,
    max_points_per_voxel: int = 20,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Assign points to voxels and compute mean features per voxel.

    Args:
        points: (N, C) tensor of point cloud data, where C >= 3 (x, y, z, ...).
        voxel_size: [vx, vy, vz] voxel dimensions in meters.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        grid_size: [Gx, Gy, Gz] number of voxels per axis.
        max_voxels: Maximum number of voxels to retain.
        max_points_per_voxel: Maximum points per voxel for averaging.

    Returns:
        voxel_features: (M, C) mean features per occupied voxel.
        voxel_coords: (M, 3) integer coordinates [ix, iy, iz] of occupied voxels.
        voxel_num_points: (M,) number of points in each voxel.
    """
    pc_range = tf.constant(point_cloud_range, dtype=tf.float32)
    vs = tf.constant(voxel_size, dtype=tf.float32)
    gs = tf.constant(grid_size, dtype=tf.int32)

    # Compute voxel indices for each point
    coords_float = (points[:, :3] - pc_range[:3]) / vs
    coords_int = tf.cast(tf.floor(coords_float), tf.int32)

    # Filter points outside the valid range
    valid_mask = tf.reduce_all(coords_int >= 0, axis=1) & tf.reduce_all(
        coords_int < gs, axis=1
    )
    valid_points = tf.boolean_mask(points, valid_mask)
    valid_coords = tf.boolean_mask(coords_int, valid_mask)

    # Compute flat voxel indices
    flat_indices = (
        valid_coords[:, 0] * gs[1] * gs[2]
        + valid_coords[:, 1] * gs[2]
        + valid_coords[:, 2]
    )

    # Get unique voxel indices and their mappings
    unique_flat, idx_mapping = tf.unique(flat_indices)
    num_unique = tf.shape(unique_flat)[0]
    num_unique = tf.minimum(num_unique, max_voxels)

    # Truncate to max_voxels
    unique_flat = unique_flat[:num_unique]

    # Compute mean features per voxel using unsorted_segment_mean
    num_features = tf.shape(valid_points)[1]
    voxel_features = tf.math.unsorted_segment_mean(
        valid_points, idx_mapping, num_segments=tf.cast(tf.shape(unique_flat)[0], tf.int32)
    )
    voxel_features = voxel_features[:num_unique]

    # Count points per voxel
    ones = tf.ones(tf.shape(idx_mapping)[0], dtype=tf.float32)
    voxel_num_points = tf.math.unsorted_segment_sum(
        ones, idx_mapping, num_segments=tf.cast(tf.shape(unique_flat)[0], tf.int32)
    )
    voxel_num_points = tf.cast(voxel_num_points[:num_unique], tf.int32)
    voxel_num_points = tf.minimum(voxel_num_points, max_points_per_voxel)

    # Recover 3D coordinates from flat indices
    iz = tf.math.floormod(unique_flat, gs[2])
    iy = tf.math.floormod(tf.math.floordiv(unique_flat, gs[2]), gs[1])
    ix = tf.math.floordiv(unique_flat, gs[1] * gs[2])
    voxel_coords = tf.stack([ix, iy, iz], axis=1)

    return voxel_features, voxel_coords, voxel_num_points


# =============================================================================
# PillarFeatureNet
# =============================================================================

class PillarFeatureNet(tf.keras.layers.Layer):
    """
    PointNet-based feature extraction for pillar representation.

    Treats each voxel column (pillar) as a set of points, applies a shared MLP
    (PointNet) and max-pools to produce a single feature vector per pillar.
    This is an alternative to the full 3D sparse backbone.
    """

    def __init__(
        self,
        in_channels: int = 5,
        feat_channels: Tuple[int, ...] = (64,),
        voxel_size: List[float] = VOXEL_SIZE,
        point_cloud_range: List[float] = POINT_CLOUD_RANGE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.mlp_layers = []
        self.bn_layers = []
        prev_ch = in_channels
        for out_ch in feat_channels:
            self.mlp_layers.append(
                tf.keras.layers.Dense(out_ch, use_bias=False)
            )
            self.bn_layers.append(tf.keras.layers.BatchNormalization())
            prev_ch = out_ch

    def call(self, voxel_features: tf.Tensor, voxel_coords: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            voxel_features: (M, C) mean point features per voxel.
            voxel_coords: (M, 3) voxel coordinates [ix, iy, iz].
            training: Whether in training mode.

        Returns:
            bev_features: (1, Gx, Gy, feat_channels[-1]) BEV feature map (pillars scattered).
        """
        x = voxel_features  # (M, C)

        for linear, bn in zip(self.mlp_layers, self.bn_layers):
            x = linear(x)
            x = bn(x, training=training)
            x = tf.nn.relu(x)

        # Scatter pillar features to BEV grid
        grid_x = GRID_SIZE[0]
        grid_y = GRID_SIZE[1]
        num_channels = self.feat_channels[-1]

        # Use only x, y coordinates (collapse z dimension)
        ix = voxel_coords[:, 0]
        iy = voxel_coords[:, 1]
        flat_bev_indices = ix * grid_y + iy

        # Scatter using tensor_scatter_nd_update
        bev_flat = tf.zeros([grid_x * grid_y, num_channels], dtype=x.dtype)
        indices = tf.expand_dims(flat_bev_indices, axis=1)

        # For duplicate indices, use max via segment_max then scatter
        unique_bev, unique_idx = tf.unique(flat_bev_indices)
        scattered_features = tf.math.unsorted_segment_max(
            x, unique_idx, num_segments=tf.shape(unique_bev)[0]
        )
        scatter_indices = tf.expand_dims(unique_bev, axis=1)
        bev_flat = tf.tensor_scatter_nd_update(bev_flat, scatter_indices, scattered_features)

        bev_map = tf.reshape(bev_flat, [1, grid_x, grid_y, num_channels])
        return bev_map


# =============================================================================
# Sparse 3D CNN Backbone (Dense with Masking)
# =============================================================================

class SparseConv3DBlock(tf.keras.layers.Layer):
    """
    A single 3D convolution block with batch normalization and ReLU.
    Uses dense 3D convolution with an occupancy mask to simulate sparsity.
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        strides: Tuple[int, int, int] = (1, 1, 1),
        padding: str = 'same',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv3D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x: tf.Tensor, mask: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            x: (B, D, H, W, C) dense feature volume.
            mask: (B, D, H, W, 1) binary occupancy mask.
            training: Training mode flag.

        Returns:
            out: (B, D', H', W', C_out) convolved features masked by updated occupancy.
            new_mask: (B, D', H', W', 1) updated occupancy mask after strided conv.
        """
        x = x * mask  # Apply mask before convolution
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        # Update mask: if stride > 1, the spatial dims shrink
        # Use max-pool on mask to propagate occupancy
        strides = self.conv.strides
        if any(s > 1 for s in strides):
            new_mask = tf.keras.layers.MaxPool3D(
                pool_size=strides, strides=strides, padding='same'
            )(mask)
        else:
            new_mask = mask

        x = x * new_mask
        return x, new_mask


class SparseCNNBackbone(tf.keras.layers.Layer):
    """
    3D Sparse Convolutional Backbone with 4 stages.

    Since TensorFlow lacks native sparse 3D convolutions, this implementation
    uses dense 3D convolutions with binary occupancy masking to approximate
    sparse behavior. Each stage doubles the stride and channel count.

    Stages:
        Stage 1: 16 channels, stride 1
        Stage 2: 32 channels, stride 2
        Stage 3: 64 channels, stride 2
        Stage 4: 128 channels, stride 2
    """

    def __init__(
        self,
        in_channels: int = 5,
        stage_channels: Tuple[int, ...] = (16, 32, 64, 128),
        blocks_per_stage: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.stage_channels = stage_channels

        # Initial projection layer
        self.initial_conv = tf.keras.layers.Conv3D(
            filters=stage_channels[0],
            kernel_size=3,
            strides=(1, 1, 1),
            padding='same',
            use_bias=False,
        )
        self.initial_bn = tf.keras.layers.BatchNormalization()

        # Build stages
        self.stages = []
        for stage_idx in range(len(stage_channels)):
            stage_blocks = []
            ch = stage_channels[stage_idx]

            # First block of each stage (except stage 0) uses stride 2
            if stage_idx > 0:
                stage_blocks.append(
                    SparseConv3DBlock(
                        out_channels=ch,
                        kernel_size=3,
                        strides=(2, 2, 2),
                        name=f'stage{stage_idx}_downsample',
                    )
                )
            else:
                stage_blocks.append(
                    SparseConv3DBlock(
                        out_channels=ch,
                        kernel_size=3,
                        strides=(1, 1, 1),
                        name=f'stage{stage_idx}_block0',
                    )
                )

            # Remaining blocks in stage maintain resolution
            for blk_idx in range(1, blocks_per_stage):
                stage_blocks.append(
                    SparseConv3DBlock(
                        out_channels=ch,
                        kernel_size=3,
                        strides=(1, 1, 1),
                        name=f'stage{stage_idx}_block{blk_idx}',
                    )
                )
            self.stages.append(stage_blocks)

    def call(
        self, voxel_features: tf.Tensor, voxel_coords: tf.Tensor, batch_size: int = 1, training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            voxel_features: (M, C) per-voxel features.
            voxel_coords: (M, 3) voxel coordinates [ix, iy, iz].
            batch_size: Number of samples in the batch.
            training: Training mode flag.

        Returns:
            spatial_features: (B, Gx/8, Gy/8, Gz/8 * 128) 3D backbone output
                reshaped ready for BEV collapse.
        """
        # Scatter voxel features into dense volume
        gx, gy, gz = GRID_SIZE
        num_features = tf.shape(voxel_features)[1]

        # Create dense volume: (B, Gx, Gy, Gz, C)
        volume = tf.zeros([batch_size, gx, gy, gz, self.in_channels], dtype=tf.float32)

        # Assume single batch for now; coords are [ix, iy, iz]
        # Add batch dimension index
        batch_indices = tf.zeros([tf.shape(voxel_coords)[0], 1], dtype=tf.int32)
        full_indices = tf.concat([batch_indices, voxel_coords], axis=1)  # (M, 4)

        # Pad or truncate features to in_channels
        feat_padded = voxel_features[:, :self.in_channels]
        if tf.shape(voxel_features)[1] < self.in_channels:
            padding = tf.zeros(
                [tf.shape(voxel_features)[0], self.in_channels - tf.shape(voxel_features)[1]],
                dtype=tf.float32,
            )
            feat_padded = tf.concat([voxel_features, padding], axis=1)

        volume = tf.tensor_scatter_nd_update(volume, full_indices, feat_padded)

        # Create occupancy mask
        mask = tf.cast(tf.reduce_any(volume != 0.0, axis=-1, keepdims=True), tf.float32)

        # Initial projection
        x = volume * mask
        x = self.initial_conv(x)
        x = self.initial_bn(x, training=training)
        x = tf.nn.relu(x)
        x = x * mask

        # Process stages
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x, mask = block(x, mask, training=training)

        # BEV collapse: reshape (B, D', H', W', C) -> (B, H', W', D' * C)
        # After 3 stride-2 stages (stages 1,2,3): spatial dims are / 8
        shape = tf.shape(x)
        b, d, h, w, c = shape[0], shape[1], shape[2], shape[3], shape[4]
        # Permute to (B, H, W, D, C) then reshape to (B, H, W, D*C)
        x = tf.transpose(x, perm=[0, 2, 3, 1, 4])  # (B, H', W', D', C)
        x = tf.reshape(x, [b, h, w, d * c])

        return x


# =============================================================================
# BEV Backbone (2D ResNet-style with Deconv Upsampling)
# =============================================================================

class ResBlock2D(tf.keras.layers.Layer):
    """2D residual block with two conv layers and skip connection."""

    def __init__(self, channels: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            channels, 3, strides=stride, padding='same', use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            channels, 3, strides=1, padding='same', use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.use_shortcut = stride != 1
        if self.use_shortcut:
            self.shortcut_conv = tf.keras.layers.Conv2D(
                channels, 1, strides=stride, padding='same', use_bias=False
            )
            self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.use_shortcut:
            identity = self.shortcut_conv(x)
            identity = self.shortcut_bn(identity, training=training)

        out = out + identity
        out = tf.nn.relu(out)
        return out


class BEVBackbone(tf.keras.layers.Layer):
    """
    2D ResNet-style backbone for BEV feature processing with deconv upsampling.

    Architecture:
        Stage 1: Maintains resolution, 128 channels, 3 residual blocks
        Stage 2: Downsamples 2x, 256 channels, 5 residual blocks
        Deconv upsample both stages to the same resolution, concatenate -> 256 ch output.
    """

    def __init__(
        self,
        in_channels: int = 640,
        stage1_channels: int = 128,
        stage2_channels: int = 256,
        stage1_blocks: int = 3,
        stage2_blocks: int = 5,
        output_channels: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels

        # Input projection to stage1 channels
        self.input_conv = tf.keras.layers.Conv2D(
            stage1_channels, 1, strides=1, padding='same', use_bias=False
        )
        self.input_bn = tf.keras.layers.BatchNormalization()

        # Stage 1: maintain resolution
        self.stage1_blocks = []
        for i in range(stage1_blocks):
            self.stage1_blocks.append(
                ResBlock2D(stage1_channels, stride=1, name=f'stage1_block{i}')
            )

        # Stage 2: downsample 2x
        self.stage2_blocks = []
        self.stage2_blocks.append(
            ResBlock2D(stage2_channels, stride=2, name='stage2_block0')
        )
        for i in range(1, stage2_blocks):
            self.stage2_blocks.append(
                ResBlock2D(stage2_channels, stride=1, name=f'stage2_block{i}')
            )

        # Deconv upsample for stage1 (identity resolution, just project)
        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            stage1_channels, kernel_size=1, strides=1, padding='same', use_bias=False
        )
        self.deconv1_bn = tf.keras.layers.BatchNormalization()

        # Deconv upsample for stage2 (upsample 2x back to stage1 resolution)
        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            stage1_channels, kernel_size=4, strides=2, padding='same', use_bias=False
        )
        self.deconv2_bn = tf.keras.layers.BatchNormalization()

        # Final projection after concatenation (128 + 128 = 256 -> output_channels)
        self.final_conv = tf.keras.layers.Conv2D(
            output_channels, 1, strides=1, padding='same', use_bias=False
        )
        self.final_bn = tf.keras.layers.BatchNormalization()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            x: (B, H, W, C_in) BEV feature map from 3D backbone.
            training: Training mode flag.

        Returns:
            out: (B, H, W, 256) processed BEV features.
        """
        # Input projection
        x = self.input_conv(x)
        x = self.input_bn(x, training=training)
        x = tf.nn.relu(x)

        # Stage 1
        stage1_out = x
        for block in self.stage1_blocks:
            stage1_out = block(stage1_out, training=training)

        # Stage 2
        stage2_out = stage1_out
        for block in self.stage2_blocks:
            stage2_out = block(stage2_out, training=training)

        # Deconv upsample
        up1 = self.deconv1(stage1_out)
        up1 = self.deconv1_bn(up1, training=training)
        up1 = tf.nn.relu(up1)

        up2 = self.deconv2(stage2_out)
        up2 = self.deconv2_bn(up2, training=training)
        up2 = tf.nn.relu(up2)

        # Concatenate and project
        concat = tf.concat([up1, up2], axis=-1)  # (B, H, W, 256)
        out = self.final_conv(concat)
        out = self.final_bn(out, training=training)
        out = tf.nn.relu(out)

        return out


# =============================================================================
# Center Head
# =============================================================================

class SeparateHead(tf.keras.layers.Layer):
    """
    A single regression/classification head with shared convolution layers
    followed by a task-specific output layer.
    """

    def __init__(self, in_channels: int, head_channels: int, out_channels: int, num_conv: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.convs = []
        self.bns = []
        prev_ch = in_channels
        for i in range(num_conv):
            self.convs.append(
                tf.keras.layers.Conv2D(head_channels, 3, padding='same', use_bias=False)
            )
            self.bns.append(tf.keras.layers.BatchNormalization())
            prev_ch = head_channels

        self.output_conv = tf.keras.layers.Conv2D(out_channels, 1, padding='same', use_bias=True)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x, training=training)
            x = tf.nn.relu(x)
        x = self.output_conv(x)
        return x


class CenterHead(tf.keras.layers.Layer):
    """
    CenterPoint detection head with separate sub-heads per task group.

    For nuScenes, there are 6 task groups. Each group outputs:
        - heatmap: (B, H, W, num_classes) center heatmap predictions
        - offset: (B, H, W, 2) sub-voxel center offset (x, y)
        - height: (B, H, W, 1) object center height (z)
        - size: (B, H, W, 3) object dimensions (l, w, h)
        - rotation: (B, H, W, 2) rotation encoding (sin, cos of yaw)
        - velocity: (B, H, W, 2) velocity (vx, vy)
    """

    def __init__(
        self,
        in_channels: int = 256,
        head_channels: int = 64,
        task_groups: List[List[str]] = None,
        num_conv: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if task_groups is None:
            task_groups = NUSCENES_TASK_GROUPS

        self.task_groups = task_groups
        self.num_tasks = len(task_groups)

        # Build heads for each task group
        self.heatmap_heads = []
        self.offset_heads = []
        self.height_heads = []
        self.size_heads = []
        self.rotation_heads = []
        self.velocity_heads = []

        for task_idx, classes in enumerate(task_groups):
            num_classes = len(classes)
            self.heatmap_heads.append(
                SeparateHead(in_channels, head_channels, num_classes, num_conv, name=f'task{task_idx}_heatmap')
            )
            self.offset_heads.append(
                SeparateHead(in_channels, head_channels, 2, num_conv, name=f'task{task_idx}_offset')
            )
            self.height_heads.append(
                SeparateHead(in_channels, head_channels, 1, num_conv, name=f'task{task_idx}_height')
            )
            self.size_heads.append(
                SeparateHead(in_channels, head_channels, 3, num_conv, name=f'task{task_idx}_size')
            )
            self.rotation_heads.append(
                SeparateHead(in_channels, head_channels, 2, num_conv, name=f'task{task_idx}_rotation')
            )
            self.velocity_heads.append(
                SeparateHead(in_channels, head_channels, 2, num_conv, name=f'task{task_idx}_velocity')
            )

    def call(self, x: tf.Tensor, training: bool = False) -> List[Dict[str, tf.Tensor]]:
        """
        Args:
            x: (B, H, W, C) BEV feature map.
            training: Training mode flag.

        Returns:
            predictions: List of dicts (one per task group), each containing:
                'heatmap', 'offset', 'height', 'size', 'rotation', 'velocity'.
        """
        predictions = []
        for task_idx in range(self.num_tasks):
            task_pred = {
                'heatmap': tf.sigmoid(self.heatmap_heads[task_idx](x, training=training)),
                'offset': self.offset_heads[task_idx](x, training=training),
                'height': self.height_heads[task_idx](x, training=training),
                'size': self.size_heads[task_idx](x, training=training),
                'rotation': self.rotation_heads[task_idx](x, training=training),
                'velocity': self.velocity_heads[task_idx](x, training=training),
            }
            predictions.append(task_pred)
        return predictions


# =============================================================================
# CenterPoint Full Model
# =============================================================================

class CenterPointModel(tf.keras.Model):
    """
    Complete CenterPoint model for 3D object detection from LiDAR point clouds.

    Pipeline:
        1. Dynamic voxelization of raw point cloud
        2. 3D Sparse CNN Backbone (dense with masking)
        3. BEV collapse (reshape 3D features to 2D)
        4. 2D BEV Backbone (ResNet-style with deconv)
        5. Center detection heads (per task group)

    Alternatively, the PillarFeatureNet can replace the 3D backbone for
    faster inference (pillar-based approach).
    """

    def __init__(
        self,
        point_channels: int = 5,
        use_pillar_backbone: bool = False,
        backbone_channels: Tuple[int, ...] = (16, 32, 64, 128),
        bev_output_channels: int = 256,
        head_channels: int = 64,
        task_groups: List[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_pillar_backbone = use_pillar_backbone
        self.point_channels = point_channels

        if use_pillar_backbone:
            self.backbone = PillarFeatureNet(
                in_channels=point_channels,
                feat_channels=(64,),
            )
            bev_in_channels = 64
        else:
            self.backbone = SparseCNNBackbone(
                in_channels=point_channels,
                stage_channels=backbone_channels,
            )
            # After 3D backbone: spatial dims / 8, last stage has 128 channels
            # BEV collapse concatenates along Z: (Gz/8) * 128 = 5 * 128 = 640
            bev_in_channels = (GRID_SIZE[2] // 8) * backbone_channels[-1]

        self.bev_backbone = BEVBackbone(
            in_channels=bev_in_channels,
            stage1_channels=128,
            stage2_channels=256,
            output_channels=bev_output_channels,
        )

        self.center_head = CenterHead(
            in_channels=bev_output_channels,
            head_channels=head_channels,
            task_groups=task_groups if task_groups is not None else NUSCENES_TASK_GROUPS,
        )

    def call(
        self,
        points: tf.Tensor,
        training: bool = False,
    ) -> List[Dict[str, tf.Tensor]]:
        """
        Args:
            points: (N, C) raw point cloud tensor with C >= 5 (x, y, z, intensity, timestamp).
            training: Training mode flag.

        Returns:
            predictions: List of task-group prediction dicts from the CenterHead.
        """
        # Step 1: Dynamic voxelization
        voxel_features, voxel_coords, voxel_num_points = dynamic_voxelization(points)

        # Step 2 & 3: Backbone + BEV collapse
        if self.use_pillar_backbone:
            bev_features = self.backbone(voxel_features, voxel_coords, training=training)
        else:
            bev_features = self.backbone(
                voxel_features, voxel_coords, batch_size=1, training=training
            )
            # bev_features is already (B, H, W, D*C) from SparseCNNBackbone

        # Step 4: 2D BEV backbone
        bev_out = self.bev_backbone(bev_features, training=training)

        # Step 5: Center detection heads
        predictions = self.center_head(bev_out, training=training)

        return predictions


# =============================================================================
# Decode Predictions
# =============================================================================

@tf.function
def _nms_heatmap(heatmap: tf.Tensor, kernel_size: int = 3) -> tf.Tensor:
    """Apply max-pool NMS on heatmap to extract peaks."""
    # heatmap: (B, H, W, C)
    hmax = tf.nn.max_pool2d(heatmap, ksize=kernel_size, strides=1, padding='SAME')
    keep = tf.cast(tf.equal(heatmap, hmax), tf.float32)
    return heatmap * keep


@tf.function
def _topk_from_heatmap(heatmap: tf.Tensor, k: int = 500) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Extract top-K peak scores and their spatial locations from heatmap.

    Args:
        heatmap: (B, H, W, C) heatmap after NMS.
        k: Number of top detections to keep.

    Returns:
        topk_scores: (B, K) top scores.
        topk_inds: (B, K) flat spatial indices.
        topk_classes: (B, K) class indices.
        topk_ys, topk_xs: (B, K) spatial coordinates.
    """
    batch_size = tf.shape(heatmap)[0]
    height = tf.shape(heatmap)[1]
    width = tf.shape(heatmap)[2]
    num_classes = tf.shape(heatmap)[3]

    # Reshape to (B, H*W*C)
    heatmap_flat = tf.reshape(heatmap, [batch_size, -1])

    # Get top-K across all classes and spatial locations
    topk_scores, topk_flat_inds = tf.math.top_k(heatmap_flat, k=k)

    # Recover class and spatial indices
    topk_classes = topk_flat_inds % num_classes
    topk_spatial_inds = topk_flat_inds // num_classes
    topk_ys = topk_spatial_inds // width
    topk_xs = topk_spatial_inds % width

    return topk_scores, topk_spatial_inds, topk_classes, topk_ys, topk_xs


def decode_predictions(
    predictions: List[Dict[str, tf.Tensor]],
    score_threshold: float = 0.1,
    top_k: int = 500,
    voxel_size: List[float] = VOXEL_SIZE,
    point_cloud_range: List[float] = POINT_CLOUD_RANGE,
    nms_kernel_size: int = 3,
) -> List[Dict[str, tf.Tensor]]:
    """
    Decode CenterPoint predictions into 3D bounding boxes.

    Performs heatmap peak extraction via max-pool NMS, then gathers regression
    values at peak locations to form final detections.

    Args:
        predictions: List of task-group prediction dicts from CenterHead.
        score_threshold: Minimum score to keep a detection.
        top_k: Number of top detections per task group.
        voxel_size: Voxel dimensions for converting pixel coords to meters.
        point_cloud_range: Point cloud spatial range.
        nms_kernel_size: Kernel size for max-pool NMS.

    Returns:
        detections: List of dicts (one per task group), each containing:
            'boxes_3d': (B, K, 9) [x, y, z, l, w, h, yaw, vx, vy]
            'scores': (B, K) detection scores
            'labels': (B, K) class labels within the task group
    """
    pc_range = tf.constant(point_cloud_range, dtype=tf.float32)
    vs = tf.constant(voxel_size, dtype=tf.float32)

    detections = []

    for task_pred in predictions:
        heatmap = task_pred['heatmap']  # (B, H, W, num_cls)
        offset = task_pred['offset']    # (B, H, W, 2)
        height = task_pred['height']    # (B, H, W, 1)
        size = task_pred['size']        # (B, H, W, 3)
        rotation = task_pred['rotation']  # (B, H, W, 2)
        velocity = task_pred['velocity']  # (B, H, W, 2)

        batch_size = tf.shape(heatmap)[0]
        h_dim = tf.shape(heatmap)[1]
        w_dim = tf.shape(heatmap)[2]

        # NMS on heatmap
        heatmap_nms = _nms_heatmap(heatmap, kernel_size=nms_kernel_size)

        # Top-K extraction
        topk_scores, topk_spatial_inds, topk_classes, topk_ys, topk_xs = _topk_from_heatmap(
            heatmap_nms, k=top_k
        )

        # Gather regression values at peak locations
        # Reshape regression maps to (B, H*W, C) for gathering
        offset_flat = tf.reshape(offset, [batch_size, -1, 2])
        height_flat = tf.reshape(height, [batch_size, -1, 1])
        size_flat = tf.reshape(size, [batch_size, -1, 3])
        rot_flat = tf.reshape(rotation, [batch_size, -1, 2])
        vel_flat = tf.reshape(velocity, [batch_size, -1, 2])

        # Gather at topk spatial indices
        # topk_spatial_inds: (B, K)
        gather_inds = tf.expand_dims(topk_spatial_inds, axis=-1)  # (B, K, 1)
        batch_inds = tf.tile(
            tf.reshape(tf.range(batch_size), [batch_size, 1, 1]),
            [1, top_k, 1],
        )
        gather_inds_full = tf.concat([batch_inds, gather_inds], axis=-1)  # (B, K, 2)

        pred_offset = tf.gather_nd(offset_flat, gather_inds_full)   # (B, K, 2)
        pred_height = tf.gather_nd(height_flat, gather_inds_full)   # (B, K, 1)
        pred_size = tf.gather_nd(size_flat, gather_inds_full)       # (B, K, 3)
        pred_rot = tf.gather_nd(rot_flat, gather_inds_full)         # (B, K, 2)
        pred_vel = tf.gather_nd(vel_flat, gather_inds_full)         # (B, K, 2)

        # Convert pixel coordinates + offset to world coordinates
        xs = tf.cast(topk_xs, tf.float32) + pred_offset[..., 0]
        ys = tf.cast(topk_ys, tf.float32) + pred_offset[..., 1]

        # Scale to metric coordinates
        x_world = xs * vs[0] + pc_range[0]
        y_world = ys * vs[1] + pc_range[1]
        z_world = pred_height[..., 0]  # Height is directly predicted in meters

        # Size is predicted as log scale
        pred_size_exp = tf.exp(pred_size)  # (B, K, 3) -> (l, w, h)

        # Rotation: atan2(sin, cos) -> yaw angle
        yaw = tf.atan2(pred_rot[..., 0], pred_rot[..., 1])  # (B, K)

        # Assemble 3D boxes: [x, y, z, l, w, h, yaw, vx, vy]
        boxes_3d = tf.stack([
            x_world, y_world, z_world,
            pred_size_exp[..., 0], pred_size_exp[..., 1], pred_size_exp[..., 2],
            yaw,
            pred_vel[..., 0], pred_vel[..., 1],
        ], axis=-1)  # (B, K, 9)

        # Filter by score threshold
        score_mask = topk_scores > score_threshold
        # Apply mask (keep all K, but mask scores to 0 for filtering downstream)
        filtered_scores = topk_scores * tf.cast(score_mask, tf.float32)

        detections.append({
            'boxes_3d': boxes_3d,
            'scores': filtered_scores,
            'labels': topk_classes,
        })

    return detections


# =============================================================================
# Loss Functions
# =============================================================================

@tf.function
def gaussian_focal_loss(
    pred_heatmap: tf.Tensor,
    target_heatmap: tf.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> tf.Tensor:
    """
    Gaussian focal loss for heatmap prediction (modified focal loss for dense detection).

    This loss handles the imbalance between positive and negative samples by
    down-weighting easy negatives and positives that are near a Gaussian peak.

    Args:
        pred_heatmap: (B, H, W, C) predicted heatmap (after sigmoid).
        target_heatmap: (B, H, W, C) ground truth heatmap with Gaussian peaks.
        alpha: Focusing parameter for hard examples.
        beta: Modulating factor for negative samples near Gaussian centers.

    Returns:
        loss: Scalar focal loss value, normalized by number of positive targets.
    """
    # Clamp predictions for numerical stability
    pred = tf.clip_by_value(pred_heatmap, 1e-6, 1.0 - 1e-6)

    # Positive locations: where target == 1
    pos_mask = tf.cast(tf.equal(target_heatmap, 1.0), tf.float32)
    neg_mask = 1.0 - pos_mask

    # Positive loss: -((1 - p)^alpha) * log(p)
    pos_loss = -tf.pow(1.0 - pred, alpha) * tf.math.log(pred) * pos_mask

    # Negative loss: -((1 - target)^beta) * (p^alpha) * log(1 - p)
    neg_loss = (
        -tf.pow(1.0 - target_heatmap, beta)
        * tf.pow(pred, alpha)
        * tf.math.log(1.0 - pred)
        * neg_mask
    )

    # Normalize by number of positive samples
    num_pos = tf.reduce_sum(pos_mask)
    num_pos = tf.maximum(num_pos, 1.0)

    loss = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / num_pos
    return loss


@tf.function
def reg_l1_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """
    Masked L1 regression loss for bounding box attributes.

    Only computes loss at locations where there is a ground truth object
    (indicated by the mask).

    Args:
        pred: (B, H, W, C) predicted regression values.
        target: (B, H, W, C) ground truth regression values.
        mask: (B, H, W, 1) binary mask indicating valid (positive) locations.

    Returns:
        loss: Scalar mean L1 loss over valid locations.
    """
    # Expand mask to match regression channels
    expanded_mask = tf.broadcast_to(mask, tf.shape(pred))

    # Compute L1 loss only at masked locations
    l1 = tf.abs(pred - target) * expanded_mask

    # Normalize by number of positive samples
    num_pos = tf.reduce_sum(mask)
    num_pos = tf.maximum(num_pos, 1.0)

    loss = tf.reduce_sum(l1) / num_pos
    return loss


# =============================================================================
# Combined Training Loss
# =============================================================================

def centerpoint_loss(
    predictions: List[Dict[str, tf.Tensor]],
    targets: List[Dict[str, tf.Tensor]],
    heatmap_weight: float = 1.0,
    offset_weight: float = 2.0,
    height_weight: float = 1.0,
    size_weight: float = 0.2,
    rotation_weight: float = 1.0,
    velocity_weight: float = 0.2,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Compute the combined CenterPoint training loss across all task groups.

    Args:
        predictions: List of prediction dicts from CenterHead.
        targets: List of ground truth dicts (same structure as predictions),
            each containing 'heatmap', 'offset', 'height', 'size', 'rotation',
            'velocity', and 'mask' (binary mask for regression targets).
        heatmap_weight: Weight for heatmap focal loss.
        offset_weight: Weight for offset regression loss.
        height_weight: Weight for height regression loss.
        size_weight: Weight for size regression loss.
        rotation_weight: Weight for rotation regression loss.
        velocity_weight: Weight for velocity regression loss.

    Returns:
        total_loss: Scalar total loss.
        loss_dict: Dictionary of individual loss components for logging.
    """
    total_loss = tf.constant(0.0)
    loss_dict = {
        'heatmap_loss': tf.constant(0.0),
        'offset_loss': tf.constant(0.0),
        'height_loss': tf.constant(0.0),
        'size_loss': tf.constant(0.0),
        'rotation_loss': tf.constant(0.0),
        'velocity_loss': tf.constant(0.0),
    }

    for pred, tgt in zip(predictions, targets):
        mask = tgt['mask']  # (B, H, W, 1)

        hm_loss = gaussian_focal_loss(pred['heatmap'], tgt['heatmap'])
        off_loss = reg_l1_loss(pred['offset'], tgt['offset'], mask)
        h_loss = reg_l1_loss(pred['height'], tgt['height'], mask)
        s_loss = reg_l1_loss(pred['size'], tgt['size'], mask)
        r_loss = reg_l1_loss(pred['rotation'], tgt['rotation'], mask)
        v_loss = reg_l1_loss(pred['velocity'], tgt['velocity'], mask)

        task_loss = (
            heatmap_weight * hm_loss
            + offset_weight * off_loss
            + height_weight * h_loss
            + size_weight * s_loss
            + rotation_weight * r_loss
            + velocity_weight * v_loss
        )
        total_loss = total_loss + task_loss

        loss_dict['heatmap_loss'] = loss_dict['heatmap_loss'] + hm_loss
        loss_dict['offset_loss'] = loss_dict['offset_loss'] + off_loss
        loss_dict['height_loss'] = loss_dict['height_loss'] + h_loss
        loss_dict['size_loss'] = loss_dict['size_loss'] + s_loss
        loss_dict['rotation_loss'] = loss_dict['rotation_loss'] + r_loss
        loss_dict['velocity_loss'] = loss_dict['velocity_loss'] + v_loss

    return total_loss, loss_dict


# =============================================================================
# Utility: Generate Gaussian Heatmap Target
# =============================================================================

def generate_gaussian_target(
    heatmap: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    min_overlap: float = 0.5,
) -> np.ndarray:
    """
    Draw a 2D Gaussian on the heatmap at the given center with specified radius.

    Args:
        heatmap: (H, W) numpy array to draw on.
        center: (cx, cy) integer pixel coordinates of the Gaussian center.
        radius: Integer radius of the Gaussian.
        min_overlap: Minimum IoU overlap (unused, kept for API compatibility).

    Returns:
        heatmap: Updated heatmap with Gaussian drawn (element-wise max).
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0

    # Generate 2D Gaussian kernel
    x = np.arange(0, diameter, dtype=np.float32)
    y = np.arange(0, diameter, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    center_x = center_y = radius
    gaussian = np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * sigma ** 2))

    height, width = heatmap.shape
    cx, cy = center

    # Clip Gaussian to heatmap boundaries
    left = max(0, cx - radius)
    right = min(width, cx + radius + 1)
    top = max(0, cy - radius)
    bottom = min(height, cy + radius + 1)

    g_left = max(0, radius - cx)
    g_right = g_left + (right - left)
    g_top = max(0, radius - cy)
    g_bottom = g_top + (bottom - top)

    heatmap[top:bottom, left:right] = np.maximum(
        heatmap[top:bottom, left:right],
        gaussian[g_top:g_bottom, g_left:g_right],
    )

    return heatmap


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.5) -> int:
    """
    Compute the Gaussian radius for a given bounding box size based on
    minimum IoU overlap criterion (CenterNet-style).

    Args:
        det_size: (height, width) of the bounding box in pixels.
        min_overlap: Minimum IoU overlap desired.

    Returns:
        radius: Integer Gaussian radius.
    """
    height, width = det_size

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2.0

    radius = int(min(r1, r2, r3))
    return max(0, radius)
