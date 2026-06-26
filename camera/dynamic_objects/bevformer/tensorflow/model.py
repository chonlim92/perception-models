"""BEVFormer: Bird's-Eye-View Transformer for 3D Object Detection from Multi-Camera Images.

A complete TensorFlow 2 / Keras implementation of BEVFormer for autonomous driving
perception. This model transforms multi-camera images into a unified bird's-eye-view
(BEV) representation and performs 3D object detection using a DETR-style decoder.

Reference: Li et al., "BEVFormer: Learning Bird's-Eye-View Representation from
Multi-Camera Images via Spatiotemporal Transformers", ECCV 2022.

Key architecture components:
    1. ResNet101 backbone + Feature Pyramid Network (FPN)
    2. BEV encoder with spatial cross-attention (deformable) and temporal self-attention
    3. DETR-style detection decoder with learnable object queries
    4. Detection heads for classification and 3D bounding box regression

Configuration (nuScenes defaults):
    - BEV grid: 200x200, embed_dims=256
    - 6 encoder layers, 6 decoder layers, 8 attention heads
    - 900 object queries, 6 cameras
    - Point cloud range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    - 10 nuScenes classes, code_size=10 (cx,cy,cz,w,l,h,sin,cos,vx,vy)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.optimize import linear_sum_assignment


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "bev_h": 200,
    "bev_w": 200,
    "embed_dims": 256,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "num_heads": 8,
    "num_queries": 900,
    "num_cameras": 6,
    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    "num_points_spatial": 4,
    "num_points_temporal": 4,
    "num_levels": 4,
    "fpn_in_channels": [512, 1024, 2048],
    "fpn_out_channels": 256,
    "fpn_num_outs": 4,
    "num_classes": 10,
    "code_size": 10,
    "dropout_rate": 0.1,
    "ffn_dim": 512,
}


# =============================================================================
# Feature Pyramid Network (FPN)
# =============================================================================


class FeaturePyramidNetwork(keras.layers.Layer):
    """Custom Feature Pyramid Network for multi-scale feature extraction.

    Takes multi-level features from the backbone (C3, C4, C5 for ResNet101)
    and produces a set of feature maps at multiple scales with uniform channel
    dimensions, suitable for the BEV encoder's multi-scale deformable attention.

    Args:
        in_channels: List of input channel dimensions from backbone levels.
        out_channels: Number of output channels for all FPN levels.
        num_outs: Number of output feature levels (extra levels via stride-2 conv).
    """

    def __init__(self, in_channels, out_channels, num_outs, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs

        self.lateral_convs = []
        self.fpn_convs = []

        for i, in_ch in enumerate(in_channels):
            lateral = layers.Conv2D(
                out_channels, 1, padding="same", name=f"lateral_conv_{i}"
            )
            fpn = layers.Conv2D(
                out_channels, 3, padding="same", name=f"fpn_conv_{i}"
            )
            self.lateral_convs.append(lateral)
            self.fpn_convs.append(fpn)

        self.extra_convs = []
        for i in range(num_outs - len(in_channels)):
            extra = layers.Conv2D(
                out_channels, 3, strides=2, padding="same", name=f"extra_conv_{i}"
            )
            self.extra_convs.append(extra)

    def call(self, inputs, training=None):
        """Forward pass through the FPN.

        Args:
            inputs: List of feature tensors from backbone levels,
                    each with shape [B, H_i, W_i, C_i].

        Returns:
            List of FPN output tensors, each [B, H_j, W_j, out_channels].
        """
        assert len(inputs) == len(self.in_channels)

        laterals = [
            self.lateral_convs[i](inputs[i]) for i in range(len(inputs))
        ]

        for i in range(len(laterals) - 2, -1, -1):
            h, w = tf.shape(laterals[i])[1], tf.shape(laterals[i])[2]
            upsampled = tf.image.resize(laterals[i + 1], [h, w], method="bilinear")
            laterals[i] = laterals[i] + upsampled

        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        if self.num_outs > len(outs):
            extra_input = outs[-1]
            for extra_conv in self.extra_convs:
                extra_input = tf.nn.relu(extra_conv(extra_input))
                outs.append(extra_input)

        return outs


# =============================================================================
# Multi-Scale Deformable Attention
# =============================================================================


class DeformableAttention(keras.layers.Layer):
    """Deformable attention mechanism with bilinear sampling.

    Implements deformable attention where each query attends to a small set of
    sampling points around a reference location. The sampling offsets are learned
    and the features are extracted via bilinear interpolation.

    Args:
        embed_dims: Embedding dimension.
        num_heads: Number of attention heads.
        num_points: Number of sampling points per attention head.
        num_levels: Number of feature map levels to attend to.
        dropout_rate: Dropout probability for attention weights.
    """

    def __init__(self, embed_dims, num_heads, num_points, num_levels, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.head_dim = embed_dims // num_heads

        self.sampling_offsets = layers.Dense(
            num_heads * num_levels * num_points * 2,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="sampling_offsets",
        )
        self.attention_weights = layers.Dense(
            num_heads * num_levels * num_points,
            name="attention_weights",
        )
        self.value_proj = layers.Dense(embed_dims, name="value_proj")
        self.output_proj = layers.Dense(embed_dims, name="output_proj")
        self.dropout = layers.Dropout(dropout_rate)

    def _bilinear_grid_sample(self, feature_map, grid):
        """Bilinear sampling from a feature map at given grid locations.

        Implements differentiable bilinear interpolation similar to
        torch.nn.functional.grid_sample or tf.image operations.

        Args:
            feature_map: Tensor of shape [B, H, W, C].
            grid: Tensor of shape [B, N, 2] with (x, y) coordinates
                  normalized to [0, 1] range.

        Returns:
            Sampled features of shape [B, N, C].
        """
        batch_size = tf.shape(feature_map)[0]
        height = tf.shape(feature_map)[1]
        width = tf.shape(feature_map)[2]
        channels = tf.shape(feature_map)[3]
        num_points_total = tf.shape(grid)[1]

        x = grid[:, :, 0] * tf.cast(width - 1, tf.float32)
        y = grid[:, :, 1] * tf.cast(height - 1, tf.float32)

        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, width - 1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y0 = tf.clip_by_value(y0, 0, height - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)

        x0f = tf.cast(x0, tf.float32)
        x1f = tf.cast(x1, tf.float32)
        y0f = tf.cast(y0, tf.float32)
        y1f = tf.cast(y1, tf.float32)

        wa = tf.expand_dims((x1f - x) * (y1f - y), axis=-1)
        wb = tf.expand_dims((x1f - x) * (y - y0f), axis=-1)
        wc = tf.expand_dims((x - x0f) * (y1f - y), axis=-1)
        wd = tf.expand_dims((x - x0f) * (y - y0f), axis=-1)

        batch_idx = tf.tile(
            tf.reshape(tf.range(batch_size), [batch_size, 1]),
            [1, num_points_total],
        )

        def gather_pixel(yy, xx):
            indices = tf.stack([batch_idx, yy, xx], axis=-1)
            return tf.gather_nd(feature_map, indices)

        Ia = gather_pixel(y0, x0)
        Ib = gather_pixel(y1, x0)
        Ic = gather_pixel(y0, x1)
        Id = gather_pixel(y1, x1)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def call(self, query, reference_points, value_list, spatial_shapes, training=None):
        """Forward pass for deformable attention.

        Args:
            query: Query tensor [B, num_queries, embed_dims].
            reference_points: Reference points [B, num_queries, num_levels, 2],
                              normalized to [0, 1].
            value_list: List of value feature maps, each [B, H_l, W_l, embed_dims].
            spatial_shapes: List of (H, W) tuples for each level.
            training: Boolean for training mode.

        Returns:
            Output tensor [B, num_queries, embed_dims].
        """
        batch_size = tf.shape(query)[0]
        num_queries = tf.shape(query)[1]

        offsets = self.sampling_offsets(query)
        offsets = tf.reshape(
            offsets,
            [batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2],
        )

        attn_weights = self.attention_weights(query)
        attn_weights = tf.reshape(
            attn_weights,
            [batch_size, num_queries, self.num_heads, self.num_levels * self.num_points],
        )
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = tf.reshape(
            attn_weights,
            [batch_size, num_queries, self.num_heads, self.num_levels, self.num_points],
        )

        sampled_values = []
        for level_idx in range(self.num_levels):
            if level_idx < len(value_list):
                value = self.value_proj(value_list[level_idx])
            else:
                value = self.value_proj(value_list[-1])

            h_l, w_l = spatial_shapes[level_idx] if level_idx < len(spatial_shapes) else spatial_shapes[-1]

            offset_normalizer = tf.constant(
                [1.0 / max(w_l, 1), 1.0 / max(h_l, 1)], dtype=tf.float32
            )

            for head_idx in range(self.num_heads):
                for point_idx in range(self.num_points):
                    offset = offsets[:, :, head_idx, level_idx, point_idx, :]
                    ref_pt = reference_points[:, :, level_idx, :] if reference_points.shape.ndims == 4 else reference_points[:, :, 0, :]

                    sampling_loc = ref_pt + offset * offset_normalizer
                    sampling_loc = tf.clip_by_value(sampling_loc, 0.0, 1.0)

                    sampled = self._bilinear_grid_sample(value, sampling_loc)
                    weight = attn_weights[:, :, head_idx, level_idx, point_idx]
                    sampled_values.append(sampled * tf.expand_dims(weight, -1))

        output = tf.add_n(sampled_values)
        output = output / tf.cast(self.num_heads, tf.float32)
        output = self.output_proj(output)
        output = self.dropout(output, training=training)
        return output


# =============================================================================
# Spatial Cross-Attention
# =============================================================================


class SpatialCrossAttention(keras.layers.Layer):
    """Spatial cross-attention for projecting image features into BEV space.

    Each BEV query attends to relevant regions across all camera images by
    projecting 3D reference points (pillar sampling along z-axis) onto 2D
    image planes and using deformable attention to sample features.

    Args:
        embed_dims: Embedding dimension.
        num_heads: Number of attention heads.
        num_points: Number of deformable sampling points.
        num_levels: Number of multi-scale feature levels.
        num_cameras: Number of camera views.
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        dropout_rate: Dropout probability.
    """

    def __init__(self, embed_dims, num_heads, num_points, num_levels,
                 num_cameras, pc_range, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.num_cameras = num_cameras
        self.pc_range = pc_range

        self.deformable_attention = DeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points,
            num_levels=num_levels,
            dropout_rate=dropout_rate,
            name="deformable_attn",
        )
        self.output_proj = layers.Dense(embed_dims, name="output_proj")
        self.camera_embed = layers.Dense(embed_dims, name="camera_embed")
        self.layer_norm = layers.LayerNormalization(name="spatial_ln")

    def _get_reference_points_3d(self, bev_h, bev_w, num_z_anchors=4, dtype=tf.float32):
        """Generate 3D reference points for BEV queries (pillar sampling).

        Creates a grid of 3D reference points by placing pillars at each BEV
        grid cell and sampling along the z-axis.

        Args:
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            num_z_anchors: Number of z-axis sampling points per pillar.
            dtype: Data type.

        Returns:
            Reference points tensor [1, bev_h*bev_w, num_z_anchors, 3],
            normalized to [0, 1] within pc_range.
        """
        zs = tf.linspace(0.0, 1.0, num_z_anchors)
        xs = tf.linspace(0.5 / tf.cast(bev_w, dtype), 1.0 - 0.5 / tf.cast(bev_w, dtype), bev_w)
        ys = tf.linspace(0.5 / tf.cast(bev_h, dtype), 1.0 - 0.5 / tf.cast(bev_h, dtype), bev_h)

        grid_y, grid_x, grid_z = tf.meshgrid(ys, xs, zs, indexing="ij")

        coords = tf.stack([grid_x, grid_y, grid_z], axis=-1)
        coords = tf.reshape(coords, [1, bev_h * bev_w, num_z_anchors, 3])
        return coords

    def _project_to_image(self, reference_points_3d, lidar2img):
        """Project 3D reference points onto 2D image planes.

        Args:
            reference_points_3d: [B, N, num_z, 3] normalized 3D coordinates.
            lidar2img: [B, num_cameras, 4, 4] transformation matrices.

        Returns:
            reference_points_cam: [B, num_cameras, N, num_z, 2] normalized
                                  2D image coordinates.
            mask: [B, num_cameras, N, num_z] boolean validity mask.
        """
        pc_range = self.pc_range
        pts_3d = reference_points_3d.numpy() if hasattr(reference_points_3d, 'numpy') else reference_points_3d

        x_range = pc_range[3] - pc_range[0]
        y_range = pc_range[4] - pc_range[1]
        z_range = pc_range[5] - pc_range[2]

        pts_x = reference_points_3d[..., 0:1] * x_range + pc_range[0]
        pts_y = reference_points_3d[..., 1:2] * y_range + pc_range[1]
        pts_z = reference_points_3d[..., 2:3] * z_range + pc_range[2]

        batch_size = tf.shape(reference_points_3d)[0]
        num_queries = tf.shape(reference_points_3d)[1]
        num_z = tf.shape(reference_points_3d)[2]

        pts = tf.concat([pts_x, pts_y, pts_z, tf.ones_like(pts_x)], axis=-1)
        pts = tf.reshape(pts, [batch_size, 1, num_queries, num_z, 4])
        pts = tf.tile(pts, [1, self.num_cameras, 1, 1, 1])

        lidar2img_expanded = tf.reshape(lidar2img, [batch_size, self.num_cameras, 1, 1, 4, 4])
        lidar2img_expanded = tf.tile(lidar2img_expanded, [1, 1, num_queries, num_z, 1, 1])

        pts_expanded = tf.expand_dims(pts, axis=-1)
        pts_cam = tf.squeeze(tf.matmul(lidar2img_expanded, pts_expanded), axis=-1)

        depth = tf.clip_by_value(pts_cam[..., 2:3], clip_value_min=1e-5, clip_value_max=1e5)
        pts_2d = pts_cam[..., :2] / depth

        img_h, img_w = 900.0, 1600.0
        pts_2d_norm = tf.stack([
            pts_2d[..., 0] / img_w,
            pts_2d[..., 1] / img_h,
        ], axis=-1)

        mask = (
            (pts_2d_norm[..., 0] >= 0.0) &
            (pts_2d_norm[..., 0] < 1.0) &
            (pts_2d_norm[..., 1] >= 0.0) &
            (pts_2d_norm[..., 1] < 1.0) &
            (pts_cam[..., 2] > 1e-5)
        )

        return pts_2d_norm, mask

    def call(self, bev_queries, multi_scale_features, lidar2img, bev_h, bev_w,
             spatial_shapes, training=None):
        """Forward pass for spatial cross-attention.

        Args:
            bev_queries: BEV query tensor [B, bev_h*bev_w, embed_dims].
            multi_scale_features: List of image features per camera,
                                  each [B*num_cameras, H_l, W_l, C].
            lidar2img: Camera projection matrices [B, num_cameras, 4, 4].
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            spatial_shapes: List of (H, W) for each feature level.
            training: Boolean for training mode.

        Returns:
            Output tensor [B, bev_h*bev_w, embed_dims].
        """
        batch_size = tf.shape(bev_queries)[0]
        num_queries = bev_h * bev_w

        ref_3d = self._get_reference_points_3d(bev_h, bev_w, num_z_anchors=4)
        ref_3d = tf.tile(ref_3d, [batch_size, 1, 1, 1])

        ref_2d, mask = self._project_to_image(ref_3d, lidar2img)

        aggregated = tf.zeros_like(bev_queries)
        count = tf.zeros([batch_size, num_queries, 1])

        for cam_idx in range(self.num_cameras):
            cam_ref = ref_2d[:, cam_idx, :, :, :]
            cam_mask = mask[:, cam_idx, :, :]

            cam_ref_mean = tf.reduce_mean(cam_ref, axis=2)
            cam_ref_levels = tf.tile(
                tf.expand_dims(cam_ref_mean, axis=2),
                [1, 1, self.num_levels, 1],
            )

            cam_features = []
            for level_idx in range(self.num_levels):
                if level_idx < len(multi_scale_features):
                    feat = multi_scale_features[level_idx]
                    feat_per_cam = feat[cam_idx::self.num_cameras]
                    cam_features.append(feat_per_cam)
                else:
                    feat = multi_scale_features[-1]
                    feat_per_cam = feat[cam_idx::self.num_cameras]
                    cam_features.append(feat_per_cam)

            cam_embed = self.camera_embed(
                tf.one_hot(
                    tf.fill([batch_size, num_queries], cam_idx),
                    depth=self.num_cameras,
                )
            )
            query_with_cam = bev_queries + cam_embed

            attn_out = self.deformable_attention(
                query=query_with_cam,
                reference_points=cam_ref_levels,
                value_list=cam_features,
                spatial_shapes=spatial_shapes,
                training=training,
            )

            cam_mask_float = tf.cast(
                tf.reduce_any(cam_mask, axis=-1), tf.float32
            )
            cam_mask_float = tf.expand_dims(cam_mask_float, axis=-1)

            aggregated = aggregated + attn_out * cam_mask_float
            count = count + cam_mask_float

        count = tf.maximum(count, 1.0)
        output = aggregated / count
        output = self.output_proj(output)
        output = self.layer_norm(output + bev_queries)
        return output


# =============================================================================
# Temporal Self-Attention
# =============================================================================


class TemporalSelfAttention(keras.layers.Layer):
    """Temporal self-attention with ego-motion alignment.

    Aligns previous BEV features to the current frame using ego-motion
    transformation (2D grid warping), then performs deformable self-attention
    between current and aligned previous BEV features.

    Args:
        embed_dims: Embedding dimension.
        num_heads: Number of attention heads.
        num_points: Number of deformable sampling points.
        dropout_rate: Dropout probability.
    """

    def __init__(self, embed_dims, num_heads, num_points, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = layers.Dense(
            num_heads * num_points * 2,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="temporal_offsets",
        )
        self.attention_weights = layers.Dense(
            num_heads * num_points * 2,
            name="temporal_attn_weights",
        )
        self.value_proj = layers.Dense(embed_dims, name="temporal_value_proj")
        self.output_proj = layers.Dense(embed_dims, name="temporal_output_proj")
        self.layer_norm = layers.LayerNormalization(name="temporal_ln")
        self.dropout = layers.Dropout(dropout_rate)

    def _warp_bev_features(self, prev_bev, ego_motion, bev_h, bev_w):
        """Warp previous BEV features to align with current frame.

        Applies a 2D affine transformation (extracted from the ego-motion matrix)
        to warp the previous BEV feature grid to the current coordinate frame.

        Args:
            prev_bev: Previous BEV features [B, bev_h*bev_w, embed_dims].
            ego_motion: Ego-motion transformation [B, 4, 4] from prev to current.
            bev_h: BEV grid height.
            bev_w: BEV grid width.

        Returns:
            Warped BEV features [B, bev_h*bev_w, embed_dims].
        """
        batch_size = tf.shape(prev_bev)[0]
        channels = tf.shape(prev_bev)[2]

        bev_2d = tf.reshape(prev_bev, [batch_size, bev_h, bev_w, channels])

        translation_x = ego_motion[:, 0, 3]
        translation_y = ego_motion[:, 1, 3]
        cos_theta = ego_motion[:, 0, 0]
        sin_theta = ego_motion[:, 0, 1]

        norm_tx = translation_x / 51.2
        norm_ty = translation_y / 51.2

        theta = tf.stack([
            tf.stack([cos_theta, -sin_theta, norm_tx], axis=-1),
            tf.stack([sin_theta, cos_theta, norm_ty], axis=-1),
        ], axis=1)

        grid_y = tf.linspace(-1.0, 1.0, bev_h)
        grid_x = tf.linspace(-1.0, 1.0, bev_w)
        grid_yy, grid_xx = tf.meshgrid(grid_y, grid_x, indexing="ij")
        ones = tf.ones_like(grid_xx)
        grid = tf.stack([grid_xx, grid_yy, ones], axis=-1)
        grid_flat = tf.reshape(grid, [-1, 3])

        grid_flat = tf.tile(tf.expand_dims(grid_flat, 0), [batch_size, 1, 1])

        transformed = tf.matmul(grid_flat, tf.transpose(theta, [0, 2, 1]))

        transformed_grid = tf.reshape(transformed, [batch_size, bev_h, bev_w, 2])

        sample_x = (transformed_grid[..., 0] + 1.0) / 2.0
        sample_y = (transformed_grid[..., 1] + 1.0) / 2.0

        sample_x = tf.clip_by_value(sample_x, 0.0, 1.0)
        sample_y = tf.clip_by_value(sample_y, 0.0, 1.0)

        ix = sample_x * tf.cast(bev_w - 1, tf.float32)
        iy = sample_y * tf.cast(bev_h - 1, tf.float32)

        ix0 = tf.cast(tf.floor(ix), tf.int32)
        ix1 = ix0 + 1
        iy0 = tf.cast(tf.floor(iy), tf.int32)
        iy1 = iy0 + 1

        ix0 = tf.clip_by_value(ix0, 0, bev_w - 1)
        ix1 = tf.clip_by_value(ix1, 0, bev_w - 1)
        iy0 = tf.clip_by_value(iy0, 0, bev_h - 1)
        iy1 = tf.clip_by_value(iy1, 0, bev_h - 1)

        ix0f = tf.cast(ix0, tf.float32)
        ix1f = tf.cast(ix1, tf.float32)
        iy0f = tf.cast(iy0, tf.float32)
        iy1f = tf.cast(iy1, tf.float32)

        wa = tf.expand_dims((ix1f - ix) * (iy1f - iy), axis=-1)
        wb = tf.expand_dims((ix1f - ix) * (iy - iy0f), axis=-1)
        wc = tf.expand_dims((ix - ix0f) * (iy1f - iy), axis=-1)
        wd = tf.expand_dims((ix - ix0f) * (iy - iy0f), axis=-1)

        batch_idx = tf.tile(
            tf.reshape(tf.range(batch_size), [batch_size, 1, 1]),
            [1, bev_h, bev_w],
        )

        def gather_2d(yy, xx):
            indices = tf.stack([batch_idx, yy, xx], axis=-1)
            return tf.gather_nd(bev_2d, indices)

        warped = (
            wa * gather_2d(iy0, ix0) +
            wb * gather_2d(iy1, ix0) +
            wc * gather_2d(iy0, ix1) +
            wd * gather_2d(iy1, ix1)
        )

        return tf.reshape(warped, [batch_size, bev_h * bev_w, channels])

    def call(self, bev_queries, prev_bev, ego_motion, bev_h, bev_w, training=None):
        """Forward pass for temporal self-attention.

        Args:
            bev_queries: Current BEV queries [B, bev_h*bev_w, embed_dims].
            prev_bev: Previous frame BEV features [B, bev_h*bev_w, embed_dims]
                      or None for the first frame.
            ego_motion: Ego-motion matrix [B, 4, 4] from previous to current frame.
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            training: Boolean for training mode.

        Returns:
            Output tensor [B, bev_h*bev_w, embed_dims].
        """
        batch_size = tf.shape(bev_queries)[0]
        num_queries = bev_h * bev_w

        if prev_bev is None:
            aligned_prev = tf.zeros_like(bev_queries)
        else:
            aligned_prev = self._warp_bev_features(prev_bev, ego_motion, bev_h, bev_w)

        value_stack = tf.concat([bev_queries, aligned_prev], axis=1)
        value = self.value_proj(value_stack)

        value = tf.reshape(value, [batch_size, 2, num_queries, self.embed_dims])

        offsets = self.sampling_offsets(bev_queries)
        offsets = tf.reshape(offsets, [batch_size, num_queries, self.num_heads, self.num_points, 2])
        offsets = tf.nn.tanh(offsets) * 0.05

        attn_weights = self.attention_weights(bev_queries)
        attn_weights = tf.reshape(
            attn_weights, [batch_size, num_queries, self.num_heads, self.num_points * 2]
        )
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = tf.reshape(
            attn_weights, [batch_size, num_queries, self.num_heads, 2, self.num_points]
        )

        ref_y = tf.cast(tf.range(bev_h), tf.float32) / tf.cast(bev_h, tf.float32)
        ref_x = tf.cast(tf.range(bev_w), tf.float32) / tf.cast(bev_w, tf.float32)
        ref_yy, ref_xx = tf.meshgrid(ref_y, ref_x, indexing="ij")
        ref_pts = tf.stack([ref_xx, ref_yy], axis=-1)
        ref_pts = tf.reshape(ref_pts, [1, num_queries, 2])
        ref_pts = tf.tile(ref_pts, [batch_size, 1, 1])

        output = tf.zeros([batch_size, num_queries, self.embed_dims])

        for temporal_idx in range(2):
            temporal_value = value[:, temporal_idx, :, :]
            temporal_value_2d = tf.reshape(
                temporal_value, [batch_size, bev_h, bev_w, self.embed_dims]
            )

            for head_idx in range(self.num_heads):
                for pt_idx in range(self.num_points):
                    offset = offsets[:, :, head_idx, pt_idx, :]
                    sample_loc = ref_pts + offset
                    sample_loc = tf.clip_by_value(sample_loc, 0.0, 1.0)

                    sampled = self._bilinear_sample_bev(temporal_value_2d, sample_loc, bev_h, bev_w)
                    w = attn_weights[:, :, head_idx, temporal_idx, pt_idx]
                    output = output + sampled * tf.expand_dims(w, -1)

        output = output / tf.cast(self.num_heads, tf.float32)
        output = self.output_proj(output)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output + bev_queries)
        return output

    def _bilinear_sample_bev(self, feature_map, grid, bev_h, bev_w):
        """Bilinear sampling from BEV feature map.

        Args:
            feature_map: [B, bev_h, bev_w, C].
            grid: [B, N, 2] with (x, y) in [0, 1].
            bev_h: Grid height.
            bev_w: Grid width.

        Returns:
            Sampled features [B, N, C].
        """
        batch_size = tf.shape(feature_map)[0]
        num_pts = tf.shape(grid)[1]

        x = grid[:, :, 0] * tf.cast(bev_w - 1, tf.float32)
        y = grid[:, :, 1] * tf.cast(bev_h - 1, tf.float32)

        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, bev_w - 1)
        x1 = tf.clip_by_value(x1, 0, bev_w - 1)
        y0 = tf.clip_by_value(y0, 0, bev_h - 1)
        y1 = tf.clip_by_value(y1, 0, bev_h - 1)

        x0f = tf.cast(x0, tf.float32)
        x1f = tf.cast(x1, tf.float32)
        y0f = tf.cast(y0, tf.float32)
        y1f = tf.cast(y1, tf.float32)

        wa = tf.expand_dims((x1f - x) * (y1f - y), -1)
        wb = tf.expand_dims((x1f - x) * (y - y0f), -1)
        wc = tf.expand_dims((x - x0f) * (y1f - y), -1)
        wd = tf.expand_dims((x - x0f) * (y - y0f), -1)

        batch_idx = tf.tile(
            tf.reshape(tf.range(batch_size), [batch_size, 1]),
            [1, num_pts],
        )

        def gather_px(yy, xx):
            indices = tf.stack([batch_idx, yy, xx], axis=-1)
            return tf.gather_nd(feature_map, indices)

        return wa * gather_px(y0, x0) + wb * gather_px(y1, x0) + wc * gather_px(y0, x1) + wd * gather_px(y1, x1)


# =============================================================================
# Feed-Forward Network
# =============================================================================


class FFN(keras.layers.Layer):
    """Position-wise Feed-Forward Network with GELU activation.

    Standard transformer FFN: Linear -> GELU -> Dropout -> Linear -> Dropout.

    Args:
        embed_dims: Input and output dimension.
        ffn_dim: Hidden layer dimension.
        dropout_rate: Dropout probability.
    """

    def __init__(self, embed_dims, ffn_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.linear1 = layers.Dense(ffn_dim, activation="gelu", name="ffn_linear1")
        self.linear2 = layers.Dense(embed_dims, name="ffn_linear2")
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.layer_norm = layers.LayerNormalization(name="ffn_ln")

    def call(self, x, training=None):
        """Forward pass.

        Args:
            x: Input tensor [B, N, embed_dims].
            training: Boolean for training mode.

        Returns:
            Output tensor [B, N, embed_dims].
        """
        residual = x
        x = self.linear1(x)
        x = self.dropout1(x, training=training)
        x = self.linear2(x)
        x = self.dropout2(x, training=training)
        return self.layer_norm(x + residual)


# =============================================================================
# BEV Encoder Layer
# =============================================================================


class BEVEncoderLayer(keras.layers.Layer):
    """Single BEV encoder layer: Temporal Self-Attention + Spatial Cross-Attention + FFN.

    Each encoder layer first aligns and attends to temporal (previous frame)
    BEV features, then performs spatial cross-attention to aggregate multi-camera
    image features, followed by a feed-forward network.

    Args:
        embed_dims: Embedding dimension.
        num_heads: Number of attention heads.
        num_points_spatial: Deformable points for spatial attention.
        num_points_temporal: Deformable points for temporal attention.
        num_levels: Number of feature map levels.
        num_cameras: Number of camera views.
        pc_range: Point cloud range.
        ffn_dim: FFN hidden dimension.
        dropout_rate: Dropout probability.
    """

    def __init__(self, embed_dims, num_heads, num_points_spatial, num_points_temporal,
                 num_levels, num_cameras, pc_range, ffn_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.temporal_self_attention = TemporalSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points_temporal,
            dropout_rate=dropout_rate,
            name="temporal_self_attn",
        )
        self.spatial_cross_attention = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points_spatial,
            num_levels=num_levels,
            num_cameras=num_cameras,
            pc_range=pc_range,
            dropout_rate=dropout_rate,
            name="spatial_cross_attn",
        )
        self.ffn = FFN(embed_dims, ffn_dim, dropout_rate, name="encoder_ffn")

    def call(self, bev_queries, prev_bev, ego_motion, multi_scale_features,
             lidar2img, bev_h, bev_w, spatial_shapes, training=None):
        """Forward pass for a single encoder layer.

        Args:
            bev_queries: BEV query tensor [B, bev_h*bev_w, embed_dims].
            prev_bev: Previous BEV features or None.
            ego_motion: Ego-motion matrix [B, 4, 4].
            multi_scale_features: Multi-scale camera features.
            lidar2img: Camera projection matrices [B, num_cameras, 4, 4].
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            spatial_shapes: List of (H, W) per feature level.
            training: Boolean for training mode.

        Returns:
            Updated BEV features [B, bev_h*bev_w, embed_dims].
        """
        bev_queries = self.temporal_self_attention(
            bev_queries=bev_queries,
            prev_bev=prev_bev,
            ego_motion=ego_motion,
            bev_h=bev_h,
            bev_w=bev_w,
            training=training,
        )

        bev_queries = self.spatial_cross_attention(
            bev_queries=bev_queries,
            multi_scale_features=multi_scale_features,
            lidar2img=lidar2img,
            bev_h=bev_h,
            bev_w=bev_w,
            spatial_shapes=spatial_shapes,
            training=training,
        )

        bev_queries = self.ffn(bev_queries, training=training)

        return bev_queries


# =============================================================================
# BEV Encoder
# =============================================================================


class BEVEncoder(keras.layers.Layer):
    """Stacked BEV encoder with multiple transformer layers.

    Processes BEV queries through a stack of encoder layers, each incorporating
    temporal and spatial attention mechanisms.

    Args:
        num_layers: Number of encoder layers.
        embed_dims: Embedding dimension.
        num_heads: Number of attention heads.
        num_points_spatial: Deformable points for spatial attention.
        num_points_temporal: Deformable points for temporal attention.
        num_levels: Number of feature map levels.
        num_cameras: Number of camera views.
        pc_range: Point cloud range.
        ffn_dim: FFN hidden dimension.
        dropout_rate: Dropout probability.
    """

    def __init__(self, num_layers, embed_dims, num_heads, num_points_spatial,
                 num_points_temporal, num_levels, num_cameras, pc_range,
                 ffn_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.layers_list = []
        for i in range(num_layers):
            layer = BEVEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_points_spatial=num_points_spatial,
                num_points_temporal=num_points_temporal,
                num_levels=num_levels,
                num_cameras=num_cameras,
                pc_range=pc_range,
                ffn_dim=ffn_dim,
                dropout_rate=dropout_rate,
                name=f"encoder_layer_{i}",
            )
            self.layers_list.append(layer)

        self.layer_norm = layers.LayerNormalization(name="encoder_final_ln")

    def call(self, bev_queries, prev_bev, ego_motion, multi_scale_features,
             lidar2img, bev_h, bev_w, spatial_shapes, training=None):
        """Forward pass through all encoder layers.

        Args:
            bev_queries: Initial BEV queries [B, bev_h*bev_w, embed_dims].
            prev_bev: Previous frame BEV features or None.
            ego_motion: Ego-motion matrix [B, 4, 4].
            multi_scale_features: Multi-scale camera features.
            lidar2img: Camera projection matrices.
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            spatial_shapes: List of (H, W) per level.
            training: Boolean for training mode.

        Returns:
            Encoded BEV features [B, bev_h*bev_w, embed_dims].
        """
        output = bev_queries
        for layer in self.layers_list:
            output = layer(
                bev_queries=output,
                prev_bev=prev_bev,
                ego_motion=ego_motion,
                multi_scale_features=multi_scale_features,
                lidar2img=lidar2img,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                training=training,
            )
        return self.layer_norm(output)


# =============================================================================
# DETR Decoder Layer
# =============================================================================


class DETRDecoderLayer(keras.layers.Layer):
    """Single DETR decoder layer: Self-Attention + Cross-Attention (to BEV) + FFN.

    Standard transformer decoder layer with learnable object queries performing
    self-attention among themselves, cross-attention to BEV features, and FFN.

    Args:
        embed_dims: Embedding dimension.
        num_heads: Number of attention heads.
        ffn_dim: FFN hidden dimension.
        dropout_rate: Dropout probability.
    """

    def __init__(self, embed_dims, num_heads, ffn_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads

        self.self_attn_qkv = layers.Dense(3 * embed_dims, name="self_attn_qkv")
        self.self_attn_out = layers.Dense(embed_dims, name="self_attn_out")
        self.self_attn_ln = layers.LayerNormalization(name="self_attn_ln")
        self.self_attn_dropout = layers.Dropout(dropout_rate)

        self.cross_attn_q = layers.Dense(embed_dims, name="cross_attn_q")
        self.cross_attn_k = layers.Dense(embed_dims, name="cross_attn_k")
        self.cross_attn_v = layers.Dense(embed_dims, name="cross_attn_v")
        self.cross_attn_out = layers.Dense(embed_dims, name="cross_attn_out")
        self.cross_attn_ln = layers.LayerNormalization(name="cross_attn_ln")
        self.cross_attn_dropout = layers.Dropout(dropout_rate)

        self.ffn = FFN(embed_dims, ffn_dim, dropout_rate, name="decoder_ffn")

    def call(self, query, bev_features, query_pos, training=None):
        """Forward pass for a single decoder layer.

        Args:
            query: Object query tensor [B, num_queries, embed_dims].
            bev_features: Encoded BEV features [B, bev_h*bev_w, embed_dims].
            query_pos: Positional embeddings for queries [B, num_queries, embed_dims].
            training: Boolean for training mode.

        Returns:
            Updated query tensor [B, num_queries, embed_dims].
        """
        batch_size = tf.shape(query)[0]
        num_queries = tf.shape(query)[1]

        q_input = query + query_pos
        qkv = self.self_attn_qkv(q_input)
        qkv = tf.reshape(qkv, [batch_size, num_queries, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q_sa, k_sa, v_sa = qkv[0], qkv[1], qkv[2]

        scale = tf.math.rsqrt(tf.cast(self.head_dim, tf.float32))
        attn_weights = tf.matmul(q_sa, k_sa, transpose_b=True) * scale
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        self_attn_out = tf.matmul(attn_weights, v_sa)
        self_attn_out = tf.transpose(self_attn_out, [0, 2, 1, 3])
        self_attn_out = tf.reshape(self_attn_out, [batch_size, num_queries, self.embed_dims])
        self_attn_out = self.self_attn_out(self_attn_out)
        self_attn_out = self.self_attn_dropout(self_attn_out, training=training)
        query = self.self_attn_ln(query + self_attn_out)

        num_bev = tf.shape(bev_features)[1]
        q_ca = self.cross_attn_q(query + query_pos)
        k_ca = self.cross_attn_k(bev_features)
        v_ca = self.cross_attn_v(bev_features)

        q_ca = tf.reshape(q_ca, [batch_size, num_queries, self.num_heads, self.head_dim])
        q_ca = tf.transpose(q_ca, [0, 2, 1, 3])
        k_ca = tf.reshape(k_ca, [batch_size, num_bev, self.num_heads, self.head_dim])
        k_ca = tf.transpose(k_ca, [0, 2, 1, 3])
        v_ca = tf.reshape(v_ca, [batch_size, num_bev, self.num_heads, self.head_dim])
        v_ca = tf.transpose(v_ca, [0, 2, 1, 3])

        scale = tf.math.rsqrt(tf.cast(self.head_dim, tf.float32))
        cross_attn_weights = tf.matmul(q_ca, k_ca, transpose_b=True) * scale
        cross_attn_weights = tf.nn.softmax(cross_attn_weights, axis=-1)

        cross_attn_out = tf.matmul(cross_attn_weights, v_ca)
        cross_attn_out = tf.transpose(cross_attn_out, [0, 2, 1, 3])
        cross_attn_out = tf.reshape(cross_attn_out, [batch_size, num_queries, self.embed_dims])
        cross_attn_out = self.cross_attn_out(cross_attn_out)
        cross_attn_out = self.cross_attn_dropout(cross_attn_out, training=training)
        query = self.cross_attn_ln(query + cross_attn_out)

        query = self.ffn(query, training=training)

        return query


# =============================================================================
# DETR Decoder
# =============================================================================


class DETRDecoder(keras.layers.Layer):
    """Stacked DETR decoder with learnable object queries.

    Performs iterative refinement of object queries through multiple decoder
    layers, each attending to the BEV feature map.

    Args:
        num_layers: Number of decoder layers.
        embed_dims: Embedding dimension.
        num_heads: Number of attention heads.
        num_queries: Number of learnable object queries.
        ffn_dim: FFN hidden dimension.
        dropout_rate: Dropout probability.
    """

    def __init__(self, num_layers, embed_dims, num_heads, num_queries,
                 ffn_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.embed_dims = embed_dims

        self.layers_list = []
        for i in range(num_layers):
            layer = DETRDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout_rate=dropout_rate,
                name=f"decoder_layer_{i}",
            )
            self.layers_list.append(layer)

        self.layer_norm = layers.LayerNormalization(name="decoder_final_ln")

    def build(self, input_shape):
        """Build learnable query embeddings and positional embeddings."""
        self.query_embedding = self.add_weight(
            name="query_embedding",
            shape=(self.num_queries, self.embed_dims),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.query_pos = self.add_weight(
            name="query_pos_embedding",
            shape=(self.num_queries, self.embed_dims),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, bev_features, training=None):
        """Forward pass through all decoder layers.

        Args:
            bev_features: Encoded BEV features [B, bev_h*bev_w, embed_dims].
            training: Boolean for training mode.

        Returns:
            Decoded query features [B, num_queries, embed_dims].
        """
        batch_size = tf.shape(bev_features)[0]

        query = tf.tile(
            tf.expand_dims(self.query_embedding, 0),
            [batch_size, 1, 1],
        )
        query_pos = tf.tile(
            tf.expand_dims(self.query_pos, 0),
            [batch_size, 1, 1],
        )

        for layer in self.layers_list:
            query = layer(
                query=query,
                bev_features=bev_features,
                query_pos=query_pos,
                training=training,
            )

        return self.layer_norm(query)


# =============================================================================
# Detection Heads
# =============================================================================


class DetectionHead(keras.layers.Layer):
    """Detection head for 3D object classification and bounding box regression.

    Produces per-query class logits (for 10 nuScenes classes) and bounding box
    parameters (cx, cy, cz, w, l, h, sin, cos, vx, vy).

    Args:
        num_classes: Number of object classes.
        code_size: Size of the bounding box code (default 10).
        embed_dims: Input embedding dimension.
    """

    def __init__(self, num_classes, code_size, embed_dims, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.code_size = code_size

        self.cls_branch = keras.Sequential([
            layers.Dense(embed_dims, activation="relu", name="cls_fc1"),
            layers.LayerNormalization(),
            layers.Dense(embed_dims, activation="relu", name="cls_fc2"),
            layers.LayerNormalization(),
            layers.Dense(num_classes, name="cls_logits"),
        ], name="cls_branch")

        self.reg_branch = keras.Sequential([
            layers.Dense(embed_dims, activation="relu", name="reg_fc1"),
            layers.LayerNormalization(),
            layers.Dense(embed_dims, activation="relu", name="reg_fc2"),
            layers.LayerNormalization(),
            layers.Dense(code_size, name="reg_output"),
        ], name="reg_branch")

    def call(self, query_features, training=None):
        """Forward pass through detection heads.

        Args:
            query_features: Decoded query features [B, num_queries, embed_dims].
            training: Boolean for training mode.

        Returns:
            cls_logits: Classification logits [B, num_queries, num_classes].
            bbox_preds: Bounding box predictions [B, num_queries, code_size].
        """
        cls_logits = self.cls_branch(query_features)
        bbox_preds = self.reg_branch(query_features)
        return cls_logits, bbox_preds


# =============================================================================
# Positional Encoding for BEV
# =============================================================================


class LearnableBEVPositionalEncoding(keras.layers.Layer):
    """Learnable 2D positional encoding for BEV queries.

    Generates additive positional embeddings for BEV grid positions using
    separate row and column embeddings that are combined.

    Args:
        bev_h: BEV grid height.
        bev_w: BEV grid width.
        embed_dims: Embedding dimension.
    """

    def __init__(self, bev_h, bev_w, embed_dims, **kwargs):
        super().__init__(**kwargs)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims

    def build(self, input_shape):
        """Build row and column embedding tables."""
        self.row_embed = self.add_weight(
            name="row_embed",
            shape=(self.bev_h, self.embed_dims // 2),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.col_embed = self.add_weight(
            name="col_embed",
            shape=(self.bev_w, self.embed_dims // 2),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, batch_size):
        """Generate positional encoding for the BEV grid.

        Args:
            batch_size: Batch size for tiling.

        Returns:
            Positional encoding tensor [B, bev_h*bev_w, embed_dims].
        """
        row_embed = tf.expand_dims(self.row_embed, axis=1)
        row_embed = tf.tile(row_embed, [1, self.bev_w, 1])

        col_embed = tf.expand_dims(self.col_embed, axis=0)
        col_embed = tf.tile(col_embed, [self.bev_h, 1, 1])

        pos = tf.concat([row_embed, col_embed], axis=-1)
        pos = tf.reshape(pos, [1, self.bev_h * self.bev_w, self.embed_dims])
        pos = tf.tile(pos, [batch_size, 1, 1])
        return pos


# =============================================================================
# Hungarian Matching
# =============================================================================


class HungarianMatcher:
    """Hungarian matching between predictions and ground truth.

    Computes cost matrix using classification and bounding box costs, then
    solves the assignment problem using scipy's linear_sum_assignment.

    Args:
        cost_class: Weight for classification cost.
        cost_bbox: Weight for L1 bounding box cost.
        focal_alpha: Alpha parameter for focal cost weighting.
        focal_gamma: Gamma parameter for focal cost weighting.
    """

    def __init__(self, cost_class=2.0, cost_bbox=5.0, focal_alpha=0.25, focal_gamma=2.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def match(self, cls_logits, bbox_preds, gt_labels, gt_bboxes):
        """Perform Hungarian matching for a single sample.

        Args:
            cls_logits: Predicted class logits [num_queries, num_classes].
            bbox_preds: Predicted bounding boxes [num_queries, code_size].
            gt_labels: Ground truth class labels [num_gt].
            gt_bboxes: Ground truth bounding boxes [num_gt, code_size].

        Returns:
            Tuple of (pred_indices, gt_indices) arrays from the optimal assignment.
        """
        num_queries = cls_logits.shape[0]
        num_gt = gt_labels.shape[0]

        if num_gt == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        cls_probs = tf.nn.sigmoid(cls_logits).numpy()

        gt_labels_np = gt_labels.numpy() if hasattr(gt_labels, 'numpy') else gt_labels
        gt_bboxes_np = gt_bboxes.numpy() if hasattr(gt_bboxes, 'numpy') else gt_bboxes
        bbox_preds_np = bbox_preds.numpy() if hasattr(bbox_preds, 'numpy') else bbox_preds

        pos_cost = self.focal_alpha * ((1 - cls_probs) ** self.focal_gamma) * (
            -np.log(np.clip(cls_probs, 1e-8, 1.0))
        )
        neg_cost = (1 - self.focal_alpha) * (cls_probs ** self.focal_gamma) * (
            -np.log(np.clip(1 - cls_probs, 1e-8, 1.0))
        )

        cls_cost = np.zeros((num_queries, num_gt))
        for j in range(num_gt):
            gt_cls = int(gt_labels_np[j])
            cls_cost[:, j] = pos_cost[:, gt_cls] - neg_cost[:, gt_cls]

        bbox_cost = np.zeros((num_queries, num_gt))
        for j in range(num_gt):
            bbox_cost[:, j] = np.sum(np.abs(bbox_preds_np - gt_bboxes_np[j:j+1]), axis=-1)

        cost_matrix = self.cost_class * cls_cost + self.cost_bbox * bbox_cost

        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        return pred_indices, gt_indices


# =============================================================================
# Loss Functions
# =============================================================================


def focal_loss(logits, targets, num_classes, alpha=0.25, gamma=2.0):
    """Compute focal loss for classification.

    Focal loss reduces the loss contribution from easy examples and focuses
    training on hard negatives.

    Args:
        logits: Predicted logits [N, num_classes].
        targets: Ground truth class indices [N] (integer labels).
        num_classes: Total number of classes.
        alpha: Balancing factor.
        gamma: Focusing parameter.

    Returns:
        Scalar focal loss value.
    """
    one_hot = tf.one_hot(targets, depth=num_classes)
    probs = tf.nn.sigmoid(logits)

    pos_loss = -alpha * ((1 - probs) ** gamma) * tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
    neg_loss = -(1 - alpha) * (probs ** gamma) * tf.math.log(tf.clip_by_value(1 - probs, 1e-8, 1.0))

    loss = one_hot * pos_loss + (1 - one_hot) * neg_loss
    return tf.reduce_sum(loss) / tf.cast(tf.maximum(tf.shape(targets)[0], 1), tf.float32)


def l1_loss(predictions, targets):
    """Compute L1 (smooth) loss for bounding box regression.

    Args:
        predictions: Predicted bounding boxes [N, code_size].
        targets: Ground truth bounding boxes [N, code_size].

    Returns:
        Scalar L1 loss value.
    """
    num_boxes = tf.cast(tf.maximum(tf.shape(predictions)[0], 1), tf.float32)
    return tf.reduce_sum(tf.abs(predictions - targets)) / num_boxes


# =============================================================================
# BEVFormer Model
# =============================================================================


class BEVFormer(keras.Model):
    """BEVFormer: Bird's-Eye-View Transformer for 3D Object Detection.

    End-to-end model that takes multi-camera images and produces 3D object
    detections in bird's-eye-view space. Architecture:
        1. ResNet101 backbone extracts multi-scale image features
        2. FPN unifies feature channel dimensions across scales
        3. BEV Encoder builds a BEV representation via spatial cross-attention
           (image->BEV) and temporal self-attention (previous frame alignment)
        4. DETR Decoder refines object queries using BEV features
        5. Detection heads predict class labels and 3D bounding boxes

    Args:
        config: Dictionary of model configuration parameters.
                See DEFAULT_CONFIG for available options.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or DEFAULT_CONFIG

        self.bev_h = self.config["bev_h"]
        self.bev_w = self.config["bev_w"]
        self.embed_dims = self.config["embed_dims"]
        self.num_cameras = self.config["num_cameras"]
        self.pc_range = self.config["pc_range"]
        self.num_classes = self.config["num_classes"]
        self.code_size = self.config["code_size"]

        self._build_backbone()
        self._build_fpn()
        self._build_bev_encoder()
        self._build_decoder()
        self._build_detection_head()
        self._build_positional_encoding()

        self.matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0)

    def _build_backbone(self):
        """Build ResNet101 backbone with multi-scale feature extraction."""
        base_model = keras.applications.ResNet101(
            include_top=False,
            weights="imagenet",
            input_shape=(None, None, 3),
        )

        layer_names = [
            "conv3_block4_out",
            "conv4_block23_out",
            "conv5_block3_out",
        ]
        outputs = [base_model.get_layer(name).output for name in layer_names]
        self.backbone = keras.Model(inputs=base_model.input, outputs=outputs, name="resnet101_backbone")

    def _build_fpn(self):
        """Build Feature Pyramid Network."""
        self.fpn = FeaturePyramidNetwork(
            in_channels=self.config["fpn_in_channels"],
            out_channels=self.config["fpn_out_channels"],
            num_outs=self.config["fpn_num_outs"],
            name="fpn",
        )

    def _build_bev_encoder(self):
        """Build BEV encoder with stacked transformer layers."""
        self.bev_encoder = BEVEncoder(
            num_layers=self.config["num_encoder_layers"],
            embed_dims=self.embed_dims,
            num_heads=self.config["num_heads"],
            num_points_spatial=self.config["num_points_spatial"],
            num_points_temporal=self.config["num_points_temporal"],
            num_levels=self.config["num_levels"],
            num_cameras=self.num_cameras,
            pc_range=self.pc_range,
            ffn_dim=self.config["ffn_dim"],
            dropout_rate=self.config["dropout_rate"],
            name="bev_encoder",
        )

    def _build_decoder(self):
        """Build DETR-style decoder."""
        self.decoder = DETRDecoder(
            num_layers=self.config["num_decoder_layers"],
            embed_dims=self.embed_dims,
            num_heads=self.config["num_heads"],
            num_queries=self.config["num_queries"],
            ffn_dim=self.config["ffn_dim"],
            dropout_rate=self.config["dropout_rate"],
            name="detr_decoder",
        )

    def _build_detection_head(self):
        """Build classification and regression heads."""
        self.detection_head = DetectionHead(
            num_classes=self.num_classes,
            code_size=self.code_size,
            embed_dims=self.embed_dims,
            name="detection_head",
        )

    def _build_positional_encoding(self):
        """Build learnable BEV positional encoding."""
        self.bev_pos_encoding = LearnableBEVPositionalEncoding(
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            embed_dims=self.embed_dims,
            name="bev_pos_enc",
        )

    def build(self, input_shape):
        """Build learnable BEV query embedding."""
        self.bev_embedding = self.add_weight(
            name="bev_embedding",
            shape=(self.bev_h * self.bev_w, self.embed_dims),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def extract_features(self, images, training=None):
        """Extract multi-scale features from multi-camera images.

        Args:
            images: Multi-camera images [B, num_cameras, H, W, 3].
            training: Boolean for training mode.

        Returns:
            multi_scale_features: List of feature tensors per level,
                                  each [B*num_cameras, H_l, W_l, fpn_out_channels].
            spatial_shapes: List of (H, W) tuples for each level.
        """
        batch_size = tf.shape(images)[0]
        num_cameras = self.num_cameras
        img_h = tf.shape(images)[2]
        img_w = tf.shape(images)[3]

        images_flat = tf.reshape(images, [batch_size * num_cameras, img_h, img_w, 3])

        backbone_features = self.backbone(images_flat, training=training)

        fpn_features = self.fpn(backbone_features, training=training)

        spatial_shapes = []
        for feat in fpn_features:
            h = feat.shape[1] if feat.shape[1] is not None else tf.shape(feat)[1]
            w = feat.shape[2] if feat.shape[2] is not None else tf.shape(feat)[2]
            spatial_shapes.append((h, w))

        return fpn_features, spatial_shapes

    def call(self, inputs, training=None):
        """Forward pass of BEVFormer.

        Args:
            inputs: Dictionary containing:
                - 'images': Multi-camera images [B, num_cameras, H, W, 3].
                - 'lidar2img': Camera projection matrices [B, num_cameras, 4, 4].
                - 'ego_motion': Ego-motion matrix [B, 4, 4] (prev->current).
                - 'prev_bev': Previous BEV features [B, bev_h*bev_w, embed_dims]
                              or None.
            training: Boolean for training mode.

        Returns:
            Dictionary containing:
                - 'cls_logits': Classification logits [B, num_queries, num_classes].
                - 'bbox_preds': Bounding box predictions [B, num_queries, code_size].
                - 'bev_features': Encoded BEV features [B, bev_h*bev_w, embed_dims].
        """
        images = inputs["images"]
        lidar2img = inputs["lidar2img"]
        ego_motion = inputs.get("ego_motion", tf.eye(4, batch_shape=[tf.shape(images)[0]]))
        prev_bev = inputs.get("prev_bev", None)

        batch_size = tf.shape(images)[0]

        multi_scale_features, spatial_shapes = self.extract_features(images, training=training)

        bev_queries = tf.tile(
            tf.expand_dims(self.bev_embedding, 0),
            [batch_size, 1, 1],
        )
        bev_pos = self.bev_pos_encoding(batch_size)
        bev_queries = bev_queries + bev_pos

        bev_features = self.bev_encoder(
            bev_queries=bev_queries,
            prev_bev=prev_bev,
            ego_motion=ego_motion,
            multi_scale_features=multi_scale_features,
            lidar2img=lidar2img,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            spatial_shapes=spatial_shapes,
            training=training,
        )

        query_features = self.decoder(bev_features, training=training)

        cls_logits, bbox_preds = self.detection_head(query_features, training=training)

        return {
            "cls_logits": cls_logits,
            "bbox_preds": bbox_preds,
            "bev_features": bev_features,
        }

    def compute_loss(self, cls_logits, bbox_preds, gt_labels_list, gt_bboxes_list):
        """Compute total training loss with Hungarian matching.

        Performs bipartite matching between predictions and ground truth for
        each sample in the batch, then computes focal loss for classification
        and L1 loss for bounding box regression on matched pairs.

        Args:
            cls_logits: Predicted class logits [B, num_queries, num_classes].
            bbox_preds: Predicted bounding boxes [B, num_queries, code_size].
            gt_labels_list: List of ground truth label tensors, one per sample.
            gt_bboxes_list: List of ground truth bbox tensors, one per sample.

        Returns:
            Dictionary containing:
                - 'total_loss': Combined weighted loss.
                - 'cls_loss': Classification focal loss.
                - 'bbox_loss': Bounding box L1 loss.
        """
        batch_size = cls_logits.shape[0]
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        total_matched = 0

        for i in range(batch_size):
            sample_cls = cls_logits[i].numpy() if hasattr(cls_logits[i], 'numpy') else cls_logits[i]
            sample_bbox = bbox_preds[i].numpy() if hasattr(bbox_preds[i], 'numpy') else bbox_preds[i]
            gt_labels = gt_labels_list[i]
            gt_bboxes = gt_bboxes_list[i]

            pred_indices, gt_indices = self.matcher.match(
                cls_logits[i], bbox_preds[i], gt_labels, gt_bboxes
            )

            num_queries = tf.shape(cls_logits[i])[0]
            bg_label = self.num_classes

            target_classes = tf.fill([num_queries], bg_label)
            if len(pred_indices) > 0:
                pred_idx_tensor = tf.constant(pred_indices, dtype=tf.int32)
                gt_idx_tensor = tf.constant(gt_indices, dtype=tf.int32)

                matched_gt_labels = tf.gather(gt_labels, gt_idx_tensor)
                target_classes = tf.tensor_scatter_nd_update(
                    target_classes,
                    tf.expand_dims(pred_idx_tensor, 1),
                    tf.cast(matched_gt_labels, tf.int32),
                )

                matched_pred_bboxes = tf.gather(bbox_preds[i], pred_idx_tensor)
                matched_gt_bboxes = tf.gather(gt_bboxes, gt_idx_tensor)
                total_bbox_loss += l1_loss(matched_pred_bboxes, matched_gt_bboxes)
                total_matched += len(pred_indices)

            fg_mask = tf.not_equal(target_classes, bg_label)
            all_target_for_focal = tf.where(fg_mask, target_classes, tf.zeros_like(target_classes))
            total_cls_loss += focal_loss(
                cls_logits[i], all_target_for_focal, self.num_classes
            )

        cls_loss = total_cls_loss / tf.cast(batch_size, tf.float32)
        bbox_loss = total_bbox_loss / tf.cast(tf.maximum(total_matched, 1), tf.float32)

        total_loss = 2.0 * cls_loss + 5.0 * bbox_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
        }

    def train_step(self, data):
        """Custom training step with Hungarian matching loss.

        Overrides keras.Model.train_step to implement the bipartite matching
        loss computation required for DETR-style training.

        Args:
            data: Tuple of (inputs, targets) where:
                - inputs: Dictionary with 'images', 'lidar2img', 'ego_motion', 'prev_bev'.
                - targets: Dictionary with 'gt_labels' (list) and 'gt_bboxes' (list).

        Returns:
            Dictionary of metric values for this training step.
        """
        inputs, targets = data
        gt_labels_list = targets["gt_labels"]
        gt_bboxes_list = targets["gt_bboxes"]

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            cls_logits = outputs["cls_logits"]
            bbox_preds = outputs["bbox_preds"]

            losses = self.compute_loss(cls_logits, bbox_preds, gt_labels_list, gt_bboxes_list)
            total_loss = losses["total_loss"]

            if self.losses:
                total_loss += tf.add_n(self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        gradients = [
            tf.clip_by_norm(g, 35.0) if g is not None else g
            for g in gradients
        ]

        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        )

        return {
            "total_loss": losses["total_loss"],
            "cls_loss": losses["cls_loss"],
            "bbox_loss": losses["bbox_loss"],
        }

    def test_step(self, data):
        """Custom evaluation step.

        Args:
            data: Tuple of (inputs, targets).

        Returns:
            Dictionary of metric values for this evaluation step.
        """
        inputs, targets = data
        gt_labels_list = targets["gt_labels"]
        gt_bboxes_list = targets["gt_bboxes"]

        outputs = self(inputs, training=False)
        cls_logits = outputs["cls_logits"]
        bbox_preds = outputs["bbox_preds"]

        losses = self.compute_loss(cls_logits, bbox_preds, gt_labels_list, gt_bboxes_list)

        return {
            "total_loss": losses["total_loss"],
            "cls_loss": losses["cls_loss"],
            "bbox_loss": losses["bbox_loss"],
        }

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 6, None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 6, 4, 4], dtype=tf.float32),
    ])
    def predict_detections(self, images, lidar2img):
        """Run inference and return decoded 3D detections.

        Args:
            images: Multi-camera images [B, 6, H, W, 3].
            lidar2img: Camera projection matrices [B, 6, 4, 4].

        Returns:
            Dictionary with:
                - 'scores': Detection confidence scores [B, num_queries].
                - 'labels': Predicted class labels [B, num_queries].
                - 'boxes': Decoded 3D bounding boxes [B, num_queries, code_size].
        """
        inputs = {
            "images": images,
            "lidar2img": lidar2img,
            "prev_bev": None,
        }
        outputs = self(inputs, training=False)

        cls_logits = outputs["cls_logits"]
        bbox_preds = outputs["bbox_preds"]

        cls_probs = tf.nn.sigmoid(cls_logits)
        scores = tf.reduce_max(cls_probs, axis=-1)
        labels = tf.argmax(cls_probs, axis=-1)

        pc_range = self.pc_range
        x_range = pc_range[3] - pc_range[0]
        y_range = pc_range[4] - pc_range[1]
        z_range = pc_range[5] - pc_range[2]

        decoded_boxes = tf.concat([
            bbox_preds[..., 0:1] * x_range + pc_range[0],
            bbox_preds[..., 1:2] * y_range + pc_range[1],
            bbox_preds[..., 2:3] * z_range + pc_range[2],
            tf.exp(bbox_preds[..., 3:6]),
            bbox_preds[..., 6:8],
            bbox_preds[..., 8:10],
        ], axis=-1)

        return {
            "scores": scores,
            "labels": labels,
            "boxes": decoded_boxes,
        }

    def get_config(self):
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({"config": self.config})
        return config


# =============================================================================
# Factory Function
# =============================================================================


def build_bevformer(config=None):
    """Build and return a BEVFormer model instance.

    Factory function that creates a BEVFormer model with the given configuration
    and compiles it with AdamW optimizer.

    Args:
        config: Optional dictionary of model parameters. Uses DEFAULT_CONFIG
                values for any missing keys.

    Returns:
        Compiled BEVFormer keras.Model instance ready for training.

    Example:
        >>> model = build_bevformer()
        >>> model.summary()
        >>> # For training:
        >>> model.fit(train_dataset, epochs=24)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        full_config = DEFAULT_CONFIG.copy()
        full_config.update(config)
        config = full_config

    model = BEVFormer(config=config)

    optimizer = keras.optimizers.AdamW(
        learning_rate=2e-4,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=35.0,
    )

    model.compile(optimizer=optimizer)

    return model


# =============================================================================
# Utility: NuScenes Class Names
# =============================================================================

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


if __name__ == "__main__":
    print("BEVFormer TensorFlow 2 Implementation")
    print("=" * 50)
    print(f"Number of classes: {len(NUSCENES_CLASSES)}")
    print(f"Classes: {NUSCENES_CLASSES}")
    print(f"Default config: {DEFAULT_CONFIG}")
    print()

    model = build_bevformer()

    batch_size = 1
    img_h, img_w = 256, 448
    dummy_inputs = {
        "images": tf.random.normal([batch_size, 6, img_h, img_w, 3]),
        "lidar2img": tf.random.normal([batch_size, 6, 4, 4]),
        "ego_motion": tf.eye(4, batch_shape=[batch_size]),
        "prev_bev": None,
    }

    print("Running forward pass...")
    outputs = model(dummy_inputs, training=False)
    print(f"cls_logits shape: {outputs['cls_logits'].shape}")
    print(f"bbox_preds shape: {outputs['bbox_preds'].shape}")
    print(f"bev_features shape: {outputs['bev_features'].shape}")
    print("Forward pass complete.")
