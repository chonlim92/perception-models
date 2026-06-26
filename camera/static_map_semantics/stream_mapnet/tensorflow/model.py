"""
StreamMapNet - TensorFlow 2 Implementation

A temporal BEV (Bird's Eye View) network for online HD map construction.
Takes multi-camera images and produces HD map element predictions (lane dividers,
pedestrian crossings, road boundaries) using temporal fusion of BEV features.

Architecture:
  Multi-camera images -> ResNet-50 backbone -> FPN neck -> LSS BEV Transform
  -> Temporal Fusion (ego-motion warping + attention) -> Transformer Decoder
  -> Map Element Heads (classification + polyline regression)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Input dimensions
    "num_cameras": 6,
    "image_height": 224,
    "image_width": 480,
    # Backbone
    "backbone_name": "resnet50",
    "backbone_frozen_stages": 1,
    # FPN
    "fpn_out_channels": 256,
    "fpn_num_outs": 4,
    # LSS BEV Transform
    "depth_channels": 64,        # Number of discrete depth bins
    "depth_min": 1.0,
    "depth_max": 60.0,
    "bev_x_range": (-30.0, 30.0),
    "bev_y_range": (-15.0, 15.0),
    "bev_resolution": 0.3,       # meters per pixel
    "bev_channels": 256,
    # Temporal Fusion
    "temporal_queue_len": 3,     # Number of historical frames to fuse
    "temporal_attn_heads": 8,
    # Transformer Decoder
    "num_queries": 100,          # Number of map element queries
    "decoder_layers": 6,
    "decoder_heads": 8,
    "decoder_ffn_dim": 512,
    "decoder_dropout": 0.1,
    # Map Element Heads
    "num_classes": 3,            # lane_divider, ped_crossing, road_boundary
    "num_points": 20,            # Points per polyline
}


# =============================================================================
# Helper Modules
# =============================================================================

class FeaturePyramidNetwork(layers.Layer):
    """Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, out_channels=256, num_outs=4, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.num_outs = num_outs

    def build(self, input_shape):
        # input_shape is a list of shapes from backbone stages
        num_inputs = len(input_shape)

        # Lateral (1x1) convolutions to reduce channel dimensions
        self.lateral_convs = []
        for i in range(num_inputs):
            self.lateral_convs.append(
                layers.Conv2D(self.out_channels, 1, padding="same",
                              name=f"lateral_conv_{i}")
            )

        # Top-down (3x3) convolutions after feature fusion
        self.fpn_convs = []
        for i in range(num_inputs):
            self.fpn_convs.append(
                layers.Conv2D(self.out_channels, 3, padding="same",
                              name=f"fpn_conv_{i}")
            )

        # Extra output levels (stride 2 conv on last feature)
        self.extra_convs = []
        for i in range(self.num_outs - num_inputs):
            self.extra_convs.append(
                layers.Conv2D(self.out_channels, 3, strides=2, padding="same",
                              name=f"extra_conv_{i}")
            )
        super().build(input_shape)

    def call(self, inputs):
        """
        Args:
            inputs: list of feature maps from backbone stages
                    Each has shape (B, H_i, W_i, C_i)
        Returns:
            list of FPN feature maps, each (B, H_i, W_i, out_channels)
        """
        # Build lateral features
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, inputs)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            h, w = tf.shape(laterals[i])[1], tf.shape(laterals[i])[2]
            upsampled = tf.image.resize(laterals[i + 1], [h, w], method="bilinear")
            laterals[i] = laterals[i] + upsampled

        # Apply 3x3 conv to each merged feature
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Extra levels
        extra_input = outs[-1]
        for conv in self.extra_convs:
            extra_input = conv(tf.nn.relu(extra_input))
            outs.append(extra_input)

        return outs[:self.num_outs]


class DepthNet(layers.Layer):
    """Predicts discrete depth distribution for each spatial location."""

    def __init__(self, in_channels, depth_channels, **kwargs):
        super().__init__(**kwargs)
        self.depth_channels = depth_channels
        self.in_channels = in_channels

    def build(self, input_shape):
        self.reduce_conv = layers.Conv2D(
            self.in_channels // 2, 3, padding="same", name="reduce_conv"
        )
        self.bn = layers.BatchNormalization(name="reduce_bn")
        self.depth_conv = layers.Conv2D(
            self.depth_channels, 1, padding="same", name="depth_conv"
        )
        super().build(input_shape)

    def call(self, x, training=False):
        """
        Args:
            x: (B*N_cam, H_feat, W_feat, C) feature map
        Returns:
            depth_prob: (B*N_cam, H_feat, W_feat, D) depth probability distribution
        """
        x = self.reduce_conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        depth_logits = self.depth_conv(x)  # (B*N, H, W, D)
        depth_prob = tf.nn.softmax(depth_logits, axis=-1)  # (B*N, H, W, D)
        return depth_prob


class LSSBEVTransform(layers.Layer):
    """
    Lift-Splat-Shoot BEV transformation.

    Lifts 2D image features to 3D using predicted depth distributions and
    camera geometry, then splats them into a 2D BEV grid via voxel pooling.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.depth_channels = config["depth_channels"]
        self.depth_min = config["depth_min"]
        self.depth_max = config["depth_max"]
        self.bev_x_range = config["bev_x_range"]
        self.bev_y_range = config["bev_y_range"]
        self.bev_resolution = config["bev_resolution"]
        self.bev_channels = config["bev_channels"]
        self.num_cameras = config["num_cameras"]

        # Compute BEV grid dimensions
        self.bev_w = int((self.bev_x_range[1] - self.bev_x_range[0]) / self.bev_resolution)
        self.bev_h = int((self.bev_y_range[1] - self.bev_y_range[0]) / self.bev_resolution)

        # Depth bin edges (uniform spacing)
        self.depth_bins = tf.cast(
            tf.linspace(self.depth_min, self.depth_max, self.depth_channels),
            tf.float32
        )

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.depth_net = DepthNet(
            in_channels=in_channels,
            depth_channels=self.depth_channels,
            name="depth_net"
        )
        # Reduce feature channels to BEV channels
        self.feat_reduce = layers.Conv2D(
            self.bev_channels, 1, padding="same", name="feat_reduce"
        )
        # BEV feature refinement after pooling
        self.bev_conv = keras.Sequential([
            layers.Conv2D(self.bev_channels, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(self.bev_channels, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name="bev_refine")
        super().build(input_shape)

    def _create_frustum(self, h_feat, w_feat):
        """
        Create a frustum grid of 3D points in camera coordinates.

        Returns:
            frustum: (D, H_feat, W_feat, 3) - (x_cam, y_cam, depth) for each cell
        """
        # Pixel coordinates grid
        xs = tf.cast(tf.linspace(0.0, 1.0, w_feat), tf.float32)
        ys = tf.cast(tf.linspace(0.0, 1.0, h_feat), tf.float32)
        # (H_feat, W_feat) grids
        grid_y, grid_x = tf.meshgrid(ys, xs, indexing="ij")

        # Expand depth dimension: (D, H_feat, W_feat)
        D = self.depth_channels
        grid_x = tf.tile(tf.expand_dims(grid_x, 0), [D, 1, 1])  # (D, H, W)
        grid_y = tf.tile(tf.expand_dims(grid_y, 0), [D, 1, 1])  # (D, H, W)
        depth_vals = tf.reshape(self.depth_bins, [D, 1, 1])
        depth_grid = tf.tile(depth_vals, [1, h_feat, w_feat])    # (D, H, W)

        # Stack to form (D, H, W, 3) with (norm_x, norm_y, depth)
        frustum = tf.stack([grid_x, grid_y, depth_grid], axis=-1)
        return frustum

    def _frustum_to_ego(self, frustum, intrinsics, extrinsics, img_h, img_w):
        """
        Transform frustum points from normalized image coords to ego frame.

        Args:
            frustum: (D, H_feat, W_feat, 3) - (norm_x, norm_y, depth)
            intrinsics: (B, N_cam, 3, 3)
            extrinsics: (B, N_cam, 4, 4) - cam-to-ego
            img_h: original image height
            img_w: original image width
        Returns:
            ego_points: (B, N_cam, D, H_feat, W_feat, 3) points in ego frame
        """
        B = tf.shape(intrinsics)[0]
        N = self.num_cameras
        D = tf.shape(frustum)[0]
        H_feat = tf.shape(frustum)[1]
        W_feat = tf.shape(frustum)[2]

        # De-normalize pixel coordinates to actual pixel space
        # frustum[..., 0] is norm_x in [0,1], frustum[..., 2] is depth
        pixel_x = frustum[..., 0] * tf.cast(img_w, tf.float32)   # (D, H, W)
        pixel_y = frustum[..., 1] * tf.cast(img_h, tf.float32)   # (D, H, W)
        depth = frustum[..., 2]                                    # (D, H, W)

        # Unproject to camera 3D coordinates: p_cam = K^-1 * [u*d, v*d, d]^T
        # Points in camera frame: X = (u - cx) * d / fx, Y = (v - cy) * d / fy, Z = d
        # intrinsics: (B, N, 3, 3)
        fx = intrinsics[:, :, 0, 0]  # (B, N)
        fy = intrinsics[:, :, 1, 1]  # (B, N)
        cx = intrinsics[:, :, 0, 2]  # (B, N)
        cy = intrinsics[:, :, 1, 2]  # (B, N)

        # Reshape for broadcasting: (B, N, 1, 1, 1)
        fx = tf.reshape(fx, [B, N, 1, 1, 1])
        fy = tf.reshape(fy, [B, N, 1, 1, 1])
        cx = tf.reshape(cx, [B, N, 1, 1, 1])
        cy = tf.reshape(cy, [B, N, 1, 1, 1])

        # Expand frustum coords for batch/camera dims: (1, 1, D, H, W)
        pixel_x = tf.reshape(pixel_x, [1, 1, D, H_feat, W_feat])
        pixel_y = tf.reshape(pixel_y, [1, 1, D, H_feat, W_feat])
        depth = tf.reshape(depth, [1, 1, D, H_feat, W_feat])

        # Camera frame coordinates
        x_cam = (pixel_x - cx) * depth / fx  # (B, N, D, H, W)
        y_cam = (pixel_y - cy) * depth / fy  # (B, N, D, H, W)
        z_cam = depth                          # (B, N, D, H, W)

        # Stack to (B, N, D, H, W, 3)
        pts_cam = tf.stack([x_cam, y_cam, z_cam], axis=-1)

        # Transform to ego frame using extrinsics (cam-to-ego)
        # extrinsics: (B, N, 4, 4), rotation (B, N, 3, 3), translation (B, N, 3, 1)
        rot = extrinsics[:, :, :3, :3]    # (B, N, 3, 3)
        trans = extrinsics[:, :, :3, 3]   # (B, N, 3)

        # Reshape for matmul: pts_cam -> (B, N, D*H*W, 3)
        pts_flat = tf.reshape(pts_cam, [B, N, D * H_feat * W_feat, 3])

        # Apply rotation: (B, N, D*H*W, 3) @ (B, N, 3, 3)^T
        pts_ego = tf.einsum("bnpc,bnrc->bnpr", pts_flat, rot)
        # Add translation
        trans_expanded = tf.reshape(trans, [B, N, 1, 3])
        pts_ego = pts_ego + trans_expanded  # (B, N, D*H*W, 3)

        # Reshape back to (B, N, D, H, W, 3)
        pts_ego = tf.reshape(pts_ego, [B, N, D, H_feat, W_feat, 3])
        return pts_ego

    def _voxel_pool(self, features, depth_prob, ego_points, batch_size):
        """
        Pool lifted 3D features into BEV grid using scatter_nd.

        Args:
            features: (B, N_cam, H_feat, W_feat, C) reduced image features
            depth_prob: (B, N_cam, H_feat, W_feat, D) depth probabilities
            ego_points: (B, N_cam, D, H_feat, W_feat, 3) points in ego frame
            batch_size: scalar batch size
        Returns:
            bev_feat: (B, bev_h, bev_w, C) BEV feature map
        """
        B = batch_size
        N = self.num_cameras
        C = tf.shape(features)[-1]
        D = self.depth_channels
        H_feat = tf.shape(features)[2]
        W_feat = tf.shape(features)[3]

        # Compute BEV grid indices from ego coordinates
        # ego_points[..., 0] = x (lateral), ego_points[..., 1] = y (longitudinal)
        x_ego = ego_points[..., 0]  # (B, N, D, H, W)
        y_ego = ego_points[..., 1]  # (B, N, D, H, W)

        # Convert to BEV grid indices
        bev_x_idx = tf.cast(
            (x_ego - self.bev_x_range[0]) / self.bev_resolution, tf.int32
        )  # (B, N, D, H, W)
        bev_y_idx = tf.cast(
            (y_ego - self.bev_y_range[0]) / self.bev_resolution, tf.int32
        )  # (B, N, D, H, W)

        # Valid mask: within BEV bounds
        valid = (
            (bev_x_idx >= 0) & (bev_x_idx < self.bev_w) &
            (bev_y_idx >= 0) & (bev_y_idx < self.bev_h)
        )  # (B, N, D, H, W)

        # Outer product of features and depth to create lifted features
        # features: (B, N, H, W, C), depth_prob: (B, N, H, W, D)
        # lifted: (B, N, D, H, W, C) = depth_prob * features expanded
        depth_prob_t = tf.transpose(depth_prob, [0, 1, 4, 2, 3])  # (B, N, D, H, W)
        features_expanded = tf.expand_dims(
            tf.transpose(features, [0, 1, 4, 2, 3]), 2
        )  # (B, N, C, 1, H, W) -- wrong, let's redo

        # Simpler approach: (B, N, H, W, D, 1) * (B, N, H, W, 1, C) -> (B, N, H, W, D, C)
        depth_expanded = tf.expand_dims(depth_prob, -1)   # (B, N, H, W, D, 1)
        feat_expanded = tf.expand_dims(features, -2)      # (B, N, H, W, 1, C)
        lifted = depth_expanded * feat_expanded           # (B, N, H, W, D, C)

        # Transpose lifted to (B, N, D, H, W, C) to match ego_points layout
        lifted = tf.transpose(lifted, [0, 1, 4, 2, 3, 5])  # (B, N, D, H, W, C)

        # Flatten spatial dimensions for scatter
        # We need batch indices for scatter_nd
        # Create batch index array
        batch_idx = tf.range(B)
        batch_idx = tf.reshape(batch_idx, [B, 1, 1, 1, 1])
        batch_idx = tf.broadcast_to(batch_idx, [B, N, D, H_feat, W_feat])

        # Flatten everything
        batch_flat = tf.reshape(batch_idx, [-1])     # (B*N*D*H*W,)
        bev_y_flat = tf.reshape(bev_y_idx, [-1])     # (B*N*D*H*W,)
        bev_x_flat = tf.reshape(bev_x_idx, [-1])     # (B*N*D*H*W,)
        valid_flat = tf.reshape(valid, [-1])          # (B*N*D*H*W,)
        lifted_flat = tf.reshape(lifted, [-1, C])    # (B*N*D*H*W, C)

        # Filter valid points
        valid_indices = tf.where(valid_flat)[:, 0]
        batch_valid = tf.gather(batch_flat, valid_indices)
        bev_y_valid = tf.gather(bev_y_flat, valid_indices)
        bev_x_valid = tf.gather(bev_x_flat, valid_indices)
        lifted_valid = tf.gather(lifted_flat, valid_indices)

        # Construct scatter indices: (num_valid, 3) -> [batch, bev_y, bev_x]
        scatter_indices = tf.stack(
            [batch_valid, bev_y_valid, bev_x_valid], axis=-1
        )  # (num_valid, 3)

        # Scatter into BEV grid
        bev_feat = tf.scatter_nd(
            indices=scatter_indices,
            updates=lifted_valid,
            shape=[B, self.bev_h, self.bev_w, C]
        )  # (B, bev_h, bev_w, C)

        return bev_feat

    def call(self, features, intrinsics, extrinsics, training=False):
        """
        Args:
            features: (B*N_cam, H_feat, W_feat, C) image features
            intrinsics: (B, N_cam, 3, 3) camera intrinsic matrices
            extrinsics: (B, N_cam, 4, 4) camera extrinsic matrices (cam-to-ego)
        Returns:
            bev_feat: (B, bev_h, bev_w, bev_channels) BEV feature map
        """
        B = tf.shape(intrinsics)[0]
        N = self.num_cameras
        H_feat = tf.shape(features)[1]
        W_feat = tf.shape(features)[2]
        C_in = tf.shape(features)[-1]

        # Predict depth distribution
        depth_prob = self.depth_net(features, training=training)  # (B*N, H, W, D)

        # Reduce feature channels
        feat_reduced = self.feat_reduce(features)  # (B*N, H, W, bev_channels)

        # Reshape to (B, N, H, W, ...)
        depth_prob = tf.reshape(depth_prob, [B, N, H_feat, W_feat, self.depth_channels])
        feat_reduced = tf.reshape(feat_reduced, [B, N, H_feat, W_feat, self.bev_channels])

        # Create frustum and project to ego frame
        frustum = self._create_frustum(H_feat, W_feat)  # (D, H, W, 3)
        img_h = self.config["image_height"]
        img_w = self.config["image_width"]
        ego_points = self._frustum_to_ego(
            frustum, intrinsics, extrinsics, img_h, img_w
        )  # (B, N, D, H, W, 3)

        # Voxel pooling
        bev_feat = self._voxel_pool(
            feat_reduced, depth_prob, ego_points, B
        )  # (B, bev_h, bev_w, bev_channels)

        # Refine BEV features
        bev_feat = self.bev_conv(bev_feat, training=training)

        return bev_feat


# =============================================================================
# Temporal Fusion
# =============================================================================

class GridSample2D(layers.Layer):
    """
    Custom grid sampling implementation for TensorFlow.
    Equivalent to torch.nn.functional.grid_sample with bilinear interpolation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, features, grid):
        """
        Bilinear sampling of features at grid locations.

        Args:
            features: (B, H, W, C) input feature map
            grid: (B, H_out, W_out, 2) sampling grid with values in [-1, 1]
                  grid[..., 0] = x (width), grid[..., 1] = y (height)
        Returns:
            sampled: (B, H_out, W_out, C) sampled features
        """
        B = tf.shape(features)[0]
        H = tf.cast(tf.shape(features)[1], tf.float32)
        W = tf.cast(tf.shape(features)[2], tf.float32)
        C = tf.shape(features)[3]
        H_out = tf.shape(grid)[1]
        W_out = tf.shape(grid)[2]

        # Convert from [-1, 1] to pixel coordinates
        # x: [-1, 1] -> [0, W-1], y: [-1, 1] -> [0, H-1]
        grid_x = (grid[..., 0] + 1.0) * 0.5 * (W - 1.0)  # (B, H_out, W_out)
        grid_y = (grid[..., 1] + 1.0) * 0.5 * (H - 1.0)  # (B, H_out, W_out)

        # Get corner pixel indices for bilinear interpolation
        x0 = tf.floor(grid_x)
        x1 = x0 + 1.0
        y0 = tf.floor(grid_y)
        y1 = y0 + 1.0

        # Compute interpolation weights
        wa = (x1 - grid_x) * (y1 - grid_y)  # (B, H_out, W_out)
        wb = (x1 - grid_x) * (grid_y - y0)
        wc = (grid_x - x0) * (y1 - grid_y)
        wd = (grid_x - x0) * (grid_y - y0)

        # Clamp coordinates
        x0 = tf.clip_by_value(x0, 0.0, W - 1.0)
        x1 = tf.clip_by_value(x1, 0.0, W - 1.0)
        y0 = tf.clip_by_value(y0, 0.0, H - 1.0)
        y1 = tf.clip_by_value(y1, 0.0, H - 1.0)

        # Convert to int for gather
        x0i = tf.cast(x0, tf.int32)
        x1i = tf.cast(x1, tf.int32)
        y0i = tf.cast(y0, tf.int32)
        y1i = tf.cast(y1, tf.int32)

        # Gather values at four corners
        # Create batch indices
        batch_idx = tf.range(B)
        batch_idx = tf.reshape(batch_idx, [B, 1, 1])
        batch_idx = tf.broadcast_to(batch_idx, [B, H_out, W_out])

        def gather_pixels(yi, xi):
            indices = tf.stack([batch_idx, yi, xi], axis=-1)  # (B, H_out, W_out, 3)
            return tf.gather_nd(features, indices)  # (B, H_out, W_out, C)

        Ia = gather_pixels(y0i, x0i)
        Ib = gather_pixels(y1i, x0i)
        Ic = gather_pixels(y0i, x1i)
        Id = gather_pixels(y1i, x1i)

        # Expand weights for channel dimension
        wa = tf.expand_dims(wa, -1)  # (B, H_out, W_out, 1)
        wb = tf.expand_dims(wb, -1)
        wc = tf.expand_dims(wc, -1)
        wd = tf.expand_dims(wd, -1)

        # Bilinear interpolation
        sampled = wa * Ia + wb * Ib + wc * Ic + wd * Id  # (B, H_out, W_out, C)
        return sampled


class TemporalFusion(layers.Layer):
    """
    Temporal fusion module that warps historical BEV features to current frame
    using ego-motion and fuses them with attention.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.bev_channels = config["bev_channels"]
        self.num_heads = config["temporal_attn_heads"]
        self.bev_x_range = config["bev_x_range"]
        self.bev_y_range = config["bev_y_range"]
        self.bev_resolution = config["bev_resolution"]
        self.bev_w = int((self.bev_x_range[1] - self.bev_x_range[0]) / self.bev_resolution)
        self.bev_h = int((self.bev_y_range[1] - self.bev_y_range[0]) / self.bev_resolution)

    def build(self, input_shape):
        self.grid_sample = GridSample2D(name="grid_sample")

        # Attention-based fusion
        self.query_proj = layers.Dense(self.bev_channels, name="attn_query_proj")
        self.key_proj = layers.Dense(self.bev_channels, name="attn_key_proj")
        self.value_proj = layers.Dense(self.bev_channels, name="attn_value_proj")
        self.out_proj = layers.Dense(self.bev_channels, name="attn_out_proj")
        self.layer_norm = layers.LayerNormalization(name="fusion_ln")
        self.ffn = keras.Sequential([
            layers.Dense(self.bev_channels * 2, activation="relu"),
            layers.Dense(self.bev_channels),
        ], name="fusion_ffn")
        self.ffn_norm = layers.LayerNormalization(name="ffn_ln")
        super().build(input_shape)

    def _warp_bev(self, prev_bev, ego_motion):
        """
        Warp previous BEV features to current frame using ego-motion.

        Args:
            prev_bev: (B, bev_h, bev_w, C) previous BEV features
            ego_motion: (B, 4, 4) ego motion matrix (prev-to-current)
        Returns:
            warped_bev: (B, bev_h, bev_w, C) warped BEV features
        """
        B = tf.shape(prev_bev)[0]
        H = self.bev_h
        W = self.bev_w

        # Create BEV coordinate grid in ego frame (x, y in meters)
        xs = tf.linspace(
            self.bev_x_range[0] + self.bev_resolution / 2,
            self.bev_x_range[1] - self.bev_resolution / 2,
            W
        )
        ys = tf.linspace(
            self.bev_y_range[0] + self.bev_resolution / 2,
            self.bev_y_range[1] - self.bev_resolution / 2,
            H
        )
        grid_y, grid_x = tf.meshgrid(ys, xs, indexing="ij")  # (H, W)

        # Create homogeneous coordinates: (H*W, 4) with z=0
        ones = tf.ones_like(grid_x)
        zeros = tf.zeros_like(grid_x)
        # Points in current frame (x, y, 0, 1)
        pts_current = tf.stack(
            [tf.reshape(grid_x, [-1]),
             tf.reshape(grid_y, [-1]),
             tf.reshape(zeros, [-1]),
             tf.reshape(ones, [-1])],
            axis=-1
        )  # (H*W, 4)

        # Transform current BEV points to previous frame
        # ego_motion maps prev -> current, so we need inverse for current -> prev
        ego_motion_inv = tf.linalg.inv(ego_motion)  # (B, 4, 4)

        # (B, 4, 4) @ (H*W, 4)^T -> (B, 4, H*W)
        pts_current_expanded = tf.expand_dims(pts_current, 0)  # (1, H*W, 4)
        pts_current_expanded = tf.tile(pts_current_expanded, [B, 1, 1])  # (B, H*W, 4)
        pts_prev = tf.einsum("bij,bnj->bni", ego_motion_inv, pts_current_expanded)
        # (B, H*W, 4)

        # Extract x, y in previous frame
        pts_prev_x = pts_prev[:, :, 0]  # (B, H*W)
        pts_prev_y = pts_prev[:, :, 1]  # (B, H*W)

        # Normalize to [-1, 1] for grid_sample
        norm_x = (pts_prev_x - self.bev_x_range[0]) / (
            self.bev_x_range[1] - self.bev_x_range[0]
        ) * 2.0 - 1.0  # (B, H*W)
        norm_y = (pts_prev_y - self.bev_y_range[0]) / (
            self.bev_y_range[1] - self.bev_y_range[0]
        ) * 2.0 - 1.0  # (B, H*W)

        # Reshape to grid format: (B, H, W, 2)
        sample_grid = tf.stack([norm_x, norm_y], axis=-1)  # (B, H*W, 2)
        sample_grid = tf.reshape(sample_grid, [B, H, W, 2])

        # Apply grid sampling
        warped_bev = self.grid_sample(prev_bev, sample_grid)  # (B, H, W, C)
        return warped_bev

    def call(self, current_bev, history_bevs, ego_motions, training=False):
        """
        Fuse current BEV with warped historical BEV features using attention.

        Args:
            current_bev: (B, bev_h, bev_w, C) current frame BEV features
            history_bevs: list of (B, bev_h, bev_w, C) historical BEV features
            ego_motions: list of (B, 4, 4) ego motions (each prev_t -> current)
        Returns:
            fused_bev: (B, bev_h, bev_w, C) fused BEV features
        """
        if len(history_bevs) == 0:
            return current_bev

        B = tf.shape(current_bev)[0]
        H = self.bev_h
        W = self.bev_w
        C = self.bev_channels

        # Warp all historical BEV features to current frame
        warped_bevs = []
        for prev_bev, ego_motion in zip(history_bevs, ego_motions):
            warped = self._warp_bev(prev_bev, ego_motion)
            warped_bevs.append(warped)

        # Stack warped features: (B, T, H*W, C)
        warped_stack = tf.stack(warped_bevs, axis=1)  # (B, T, H, W, C)
        T = len(warped_bevs)
        warped_flat = tf.reshape(warped_stack, [B, T * H * W, C])

        # Current BEV as query: (B, H*W, C)
        current_flat = tf.reshape(current_bev, [B, H * W, C])

        # Multi-head cross-attention: current attends to warped history
        Q = self.query_proj(current_flat)    # (B, H*W, C)
        K = self.key_proj(warped_flat)       # (B, T*H*W, C)
        V = self.value_proj(warped_flat)     # (B, T*H*W, C)

        # Reshape for multi-head attention
        head_dim = C // self.num_heads
        Q = tf.reshape(Q, [B, H * W, self.num_heads, head_dim])
        Q = tf.transpose(Q, [0, 2, 1, 3])  # (B, heads, H*W, head_dim)
        K = tf.reshape(K, [B, T * H * W, self.num_heads, head_dim])
        K = tf.transpose(K, [0, 2, 1, 3])  # (B, heads, T*H*W, head_dim)
        V = tf.reshape(V, [B, T * H * W, self.num_heads, head_dim])
        V = tf.transpose(V, [0, 2, 1, 3])  # (B, heads, T*H*W, head_dim)

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(head_dim, tf.float32))
        attn_weights = tf.matmul(Q, K, transpose_b=True) / scale  # (B, heads, H*W, T*H*W)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        attn_out = tf.matmul(attn_weights, V)  # (B, heads, H*W, head_dim)
        attn_out = tf.transpose(attn_out, [0, 2, 1, 3])  # (B, H*W, heads, head_dim)
        attn_out = tf.reshape(attn_out, [B, H * W, C])

        attn_out = self.out_proj(attn_out)  # (B, H*W, C)

        # Residual connection + layer norm
        fused_flat = self.layer_norm(current_flat + attn_out)

        # FFN with residual
        fused_flat = self.ffn_norm(fused_flat + self.ffn(fused_flat, training=training))

        # Reshape back to spatial
        fused_bev = tf.reshape(fused_flat, [B, H, W, C])
        return fused_bev


# =============================================================================
# Transformer Decoder
# =============================================================================

class TransformerDecoderLayer(layers.Layer):
    """Single transformer decoder layer with self-attention and cross-attention."""

    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        # Self-attention
        self.self_attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name="self_attn"
        )
        self.self_attn_norm = layers.LayerNormalization(name="self_attn_norm")
        self.self_attn_dropout = layers.Dropout(self.dropout_rate)

        # Cross-attention to BEV features
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name="cross_attn"
        )
        self.cross_attn_norm = layers.LayerNormalization(name="cross_attn_norm")
        self.cross_attn_dropout = layers.Dropout(self.dropout_rate)

        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(self.ffn_dim, activation="relu"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.d_model),
            layers.Dropout(self.dropout_rate),
        ], name="ffn")
        self.ffn_norm = layers.LayerNormalization(name="ffn_norm")
        super().build(input_shape)

    def call(self, queries, bev_features, training=False):
        """
        Args:
            queries: (B, N_queries, d_model) map element queries
            bev_features: (B, H*W, d_model) flattened BEV features
        Returns:
            queries: (B, N_queries, d_model) updated queries
        """
        # Self-attention among queries
        q_norm = self.self_attn_norm(queries)
        self_attn_out = self.self_attn(
            query=q_norm, value=q_norm, key=q_norm, training=training
        )
        queries = queries + self.self_attn_dropout(self_attn_out, training=training)

        # Cross-attention: queries attend to BEV features
        q_norm = self.cross_attn_norm(queries)
        cross_attn_out = self.cross_attn(
            query=q_norm, value=bev_features, key=bev_features, training=training
        )
        queries = queries + self.cross_attn_dropout(cross_attn_out, training=training)

        # FFN
        q_norm = self.ffn_norm(queries)
        ffn_out = self.ffn(q_norm, training=training)
        queries = queries + ffn_out

        return queries


class MapTransformerDecoder(layers.Layer):
    """Transformer decoder for map element prediction."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.d_model = config["bev_channels"]
        self.num_queries = config["num_queries"]
        self.num_layers = config["decoder_layers"]
        self.num_heads = config["decoder_heads"]
        self.ffn_dim = config["decoder_ffn_dim"]
        self.dropout = config["decoder_dropout"]

    def build(self, input_shape):
        # Learnable map element queries
        self.query_embed = self.add_weight(
            name="query_embed",
            shape=(self.num_queries, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Learnable positional embedding for queries
        self.query_pos = self.add_weight(
            name="query_pos",
            shape=(self.num_queries, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

        # BEV positional encoding (learned)
        bev_h = int(
            (self.config["bev_y_range"][1] - self.config["bev_y_range"][0])
            / self.config["bev_resolution"]
        )
        bev_w = int(
            (self.config["bev_x_range"][1] - self.config["bev_x_range"][0])
            / self.config["bev_resolution"]
        )
        self.bev_pos = self.add_weight(
            name="bev_pos",
            shape=(bev_h * bev_w, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
                name=f"decoder_layer_{i}"
            )
            for i in range(self.num_layers)
        ]

        # Final layer norm
        self.final_norm = layers.LayerNormalization(name="final_norm")
        super().build(input_shape)

    def call(self, bev_features, training=False):
        """
        Args:
            bev_features: (B, bev_h, bev_w, C) BEV feature map
        Returns:
            decoded_queries: (B, N_queries, d_model)
        """
        B = tf.shape(bev_features)[0]
        H = tf.shape(bev_features)[1]
        W = tf.shape(bev_features)[2]

        # Flatten BEV features to sequence: (B, H*W, C)
        bev_flat = tf.reshape(bev_features, [B, H * W, self.d_model])

        # Add positional encoding to BEV features
        bev_flat = bev_flat + tf.expand_dims(self.bev_pos, 0)  # (B, H*W, C)

        # Initialize queries: (B, N_queries, d_model)
        queries = tf.expand_dims(self.query_embed, 0)  # (1, N, C)
        queries = tf.tile(queries, [B, 1, 1])          # (B, N, C)

        # Add positional embedding to queries
        query_pos = tf.expand_dims(self.query_pos, 0)  # (1, N, C)
        queries = queries + query_pos                   # (B, N, C)

        # Apply decoder layers
        for dec_layer in self.decoder_layers:
            queries = dec_layer(queries, bev_flat, training=training)

        queries = self.final_norm(queries)  # (B, N_queries, d_model)
        return queries


# =============================================================================
# Map Element Prediction Heads
# =============================================================================

class MapElementHeads(layers.Layer):
    """Classification and point regression heads for map elements."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = config["num_classes"]
        self.num_points = config["num_points"]
        self.d_model = config["bev_channels"]

    def build(self, input_shape):
        # Classification head: predicts class logits (including background)
        self.cls_head = keras.Sequential([
            layers.Dense(self.d_model, activation="relu"),
            layers.LayerNormalization(),
            layers.Dense(self.d_model // 2, activation="relu"),
            layers.Dense(self.num_classes + 1),  # +1 for background/no-object
        ], name="cls_head")

        # Point regression head: predicts polyline points
        self.pts_head = keras.Sequential([
            layers.Dense(self.d_model, activation="relu"),
            layers.LayerNormalization(),
            layers.Dense(self.d_model, activation="relu"),
            layers.Dense(self.num_points * 2),  # (x, y) for each point
        ], name="pts_head")
        super().build(input_shape)

    def call(self, queries, training=False):
        """
        Args:
            queries: (B, N_queries, d_model) decoded map element queries
        Returns:
            logits: (B, N_queries, num_classes+1) classification logits
            points: (B, N_queries, num_points, 2) predicted polyline points
        """
        B = tf.shape(queries)[0]
        N = tf.shape(queries)[1]

        # Classification
        logits = self.cls_head(queries)  # (B, N, num_classes+1)

        # Point regression
        pts_raw = self.pts_head(queries)  # (B, N, num_points*2)
        points = tf.reshape(pts_raw, [B, N, self.num_points, 2])
        # Sigmoid to normalize points to [0, 1] range
        points = tf.sigmoid(points)  # (B, N, K, 2)

        return logits, points


# =============================================================================
# Main StreamMapNet Model
# =============================================================================

class StreamMapNet(keras.Model):
    """
    StreamMapNet: Temporal BEV network for online HD map construction.

    Takes multi-camera images and produces HD map element predictions using
    temporal fusion of BEV features.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or DEFAULT_CONFIG.copy()

        # Build components
        self._build_backbone()
        self.fpn = FeaturePyramidNetwork(
            out_channels=self.config["fpn_out_channels"],
            num_outs=self.config["fpn_num_outs"],
            name="fpn"
        )
        self.bev_transform = LSSBEVTransform(self.config, name="lss_bev")
        self.temporal_fusion = TemporalFusion(self.config, name="temporal_fusion")
        self.decoder = MapTransformerDecoder(self.config, name="map_decoder")
        self.heads = MapElementHeads(self.config, name="map_heads")

        # Temporal state: stores previous BEV features and cumulative ego motions
        self._bev_history = []
        self._ego_motion_history = []
        self._temporal_queue_len = self.config["temporal_queue_len"]

    def _build_backbone(self):
        """Build ResNet-50 backbone with multi-stage feature extraction."""
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(
                self.config["image_height"],
                self.config["image_width"],
                3
            ),
        )

        # Extract features from multiple stages (C2, C3, C4, C5)
        # ResNet50 layer names for stage outputs:
        stage_outputs = [
            "conv2_block3_out",   # C2: stride 4,  64 channels  -> H/4, W/4
            "conv3_block4_out",   # C3: stride 8,  128 channels -> H/8, W/8
            "conv4_block6_out",   # C4: stride 16, 256 channels -> H/16, W/16
            "conv5_block3_out",   # C5: stride 32, 512 channels -> H/32, W/32
        ]

        outputs = [base_model.get_layer(name).output for name in stage_outputs]
        self.backbone = keras.Model(
            inputs=base_model.input,
            outputs=outputs,
            name="resnet50_backbone"
        )

        # Optionally freeze early stages
        frozen_stages = self.config.get("backbone_frozen_stages", 1)
        if frozen_stages >= 1:
            # Freeze all layers up to conv2 (first residual stage)
            for layer in self.backbone.layers:
                if "conv1" in layer.name or "bn_conv1" in layer.name:
                    layer.trainable = False
                if frozen_stages >= 2 and "conv2" in layer.name:
                    layer.trainable = False

    def reset_temporal_state(self):
        """Reset temporal BEV feature history. Call at sequence boundaries."""
        self._bev_history = []
        self._ego_motion_history = []

    def _extract_features(self, images, training=False):
        """
        Extract multi-scale features from all camera images.

        Args:
            images: (B, N_cam, H, W, 3) multi-camera images
        Returns:
            features: (B*N_cam, H_feat, W_feat, C) selected FPN features
        """
        B = tf.shape(images)[0]
        N = self.config["num_cameras"]
        H = self.config["image_height"]
        W = self.config["image_width"]

        # Reshape to process all cameras together: (B*N, H, W, 3)
        imgs_flat = tf.reshape(images, [B * N, H, W, 3])

        # Backbone: returns list of [C2, C3, C4, C5]
        multi_scale_feats = self.backbone(imgs_flat, training=training)

        # FPN: combine multi-scale features
        fpn_feats = self.fpn(multi_scale_feats, training=training)

        # Use the second FPN level (P3) as primary feature for BEV transform
        # P3 is a good balance of resolution and semantic content
        # Shape: (B*N, H/8, W/8, fpn_out_channels)
        selected_feat = fpn_feats[1]

        return selected_feat

    def call(self, inputs, training=False):
        """
        Forward pass of StreamMapNet.

        Args:
            inputs: dict with keys:
                - "images": (B, 6, H, W, 3) multi-camera images
                - "intrinsics": (B, 6, 3, 3) camera intrinsic matrices
                - "extrinsics": (B, 6, 4, 4) camera extrinsic matrices (cam-to-ego)
                - "ego_motion": (B, 4, 4) ego motion from previous to current frame
        Returns:
            dict with keys:
                - "logits": (B, N_queries, num_classes+1) classification logits
                - "points": (B, N_queries, num_points, 2) predicted polyline points
        """
        images = inputs["images"]         # (B, 6, H, W, 3)
        intrinsics = inputs["intrinsics"] # (B, 6, 3, 3)
        extrinsics = inputs["extrinsics"] # (B, 6, 4, 4)
        ego_motion = inputs["ego_motion"] # (B, 4, 4)

        # Step 1: Extract multi-scale image features
        # features: (B*N_cam, H_feat, W_feat, fpn_out_channels)
        features = self._extract_features(images, training=training)

        # Step 2: LSS BEV Transform
        # bev_feat: (B, bev_h, bev_w, bev_channels)
        bev_feat = self.bev_transform(
            features, intrinsics, extrinsics, training=training
        )

        # Step 3: Temporal Fusion
        # Fuse with historical BEV features
        fused_bev = self.temporal_fusion(
            bev_feat, self._bev_history, self._ego_motion_history,
            training=training
        )

        # Update temporal state (only during inference or when not training)
        # During training, temporal state management is handled externally
        if not training:
            self._update_temporal_state(bev_feat, ego_motion)

        # Step 4: Transformer Decoder
        # decoded_queries: (B, N_queries, d_model)
        decoded_queries = self.decoder(fused_bev, training=training)

        # Step 5: Prediction Heads
        # logits: (B, N_queries, num_classes+1)
        # points: (B, N_queries, num_points, 2)
        logits, points = self.heads(decoded_queries, training=training)

        return {"logits": logits, "points": points}

    def _update_temporal_state(self, current_bev, ego_motion):
        """
        Update the temporal BEV feature history.

        Maintains a queue of historical BEV features and the cumulative
        ego motions needed to warp each historical frame to the current frame.

        Args:
            current_bev: (B, bev_h, bev_w, C) current BEV features
            ego_motion: (B, 4, 4) ego motion (prev -> current)
        """
        # Update ego motions for existing history entries
        # Each historical ego_motion needs to be composed with the new one
        updated_ego_motions = []
        for hist_ego in self._ego_motion_history:
            # Compose: new cumulative = current_ego @ historical_ego
            composed = tf.matmul(ego_motion, hist_ego)
            updated_ego_motions.append(composed)
        self._ego_motion_history = updated_ego_motions

        # Add current BEV to history with identity ego motion
        # (it will be warped by future ego motions)
        B = tf.shape(current_bev)[0]
        identity = tf.eye(4, batch_shape=[B])
        self._bev_history.append(current_bev)
        self._ego_motion_history.append(identity)

        # Trim to queue length
        if len(self._bev_history) > self._temporal_queue_len:
            self._bev_history = self._bev_history[-self._temporal_queue_len:]
            self._ego_motion_history = self._ego_motion_history[-self._temporal_queue_len:]

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config
        return config


# =============================================================================
# Main: Verification
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("StreamMapNet TensorFlow 2 Implementation - Shape Verification")
    print("=" * 70)

    # Use default config
    config = DEFAULT_CONFIG.copy()
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create model
    print("\nBuilding model...")
    model = StreamMapNet(config)

    # Create dummy inputs
    B = 2  # batch size
    N = config["num_cameras"]
    H = config["image_height"]
    W = config["image_width"]

    dummy_inputs = {
        "images": tf.random.normal([B, N, H, W, 3]),
        "intrinsics": tf.eye(3, batch_shape=[B, N]),
        "extrinsics": tf.eye(4, batch_shape=[B, N]),
        "ego_motion": tf.eye(4, batch_shape=[B]),
    }

    # Set realistic intrinsics (focal length ~500, principal point at center)
    fx, fy = 500.0, 500.0
    cx, cy = W / 2.0, H / 2.0
    intrinsics_np = np.zeros((B, N, 3, 3), dtype=np.float32)
    intrinsics_np[:, :, 0, 0] = fx
    intrinsics_np[:, :, 1, 1] = fy
    intrinsics_np[:, :, 0, 2] = cx
    intrinsics_np[:, :, 1, 2] = cy
    intrinsics_np[:, :, 2, 2] = 1.0
    dummy_inputs["intrinsics"] = tf.constant(intrinsics_np)

    # Set extrinsics with small translations to simulate camera positions
    extrinsics_np = np.zeros((B, N, 4, 4), dtype=np.float32)
    for i in range(N):
        extrinsics_np[:, i] = np.eye(4)
        angle = (2 * np.pi * i) / N
        extrinsics_np[:, i, 0, 3] = 1.5 * np.cos(angle)  # x offset
        extrinsics_np[:, i, 1, 3] = 1.5 * np.sin(angle)  # y offset
        extrinsics_np[:, i, 2, 3] = 1.5                   # height
    dummy_inputs["extrinsics"] = tf.constant(extrinsics_np)

    print(f"\nInput shapes:")
    print(f"  images:     {dummy_inputs['images'].shape}")
    print(f"  intrinsics: {dummy_inputs['intrinsics'].shape}")
    print(f"  extrinsics: {dummy_inputs['extrinsics'].shape}")
    print(f"  ego_motion: {dummy_inputs['ego_motion'].shape}")

    # First forward pass (no temporal history)
    print("\n--- Forward pass 1 (no temporal history) ---")
    model.reset_temporal_state()
    outputs = model(dummy_inputs, training=False)

    logits = outputs["logits"]
    points = outputs["points"]
    print(f"  logits shape: {logits.shape}  (expected: [{B}, {config['num_queries']}, {config['num_classes'] + 1}])")
    print(f"  points shape: {points.shape}  (expected: [{B}, {config['num_queries']}, {config['num_points']}, 2])")

    assert logits.shape == (B, config["num_queries"], config["num_classes"] + 1), \
        f"Logits shape mismatch: {logits.shape}"
    assert points.shape == (B, config["num_queries"], config["num_points"], 2), \
        f"Points shape mismatch: {points.shape}"

    # Second forward pass (with temporal history from first pass)
    print("\n--- Forward pass 2 (with temporal history) ---")
    # Simulate small ego motion (slight forward movement)
    ego_np = np.eye(4, dtype=np.float32)
    ego_np[0, 3] = 0.5  # 0.5m forward
    dummy_inputs["ego_motion"] = tf.constant(
        np.tile(ego_np, (B, 1, 1))
    )
    outputs2 = model(dummy_inputs, training=False)
    logits2 = outputs2["logits"]
    points2 = outputs2["points"]
    print(f"  logits shape: {logits2.shape}  (expected: [{B}, {config['num_queries']}, {config['num_classes'] + 1}])")
    print(f"  points shape: {points2.shape}  (expected: [{B}, {config['num_queries']}, {config['num_points']}, 2])")

    assert logits2.shape == (B, config["num_queries"], config["num_classes"] + 1)
    assert points2.shape == (B, config["num_queries"], config["num_points"], 2)

    # Third forward pass (queue length test)
    print("\n--- Forward pass 3 (building temporal queue) ---")
    outputs3 = model(dummy_inputs, training=False)
    print(f"  Temporal history length: {len(model._bev_history)}")
    print(f"  logits shape: {outputs3['logits'].shape}")
    print(f"  points shape: {outputs3['points'].shape}")

    # Verify point values are in [0, 1] (sigmoid output)
    assert tf.reduce_all(points >= 0.0) and tf.reduce_all(points <= 1.0), \
        "Points should be normalized to [0, 1]"
    print("\n  Point values verified in [0, 1] range.")

    # Print model summary
    print("\n--- Model Summary ---")
    print(f"  Total parameters: {model.count_params():,}")

    # BEV grid dimensions
    bev_h = int((config["bev_y_range"][1] - config["bev_y_range"][0]) / config["bev_resolution"])
    bev_w = int((config["bev_x_range"][1] - config["bev_x_range"][0]) / config["bev_resolution"])
    print(f"  BEV grid size: {bev_h} x {bev_w}")
    print(f"  Num map queries: {config['num_queries']}")
    print(f"  Num classes: {config['num_classes']} (+ background)")
    print(f"  Points per polyline: {config['num_points']}")

    # Reset and verify
    model.reset_temporal_state()
    assert len(model._bev_history) == 0, "History should be empty after reset"
    print("\n  Temporal state reset verified.")

    print("\n" + "=" * 70)
    print("All shape checks PASSED.")
    print("=" * 70)
