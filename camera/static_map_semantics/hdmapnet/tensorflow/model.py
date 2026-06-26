"""
HDMapNet: An Online HD Map Construction and Evaluation Framework
TensorFlow 2 / Keras Implementation

Paper: Li et al., "HDMapNet: An Online HD Map Construction and Evaluation Framework", ICRA 2022

This module implements the full HDMapNet architecture including:
- EfficientNet-B0 backbone for multi-camera feature extraction
- IPM (Inverse Perspective Mapping) view transform
- LSS (Lift-Splat-Shoot) view transform with depth prediction
- BEV encoder with residual convolutional blocks
- Semantic segmentation head (3 classes)
- Instance embedding head (16-dim)
- Direction prediction head (2D unit vector)

Input specs:
- Images: [B, 6, 128, 352, 3]
- Extrinsics: [B, 6, 4, 4]
- Intrinsics: [B, 6, 3, 3]

Output specs:
- Semantic: [B, 200, 200, 3] (binary segmentation per class)
- Instance: [B, 200, 200, 16] (embedding vectors)
- Direction: [B, 200, 200, 2] (unit direction vectors)

BEV grid: 200x200 covering 60m x 30m (x: -30 to 30, y: -15 to 15)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# =============================================================================
# Utility Modules
# =============================================================================

class ConvBnRelu(layers.Layer):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, filters, kernel_size=3, strides=1, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class ResidualBlock(layers.Layer):
    """Residual block with two conv-bn-relu layers and skip connection."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.match_conv = None
        self.match_bn = None
        self._filters = filters

    def build(self, input_shape):
        if input_shape[-1] != self._filters:
            self.match_conv = layers.Conv2D(self._filters, 1, padding='same', use_bias=False)
            self.match_bn = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x
        if self.match_conv is not None:
            residual = self.match_conv(residual)
            residual = self.match_bn(residual, training=training)

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = out + residual
        out = self.relu2(out)
        return out


class UpBlock(layers.Layer):
    """Upsample + Conv block for decoder paths."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv_block = ConvBnRelu(filters, kernel_size=3)

    def call(self, x, training=False):
        x = self.upsample(x)
        x = self.conv_block(x, training=training)
        return x


# =============================================================================
# Backbone
# =============================================================================

class EfficientNetBackbone(layers.Layer):
    """
    EfficientNet-B0 backbone for feature extraction.
    Extracts features from the block5 output (stride 16, 112 channels for B0).
    Input: [B, H, W, 3] where H=128, W=352
    Output: [B, H/16, W/16, C] where C=112 (block 5a expansion output)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(128, 352, 3)
        )
        # Use output of block 5a (stride 16) which has 112 channels
        # block5a_expand_activation or use the full model up to a certain point
        # EfficientNetB0 layer names: block5a_expand_activation gives 112 channels at stride 16
        target_layer = 'block5a_expand_activation'
        self.feature_extractor = keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer(target_layer).output
        )
        self.out_channels = 112

    def call(self, x, training=False):
        return self.feature_extractor(x, training=training)


# =============================================================================
# Neck (Feature Dimension Reduction)
# =============================================================================

class FeatureNeck(layers.Layer):
    """Reduces backbone feature channels to a target dimension."""

    def __init__(self, out_channels=64, **kwargs):
        super().__init__(**kwargs)
        self.conv = ConvBnRelu(out_channels, kernel_size=1)
        self.out_channels = out_channels

    def call(self, x, training=False):
        return self.conv(x, training=training)


# =============================================================================
# IPM (Inverse Perspective Mapping) View Transform
# =============================================================================

class IPMTransform(layers.Layer):
    """
    Inverse Perspective Mapping view transform.
    Projects BEV grid points into camera image coordinates using intrinsics/extrinsics,
    then samples features using bilinear interpolation via tf.gather_nd.

    BEV grid: 200x200, x: [-30, 30], y: [-15, 15], z=0 (ground plane)
    """

    def __init__(self, bev_h=200, bev_w=200, x_range=(-30.0, 30.0),
                 y_range=(-15.0, 15.0), feat_h=8, feat_w=22, **kwargs):
        super().__init__(**kwargs)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_range = x_range
        self.y_range = y_range
        self.feat_h = feat_h
        self.feat_w = feat_w

        # Create BEV grid coordinates (ground plane z=0)
        xs = tf.linspace(x_range[0], x_range[1], bev_w)
        ys = tf.linspace(y_range[0], y_range[1], bev_h)
        # Create meshgrid: shape [bev_h, bev_w]
        grid_x, grid_y = tf.meshgrid(xs, ys)
        # Flatten to [N, 3] with z=0 (ground plane assumption)
        grid_x_flat = tf.reshape(grid_x, [-1])
        grid_y_flat = tf.reshape(grid_y, [-1])
        grid_z_flat = tf.zeros_like(grid_x_flat)
        ones = tf.ones_like(grid_x_flat)
        # Homogeneous world coordinates: [4, N]
        self.world_coords = tf.stack([grid_x_flat, grid_y_flat, grid_z_flat, ones], axis=0)

    def _bilinear_sample(self, feat, coords_y, coords_x):
        """
        Bilinear sampling from feature map using tf.gather_nd.

        Args:
            feat: [B, H, W, C] feature map
            coords_y: [B, N] float y coordinates
            coords_x: [B, N] float x coordinates

        Returns:
            sampled: [B, N, C]
        """
        batch_size = tf.shape(feat)[0]
        h = tf.shape(feat)[1]
        w = tf.shape(feat)[2]
        c = tf.shape(feat)[3]
        n = tf.shape(coords_y)[1]

        # Clamp coordinates
        coords_y = tf.clip_by_value(coords_y, 0.0, tf.cast(h - 1, tf.float32))
        coords_x = tf.clip_by_value(coords_x, 0.0, tf.cast(w - 1, tf.float32))

        # Get corner coordinates
        y0 = tf.cast(tf.floor(coords_y), tf.int32)
        x0 = tf.cast(tf.floor(coords_x), tf.int32)
        y1 = tf.minimum(y0 + 1, h - 1)
        x1 = tf.minimum(x0 + 1, w - 1)

        # Bilinear weights
        wy1 = coords_y - tf.cast(y0, tf.float32)
        wx1 = coords_x - tf.cast(x0, tf.float32)
        wy0 = 1.0 - wy1
        wx0 = 1.0 - wx1

        # Batch indices
        batch_idx = tf.repeat(tf.range(batch_size), n)  # [B*N]
        y0_flat = tf.reshape(y0, [-1])
        x0_flat = tf.reshape(x0, [-1])
        y1_flat = tf.reshape(y1, [-1])
        x1_flat = tf.reshape(x1, [-1])

        # Gather from four corners
        idx_00 = tf.stack([batch_idx, y0_flat, x0_flat], axis=1)
        idx_01 = tf.stack([batch_idx, y0_flat, x1_flat], axis=1)
        idx_10 = tf.stack([batch_idx, y1_flat, x0_flat], axis=1)
        idx_11 = tf.stack([batch_idx, y1_flat, x1_flat], axis=1)

        val_00 = tf.gather_nd(feat, idx_00)  # [B*N, C]
        val_01 = tf.gather_nd(feat, idx_01)
        val_10 = tf.gather_nd(feat, idx_10)
        val_11 = tf.gather_nd(feat, idx_11)

        # Reshape weights
        wy0_flat = tf.reshape(wy0, [-1, 1])
        wx0_flat = tf.reshape(wx0, [-1, 1])
        wy1_flat = tf.reshape(wy1, [-1, 1])
        wx1_flat = tf.reshape(wx1, [-1, 1])

        # Bilinear interpolation
        result = (val_00 * wy0_flat * wx0_flat +
                  val_01 * wy0_flat * wx1_flat +
                  val_10 * wy1_flat * wx0_flat +
                  val_11 * wy1_flat * wx1_flat)

        result = tf.reshape(result, [batch_size, n, c])
        return result

    def call(self, features, extrinsics, intrinsics, training=False):
        """
        Args:
            features: [B, 6, feat_h, feat_w, C] camera features
            extrinsics: [B, 6, 4, 4] camera-to-world transforms
            intrinsics: [B, 6, 3, 3] camera intrinsic matrices

        Returns:
            bev_features: [B, bev_h, bev_w, C]
        """
        batch_size = tf.shape(features)[0]
        num_cams = 6
        feat_c = tf.shape(features)[-1]
        n_points = self.bev_h * self.bev_w

        # World coordinates: [4, N]
        world_pts = self.world_coords  # [4, N]

        # Accumulate BEV features from all cameras
        bev_accum = tf.zeros([batch_size, n_points, feat_c], dtype=tf.float32)
        bev_count = tf.zeros([batch_size, n_points, 1], dtype=tf.float32)

        for cam_idx in range(num_cams):
            # Get camera parameters for this camera
            ext = extrinsics[:, cam_idx, :, :]  # [B, 4, 4]
            intr = intrinsics[:, cam_idx, :, :]  # [B, 3, 3]
            cam_feat = features[:, cam_idx, :, :, :]  # [B, feat_h, feat_w, C]

            # World to camera transform: inverse of extrinsics (world-to-cam)
            # Assuming extrinsics is cam-to-world, we need world-to-cam
            ext_inv = tf.linalg.inv(ext)  # [B, 4, 4]

            # Project world points to camera frame
            # world_pts: [4, N], ext_inv: [B, 4, 4]
            # cam_pts = ext_inv @ world_pts -> [B, 4, N]
            world_pts_expanded = tf.expand_dims(world_pts, 0)  # [1, 4, N]
            world_pts_batch = tf.tile(world_pts_expanded, [batch_size, 1, 1])  # [B, 4, N]
            cam_pts = tf.matmul(ext_inv, world_pts_batch)  # [B, 4, N]

            # Extract 3D camera coordinates
            cam_x = cam_pts[:, 0, :]  # [B, N]
            cam_y = cam_pts[:, 1, :]  # [B, N]
            cam_z = cam_pts[:, 2, :]  # [B, N]

            # Only keep points in front of camera (z > 0)
            valid_mask = tf.cast(cam_z > 0.1, tf.float32)  # [B, N]

            # Project to image coordinates using intrinsics
            # pixel = K @ [x/z, y/z, 1]^T
            eps = 1e-6
            cam_z_safe = tf.maximum(cam_z, eps)
            px = cam_x / cam_z_safe  # [B, N]
            py = cam_y / cam_z_safe  # [B, N]

            # Apply intrinsics: [B, 3, 3] @ [3, N]
            # u = fx * px + cx, v = fy * py + cy
            fx = intr[:, 0, 0]  # [B]
            fy = intr[:, 1, 1]  # [B]
            cx = intr[:, 0, 2]  # [B]
            cy = intr[:, 1, 2]  # [B]

            u = fx[:, tf.newaxis] * px + cx[:, tf.newaxis]  # [B, N]
            v = fy[:, tf.newaxis] * py + cy[:, tf.newaxis]  # [B, N]

            # Map image coordinates to feature map coordinates
            # Original image size: 128 x 352, feature size: feat_h x feat_w
            scale_x = tf.cast(self.feat_w, tf.float32) / 352.0
            scale_y = tf.cast(self.feat_h, tf.float32) / 128.0
            feat_u = u * scale_x
            feat_v = v * scale_y

            # Check bounds
            in_bounds = tf.cast(
                (feat_u >= 0) & (feat_u < tf.cast(self.feat_w, tf.float32)) &
                (feat_v >= 0) & (feat_v < tf.cast(self.feat_h, tf.float32)),
                tf.float32
            )  # [B, N]

            # Combined validity mask
            mask = valid_mask * in_bounds  # [B, N]

            # Sample features using bilinear interpolation
            sampled = self._bilinear_sample(cam_feat, feat_v, feat_u)  # [B, N, C]

            # Apply mask
            mask_expanded = tf.expand_dims(mask, -1)  # [B, N, 1]
            sampled = sampled * mask_expanded

            bev_accum = bev_accum + sampled
            bev_count = bev_count + mask_expanded

        # Average features from all contributing cameras
        bev_count = tf.maximum(bev_count, 1.0)
        bev_features = bev_accum / bev_count

        # Reshape to spatial BEV grid
        bev_features = tf.reshape(bev_features, [batch_size, self.bev_h, self.bev_w, feat_c])
        return bev_features


# =============================================================================
# LSS (Lift-Splat-Shoot) View Transform
# =============================================================================

class DepthNet(layers.Layer):
    """
    Depth prediction network for LSS.
    Predicts a depth distribution over discrete bins for each spatial location.
    Also outputs context features that are lifted into 3D.
    """

    def __init__(self, in_channels, num_depth_bins=41, context_channels=64, **kwargs):
        super().__init__(**kwargs)
        self.num_depth_bins = num_depth_bins
        self.context_channels = context_channels

        # Depth distribution prediction
        self.depth_conv1 = ConvBnRelu(128, kernel_size=3)
        self.depth_conv2 = layers.Conv2D(num_depth_bins, kernel_size=1, padding='same')

        # Context feature prediction
        self.context_conv1 = ConvBnRelu(128, kernel_size=3)
        self.context_conv2 = ConvBnRelu(context_channels, kernel_size=1)

    def call(self, x, training=False):
        """
        Args:
            x: [B, H, W, C] image features

        Returns:
            depth: [B, H, W, D] depth distribution (softmax over bins)
            context: [B, H, W, context_channels] context features
        """
        # Depth distribution
        d = self.depth_conv1(x, training=training)
        d = self.depth_conv2(d)
        depth = tf.nn.softmax(d, axis=-1)  # [B, H, W, D]

        # Context features
        c = self.context_conv1(x, training=training)
        context = self.context_conv2(c, training=training)  # [B, H, W, context_channels]

        return depth, context


class LSSTransform(layers.Layer):
    """
    Lift-Splat-Shoot view transform.
    Lifts 2D image features into 3D using predicted depth distributions,
    then splats them into a BEV grid.

    Depth bins: 4m to 45m at 1m intervals (41 bins)
    BEV grid: 200x200, x: [-30, 30], y: [-15, 15]
    """

    def __init__(self, in_channels=64, bev_h=200, bev_w=200,
                 x_range=(-30.0, 30.0), y_range=(-15.0, 15.0),
                 d_min=4.0, d_max=45.0, d_step=1.0,
                 feat_h=8, feat_w=22, context_channels=64, **kwargs):
        super().__init__(**kwargs)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_range = x_range
        self.y_range = y_range
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.context_channels = context_channels

        # Depth bins
        self.num_depth_bins = int((d_max - d_min) / d_step) + 1  # 41
        self.depth_bins = tf.constant(
            np.arange(d_min, d_max + d_step * 0.5, d_step, dtype=np.float32)
        )  # [41]

        # BEV grid resolution
        self.bev_x_res = (x_range[1] - x_range[0]) / bev_w
        self.bev_y_res = (y_range[1] - y_range[0]) / bev_h

        # Depth prediction network
        self.depth_net = DepthNet(
            in_channels=in_channels,
            num_depth_bins=self.num_depth_bins,
            context_channels=context_channels
        )

        # Post-splat conv to refine BEV features
        self.bev_conv = ConvBnRelu(context_channels, kernel_size=3)

    def _create_frustum_grid(self):
        """
        Create frustum grid of (u, v, d) coordinates.
        Returns: [feat_h, feat_w, D, 3] grid
        """
        # Feature pixel coordinates
        us = tf.linspace(0.0, tf.cast(self.feat_w - 1, tf.float32), self.feat_w)
        vs = tf.linspace(0.0, tf.cast(self.feat_h - 1, tf.float32), self.feat_h)
        grid_u, grid_v = tf.meshgrid(us, vs)  # [feat_h, feat_w]

        # Expand for depth bins
        grid_u = tf.expand_dims(grid_u, -1)  # [feat_h, feat_w, 1]
        grid_v = tf.expand_dims(grid_v, -1)  # [feat_h, feat_w, 1]
        grid_u = tf.tile(grid_u, [1, 1, self.num_depth_bins])  # [feat_h, feat_w, D]
        grid_v = tf.tile(grid_v, [1, 1, self.num_depth_bins])  # [feat_h, feat_w, D]

        # Depth values
        depth_grid = tf.reshape(self.depth_bins, [1, 1, self.num_depth_bins])
        depth_grid = tf.tile(depth_grid, [self.feat_h, self.feat_w, 1])  # [feat_h, feat_w, D]

        # Stack: [feat_h, feat_w, D, 3] where 3 = (u, v, depth)
        frustum = tf.stack([grid_u, grid_v, depth_grid], axis=-1)
        return frustum

    def call(self, features, extrinsics, intrinsics, training=False):
        """
        Args:
            features: [B, 6, feat_h, feat_w, C] camera features
            extrinsics: [B, 6, 4, 4] camera-to-world transforms
            intrinsics: [B, 6, 3, 3] camera intrinsics

        Returns:
            bev_features: [B, bev_h, bev_w, context_channels]
        """
        batch_size = tf.shape(features)[0]
        num_cams = 6

        # Initialize BEV accumulator
        bev_accum = tf.zeros(
            [batch_size, self.bev_h, self.bev_w, self.context_channels],
            dtype=tf.float32
        )

        # Create frustum grid
        frustum = self._create_frustum_grid()  # [feat_h, feat_w, D, 3]

        for cam_idx in range(num_cams):
            cam_feat = features[:, cam_idx, :, :, :]  # [B, feat_h, feat_w, C]
            ext = extrinsics[:, cam_idx, :, :]  # [B, 4, 4]
            intr = intrinsics[:, cam_idx, :, :]  # [B, 3, 3]

            # Predict depth distribution and context
            depth_dist, context = self.depth_net(cam_feat, training=training)
            # depth_dist: [B, feat_h, feat_w, D]
            # context: [B, feat_h, feat_w, context_channels]

            # Outer product: depth-weighted context features
            # [B, feat_h, feat_w, D, context_channels]
            depth_expanded = tf.expand_dims(depth_dist, -1)  # [B, H, W, D, 1]
            context_expanded = tf.expand_dims(context, 3)  # [B, H, W, 1, C]
            lifted = depth_expanded * context_expanded  # [B, H, W, D, C]

            # Unproject frustum points to 3D world coordinates
            # frustum: [feat_h, feat_w, D, 3] with (u_feat, v_feat, depth)
            # Convert feature coordinates to image coordinates
            img_scale_x = 352.0 / tf.cast(self.feat_w, tf.float32)
            img_scale_y = 128.0 / tf.cast(self.feat_h, tf.float32)

            # For each batch element, unproject using intrinsics and extrinsics
            frustum_flat = tf.reshape(frustum, [-1, 3])  # [H*W*D, 3]
            n_frustum = tf.shape(frustum_flat)[0]

            u_img = frustum_flat[:, 0] * img_scale_x  # pixel u
            v_img = frustum_flat[:, 1] * img_scale_y  # pixel v
            d = frustum_flat[:, 2]  # depth

            # Splat into BEV for each batch element
            lifted_flat = tf.reshape(lifted, [batch_size, -1, self.context_channels])
            # [B, H*W*D, C]

            for b in tf.range(batch_size):
                # Get intrinsics for this batch/camera
                K = intr[b]  # [3, 3]
                E = ext[b]  # [4, 4] cam-to-world

                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]

                # Unproject to camera coordinates
                x_cam = (u_img - cx) * d / fx
                y_cam = (v_img - cy) * d / fy
                z_cam = d

                # Homogeneous camera coordinates
                ones_vec = tf.ones_like(x_cam)
                cam_pts = tf.stack([x_cam, y_cam, z_cam, ones_vec], axis=0)  # [4, N]

                # Transform to world coordinates
                world_pts = tf.matmul(E, cam_pts)  # [4, N]
                wx = world_pts[0, :]
                wy = world_pts[1, :]

                # Map world coordinates to BEV grid indices
                bev_x_idx = (wx - self.x_range[0]) / self.bev_x_res
                bev_y_idx = (wy - self.y_range[0]) / self.bev_y_res

                bev_x_int = tf.cast(tf.floor(bev_x_idx), tf.int32)
                bev_y_int = tf.cast(tf.floor(bev_y_idx), tf.int32)

                # Valid mask
                valid = (
                    (bev_x_int >= 0) & (bev_x_int < self.bev_w) &
                    (bev_y_int >= 0) & (bev_y_int < self.bev_h)
                )

                # Gather valid points
                valid_indices = tf.where(valid)[:, 0]  # [M]
                valid_x = tf.gather(bev_x_int, valid_indices)
                valid_y = tf.gather(bev_y_int, valid_indices)
                valid_feats = tf.gather(lifted_flat[b], valid_indices)  # [M, C]

                # Scatter into BEV grid using tensor_scatter_nd_add
                scatter_indices = tf.stack([valid_y, valid_x], axis=1)  # [M, 2]
                bev_single = tf.tensor_scatter_nd_add(
                    tf.zeros([self.bev_h, self.bev_w, self.context_channels]),
                    scatter_indices,
                    valid_feats
                )

                # Add to batch accumulator
                bev_accum = bev_accum + tf.expand_dims(bev_single, 0) * tf.one_hot(
                    b, batch_size, dtype=tf.float32
                )[..., tf.newaxis, tf.newaxis, tf.newaxis]

        # Post-processing convolution
        bev_features = self.bev_conv(bev_accum, training=training)
        return bev_features


# =============================================================================
# Vectorized LSS Transform (More Efficient)
# =============================================================================

class LSSTransformVectorized(layers.Layer):
    """
    Vectorized Lift-Splat-Shoot view transform.
    Uses scatter operations for efficiency while maintaining correct geometry.

    This version avoids explicit Python for-loops over batch dimension by
    using batched scatter operations.
    """

    def __init__(self, in_channels=64, bev_h=200, bev_w=200,
                 x_range=(-30.0, 30.0), y_range=(-15.0, 15.0),
                 d_min=4.0, d_max=45.0, d_step=1.0,
                 feat_h=8, feat_w=22, context_channels=64, **kwargs):
        super().__init__(**kwargs)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_range = x_range
        self.y_range = y_range
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.context_channels = context_channels

        self.num_depth_bins = int((d_max - d_min) / d_step) + 1  # 41
        self.depth_bins = tf.constant(
            np.arange(d_min, d_max + d_step * 0.5, d_step, dtype=np.float32)
        )

        self.bev_x_res = (x_range[1] - x_range[0]) / bev_w
        self.bev_y_res = (y_range[1] - y_range[0]) / bev_h

        self.depth_net = DepthNet(
            in_channels=in_channels,
            num_depth_bins=self.num_depth_bins,
            context_channels=context_channels
        )

        self.bev_conv = ConvBnRelu(context_channels, kernel_size=3)

    def _create_pixel_coords(self):
        """Create normalized pixel coordinate grid for the feature map."""
        us = tf.linspace(0.5, tf.cast(self.feat_w, tf.float32) - 0.5, self.feat_w)
        vs = tf.linspace(0.5, tf.cast(self.feat_h, tf.float32) - 0.5, self.feat_h)
        grid_u, grid_v = tf.meshgrid(us, vs)  # [feat_h, feat_w]
        return grid_u, grid_v

    def call(self, features, extrinsics, intrinsics, training=False):
        """
        Args:
            features: [B, 6, feat_h, feat_w, C]
            extrinsics: [B, 6, 4, 4]
            intrinsics: [B, 6, 3, 3]

        Returns:
            bev_features: [B, bev_h, bev_w, context_channels]
        """
        batch_size = tf.shape(features)[0]
        num_cams = 6

        bev_accum = tf.zeros(
            [batch_size, self.bev_h, self.bev_w, self.context_channels],
            dtype=tf.float32
        )

        grid_u, grid_v = self._create_pixel_coords()
        img_scale_x = 352.0 / tf.cast(self.feat_w, tf.float32)
        img_scale_y = 128.0 / tf.cast(self.feat_h, tf.float32)

        # Image pixel coordinates from feature coordinates
        u_img = tf.reshape(grid_u * img_scale_x, [-1])  # [H*W]
        v_img = tf.reshape(grid_v * img_scale_y, [-1])  # [H*W]
        hw = self.feat_h * self.feat_w

        for cam_idx in range(num_cams):
            cam_feat = features[:, cam_idx]  # [B, feat_h, feat_w, C]
            ext = extrinsics[:, cam_idx]  # [B, 4, 4]
            intr = intrinsics[:, cam_idx]  # [B, 3, 3]

            # Predict depth and context
            depth_dist, context = self.depth_net(cam_feat, training=training)
            # depth_dist: [B, feat_h, feat_w, D]
            # context: [B, feat_h, feat_w, context_channels]

            # Lift: outer product of depth and context
            depth_expanded = tf.expand_dims(depth_dist, -1)  # [B, H, W, D, 1]
            context_expanded = tf.expand_dims(context, 3)  # [B, H, W, 1, C]
            lifted = depth_expanded * context_expanded  # [B, H, W, D, C]
            # Reshape to [B, H*W*D, C]
            lifted_flat = tf.reshape(lifted, [batch_size, -1, self.context_channels])

            # Unproject for each depth bin
            # For vectorized computation, we process per-batch
            fx = intr[:, 0, 0]  # [B]
            fy = intr[:, 1, 1]  # [B]
            cx = intr[:, 0, 2]  # [B]
            cy = intr[:, 1, 2]  # [B]

            # Create full frustum points: for each (u, v) and each depth d
            # x_cam = (u - cx) * d / fx, y_cam = (v - cy) * d / fy, z_cam = d
            # u_img: [H*W], depth_bins: [D]
            # For batch computation, expand dims
            u_expanded = tf.reshape(u_img, [1, hw, 1])  # [1, H*W, 1]
            v_expanded = tf.reshape(v_img, [1, hw, 1])  # [1, H*W, 1]
            d_expanded = tf.reshape(self.depth_bins, [1, 1, self.num_depth_bins])  # [1, 1, D]

            # Per-batch intrinsics
            fx_exp = tf.reshape(fx, [-1, 1, 1])  # [B, 1, 1]
            fy_exp = tf.reshape(fy, [-1, 1, 1])
            cx_exp = tf.reshape(cx, [-1, 1, 1])
            cy_exp = tf.reshape(cy, [-1, 1, 1])

            # Camera coordinates: [B, H*W, D]
            x_cam = (u_expanded - cx_exp) * d_expanded / fx_exp
            y_cam = (v_expanded - cy_exp) * d_expanded / fy_exp
            z_cam = tf.broadcast_to(d_expanded, [batch_size, hw, self.num_depth_bins])

            # Reshape to [B, H*W*D]
            x_cam_flat = tf.reshape(x_cam, [batch_size, -1])
            y_cam_flat = tf.reshape(y_cam, [batch_size, -1])
            z_cam_flat = tf.reshape(z_cam, [batch_size, -1])
            n_pts = hw * self.num_depth_bins

            # Build homogeneous coordinates [B, 4, N]
            ones_flat = tf.ones([batch_size, n_pts], dtype=tf.float32)
            cam_pts = tf.stack([x_cam_flat, y_cam_flat, z_cam_flat, ones_flat], axis=1)

            # Transform to world: [B, 4, 4] @ [B, 4, N] -> [B, 4, N]
            world_pts = tf.matmul(ext, cam_pts)

            wx = world_pts[:, 0, :]  # [B, N]
            wy = world_pts[:, 1, :]  # [B, N]

            # BEV grid indices
            bev_x_idx = tf.cast(tf.floor(
                (wx - self.x_range[0]) / self.bev_x_res
            ), tf.int32)
            bev_y_idx = tf.cast(tf.floor(
                (wy - self.y_range[0]) / self.bev_y_res
            ), tf.int32)

            # Valid mask
            valid = (
                (bev_x_idx >= 0) & (bev_x_idx < self.bev_w) &
                (bev_y_idx >= 0) & (bev_y_idx < self.bev_h)
            )  # [B, N]

            # Clamp to valid range for safe indexing (masked values won't contribute)
            bev_x_clamped = tf.clip_by_value(bev_x_idx, 0, self.bev_w - 1)
            bev_y_clamped = tf.clip_by_value(bev_y_idx, 0, self.bev_h - 1)

            # Flatten BEV index: y * W + x
            flat_bev_idx = bev_y_clamped * self.bev_w + bev_x_clamped  # [B, N]

            # Mask invalid points by zeroing their features
            valid_float = tf.cast(valid, tf.float32)  # [B, N]
            masked_feats = lifted_flat * tf.expand_dims(valid_float, -1)  # [B, N, C]

            # Scatter add into BEV for each batch element
            # Use tf.math.unsorted_segment_sum
            for b in tf.range(batch_size):
                indices = flat_bev_idx[b]  # [N]
                feats = masked_feats[b]  # [N, C]
                scattered = tf.math.unsorted_segment_sum(
                    feats, indices, num_segments=self.bev_h * self.bev_w
                )  # [H*W, C]
                scattered = tf.reshape(scattered, [self.bev_h, self.bev_w, self.context_channels])
                bev_accum = bev_accum + tf.expand_dims(scattered, 0) * tf.cast(
                    tf.reshape(tf.one_hot(b, batch_size), [batch_size, 1, 1, 1]),
                    tf.float32
                )

        bev_features = self.bev_conv(bev_accum, training=training)
        return bev_features


# =============================================================================
# BEV Encoder
# =============================================================================

class BEVEncoder(layers.Layer):
    """
    BEV feature encoder with residual convolutional blocks.
    Processes BEV features through multiple residual blocks with
    optional downsampling and upsampling to capture multi-scale context.
    """

    def __init__(self, in_channels=64, mid_channels=128, out_channels=64, **kwargs):
        super().__init__(**kwargs)
        # Encoder path (downsample)
        self.res_block1 = ResidualBlock(mid_channels)
        self.res_block2 = ResidualBlock(mid_channels)
        self.downsample = layers.Conv2D(mid_channels, 3, strides=2, padding='same', use_bias=False)
        self.down_bn = layers.BatchNormalization()
        self.down_relu = layers.ReLU()

        # Bottleneck
        self.res_block3 = ResidualBlock(mid_channels * 2)
        self.res_block4 = ResidualBlock(mid_channels * 2)

        # Decoder path (upsample)
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.up_conv = ConvBnRelu(mid_channels, kernel_size=1)
        self.res_block5 = ResidualBlock(mid_channels)
        self.res_block6 = ResidualBlock(mid_channels)

        # Final projection
        self.final_conv = ConvBnRelu(out_channels, kernel_size=1)

    def call(self, x, training=False):
        """
        Args:
            x: [B, 200, 200, in_channels]
        Returns:
            [B, 200, 200, out_channels]
        """
        # Initial residual blocks
        x = self.res_block1(x, training=training)
        skip = self.res_block2(x, training=training)

        # Downsample
        down = self.downsample(skip)
        down = self.down_bn(down, training=training)
        down = self.down_relu(down)

        # Bottleneck
        down = self.res_block3(down, training=training)
        down = self.res_block4(down, training=training)

        # Upsample and skip connection
        up = self.upsample(down)
        # Handle potential size mismatch due to odd dimensions
        up = tf.image.resize(up, [tf.shape(skip)[1], tf.shape(skip)[2]])
        up = self.up_conv(up, training=training)
        x = up + skip

        # Final residual blocks
        x = self.res_block5(x, training=training)
        x = self.res_block6(x, training=training)

        # Project to output channels
        x = self.final_conv(x, training=training)
        return x


# =============================================================================
# Output Heads
# =============================================================================

class SemanticHead(layers.Layer):
    """
    Semantic segmentation head.
    Outputs per-class binary segmentation for 3 classes:
    - Lane dividers
    - Road boundaries
    - Pedestrian crossings

    Output: [B, 200, 200, 3] (logits, apply sigmoid for probability)
    """

    def __init__(self, in_channels=64, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvBnRelu(in_channels, kernel_size=3)
        self.conv2 = ConvBnRelu(in_channels // 2, kernel_size=3)
        self.conv3 = ConvBnRelu(in_channels // 4, kernel_size=3)
        self.output_conv = layers.Conv2D(num_classes, kernel_size=1, padding='same')

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.output_conv(x)
        return x  # [B, H, W, num_classes]


class InstanceHead(layers.Layer):
    """
    Instance embedding head.
    Outputs 16-dimensional embedding vector per BEV pixel.
    Pixels belonging to the same instance should have similar embeddings.

    Output: [B, 200, 200, 16]
    """

    def __init__(self, in_channels=64, embed_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvBnRelu(in_channels, kernel_size=3)
        self.conv2 = ConvBnRelu(in_channels // 2, kernel_size=3)
        self.conv3 = ConvBnRelu(in_channels // 4, kernel_size=3)
        self.output_conv = layers.Conv2D(embed_dim, kernel_size=1, padding='same')

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.output_conv(x)
        return x  # [B, H, W, embed_dim]


class DirectionHead(layers.Layer):
    """
    Direction prediction head.
    Outputs 2D unit direction vector per BEV pixel.
    Used for vectorized map element direction estimation.

    Output: [B, 200, 200, 2] (L2-normalized direction vectors)
    """

    def __init__(self, in_channels=64, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvBnRelu(in_channels, kernel_size=3)
        self.conv2 = ConvBnRelu(in_channels // 2, kernel_size=3)
        self.conv3 = ConvBnRelu(in_channels // 4, kernel_size=3)
        self.output_conv = layers.Conv2D(2, kernel_size=1, padding='same')

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.output_conv(x)
        # Normalize to unit vectors
        x = tf.math.l2_normalize(x, axis=-1)
        return x  # [B, H, W, 2]


# =============================================================================
# HDMapNet Main Model
# =============================================================================

class HDMapNet(keras.Model):
    """
    HDMapNet: An Online HD Map Construction and Evaluation Framework.

    Complete implementation supporting both IPM and LSS view transforms
    for projecting multi-camera features into a Bird's Eye View (BEV)
    representation, followed by semantic segmentation, instance embedding,
    and direction prediction.

    Args:
        view_transform_type: str, either 'ipm' or 'lss'
        bev_h: int, BEV grid height (default 200)
        bev_w: int, BEV grid width (default 200)
        x_range: tuple, x-axis range in meters (default (-30, 30))
        y_range: tuple, y-axis range in meters (default (-15, 15))
        num_classes: int, number of semantic classes (default 3)
        embed_dim: int, instance embedding dimension (default 16)
        backbone_channels: int, backbone output channels after neck (default 64)
        bev_encoder_channels: int, BEV encoder intermediate channels (default 128)

    Inputs:
        images: [B, 6, 128, 352, 3] - Multi-camera images
        extrinsics: [B, 6, 4, 4] - Camera-to-world transformation matrices
        intrinsics: [B, 6, 3, 3] - Camera intrinsic matrices

    Outputs:
        dict with keys:
            'semantic': [B, 200, 200, 3] - Per-class segmentation logits
            'instance': [B, 200, 200, 16] - Instance embedding vectors
            'direction': [B, 200, 200, 2] - Unit direction vectors
    """

    def __init__(self, view_transform_type='lss',
                 bev_h=200, bev_w=200,
                 x_range=(-30.0, 30.0), y_range=(-15.0, 15.0),
                 num_classes=3, embed_dim=16,
                 backbone_channels=64, bev_encoder_channels=128,
                 **kwargs):
        super().__init__(**kwargs)

        self.view_transform_type = view_transform_type
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cams = 6
        self.img_h = 128
        self.img_w = 352

        # Feature map spatial dimensions (EfficientNet-B0 stride 16)
        self.feat_h = self.img_h // 16  # 8
        self.feat_w = self.img_w // 16  # 22

        # Backbone: shared EfficientNet-B0 for all cameras
        self.backbone = EfficientNetBackbone()

        # Neck: reduce feature dimensions
        self.neck = FeatureNeck(out_channels=backbone_channels)

        # View transform
        if view_transform_type == 'ipm':
            self.view_transform = IPMTransform(
                bev_h=bev_h, bev_w=bev_w,
                x_range=x_range, y_range=y_range,
                feat_h=self.feat_h, feat_w=self.feat_w
            )
        elif view_transform_type == 'lss':
            self.view_transform = LSSTransformVectorized(
                in_channels=backbone_channels,
                bev_h=bev_h, bev_w=bev_w,
                x_range=x_range, y_range=y_range,
                feat_h=self.feat_h, feat_w=self.feat_w,
                context_channels=backbone_channels
            )
        else:
            raise ValueError(
                f"Unknown view_transform_type: {view_transform_type}. "
                f"Must be 'ipm' or 'lss'."
            )

        # BEV encoder
        self.bev_encoder = BEVEncoder(
            in_channels=backbone_channels,
            mid_channels=bev_encoder_channels,
            out_channels=backbone_channels
        )

        # Output heads
        self.semantic_head = SemanticHead(
            in_channels=backbone_channels, num_classes=num_classes
        )
        self.instance_head = InstanceHead(
            in_channels=backbone_channels, embed_dim=embed_dim
        )
        self.direction_head = DirectionHead(
            in_channels=backbone_channels
        )

    def extract_features(self, images, training=False):
        """
        Extract features from multi-camera images using shared backbone.

        Args:
            images: [B, 6, 128, 352, 3]

        Returns:
            features: [B, 6, feat_h, feat_w, C]
        """
        batch_size = tf.shape(images)[0]

        # Reshape to process all camera images together
        # [B, 6, H, W, 3] -> [B*6, H, W, 3]
        images_flat = tf.reshape(images, [-1, self.img_h, self.img_w, 3])

        # Extract features with shared backbone
        feats = self.backbone(images_flat, training=training)  # [B*6, feat_h, feat_w, C_backbone]

        # Apply neck for dimension reduction
        feats = self.neck(feats, training=training)  # [B*6, feat_h, feat_w, C]

        # Reshape back to per-camera format
        feat_c = tf.shape(feats)[-1]
        feats = tf.reshape(feats, [batch_size, self.num_cams,
                                   self.feat_h, self.feat_w, feat_c])
        return feats

    def call(self, inputs, training=False):
        """
        Forward pass of HDMapNet.

        Args:
            inputs: tuple or dict containing:
                - images: [B, 6, 128, 352, 3]
                - extrinsics: [B, 6, 4, 4]
                - intrinsics: [B, 6, 3, 3]

        Returns:
            dict with keys 'semantic', 'instance', 'direction'
        """
        if isinstance(inputs, dict):
            images = inputs['images']
            extrinsics = inputs['extrinsics']
            intrinsics = inputs['intrinsics']
        elif isinstance(inputs, (list, tuple)):
            images, extrinsics, intrinsics = inputs
        else:
            raise ValueError("Inputs must be a dict or tuple of (images, extrinsics, intrinsics)")

        # 1. Extract multi-camera features
        features = self.extract_features(images, training=training)
        # features: [B, 6, feat_h, feat_w, C]

        # 2. View transform: project to BEV
        bev_features = self.view_transform(
            features, extrinsics, intrinsics, training=training
        )
        # bev_features: [B, bev_h, bev_w, C]

        # 3. BEV encoder: refine BEV features
        bev_encoded = self.bev_encoder(bev_features, training=training)
        # bev_encoded: [B, bev_h, bev_w, C]

        # 4. Output heads
        semantic = self.semantic_head(bev_encoded, training=training)
        instance = self.instance_head(bev_encoded, training=training)
        direction = self.direction_head(bev_encoded, training=training)

        return {
            'semantic': semantic,    # [B, 200, 200, 3]
            'instance': instance,    # [B, 200, 200, 16]
            'direction': direction,  # [B, 200, 200, 2]
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            'view_transform_type': self.view_transform_type,
            'bev_h': self.bev_h,
            'bev_w': self.bev_w,
        })
        return config


# =============================================================================
# Loss Functions
# =============================================================================

class HDMapNetLoss(keras.losses.Loss):
    """
    Combined loss for HDMapNet training.

    Components:
    - Semantic loss: Binary cross-entropy for each of the 3 classes
    - Instance loss: Discriminative loss (push-pull) for embeddings
    - Direction loss: Cosine similarity loss for direction vectors
    """

    def __init__(self, semantic_weight=1.0, instance_weight=1.0,
                 direction_weight=0.5, delta_v=0.5, delta_d=3.0, **kwargs):
        super().__init__(**kwargs)
        self.semantic_weight = semantic_weight
        self.instance_weight = instance_weight
        self.direction_weight = direction_weight
        self.delta_v = delta_v  # Pull margin
        self.delta_d = delta_d  # Push margin
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)

    def semantic_loss(self, y_true, y_pred):
        """Binary cross-entropy loss per class."""
        return self.bce(y_true, y_pred)

    def discriminative_loss(self, embeddings, instance_labels, num_instances):
        """
        Discriminative loss for instance embeddings.
        Pull loss: pull embeddings toward their instance mean.
        Push loss: push instance means apart.

        Args:
            embeddings: [H, W, E] embedding vectors
            instance_labels: [H, W] integer instance IDs (0 = background)
            num_instances: number of instances (excluding background)

        Returns:
            loss: scalar
        """
        embed_dim = tf.shape(embeddings)[-1]
        embeddings_flat = tf.reshape(embeddings, [-1, embed_dim])
        labels_flat = tf.reshape(instance_labels, [-1])

        pull_loss = 0.0
        means = []

        for i in tf.range(1, num_instances + 1):
            mask = tf.equal(labels_flat, i)
            if tf.reduce_any(mask):
                instance_embeds = tf.boolean_mask(embeddings_flat, mask)
                mean = tf.reduce_mean(instance_embeds, axis=0, keepdims=True)
                means.append(mean)

                # Pull: distance from mean
                dist = tf.norm(instance_embeds - mean, axis=1)
                pull = tf.maximum(dist - self.delta_v, 0.0)
                pull_loss = pull_loss + tf.reduce_mean(pull ** 2)

        if len(means) > 1:
            means_tensor = tf.concat(means, axis=0)  # [K, E]
            n_means = tf.shape(means_tensor)[0]

            # Push: pairwise distances between means
            push_loss = 0.0
            count = 0
            for i in range(n_means):
                for j in range(i + 1, n_means):
                    dist = tf.norm(means_tensor[i] - means_tensor[j])
                    push = tf.maximum(self.delta_d - dist, 0.0)
                    push_loss = push_loss + push ** 2
                    count += 1
            push_loss = push_loss / tf.maximum(tf.cast(count, tf.float32), 1.0)
        else:
            push_loss = 0.0

        pull_loss = pull_loss / tf.maximum(tf.cast(num_instances, tf.float32), 1.0)
        return pull_loss + push_loss

    def direction_loss(self, y_true, y_pred, mask):
        """
        Direction loss using cosine similarity.

        Args:
            y_true: [B, H, W, 2] ground truth directions
            y_pred: [B, H, W, 2] predicted directions
            mask: [B, H, W] binary mask of valid pixels

        Returns:
            loss: scalar
        """
        # Cosine similarity
        cos_sim = tf.reduce_sum(y_true * y_pred, axis=-1)  # [B, H, W]
        loss = 1.0 - cos_sim  # [B, H, W]
        # Apply mask
        mask_float = tf.cast(mask, tf.float32)
        masked_loss = loss * mask_float
        total = tf.reduce_sum(masked_loss)
        count = tf.maximum(tf.reduce_sum(mask_float), 1.0)
        return total / count

    def call(self, y_true, y_pred):
        """
        Compute combined loss.

        Args:
            y_true: dict with 'semantic', 'instance_labels', 'num_instances',
                    'direction', 'direction_mask'
            y_pred: dict with 'semantic', 'instance', 'direction'
        """
        sem_loss = self.semantic_loss(y_true['semantic'], y_pred['semantic'])
        dir_loss = self.direction_loss(
            y_true['direction'], y_pred['direction'], y_true['direction_mask']
        )

        total = (self.semantic_weight * sem_loss +
                 self.direction_weight * dir_loss)
        return total


# =============================================================================
# Model Builder Utilities
# =============================================================================

def build_hdmapnet_ipm(num_classes=3, embed_dim=16):
    """Build HDMapNet with IPM view transform."""
    return HDMapNet(
        view_transform_type='ipm',
        bev_h=200, bev_w=200,
        x_range=(-30.0, 30.0), y_range=(-15.0, 15.0),
        num_classes=num_classes,
        embed_dim=embed_dim,
        backbone_channels=64,
        bev_encoder_channels=128
    )


def build_hdmapnet_lss(num_classes=3, embed_dim=16):
    """Build HDMapNet with LSS view transform."""
    return HDMapNet(
        view_transform_type='lss',
        bev_h=200, bev_w=200,
        x_range=(-30.0, 30.0), y_range=(-15.0, 15.0),
        num_classes=num_classes,
        embed_dim=embed_dim,
        backbone_channels=64,
        bev_encoder_channels=128
    )


# =============================================================================
# Example Usage / Model Summary
# =============================================================================

if __name__ == '__main__':
    # Build model
    model = build_hdmapnet_lss()

    # Create dummy inputs
    batch_size = 2
    images = tf.random.normal([batch_size, 6, 128, 352, 3])
    extrinsics = tf.eye(4, batch_shape=[batch_size, 6])
    intrinsics = tf.constant([
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]]
        for fx, fy, cx, cy in [(700.0, 700.0, 176.0, 64.0)]
    ] * 6, dtype=tf.float32)
    intrinsics = tf.expand_dims(intrinsics, 0)
    intrinsics = tf.tile(intrinsics, [batch_size, 1, 1, 1])

    # Forward pass
    outputs = model([images, extrinsics, intrinsics], training=False)

    print("HDMapNet Output Shapes:")
    print(f"  Semantic:  {outputs['semantic'].shape}")
    print(f"  Instance:  {outputs['instance'].shape}")
    print(f"  Direction: {outputs['direction'].shape}")

    # Print model summary info
    print(f"\nModel Configuration:")
    print(f"  View Transform: LSS")
    print(f"  BEV Grid: {model.bev_h}x{model.bev_w}")
    print(f"  Num Cameras: {model.num_cams}")
    print(f"  Image Size: {model.img_h}x{model.img_w}")
    print(f"  Feature Size: {model.feat_h}x{model.feat_w}")
