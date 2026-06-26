"""Cylinder3D TensorFlow 2 model implementation.

Complete implementation of the Cylinder3D architecture for LiDAR point cloud
semantic segmentation, including:
- Cylindrical partition and voxelization
- Asymmetric convolution blocks
- Dimension-decomposition context modeling (DDCM)
- U-Net encoder-decoder backbone
- Point-level refinement module
- Lovasz-Softmax and combined loss functions
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# SemanticKITTI class labels (20 classes including unlabeled)
SEMANTICKITTI_NUM_CLASSES = 20

# Default cylindrical grid dimensions
DEFAULT_GRID_SIZE = [480, 360, 32]

# Default cylindrical coordinate ranges
DEFAULT_CYLINDRICAL_RANGE = {
    "rho": [0.0, 50.0],
    "theta": [-np.pi, np.pi],
    "z": [-3.0, 1.0],
}


class CylindricalPartition(layers.Layer):
    """Converts raw point clouds to cylindrical voxel grid representation.

    Maps (x, y, z, intensity) points to cylindrical coordinates (rho, theta, z),
    quantizes into a voxel grid of shape [H, W, D], and aggregates features
    within each voxel using scatter mean.
    """

    def __init__(
        self,
        grid_size=None,
        rho_range=None,
        theta_range=None,
        z_range=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.grid_size = grid_size or DEFAULT_GRID_SIZE
        self.rho_range = rho_range or DEFAULT_CYLINDRICAL_RANGE["rho"]
        self.theta_range = theta_range or DEFAULT_CYLINDRICAL_RANGE["theta"]
        self.z_range = z_range or DEFAULT_CYLINDRICAL_RANGE["z"]

        # Compute voxel sizes
        self.rho_res = (self.rho_range[1] - self.rho_range[0]) / self.grid_size[0]
        self.theta_res = (self.theta_range[1] - self.theta_range[0]) / self.grid_size[1]
        self.z_res = (self.z_range[1] - self.z_range[0]) / self.grid_size[2]

    def call(self, points):
        """Convert point cloud to cylindrical voxel grid.

        Args:
            points: tf.Tensor of shape [B, N, 4] with (x, y, z, intensity)

        Returns:
            voxel_features: tf.Tensor of shape [B, H, W, D, C] with aggregated features
            voxel_coords: tf.Tensor of shape [B, N, 3] with (rho_idx, theta_idx, z_idx)
            point_features: tf.Tensor of shape [B, N, 9] with augmented point features
        """
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        intensity = points[..., 3]

        # Convert to cylindrical coordinates
        rho = tf.sqrt(x ** 2 + y ** 2)
        theta = tf.atan2(y, x)

        # Compute grid indices
        rho_idx = tf.cast(
            tf.clip_by_value(
                (rho - self.rho_range[0]) / self.rho_res,
                0.0,
                float(self.grid_size[0] - 1),
            ),
            tf.int32,
        )
        theta_idx = tf.cast(
            tf.clip_by_value(
                (theta - self.theta_range[0]) / self.theta_res,
                0.0,
                float(self.grid_size[1] - 1),
            ),
            tf.int32,
        )
        z_idx = tf.cast(
            tf.clip_by_value(
                (z - self.z_range[0]) / self.z_res,
                0.0,
                float(self.grid_size[2] - 1),
            ),
            tf.int32,
        )

        # Stack voxel coordinates
        voxel_coords = tf.stack([rho_idx, theta_idx, z_idx], axis=-1)  # [B, N, 3]

        # Augmented point features: (x, y, z, intensity, rho, theta, rho_center, theta_center, z_center)
        rho_center = (tf.cast(rho_idx, tf.float32) + 0.5) * self.rho_res + self.rho_range[0]
        theta_center = (tf.cast(theta_idx, tf.float32) + 0.5) * self.theta_res + self.theta_range[0]
        z_center = (tf.cast(z_idx, tf.float32) + 0.5) * self.z_res + self.z_range[0]

        point_features = tf.stack(
            [x, y, z, intensity, rho, theta, rho - rho_center, theta - theta_center, z - z_center],
            axis=-1,
        )  # [B, N, 9]

        # Scatter mean to build voxel features
        batch_size = tf.shape(points)[0]
        num_points = tf.shape(points)[1]
        num_features = 9

        # Flatten batch for scatter operation
        voxel_features = self._scatter_mean(
            point_features, voxel_coords, batch_size, num_points, num_features
        )

        return voxel_features, voxel_coords, point_features

    def _scatter_mean(self, features, coords, batch_size, num_points, num_features):
        """Scatter mean aggregation of point features into voxel grid.

        Args:
            features: [B, N, C] point features
            coords: [B, N, 3] voxel indices
            batch_size: int
            num_points: int
            num_features: int

        Returns:
            voxel_grid: [B, H, W, D, C] aggregated features
        """
        H, W, D = self.grid_size

        # Initialize output grid and count grid
        voxel_grid = tf.zeros([batch_size, H, W, D, num_features], dtype=tf.float32)
        count_grid = tf.zeros([batch_size, H, W, D, 1], dtype=tf.float32)

        # Build linear indices for scatter
        batch_indices = tf.repeat(
            tf.range(batch_size)[:, tf.newaxis], num_points, axis=1
        )  # [B, N]

        # Flatten indices
        flat_batch = tf.reshape(batch_indices, [-1])
        flat_rho = tf.reshape(coords[..., 0], [-1])
        flat_theta = tf.reshape(coords[..., 1], [-1])
        flat_z = tf.reshape(coords[..., 2], [-1])
        flat_features = tf.reshape(features, [-1, num_features])

        # Scatter nd indices
        indices = tf.stack([flat_batch, flat_rho, flat_theta, flat_z], axis=1)

        # Sum features
        voxel_grid = tf.tensor_scatter_nd_add(voxel_grid, indices, flat_features)
        ones = tf.ones([tf.shape(flat_batch)[0], 1], dtype=tf.float32)
        count_grid = tf.tensor_scatter_nd_add(count_grid, indices, ones)

        # Mean (avoid division by zero)
        count_grid = tf.maximum(count_grid, 1.0)
        voxel_grid = voxel_grid / count_grid

        return voxel_grid

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_size": self.grid_size,
                "rho_range": self.rho_range,
                "theta_range": self.theta_range,
                "z_range": self.z_range,
            }
        )
        return config


class AsymmetricConvBlock(layers.Layer):
    """Asymmetric convolution block with parallel decomposed 3D kernels.

    Uses three parallel branches with kernels (1,3,3), (3,1,3), and (3,3,1)
    to capture spatial features efficiently while reducing parameters compared
    to a full 3x3x3 convolution.
    """

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        # Branch 1: (1, 3, 3) kernel
        self.conv_133 = layers.Conv3D(
            self.out_channels,
            kernel_size=(1, 3, 3),
            padding="same",
            use_bias=False,
        )
        self.bn_133 = layers.BatchNormalization()

        # Branch 2: (3, 1, 3) kernel
        self.conv_313 = layers.Conv3D(
            self.out_channels,
            kernel_size=(3, 1, 3),
            padding="same",
            use_bias=False,
        )
        self.bn_313 = layers.BatchNormalization()

        # Branch 3: (3, 3, 1) kernel
        self.conv_331 = layers.Conv3D(
            self.out_channels,
            kernel_size=(3, 3, 1),
            padding="same",
            use_bias=False,
        )
        self.bn_331 = layers.BatchNormalization()

        self.leaky_relu = layers.LeakyReLU(negative_slope=0.1)
        super().build(input_shape)

    def call(self, x, training=False):
        """Forward pass with parallel asymmetric convolutions.

        Args:
            x: [B, H, W, D, C] input tensor
            training: bool

        Returns:
            [B, H, W, D, out_channels] summed output
        """
        out_133 = self.leaky_relu(self.bn_133(self.conv_133(x), training=training))
        out_313 = self.leaky_relu(self.bn_313(self.conv_313(x), training=training))
        out_331 = self.leaky_relu(self.bn_331(self.conv_331(x), training=training))

        return out_133 + out_313 + out_331

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels})
        return config


class DDCMod(layers.Layer):
    """Dimension-Decomposition Context Modeling module.

    Projects the 3D volume onto three 2D planes (rho-theta, rho-z, theta-z),
    applies 2D convolutions for context modeling, and broadcasts back to 3D
    to enrich voxel features with global context.
    """

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        in_channels = input_shape[-1]

        # 2D convolutions for each projected plane
        self.conv_rho_theta = layers.Conv2D(
            self.out_channels, kernel_size=3, padding="same", use_bias=False
        )
        self.bn_rho_theta = layers.BatchNormalization()

        self.conv_rho_z = layers.Conv2D(
            self.out_channels, kernel_size=3, padding="same", use_bias=False
        )
        self.bn_rho_z = layers.BatchNormalization()

        self.conv_theta_z = layers.Conv2D(
            self.out_channels, kernel_size=3, padding="same", use_bias=False
        )
        self.bn_theta_z = layers.BatchNormalization()

        # Fusion 1x1x1 conv
        self.fusion_conv = layers.Conv3D(
            self.out_channels, kernel_size=1, padding="same", use_bias=False
        )
        self.fusion_bn = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU(negative_slope=0.1)

        super().build(input_shape)

    def call(self, x, training=False):
        """Apply dimension-decomposition context modeling.

        Args:
            x: [B, H, W, D, C] 3D volume
            training: bool

        Returns:
            [B, H, W, D, out_channels] context-enriched volume
        """
        # x shape: [B, H, W, D, C]
        # Project to 2D planes by reducing one dimension (mean pooling)

        # rho-theta plane: reduce over z (axis 3)
        proj_rho_theta = tf.reduce_mean(x, axis=3)  # [B, H, W, C]
        proj_rho_theta = self.leaky_relu(
            self.bn_rho_theta(self.conv_rho_theta(proj_rho_theta), training=training)
        )  # [B, H, W, out_channels]

        # rho-z plane: reduce over theta (axis 2)
        proj_rho_z = tf.reduce_mean(x, axis=2)  # [B, H, D, C]
        proj_rho_z = self.leaky_relu(
            self.bn_rho_z(self.conv_rho_z(proj_rho_z), training=training)
        )  # [B, H, D, out_channels]

        # theta-z plane: reduce over rho (axis 1)
        proj_theta_z = tf.reduce_mean(x, axis=1)  # [B, W, D, C]
        proj_theta_z = self.leaky_relu(
            self.bn_theta_z(self.conv_theta_z(proj_theta_z), training=training)
        )  # [B, W, D, out_channels]

        # Broadcast back to 3D
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        D = tf.shape(x)[3]

        # [B, H, W, out_channels] -> [B, H, W, D, out_channels]
        rho_theta_3d = tf.repeat(proj_rho_theta[:, :, :, tf.newaxis, :], D, axis=3)

        # [B, H, D, out_channels] -> [B, H, W, D, out_channels]
        rho_z_3d = tf.repeat(proj_rho_z[:, :, tf.newaxis, :, :], W, axis=2)

        # [B, W, D, out_channels] -> [B, H, W, D, out_channels]
        theta_z_3d = tf.repeat(proj_theta_z[:, tf.newaxis, :, :, :], H, axis=1)

        # Sum and fuse
        combined = rho_theta_3d + rho_z_3d + theta_z_3d
        out = self.leaky_relu(
            self.fusion_bn(self.fusion_conv(combined), training=training)
        )

        return out

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels})
        return config


class AsymmetricResBlock(layers.Layer):
    """Residual block using asymmetric convolutions.

    Applies two asymmetric conv blocks with a skip connection.
    If input channels differ from output channels, a 1x1x1 projection is used.
    """

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.asym_conv1 = AsymmetricConvBlock(self.out_channels)
        self.asym_conv2 = AsymmetricConvBlock(self.out_channels)

        self.use_projection = in_channels != self.out_channels
        if self.use_projection:
            self.projection = layers.Conv3D(
                self.out_channels, kernel_size=1, use_bias=False
            )
            self.proj_bn = layers.BatchNormalization()

        self.leaky_relu = layers.LeakyReLU(negative_slope=0.1)
        super().build(input_shape)

    def call(self, x, training=False):
        """Forward pass with residual connection.

        Args:
            x: [B, H, W, D, C] input
            training: bool

        Returns:
            [B, H, W, D, out_channels] output with skip connection
        """
        identity = x

        out = self.asym_conv1(x, training=training)
        out = self.asym_conv2(out, training=training)

        if self.use_projection:
            identity = self.leaky_relu(
                self.proj_bn(self.projection(identity), training=training)
            )

        return self.leaky_relu(out + identity)

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels})
        return config


class EncoderBlock(layers.Layer):
    """Encoder stage: downsample + asymmetric residual blocks + DDCM."""

    def __init__(self, out_channels, num_blocks=2, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.num_blocks = num_blocks

    def build(self, input_shape):
        # Strided convolution for downsampling
        self.downsample = layers.Conv3D(
            self.out_channels,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
        )
        self.down_bn = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU(negative_slope=0.1)

        # Residual blocks
        self.res_blocks = []
        for i in range(self.num_blocks):
            self.res_blocks.append(AsymmetricResBlock(self.out_channels))

        # DDCM for context modeling
        self.ddcm = DDCMod(self.out_channels)

        super().build(input_shape)

    def call(self, x, training=False):
        """Encode: downsample, apply residual blocks, add DDCM context.

        Args:
            x: [B, H, W, D, C] input volume
            training: bool

        Returns:
            out: [B, H/2, W/2, D/2, out_channels] encoded volume
        """
        x = self.leaky_relu(self.down_bn(self.downsample(x), training=training))

        for block in self.res_blocks:
            x = block(x, training=training)

        context = self.ddcm(x, training=training)
        out = x + context

        return out

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels, "num_blocks": self.num_blocks})
        return config


class DecoderBlock(layers.Layer):
    """Decoder stage: upsample via transpose conv + skip connection + residual blocks."""

    def __init__(self, out_channels, num_blocks=2, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.num_blocks = num_blocks

    def build(self, input_shape):
        # Transpose convolution for upsampling
        self.upsample = layers.Conv3DTranspose(
            self.out_channels,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
        )
        self.up_bn = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU(negative_slope=0.1)

        # 1x1x1 conv to fuse skip connection
        self.skip_fusion = layers.Conv3D(
            self.out_channels, kernel_size=1, use_bias=False
        )
        self.skip_bn = layers.BatchNormalization()

        # Residual blocks
        self.res_blocks = []
        for i in range(self.num_blocks):
            self.res_blocks.append(AsymmetricResBlock(self.out_channels))

        super().build(input_shape)

    def call(self, x, skip, training=False):
        """Decode: upsample, fuse skip, apply residual blocks.

        Args:
            x: [B, H, W, D, C] lower-resolution input
            skip: [B, 2H, 2W, 2D, C_skip] skip connection from encoder
            training: bool

        Returns:
            out: [B, 2H, 2W, 2D, out_channels] decoded volume
        """
        x = self.leaky_relu(self.up_bn(self.upsample(x), training=training))

        # Pad/crop to match skip size if needed
        skip_shape = tf.shape(skip)
        x = x[:, : skip_shape[1], : skip_shape[2], : skip_shape[3], :]

        # Concatenate and fuse
        combined = tf.concat([x, skip], axis=-1)
        combined = self.leaky_relu(
            self.skip_bn(self.skip_fusion(combined), training=training)
        )

        for block in self.res_blocks:
            combined = block(combined, training=training)

        return combined

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels, "num_blocks": self.num_blocks})
        return config


class Cylinder3DBackbone(layers.Layer):
    """U-Net style encoder-decoder backbone with 4 stages.

    Encoder channels: 32 -> 64 -> 128 -> 256 -> 256
    Decoder mirrors with skip connections at each stage.
    """

    def __init__(self, num_classes=SEMANTICKITTI_NUM_CLASSES, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        # Initial projection
        self.init_conv = layers.Conv3D(32, kernel_size=3, padding="same", use_bias=False)
        self.init_bn = layers.BatchNormalization()
        self.init_act = layers.LeakyReLU(negative_slope=0.1)

        # Encoder stages
        self.encoder1 = EncoderBlock(64, num_blocks=2)
        self.encoder2 = EncoderBlock(128, num_blocks=2)
        self.encoder3 = EncoderBlock(256, num_blocks=2)
        self.encoder4 = EncoderBlock(256, num_blocks=2)

        # Decoder stages
        self.decoder4 = DecoderBlock(128, num_blocks=2)
        self.decoder3 = DecoderBlock(64, num_blocks=2)
        self.decoder2 = DecoderBlock(32, num_blocks=2)
        self.decoder1 = DecoderBlock(32, num_blocks=2)

        # Output head for voxel-level predictions
        self.output_conv = layers.Conv3D(
            self.num_classes, kernel_size=1, padding="same"
        )

        super().build(input_shape)

    def call(self, x, training=False):
        """Forward pass through U-Net backbone.

        Args:
            x: [B, H, W, D, C] voxelized input
            training: bool

        Returns:
            voxel_logits: [B, H, W, D, num_classes] per-voxel predictions
            voxel_features: [B, H, W, D, 32] final decoder features
        """
        # Initial convolution
        x0 = self.init_act(self.init_bn(self.init_conv(x), training=training))

        # Encoder
        x1 = self.encoder1(x0, training=training)
        x2 = self.encoder2(x1, training=training)
        x3 = self.encoder3(x2, training=training)
        x4 = self.encoder4(x3, training=training)

        # Decoder with skip connections
        d4 = self.decoder4(x4, x3, training=training)
        d3 = self.decoder3(d4, x2, training=training)
        d2 = self.decoder2(d3, x1, training=training)
        d1 = self.decoder1(d2, x0, training=training)

        # Voxel-level logits
        voxel_logits = self.output_conv(d1)

        return voxel_logits, d1

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


class PointRefinementModule(layers.Layer):
    """Point-level refinement MLP.

    Concatenates per-point voxel features (gathered from the voxel grid)
    with the original point features, then applies an MLP to produce
    per-point class predictions.
    """

    def __init__(self, num_classes=SEMANTICKITTI_NUM_CLASSES, hidden_dims=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims or [64, 128, 64]

    def build(self, input_shape):
        self.mlp_layers = []
        for dim in self.hidden_dims:
            self.mlp_layers.append(layers.Dense(dim, use_bias=False))
            self.mlp_layers.append(layers.BatchNormalization())
            self.mlp_layers.append(layers.LeakyReLU(negative_slope=0.1))
            self.mlp_layers.append(layers.Dropout(0.1))

        self.output_layer = layers.Dense(self.num_classes)
        super().build(input_shape)

    def call(self, point_features, voxel_features, voxel_coords, training=False):
        """Refine predictions at point level.

        Args:
            point_features: [B, N, C_point] raw point features
            voxel_features: [B, H, W, D, C_voxel] voxel-level features from backbone
            voxel_coords: [B, N, 3] voxel indices for each point
            training: bool

        Returns:
            point_logits: [B, N, num_classes] per-point class predictions
        """
        # Gather voxel features for each point
        batch_size = tf.shape(point_features)[0]
        num_points = tf.shape(point_features)[1]

        # Build gather indices
        batch_indices = tf.repeat(
            tf.range(batch_size)[:, tf.newaxis], num_points, axis=1
        )  # [B, N]

        gather_indices = tf.stack(
            [
                tf.reshape(batch_indices, [-1]),
                tf.reshape(voxel_coords[..., 0], [-1]),
                tf.reshape(voxel_coords[..., 1], [-1]),
                tf.reshape(voxel_coords[..., 2], [-1]),
            ],
            axis=1,
        )  # [B*N, 4]

        gathered = tf.gather_nd(voxel_features, gather_indices)  # [B*N, C_voxel]
        gathered = tf.reshape(gathered, [batch_size, num_points, -1])  # [B, N, C_voxel]

        # Concatenate point features with voxel features
        combined = tf.concat([point_features, gathered], axis=-1)  # [B, N, C_point + C_voxel]

        # MLP
        x = combined
        for layer in self.mlp_layers:
            if isinstance(layer, layers.BatchNormalization):
                x = layer(x, training=training)
            elif isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)

        point_logits = self.output_layer(x)

        return point_logits

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_classes": self.num_classes, "hidden_dims": self.hidden_dims}
        )
        return config


class LovaszSoftmaxLoss(tf.keras.losses.Loss):
    """Lovasz-Softmax loss for semantic segmentation.

    Optimizes the mean intersection-over-union (mIoU) metric directly
    via a surrogate convex loss based on the Lovasz extension of submodular
    set functions.
    """

    def __init__(self, num_classes=SEMANTICKITTI_NUM_CLASSES, ignore_index=0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def call(self, y_true, y_pred):
        """Compute Lovasz-Softmax loss.

        Args:
            y_true: [B, N] integer class labels
            y_pred: [B, N, C] logits (pre-softmax)

        Returns:
            Scalar loss value
        """
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.reshape(y_pred, [-1, self.num_classes])

        # Filter ignored points
        valid_mask = tf.not_equal(y_true, self.ignore_index)
        y_true_valid = tf.boolean_mask(y_true, valid_mask)
        y_pred_valid = tf.boolean_mask(y_pred, valid_mask)

        probas = tf.nn.softmax(y_pred_valid, axis=-1)

        losses = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            fg = tf.cast(tf.equal(y_true_valid, c), tf.float32)
            if tf.reduce_sum(fg) == 0 and tf.reduce_sum(1.0 - fg) == 0:
                continue
            errors = tf.abs(fg - probas[:, c])
            errors_sorted, perm = tf.math.top_k(errors, k=tf.shape(errors)[0])
            fg_sorted = tf.gather(fg, perm)
            losses.append(self._lovasz_grad(fg_sorted, errors_sorted))

        if len(losses) == 0:
            return tf.constant(0.0)

        return tf.reduce_mean(tf.stack(losses))

    @staticmethod
    def _lovasz_grad(gt_sorted, errors_sorted):
        """Compute Lovasz gradient and loss for one class.

        Args:
            gt_sorted: sorted ground truth (1s and 0s)
            errors_sorted: sorted prediction errors

        Returns:
            Scalar loss for this class
        """
        n = tf.shape(gt_sorted)[0]
        gts = tf.reduce_sum(gt_sorted)

        # Intersection and union
        intersection = gts - tf.cumsum(gt_sorted)
        union = gts + tf.cumsum(1.0 - gt_sorted)
        jaccard = 1.0 - intersection / (union + 1e-8)

        # Compute gradient
        jaccard_shifted = tf.concat([jaccard[:1], jaccard[1:] - jaccard[:-1]], axis=0)
        jaccard_shifted = tf.clip_by_value(jaccard_shifted, 0.0, 1.0)

        loss = tf.reduce_sum(errors_sorted * tf.stop_gradient(jaccard_shifted))
        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_classes": self.num_classes, "ignore_index": self.ignore_index}
        )
        return config


class CombinedLoss(tf.keras.losses.Loss):
    """Combined Cross-Entropy + Lovasz-Softmax loss.

    The weighted sum of standard cross-entropy loss and Lovasz-Softmax loss
    provides both stable gradients (CE) and direct mIoU optimization (Lovasz).
    """

    def __init__(
        self,
        num_classes=SEMANTICKITTI_NUM_CLASSES,
        ignore_index=0,
        ce_weight=1.0,
        lovasz_weight=1.0,
        class_weights=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.class_weights = class_weights
        self.lovasz_loss = LovaszSoftmaxLoss(
            num_classes=num_classes, ignore_index=ignore_index
        )

    def call(self, y_true, y_pred):
        """Compute combined CE + Lovasz loss.

        Args:
            y_true: [B, N] integer class labels
            y_pred: [B, N, C] logits

        Returns:
            Scalar combined loss
        """
        # Flatten
        y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_flat = tf.reshape(y_pred, [-1, self.num_classes])

        # Valid mask (ignore unlabeled)
        valid_mask = tf.not_equal(y_true_flat, self.ignore_index)
        y_true_valid = tf.boolean_mask(y_true_flat, valid_mask)
        y_pred_valid = tf.boolean_mask(y_pred_flat, valid_mask)

        # Cross entropy
        if self.class_weights is not None:
            weights = tf.constant(self.class_weights, dtype=tf.float32)
            sample_weights = tf.gather(weights, y_true_valid)
            ce_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_true_valid, logits=y_pred_valid
                )
                * sample_weights
            )
        else:
            ce_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_true_valid, logits=y_pred_valid
                )
            )

        # Lovasz loss
        lovasz_loss = self.lovasz_loss(y_true, y_pred)

        return self.ce_weight * ce_loss + self.lovasz_weight * lovasz_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "ignore_index": self.ignore_index,
                "ce_weight": self.ce_weight,
                "lovasz_weight": self.lovasz_weight,
                "class_weights": self.class_weights,
            }
        )
        return config


class Cylinder3DModel(Model):
    """Complete Cylinder3D model for LiDAR point cloud semantic segmentation.

    Integrates cylindrical partition, U-Net backbone with asymmetric convolutions
    and DDCM, and point-level refinement for end-to-end 3D semantic segmentation.
    """

    def __init__(
        self,
        num_classes=SEMANTICKITTI_NUM_CLASSES,
        grid_size=None,
        rho_range=None,
        theta_range=None,
        z_range=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.grid_size = grid_size or DEFAULT_GRID_SIZE

        self.partition = CylindricalPartition(
            grid_size=self.grid_size,
            rho_range=rho_range,
            theta_range=theta_range,
            z_range=z_range,
        )
        self.backbone = Cylinder3DBackbone(num_classes=num_classes)
        self.refinement = PointRefinementModule(num_classes=num_classes)

    def call(self, points, training=False):
        """Process point cloud end-to-end.

        Args:
            points: tf.Tensor of shape [B, N, 4] with (x, y, z, intensity)
            training: bool

        Returns:
            point_logits: [B, N, num_classes] per-point class predictions
            voxel_logits: [B, H, W, D, num_classes] per-voxel predictions
        """
        # Step 1: Cylindrical partition
        voxel_features, voxel_coords, point_features = self.partition(points)

        # Step 2: Backbone (U-Net with asymmetric convs + DDCM)
        voxel_logits, decoder_features = self.backbone(
            voxel_features, training=training
        )

        # Step 3: Point-level refinement
        point_logits = self.refinement(
            point_features, decoder_features, voxel_coords, training=training
        )

        return point_logits, voxel_logits

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32)])
    def predict_points(self, points):
        """Inference-optimized prediction (no training ops).

        Args:
            points: [B, N, 4] point cloud

        Returns:
            predictions: [B, N] predicted class indices
        """
        point_logits, _ = self.call(points, training=False)
        predictions = tf.argmax(point_logits, axis=-1)
        return predictions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "grid_size": self.grid_size,
            }
        )
        return config
