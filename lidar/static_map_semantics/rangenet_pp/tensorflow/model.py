"""
RangeNet++ TensorFlow 2 / Keras Implementation.

Based on: Milioto et al., "RangeNet++: Fast and Accurate LiDAR Semantic
Segmentation", IROS 2019.

Architecture:
    - DarkNet-53 encoder operating on range images (H=64, W=2048).
    - U-Net style decoder with skip connections from encoder stages.
    - Input: (B, H, W, 5) range image with channels [x, y, z, intensity, range].
    - Output: (B, H, W, num_classes) logits.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBNLeakyReLU(layers.Layer):
    """Conv2D -> BatchNorm -> LeakyReLU(0.1)."""

    def __init__(self, filters, kernel_size, strides=1, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU(alpha=0.1)

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x


class DarkNetResidualBlock(layers.Layer):
    """Single DarkNet-53 residual block.

    Structure:
        input -> Conv1x1 (filters//2) -> Conv3x3 (filters) -> Add(input) -> output
    """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvBNLeakyReLU(filters // 2, kernel_size=1)
        self.conv2 = ConvBNLeakyReLU(filters, kernel_size=3)

    def call(self, x, training=False):
        shortcut = x
        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)
        return out + shortcut


class DarkNetStage(layers.Layer):
    """One encoder stage: downsample (stride-2 conv) followed by N residual blocks."""

    def __init__(self, filters, num_blocks, **kwargs):
        super().__init__(**kwargs)
        self.downsample = ConvBNLeakyReLU(filters, kernel_size=3, strides=2)
        self.blocks = [
            DarkNetResidualBlock(filters, name=f"res_block_{i}")
            for i in range(num_blocks)
        ]

    def call(self, x, training=False):
        x = self.downsample(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return x


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------


class DecoderBlock(layers.Layer):
    """Decoder block: upsample -> concatenate with skip -> Conv3x3 -> Conv3x3."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.conv1 = ConvBNLeakyReLU(filters, kernel_size=1)
        self.conv2 = ConvBNLeakyReLU(filters, kernel_size=3)
        self.conv3 = ConvBNLeakyReLU(filters, kernel_size=3)

    def call(self, x, skip, training=False):
        x = self.upsample(x)

        # Handle potential size mismatch after upsampling due to odd dimensions
        skip_h = tf.shape(skip)[1]
        skip_w = tf.shape(skip)[2]
        x = x[:, :skip_h, :skip_w, :]

        x = self.conv1(x, training=training)
        x = layers.concatenate([x, skip])
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


# ---------------------------------------------------------------------------
# RangeNet++ Model
# ---------------------------------------------------------------------------


class RangeNetPP(Model):
    """RangeNet++ for LiDAR semantic segmentation on range images.

    Args:
        num_classes: Number of output semantic classes (default: 20).
        input_height: Height of the range image (default: 64).
        input_width: Width of the range image (default: 2048).
        input_channels: Number of input channels (default: 5).
    """

    def __init__(
        self,
        num_classes=20,
        input_height=64,
        input_width=2048,
        input_channels=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        # Channel progression: 32 -> 64 -> 128 -> 256 -> 512
        # Block counts:         1     2     8      8     4
        encoder_filters = [32, 64, 128, 256, 512]
        encoder_blocks = [1, 2, 8, 8, 4]

        # Initial convolution (no downsampling)
        self.initial_conv = ConvBNLeakyReLU(32, kernel_size=3, name="initial_conv")

        # Encoder stages
        self.encoder_stages = []
        for i, (f, n) in enumerate(zip(encoder_filters, encoder_blocks)):
            self.encoder_stages.append(
                DarkNetStage(f, n, name=f"encoder_stage_{i}")
            )

        # Decoder stages (reverse order, matching encoder skip connections)
        # Decoder produces: 256 -> 128 -> 64 -> 32 -> 32
        decoder_filters = [256, 128, 64, 32, 32]
        self.decoder_blocks = []
        for i, f in enumerate(decoder_filters):
            self.decoder_blocks.append(
                DecoderBlock(f, name=f"decoder_block_{i}")
            )

        # Dropout for regularization
        self.dropout = layers.Dropout(0.2)

        # Final classification head
        self.final_conv = layers.Conv2D(
            num_classes,
            kernel_size=1,
            padding="same",
            kernel_initializer="he_normal",
            name="logits",
        )

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Tensor of shape (B, H, W, 5).
            training: Boolean flag for batch normalization and dropout.

        Returns:
            Logits tensor of shape (B, H, W, num_classes).
        """
        # Initial feature extraction
        x = self.initial_conv(inputs, training=training)
        skip0 = x  # (B, H, W, 32)

        # Encoder path - collect skip connections
        skips = [skip0]
        for stage in self.encoder_stages:
            x = stage(x, training=training)
            skips.append(x)

        # skips[0]: (B, H, W, 32)       - after initial conv
        # skips[1]: (B, H/2, W/2, 32)   - after stage 0
        # skips[2]: (B, H/4, W/4, 64)   - after stage 1
        # skips[3]: (B, H/8, W/8, 128)  - after stage 2
        # skips[4]: (B, H/16, W/16, 256) - after stage 3
        # skips[5]: (B, H/32, W/32, 512) - after stage 4 (bottleneck)

        # Decoder path - use skip connections in reverse
        # decoder_block_0: upsample skips[5] and concat with skips[4]
        # decoder_block_1: upsample result and concat with skips[3]
        # decoder_block_2: upsample result and concat with skips[2]
        # decoder_block_3: upsample result and concat with skips[1]
        # decoder_block_4: upsample result and concat with skips[0]

        x = skips[-1]  # bottleneck features
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_connection = skips[-(i + 2)]
            x = decoder_block(x, skip_connection, training=training)

        # Apply dropout before final classification
        x = self.dropout(x, training=training)

        # Final 1x1 convolution to produce per-pixel class logits
        logits = self.final_conv(x)

        return logits

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "input_height": self.input_height,
                "input_width": self.input_width,
                "input_channels": self.input_channels,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def build_rangenet_pp(
    num_classes=20,
    input_height=64,
    input_width=2048,
    input_channels=5,
):
    """Build and return a RangeNet++ model.

    Args:
        num_classes: Number of semantic classes.
        input_height: Range image height (default 64).
        input_width: Range image width (default 2048).
        input_channels: Number of input channels (default 5).

    Returns:
        A compiled-ready RangeNet++ tf.keras.Model instance.
    """
    model = RangeNetPP(
        num_classes=num_classes,
        input_height=input_height,
        input_width=input_width,
        input_channels=input_channels,
    )
    # Build the model by running a dummy forward pass
    dummy_input = tf.zeros((1, input_height, input_width, input_channels))
    _ = model(dummy_input, training=False)
    return model


def rangenet_pp_loss(y_true, y_pred, class_weights=None):
    """Weighted cross-entropy loss for RangeNet++.

    Args:
        y_true: Ground truth labels of shape (B, H, W) with integer class indices.
        y_pred: Predicted logits of shape (B, H, W, num_classes).
        class_weights: Optional tensor of shape (num_classes,) for class balancing.

    Returns:
        Scalar loss value.
    """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(y_true, tf.int32), logits=y_pred
    )

    if class_weights is not None:
        class_weights = tf.cast(class_weights, tf.float32)
        weights = tf.gather(class_weights, tf.cast(y_true, tf.int32))
        loss = loss * weights

    return tf.reduce_mean(loss)


# ---------------------------------------------------------------------------
# Main entry point (for quick testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building RangeNet++ model...")
    model = build_rangenet_pp(num_classes=20, input_height=64, input_width=2048)
    model.summary(line_length=120)

    # Verify output shape
    test_input = tf.random.normal((2, 64, 2048, 5))
    output = model(test_input, training=False)
    print(f"\nInput shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 64, 2048, 20), f"Unexpected output shape: {output.shape}"
    print("Shape verification passed.")
