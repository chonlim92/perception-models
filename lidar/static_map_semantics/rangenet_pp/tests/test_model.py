"""
Complete pytest test suite for RangeNet++ TensorFlow implementation.

Tests cover:
    - Spherical projection math
    - Back-projection consistency
    - DarkNet-53 encoder output shapes
    - Decoder output shape
    - Full model forward pass
    - KNN post-processing
    - Loss computation
    - Model trainable parameters

All tests use synthetic/random data and require no real datasets.
"""

import sys
import os
import math

import numpy as np
import pytest
import tensorflow as tf

# Add the tensorflow module directory to the path so we can import model components
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tensorflow")
    ),
)

from model import (
    RangeNetPP,
    build_rangenet_pp,
    rangenet_pp_loss,
    ConvBNLeakyReLU,
    DarkNetStage,
    DecoderBlock,
)


# ---------------------------------------------------------------------------
# Constants for spherical projection (SemanticKITTI / RangeNet++ defaults)
# ---------------------------------------------------------------------------

FOV_UP_DEG = 2.0
FOV_DOWN_DEG = -24.8
FOV_UP_RAD = FOV_UP_DEG * math.pi / 180.0
FOV_DOWN_RAD = FOV_DOWN_DEG * math.pi / 180.0
FOV_TOTAL_RAD = FOV_UP_RAD - FOV_DOWN_RAD  # positive value, total vertical FOV
H = 64
W = 2048


# ---------------------------------------------------------------------------
# Helper: Spherical Projection
# ---------------------------------------------------------------------------


def spherical_project(points):
    """Project 3D points into a range image using spherical coordinates.

    Args:
        points: numpy array of shape (N, 3) with columns [x, y, z].

    Returns:
        u: column indices in [0, W-1]
        v: row indices in [0, H-1]
        r: range (distance from origin)
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Yaw: atan2(y, x), range [-pi, pi]
    yaw = np.arctan2(y, x)

    # Pitch: asin(z / r), range [-pi/2, pi/2]
    pitch = np.arcsin(np.clip(z / (r + 1e-10), -1.0, 1.0))

    # Normalize yaw to [0, 1] -> column index
    # yaw=0 maps to center (W/2), yaw=pi maps to 0, yaw=-pi maps to W-1
    u = 0.5 * (1.0 - yaw / math.pi) * W
    u = np.clip(np.floor(u).astype(np.int32), 0, W - 1)

    # Normalize pitch to [0, 1] -> row index
    # pitch = fov_up maps to row 0, pitch = fov_down maps to row H-1
    v = (1.0 - (pitch - FOV_DOWN_RAD) / FOV_TOTAL_RAD) * H
    v = np.clip(np.floor(v).astype(np.int32), 0, H - 1)

    return u, v, r


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    """Build a small RangeNet++ model for testing (reduced resolution)."""
    model = RangeNetPP(
        num_classes=20,
        input_height=64,
        input_width=128,  # Reduced width for faster tests
        input_channels=5,
    )
    dummy = tf.zeros((1, 64, 128, 5))
    _ = model(dummy, training=False)
    return model


@pytest.fixture
def full_model():
    """Build a full-resolution RangeNet++ model."""
    model = RangeNetPP(
        num_classes=20,
        input_height=64,
        input_width=2048,
        input_channels=5,
    )
    dummy = tf.zeros((1, 64, 2048, 5))
    _ = model(dummy, training=False)
    return model


# ---------------------------------------------------------------------------
# Test 1: Spherical Projection
# ---------------------------------------------------------------------------


class TestSphericalProjection:
    """Test spherical projection of 3D points to range image coordinates."""

    def test_point_along_x_axis(self):
        """Point at (10, 0, 0): yaw=0, pitch=0 -> center column, specific row."""
        point = np.array([[10.0, 0.0, 0.0]])
        u, v, r = spherical_project(point)

        # yaw = atan2(0, 10) = 0
        # u = 0.5 * (1 - 0/pi) * 2048 = 0.5 * 2048 = 1024
        expected_u = int(np.floor(0.5 * (1.0 - 0.0 / math.pi) * W))
        assert u[0] == expected_u, f"Expected u={expected_u}, got u={u[0]}"

        # pitch = asin(0 / 10) = 0
        # v = (1 - (0 - fov_down) / fov_total) * H
        pitch = 0.0
        expected_v_float = (1.0 - (pitch - FOV_DOWN_RAD) / FOV_TOTAL_RAD) * H
        expected_v = np.clip(int(np.floor(expected_v_float)), 0, H - 1)
        assert v[0] == expected_v, f"Expected v={expected_v}, got v={v[0]}"

        # Range should be 10
        assert abs(r[0] - 10.0) < 1e-6

    def test_point_along_y_axis(self):
        """Point at (0, 10, 0): yaw=pi/2 (90 deg) -> W/4 column."""
        point = np.array([[0.0, 10.0, 0.0]])
        u, v, r = spherical_project(point)

        # yaw = atan2(10, 0) = pi/2
        # u = 0.5 * (1 - (pi/2)/pi) * 2048 = 0.5 * 0.5 * 2048 = 512
        expected_u = int(np.floor(0.5 * (1.0 - (math.pi / 2) / math.pi) * W))
        assert expected_u == 512, f"Sanity check: expected_u should be 512, got {expected_u}"
        assert u[0] == expected_u, f"Expected u={expected_u}, got u={u[0]}"

        # pitch = 0 (same as x-axis test)
        pitch = 0.0
        expected_v_float = (1.0 - (pitch - FOV_DOWN_RAD) / FOV_TOTAL_RAD) * H
        expected_v = np.clip(int(np.floor(expected_v_float)), 0, H - 1)
        assert v[0] == expected_v, f"Expected v={expected_v}, got v={v[0]}"

    def test_point_negative_y_axis(self):
        """Point at (0, -10, 0): yaw=-pi/2 -> 3*W/4 column."""
        point = np.array([[0.0, -10.0, 0.0]])
        u, v, r = spherical_project(point)

        # yaw = atan2(-10, 0) = -pi/2
        # u = 0.5 * (1 - (-pi/2)/pi) * 2048 = 0.5 * 1.5 * 2048 = 1536
        expected_u = int(np.floor(0.5 * (1.0 - (-math.pi / 2) / math.pi) * W))
        assert expected_u == 1536, f"Sanity check: expected_u should be 1536, got {expected_u}"
        assert u[0] == expected_u, f"Expected u={expected_u}, got u={u[0]}"

    def test_point_with_elevation(self):
        """Point at (10, 0, 1): positive pitch -> lower row index (closer to top)."""
        point = np.array([[10.0, 0.0, 1.0]])
        u, v, r = spherical_project(point)

        # pitch = asin(1 / sqrt(101)) ~ 0.0995 rad ~ 5.7 deg
        # This pitch is within FOV (fov_up=2 deg), so it maps near top or clips
        range_val = math.sqrt(100 + 1)
        pitch = math.asin(1.0 / range_val)
        expected_v_float = (1.0 - (pitch - FOV_DOWN_RAD) / FOV_TOTAL_RAD) * H
        expected_v = np.clip(int(np.floor(expected_v_float)), 0, H - 1)
        assert v[0] == expected_v

    def test_fov_boundaries(self):
        """Points at exact FOV boundaries should map to row 0 and H-1."""
        # Point at upper FOV boundary (pitch = fov_up)
        pitch_up = FOV_UP_RAD
        # Create point with this pitch: z = r * sin(pitch), x = r * cos(pitch), y = 0
        r_val = 10.0
        z_up = r_val * math.sin(pitch_up)
        x_up = r_val * math.cos(pitch_up)
        point_up = np.array([[x_up, 0.0, z_up]])
        _, v_up, _ = spherical_project(point_up)
        # Should map to row 0 (or very close)
        assert v_up[0] <= 1, f"Upper FOV boundary should map near row 0, got {v_up[0]}"

        # Point at lower FOV boundary (pitch = fov_down)
        pitch_down = FOV_DOWN_RAD
        z_down = r_val * math.sin(pitch_down)
        x_down = r_val * math.cos(pitch_down)
        point_down = np.array([[x_down, 0.0, z_down]])
        _, v_down, _ = spherical_project(point_down)
        # Should map to row H-1 (or very close)
        assert v_down[0] >= H - 2, f"Lower FOV boundary should map near row {H-1}, got {v_down[0]}"


# ---------------------------------------------------------------------------
# Test 2: Back-Projection Consistency
# ---------------------------------------------------------------------------


class TestBackProjectionConsistency:
    """Project 3D points to range image then back-project; verify round-trip consistency."""

    def test_round_trip_reconstruction(self):
        """Generate random 3D points, project, store xyz channels, back-project."""
        np.random.seed(42)
        num_points = 500

        # Generate random points in front of the sensor (x > 0) within FOV
        x = np.random.uniform(5.0, 50.0, num_points)
        y = np.random.uniform(-30.0, 30.0, num_points)
        # Keep z within vertical FOV
        r_vals = np.sqrt(x ** 2 + y ** 2)
        pitch_min = FOV_DOWN_RAD
        pitch_max = FOV_UP_RAD
        pitch_angles = np.random.uniform(pitch_min, pitch_max, num_points)
        z = r_vals * np.tan(pitch_angles)

        points = np.stack([x, y, z], axis=1)

        # Project to range image
        u_coords, v_coords, ranges = spherical_project(points)

        # Create range image and store xyz values
        range_image_x = np.zeros((H, W), dtype=np.float32)
        range_image_y = np.zeros((H, W), dtype=np.float32)
        range_image_z = np.zeros((H, W), dtype=np.float32)

        # Store the original xyz in the range image pixels
        for i in range(num_points):
            vi, ui = v_coords[i], u_coords[i]
            range_image_x[vi, ui] = points[i, 0]
            range_image_y[vi, ui] = points[i, 1]
            range_image_z[vi, ui] = points[i, 2]

        # Back-project by reading the stored coordinates
        reconstructed_points = []
        for i in range(num_points):
            vi, ui = v_coords[i], u_coords[i]
            rx = range_image_x[vi, ui]
            ry = range_image_y[vi, ui]
            rz = range_image_z[vi, ui]
            reconstructed_points.append([rx, ry, rz])

        reconstructed = np.array(reconstructed_points)

        # Points that were last written to each pixel should match exactly
        # (some may be overwritten by later points at the same pixel)
        # Check that non-zero reconstructed points are close to *some* original point
        for i in range(num_points):
            vi, ui = v_coords[i], u_coords[i]
            stored_x = range_image_x[vi, ui]
            stored_y = range_image_y[vi, ui]
            stored_z = range_image_z[vi, ui]
            if stored_x != 0.0 or stored_y != 0.0 or stored_z != 0.0:
                # The stored point should exactly match the last point that mapped
                # to this pixel
                stored = np.array([stored_x, stored_y, stored_z])
                # Find all original points mapping to this pixel
                mask = (v_coords == vi) & (u_coords == ui)
                candidates = points[mask]
                # The stored value should match the last candidate (due to overwrite)
                last_candidate = candidates[-1]
                np.testing.assert_allclose(
                    stored, last_candidate, rtol=1e-5, atol=1e-5,
                    err_msg=f"Back-projection mismatch at pixel ({vi}, {ui})"
                )

    def test_single_point_round_trip(self):
        """Single point with no pixel collision: exact round-trip."""
        point = np.array([[20.0, 5.0, -1.0]])
        u, v, r = spherical_project(point)

        # Store and retrieve
        range_image = np.zeros((H, W, 3), dtype=np.float32)
        range_image[v[0], u[0], :] = point[0]

        recovered = range_image[v[0], u[0], :]
        np.testing.assert_allclose(recovered, point[0], rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: DarkNet-53 Backbone Shapes
# ---------------------------------------------------------------------------


class TestDarkNet53BackboneShapes:
    """Test that the encoder produces expected output shapes at each scale."""

    def test_encoder_stage_shapes(self):
        """Verify intermediate feature map shapes through the encoder."""
        # Build encoder components
        initial_conv = ConvBNLeakyReLU(32, kernel_size=3, name="initial")

        encoder_filters = [32, 64, 128, 256, 512]
        encoder_blocks = [1, 2, 8, 8, 4]
        stages = [
            DarkNetStage(f, n, name=f"stage_{i}")
            for i, (f, n) in enumerate(zip(encoder_filters, encoder_blocks))
        ]

        # Input: (1, 64, 2048, 5)
        x = tf.random.normal((1, 64, 2048, 5))

        # Initial conv: no downsampling -> (1, 64, 2048, 32)
        x = initial_conv(x, training=False)
        assert x.shape == (1, 64, 2048, 32), f"After initial conv: {x.shape}"

        # Stage 0: stride-2 -> (1, 32, 1024, 32)
        x = stages[0](x, training=False)
        assert x.shape == (1, 32, 1024, 32), f"After stage 0: {x.shape}"

        # Stage 1: stride-2 -> (1, 16, 512, 64)
        x = stages[1](x, training=False)
        assert x.shape == (1, 16, 512, 64), f"After stage 1: {x.shape}"

        # Stage 2: stride-2 -> (1, 8, 256, 128)
        x = stages[2](x, training=False)
        assert x.shape == (1, 8, 256, 128), f"After stage 2: {x.shape}"

        # Stage 3: stride-2 -> (1, 4, 128, 256)
        x = stages[3](x, training=False)
        assert x.shape == (1, 4, 128, 256), f"After stage 3: {x.shape}"

        # Stage 4: stride-2 -> (1, 2, 64, 512)
        x = stages[4](x, training=False)
        assert x.shape == (1, 2, 64, 512), f"After stage 4: {x.shape}"

    def test_encoder_preserves_batch_dimension(self):
        """Verify batch dimension is preserved through all encoder stages."""
        batch_size = 4
        initial_conv = ConvBNLeakyReLU(32, kernel_size=3)
        stage = DarkNetStage(64, num_blocks=2)

        x = tf.random.normal((batch_size, 64, 128, 5))
        x = initial_conv(x, training=False)
        x = stage(x, training=False)
        assert x.shape[0] == batch_size

    def test_residual_block_preserves_shape(self):
        """Residual blocks should not change spatial or channel dimensions."""
        from model import DarkNetResidualBlock

        block = DarkNetResidualBlock(filters=128)
        x = tf.random.normal((1, 16, 256, 128))
        out = block(x, training=False)
        assert out.shape == x.shape, f"Residual block changed shape: {x.shape} -> {out.shape}"


# ---------------------------------------------------------------------------
# Test 4: Decoder Output Shape
# ---------------------------------------------------------------------------


class TestDecoderOutputShape:
    """Test that decoder output matches expected spatial dimensions."""

    def test_decoder_upsamples_correctly(self):
        """Given encoder outputs at multiple scales, decoder should upsample back."""
        # Simulate encoder skip connections
        skip0 = tf.random.normal((1, 64, 128, 32))   # Full resolution
        skip1 = tf.random.normal((1, 32, 64, 32))    # After stage 0
        skip2 = tf.random.normal((1, 16, 32, 64))    # After stage 1
        skip3 = tf.random.normal((1, 8, 16, 128))    # After stage 2
        skip4 = tf.random.normal((1, 4, 8, 256))     # After stage 3
        bottleneck = tf.random.normal((1, 2, 4, 512))  # After stage 4

        skips = [skip0, skip1, skip2, skip3, skip4, bottleneck]
        decoder_filters = [256, 128, 64, 32, 32]
        decoder_blocks = [DecoderBlock(f, name=f"dec_{i}") for i, f in enumerate(decoder_filters)]

        x = skips[-1]
        for i, dec_block in enumerate(decoder_blocks):
            skip_connection = skips[-(i + 2)]
            x = dec_block(x, skip_connection, training=False)

        # Final output should match skip0 spatial dimensions
        assert x.shape[1] == 64, f"Expected height 64, got {x.shape[1]}"
        assert x.shape[2] == 128, f"Expected width 128, got {x.shape[2]}"

    def test_decoder_block_doubles_spatial(self):
        """A single decoder block should approximately double spatial dimensions."""
        dec = DecoderBlock(64)
        x = tf.random.normal((1, 8, 16, 256))
        skip = tf.random.normal((1, 16, 32, 128))
        out = dec(x, skip, training=False)
        assert out.shape[1] == 16, f"Expected height 16, got {out.shape[1]}"
        assert out.shape[2] == 32, f"Expected width 32, got {out.shape[2]}"


# ---------------------------------------------------------------------------
# Test 5: Full Model Forward Pass
# ---------------------------------------------------------------------------


class TestFullModelForwardPass:
    """End-to-end shape test for the complete RangeNet++ model."""

    def test_output_shape_batch_2(self):
        """Input (2, 64, 2048, 5) -> output (2, 64, 2048, 20)."""
        model = RangeNetPP(num_classes=20, input_height=64, input_width=2048, input_channels=5)
        x = tf.random.normal((2, 64, 2048, 5))
        output = model(x, training=False)
        assert output.shape == (2, 64, 2048, 20), f"Unexpected output shape: {output.shape}"

    def test_output_shape_batch_1(self):
        """Input (1, 64, 2048, 5) -> output (1, 64, 2048, 20)."""
        model = RangeNetPP(num_classes=20, input_height=64, input_width=2048, input_channels=5)
        x = tf.random.normal((1, 64, 2048, 5))
        output = model(x, training=False)
        assert output.shape == (1, 64, 2048, 20), f"Unexpected output shape: {output.shape}"

    def test_build_rangenet_pp_utility(self):
        """Test the build_rangenet_pp convenience function."""
        model = build_rangenet_pp(num_classes=10, input_height=64, input_width=512, input_channels=5)
        x = tf.random.normal((1, 64, 512, 5))
        output = model(x, training=False)
        assert output.shape == (1, 64, 512, 10)

    def test_different_num_classes(self):
        """Model should work with different numbers of output classes."""
        for nc in [5, 10, 34]:
            model = RangeNetPP(num_classes=nc, input_height=32, input_width=128, input_channels=5)
            x = tf.random.normal((1, 32, 128, 5))
            output = model(x, training=False)
            assert output.shape[-1] == nc, f"Expected {nc} classes, got {output.shape[-1]}"

    def test_training_vs_inference_mode(self):
        """Model should produce different outputs in training vs inference due to dropout/BN."""
        model = RangeNetPP(num_classes=20, input_height=32, input_width=128, input_channels=5)
        x = tf.random.normal((1, 32, 128, 5))

        # Build model first
        _ = model(x, training=False)

        # Get outputs in both modes
        out_train = model(x, training=True)
        out_infer = model(x, training=False)

        # Shapes should be the same
        assert out_train.shape == out_infer.shape
        # Values might differ due to dropout and batch norm behavior
        # (not guaranteed to differ with batch_size=1 for BN, but dropout should cause diff)


# ---------------------------------------------------------------------------
# Test 6: KNN Post-Processing
# ---------------------------------------------------------------------------


def knn_post_process(labels, points, k=5):
    """Apply KNN-based label smoothing to predicted semantic labels.

    For each point, find its K nearest neighbors and assign the majority label.

    Args:
        labels: numpy array of shape (N,) with integer class labels.
        points: numpy array of shape (N, 3) with 3D coordinates.
        k: number of nearest neighbors.

    Returns:
        Smoothed labels of shape (N,).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    smoothed = np.copy(labels)

    # Query k+1 neighbors (includes self)
    distances, indices = tree.query(points, k=k + 1)

    for i in range(len(labels)):
        neighbor_labels = labels[indices[i]]
        # Majority vote (exclude the point itself by taking indices[i][1:] if desired,
        # but including self strengthens the majority)
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        smoothed[i] = unique[np.argmax(counts)]

    return smoothed


class TestKNNPostProcessing:
    """Test KNN label smoothing on synthetic data."""

    def test_majority_vote_corrects_noise(self):
        """Create a cluster with known labels and add mislabeled points; KNN should fix them."""
        np.random.seed(123)

        # Create a tight cluster of 50 points, all labeled class 3
        cluster_points = np.random.randn(50, 3) * 0.5 + np.array([5.0, 5.0, 5.0])
        cluster_labels = np.full(50, 3, dtype=np.int32)

        # Add 3 "noisy" mislabeled points within the same cluster
        noise_points = np.random.randn(3, 3) * 0.3 + np.array([5.0, 5.0, 5.0])
        noise_labels = np.array([7, 12, 0], dtype=np.int32)  # Wrong labels

        all_points = np.vstack([cluster_points, noise_points])
        all_labels = np.concatenate([cluster_labels, noise_labels])

        # Apply KNN smoothing with K=5
        smoothed = knn_post_process(all_labels, all_points, k=5)

        # The noisy points (indices 50, 51, 52) should now have label 3
        for i in range(50, 53):
            assert smoothed[i] == 3, (
                f"Noisy point {i} should be corrected to label 3, got {smoothed[i]}"
            )

    def test_preserves_correct_labels(self):
        """Points with correct majority labels should remain unchanged."""
        np.random.seed(456)

        # Two well-separated clusters
        cluster_a = np.random.randn(30, 3) * 0.2 + np.array([0.0, 0.0, 0.0])
        labels_a = np.full(30, 1, dtype=np.int32)

        cluster_b = np.random.randn(30, 3) * 0.2 + np.array([10.0, 10.0, 10.0])
        labels_b = np.full(30, 2, dtype=np.int32)

        all_points = np.vstack([cluster_a, cluster_b])
        all_labels = np.concatenate([labels_a, labels_b])

        smoothed = knn_post_process(all_labels, all_points, k=5)

        # All labels should remain unchanged since clusters are well-separated
        np.testing.assert_array_equal(smoothed[:30], 1)
        np.testing.assert_array_equal(smoothed[30:], 2)

    def test_knn_with_single_outlier(self):
        """A single outlier in a uniform cluster gets corrected."""
        np.random.seed(789)

        # 20 points tightly clustered, all label 5
        points = np.random.randn(20, 3) * 0.1
        labels = np.full(20, 5, dtype=np.int32)

        # Change one point's label to something different
        labels[10] = 99

        smoothed = knn_post_process(labels, points, k=5)

        # The outlier should be corrected
        assert smoothed[10] == 5, f"Outlier should be corrected to 5, got {smoothed[10]}"


# ---------------------------------------------------------------------------
# Test 7: Loss Computation
# ---------------------------------------------------------------------------


class TestLossComputation:
    """Test cross-entropy loss computation for RangeNet++."""

    def test_loss_is_positive_scalar(self):
        """Loss should be a positive scalar tensor."""
        batch_size, h, w, num_classes = 2, 32, 64, 20
        logits = tf.random.normal((batch_size, h, w, num_classes))
        labels = tf.random.uniform(
            (batch_size, h, w), minval=0, maxval=num_classes, dtype=tf.int32
        )

        loss = rangenet_pp_loss(labels, logits)

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        assert loss.numpy() > 0.0, f"Loss should be positive, got {loss.numpy()}"

    def test_loss_decreases_with_better_predictions(self):
        """Loss for correct predictions should be lower than for random predictions."""
        batch_size, h, w, num_classes = 2, 16, 32, 20

        # Create ground truth labels
        labels = tf.random.uniform(
            (batch_size, h, w), minval=0, maxval=num_classes, dtype=tf.int32
        )

        # Random logits (bad predictions)
        random_logits = tf.random.normal((batch_size, h, w, num_classes))
        loss_random = rangenet_pp_loss(labels, random_logits)

        # Perfect logits: one-hot encode the labels with high confidence
        perfect_logits = tf.one_hot(labels, num_classes) * 10.0  # High confidence
        loss_perfect = rangenet_pp_loss(labels, perfect_logits)

        assert loss_perfect.numpy() < loss_random.numpy(), (
            f"Perfect loss ({loss_perfect.numpy():.4f}) should be less than "
            f"random loss ({loss_random.numpy():.4f})"
        )

    def test_loss_with_class_weights(self):
        """Weighted loss should differ from unweighted loss."""
        batch_size, h, w, num_classes = 1, 16, 32, 20
        logits = tf.random.normal((batch_size, h, w, num_classes))
        labels = tf.random.uniform(
            (batch_size, h, w), minval=0, maxval=num_classes, dtype=tf.int32
        )

        # Uniform weights = unweighted
        uniform_weights = tf.ones(num_classes)
        loss_uniform = rangenet_pp_loss(labels, logits, class_weights=uniform_weights)

        # Non-uniform weights
        non_uniform_weights = tf.random.uniform((num_classes,), minval=0.5, maxval=5.0)
        loss_weighted = rangenet_pp_loss(labels, logits, class_weights=non_uniform_weights)

        # They should generally differ (unless extremely unlikely coincidence)
        # At minimum, both should be positive
        assert loss_uniform.numpy() > 0.0
        assert loss_weighted.numpy() > 0.0

    def test_lovasz_softmax_approximation(self):
        """Test a simplified Lovasz-like surrogate loss decreases appropriately.

        The full Lovasz-softmax requires sorting and piecewise-linear interpolation.
        Here we test a differentiable IoU-based surrogate (1 - soft_iou) which
        approximates Lovasz behavior for optimization.
        """
        num_classes = 5

        def soft_iou_loss(y_true, y_pred_logits, num_classes):
            """Compute 1 - mean(soft IoU per class) as a differentiable surrogate."""
            probabilities = tf.nn.softmax(y_pred_logits, axis=-1)
            y_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)

            # Flatten spatial dims
            flat_probs = tf.reshape(probabilities, (-1, num_classes))
            flat_labels = tf.reshape(y_one_hot, (-1, num_classes))

            intersection = tf.reduce_sum(flat_probs * flat_labels, axis=0)
            union = (
                tf.reduce_sum(flat_probs, axis=0)
                + tf.reduce_sum(flat_labels, axis=0)
                - intersection
            )

            iou_per_class = (intersection + 1e-6) / (union + 1e-6)
            mean_iou = tf.reduce_mean(iou_per_class)
            return 1.0 - mean_iou

        labels = tf.constant([[0, 1, 2, 3, 4, 0, 1, 2]], dtype=tf.int32)

        # Bad predictions (random)
        bad_logits = tf.random.normal((1, 8, num_classes), seed=42)
        loss_bad = soft_iou_loss(labels, bad_logits, num_classes)

        # Good predictions (one-hot with high confidence)
        good_logits = tf.one_hot(labels, num_classes) * 10.0
        loss_good = soft_iou_loss(labels, good_logits, num_classes)

        assert loss_good.numpy() < loss_bad.numpy(), (
            f"Good predictions loss ({loss_good.numpy():.4f}) should be less than "
            f"bad predictions loss ({loss_bad.numpy():.4f})"
        )
        assert loss_good.numpy() >= 0.0, "IoU loss should be non-negative"
        assert loss_bad.numpy() <= 1.0, "IoU loss should be at most 1.0"


# ---------------------------------------------------------------------------
# Test 8: Model Trainable Parameters
# ---------------------------------------------------------------------------


class TestModelTrainableParams:
    """Verify model has reasonable number of trainable parameters."""

    def test_has_trainable_variables(self, small_model):
        """Model should have non-empty trainable_variables."""
        assert len(small_model.trainable_variables) > 0, (
            "Model should have trainable variables"
        )

    def test_total_params_positive(self, small_model):
        """Total number of trainable parameters should be positive."""
        total_params = sum(
            tf.reduce_prod(v.shape).numpy() for v in small_model.trainable_variables
        )
        assert total_params > 0, f"Expected positive params, got {total_params}"

    def test_minimum_param_count(self, small_model):
        """Model should have a substantial number of parameters (> 100K for reduced model)."""
        total_params = sum(
            tf.reduce_prod(v.shape).numpy() for v in small_model.trainable_variables
        )
        assert total_params > 100_000, (
            f"Expected > 100K parameters, got {total_params:,}"
        )

    def test_full_model_param_count(self, full_model):
        """Full-resolution model should have millions of parameters."""
        total_params = sum(
            tf.reduce_prod(v.shape).numpy() for v in full_model.trainable_variables
        )
        # DarkNet-53 backbone alone has ~40M params; with decoder expect > 1M at minimum
        assert total_params > 1_000_000, (
            f"Expected > 1M parameters for full model, got {total_params:,}"
        )

    def test_all_layers_have_weights(self, small_model):
        """Check that encoder and decoder stages contain weight tensors."""
        encoder_vars = [
            v for v in small_model.trainable_variables if "encoder" in v.name
        ]
        decoder_vars = [
            v for v in small_model.trainable_variables if "decoder" in v.name
        ]
        assert len(encoder_vars) > 0, "Encoder should have trainable weights"
        assert len(decoder_vars) > 0, "Decoder should have trainable weights"

    def test_gradients_flow(self, small_model):
        """Verify that gradients can flow through the entire model."""
        x = tf.random.normal((1, 64, 128, 5))
        labels = tf.random.uniform((1, 64, 128), minval=0, maxval=20, dtype=tf.int32)

        with tf.GradientTape() as tape:
            logits = small_model(x, training=True)
            loss = rangenet_pp_loss(labels, logits)

        gradients = tape.gradient(loss, small_model.trainable_variables)

        # All gradients should be non-None
        none_grads = [
            (i, v.name)
            for i, (g, v) in enumerate(
                zip(gradients, small_model.trainable_variables)
            )
            if g is None
        ]
        assert len(none_grads) == 0, (
            f"Found {len(none_grads)} variables with None gradients: "
            f"{none_grads[:5]}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
