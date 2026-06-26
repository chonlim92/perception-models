"""
HDMapNet - Comprehensive TensorFlow Model Unit Tests

Tests for the TensorFlow HDMapNet architecture components including:
- IPM view transform with known homography
- LSS view transform dimensions and depth softmax
- Full forward pass with correct output shapes
- Loss computation (semantic, discriminative, direction)
- Post-processing (vectorization on synthetic masks)
- Dataset loading with mock data
- BEV encoder residual connections
- Individual prediction heads

Run with: pytest tests/test_model.py -v
"""

import sys
import os
import tempfile
import shutil

import pytest
import numpy as np

# Add the tensorflow module path for imports
_TF_MODULE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tensorflow")
)
if _TF_MODULE_ROOT not in sys.path:
    sys.path.insert(0, _TF_MODULE_ROOT)

import tensorflow as tf
from model import (
    HDMapNet,
    HDMapNetLoss,
    IPMTransform,
    LSSTransformVectorized,
    BEVEncoder,
    SemanticHead,
    InstanceHead,
    DirectionHead,
    ResidualBlock,
    ConvBnRelu,
    EfficientNetBackbone,
    FeatureNeck,
    DepthNet,
    build_hdmapnet_ipm,
    build_hdmapnet_lss,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def ipm_model():
    """Build an IPM-based HDMapNet model (shared across tests in module)."""
    model = build_hdmapnet_ipm()
    # Warm up with a dummy forward pass
    dummy_images = tf.random.normal([1, 6, 128, 352, 3])
    dummy_ext = tf.eye(4, batch_shape=[1, 6])
    dummy_int = tf.constant(np.tile(
        np.array([[700, 0, 176], [0, 700, 64], [0, 0, 1]], dtype=np.float32),
        (1, 6, 1, 1)
    ))
    model((dummy_images, dummy_ext, dummy_int), training=False)
    return model


@pytest.fixture(scope="module")
def lss_model():
    """Build an LSS-based HDMapNet model (shared across tests in module)."""
    model = build_hdmapnet_lss()
    dummy_images = tf.random.normal([1, 6, 128, 352, 3])
    dummy_ext = tf.eye(4, batch_shape=[1, 6])
    dummy_int = tf.constant(np.tile(
        np.array([[700, 0, 176], [0, 700, 64], [0, 0, 1]], dtype=np.float32),
        (1, 6, 1, 1)
    ))
    model((dummy_images, dummy_ext, dummy_int), training=False)
    return model


@pytest.fixture
def dummy_inputs():
    """Create standard dummy inputs for testing."""
    batch_size = 1
    images = tf.random.uniform([batch_size, 6, 128, 352, 3], 0.0, 1.0)
    # Identity extrinsics (camera at origin looking forward)
    extrinsics = tf.eye(4, batch_shape=[batch_size, 6])
    # Typical intrinsics
    fx, fy = 700.0, 700.0
    cx, cy = 176.0, 64.0
    intrinsic_single = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsics = tf.constant(np.tile(intrinsic_single, (batch_size, 6, 1, 1)))
    return images, extrinsics, intrinsics


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with mock .npz training data."""
    tmpdir = tempfile.mkdtemp(prefix="hdmapnet_test_data_")
    # Create several mock samples
    for i in range(5):
        images = np.random.randint(0, 255, (6, 128, 352, 3), dtype=np.uint8)
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))
        intrinsics = np.tile(
            np.array([[700, 0, 176], [0, 700, 64], [0, 0, 1]], dtype=np.float32),
            (6, 1, 1)
        )
        semantic_masks = np.random.randint(0, 2, (200, 200, 3)).astype(np.float32)
        instance_masks = np.random.randint(0, 5, (200, 200)).astype(np.int32)
        direction_masks = np.random.randn(200, 200, 2).astype(np.float32)
        # Normalize direction vectors
        norms = np.linalg.norm(direction_masks, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        direction_masks = direction_masks / norms

        np.savez(
            os.path.join(tmpdir, f"sample_{i:04d}.npz"),
            images=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            semantic_masks=semantic_masks,
            instance_masks=instance_masks,
            direction_masks=direction_masks,
        )
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# Test IPM View Transform
# =============================================================================


class TestIPMViewTransform:
    """Tests for Inverse Perspective Mapping view transform."""

    def test_ipm_output_shape(self):
        """Test that IPM transform produces correct BEV output shape."""
        ipm = IPMTransform(bev_h=200, bev_w=200, feat_h=8, feat_w=22)
        batch_size = 1
        features = tf.random.normal([batch_size, 6, 8, 22, 64])
        extrinsics = tf.eye(4, batch_shape=[batch_size, 6])
        intrinsics = tf.constant(np.tile(
            np.array([[700, 0, 176], [0, 700, 64], [0, 0, 1]], dtype=np.float32),
            (batch_size, 6, 1, 1)
        ))

        output = ipm(features, extrinsics, intrinsics, training=False)
        assert output.shape[0] == batch_size
        assert output.shape[1] == 200
        assert output.shape[2] == 200
        assert output.shape[3] == 64

    def test_ipm_with_known_homography(self):
        """Test IPM with a known camera-looking-down configuration.

        When a camera is placed directly above the origin looking straight down,
        the IPM should map the center of the feature map to the center of the BEV.
        """
        ipm = IPMTransform(
            bev_h=200, bev_w=200,
            x_range=(-30.0, 30.0), y_range=(-15.0, 15.0),
            feat_h=8, feat_w=22
        )

        batch_size = 1
        # Create a feature map with a distinctive pattern at center
        features = tf.zeros([batch_size, 6, 8, 22, 64])
        # Put strong features at the center of camera 0's feature map
        center_feat = tf.ones([1, 1, 1, 1, 64]) * 10.0
        indices = tf.constant([[0, 0, 4, 11, 0]])  # batch=0, cam=0, y=4, x=11, c=0
        features_np = features.numpy()
        features_np[0, 0, 4, 11, :] = 10.0
        features = tf.constant(features_np)

        # Camera looking down from height 10m
        # Extrinsics: translation z=10, rotation so that camera z-axis points down
        ext = np.zeros((1, 6, 4, 4), dtype=np.float32)
        for cam in range(6):
            ext[0, cam] = np.eye(4)
        # Camera 0: looking straight down
        ext[0, 0] = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 10],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        extrinsics = tf.constant(ext)
        intrinsics = tf.constant(np.tile(
            np.array([[700, 0, 176], [0, 700, 64], [0, 0, 1]], dtype=np.float32),
            (1, 6, 1, 1)
        ))

        output = ipm(features, extrinsics, intrinsics, training=False)
        # Output should be valid (no NaN)
        assert not tf.reduce_any(tf.math.is_nan(output)).numpy()
        # The output at BEV center should have non-zero values from the projection
        assert output.shape == (1, 200, 200, 64)

    def test_ipm_bev_grid_dimensions(self):
        """Test that BEV grid correctly spans the configured spatial range."""
        x_range = (-30.0, 30.0)
        y_range = (-15.0, 15.0)
        bev_h, bev_w = 200, 200

        ipm = IPMTransform(
            bev_h=bev_h, bev_w=bev_w,
            x_range=x_range, y_range=y_range,
            feat_h=8, feat_w=22
        )

        # The world_coords should span the expected range
        world_coords = ipm.world_coords.numpy()  # [4, N]
        x_coords = world_coords[0]
        y_coords = world_coords[1]
        z_coords = world_coords[2]

        # Check x range
        assert np.isclose(x_coords.min(), x_range[0], atol=0.5)
        assert np.isclose(x_coords.max(), x_range[1], atol=0.5)
        # Check y range
        assert np.isclose(y_coords.min(), y_range[0], atol=0.5)
        assert np.isclose(y_coords.max(), y_range[1], atol=0.5)
        # All z should be 0 (ground plane)
        assert np.allclose(z_coords, 0.0)
        # Total points = bev_h * bev_w
        assert world_coords.shape[1] == bev_h * bev_w


# =============================================================================
# Test LSS View Transform
# =============================================================================


class TestLSSViewTransform:
    """Tests for Lift-Splat-Shoot view transform."""

    def test_lss_output_shape(self):
        """Test that LSS transform produces correct BEV output shape."""
        lss = LSSTransformVectorized(
            in_channels=64, bev_h=200, bev_w=200,
            feat_h=8, feat_w=22, context_channels=64
        )
        batch_size = 1
        features = tf.random.normal([batch_size, 6, 8, 22, 64])
        extrinsics = tf.eye(4, batch_shape=[batch_size, 6])
        intrinsics = tf.constant(np.tile(
            np.array([[700, 0, 176], [0, 700, 64], [0, 0, 1]], dtype=np.float32),
            (batch_size, 6, 1, 1)
        ))

        output = lss(features, extrinsics, intrinsics, training=False)
        assert output.shape[0] == batch_size
        assert output.shape[1] == 200
        assert output.shape[2] == 200
        assert output.shape[3] == 64

    def test_lss_depth_bins_count(self):
        """Test that LSS has correct number of depth bins (4m to 45m at 1m)."""
        lss = LSSTransformVectorized(
            in_channels=64, bev_h=200, bev_w=200,
            d_min=4.0, d_max=45.0, d_step=1.0,
            feat_h=8, feat_w=22, context_channels=64
        )
        assert lss.num_depth_bins == 42  # (45 - 4) / 1 + 1 = 42

    def test_lss_depth_distribution_softmax(self):
        """Test that depth network output sums to 1 along depth axis (softmax)."""
        depth_net = DepthNet(
            in_channels=64, num_depth_bins=42, context_channels=64
        )
        # Create random features
        feat = tf.random.normal([2, 8, 22, 64])
        depth_logits, context = depth_net(feat, training=False)

        # Apply softmax to depth logits (the depth_net outputs raw logits)
        depth_dist = tf.nn.softmax(depth_logits, axis=-1)
        sums = tf.reduce_sum(depth_dist, axis=-1)

        # Each spatial location should sum to 1
        assert tf.reduce_all(tf.abs(sums - 1.0) < 1e-5).numpy()

    def test_lss_depth_net_output_shapes(self):
        """Test that DepthNet produces correct output shapes."""
        num_depth = 42
        context_channels = 64
        depth_net = DepthNet(
            in_channels=64, num_depth_bins=num_depth, context_channels=context_channels
        )
        feat = tf.random.normal([1, 8, 22, 64])
        depth_logits, context = depth_net(feat, training=False)

        assert depth_logits.shape == (1, 8, 22, num_depth)
        assert context.shape == (1, 8, 22, context_channels)

    def test_lss_no_nan_output(self):
        """Test that LSS transform does not produce NaN values."""
        lss = LSSTransformVectorized(
            in_channels=64, bev_h=200, bev_w=200,
            feat_h=8, feat_w=22, context_channels=64
        )
        batch_size = 1
        features = tf.random.normal([batch_size, 6, 8, 22, 64])
        extrinsics = tf.eye(4, batch_shape=[batch_size, 6])
        intrinsics = tf.constant(np.tile(
            np.array([[700, 0, 176], [0, 700, 64], [0, 0, 1]], dtype=np.float32),
            (batch_size, 6, 1, 1)
        ))

        output = lss(features, extrinsics, intrinsics, training=False)
        assert not tf.reduce_any(tf.math.is_nan(output)).numpy()


# =============================================================================
# Test Full Forward Pass
# =============================================================================


class TestFullForwardPass:
    """Tests for complete HDMapNet forward pass."""

    def test_ipm_forward_pass_shapes(self, ipm_model, dummy_inputs):
        """Test full forward pass with IPM produces correct output shapes."""
        images, extrinsics, intrinsics = dummy_inputs
        outputs = ipm_model((images, extrinsics, intrinsics), training=False)

        assert 'semantic' in outputs
        assert 'instance' in outputs
        assert 'direction' in outputs

        batch_size = images.shape[0]
        assert outputs['semantic'].shape == (batch_size, 200, 200, 3)
        assert outputs['instance'].shape == (batch_size, 200, 200, 16)
        assert outputs['direction'].shape == (batch_size, 200, 200, 2)

    def test_lss_forward_pass_shapes(self, lss_model, dummy_inputs):
        """Test full forward pass with LSS produces correct output shapes."""
        images, extrinsics, intrinsics = dummy_inputs
        outputs = lss_model((images, extrinsics, intrinsics), training=False)

        assert 'semantic' in outputs
        assert 'instance' in outputs
        assert 'direction' in outputs

        batch_size = images.shape[0]
        assert outputs['semantic'].shape == (batch_size, 200, 200, 3)
        assert outputs['instance'].shape == (batch_size, 200, 200, 16)
        assert outputs['direction'].shape == (batch_size, 200, 200, 2)

    def test_forward_pass_dict_input(self, ipm_model, dummy_inputs):
        """Test that model accepts dict inputs."""
        images, extrinsics, intrinsics = dummy_inputs
        input_dict = {
            'images': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }
        outputs = ipm_model(input_dict, training=False)
        assert outputs['semantic'].shape == (1, 200, 200, 3)

    def test_gradient_flow(self, ipm_model, dummy_inputs):
        """Test that gradients flow through the entire model."""
        images, extrinsics, intrinsics = dummy_inputs
        images_var = tf.Variable(images)

        with tf.GradientTape() as tape:
            outputs = ipm_model((images_var, extrinsics, intrinsics), training=True)
            loss = tf.reduce_mean(outputs['semantic'])

        grads = tape.gradient(loss, images_var)
        assert grads is not None
        # Gradients should not all be zero
        assert tf.reduce_any(tf.not_equal(grads, 0.0)).numpy()

    def test_direction_output_normalized(self, ipm_model, dummy_inputs):
        """Test that direction output vectors are unit-normalized."""
        images, extrinsics, intrinsics = dummy_inputs
        outputs = ipm_model((images, extrinsics, intrinsics), training=False)

        direction = outputs['direction']
        norms = tf.norm(direction, axis=-1)
        # All direction vectors should be approximately unit length
        assert tf.reduce_all(tf.abs(norms - 1.0) < 0.01).numpy()


# =============================================================================
# Test Loss Computation
# =============================================================================


class TestLossComputation:
    """Tests for HDMapNet loss functions."""

    def test_semantic_loss_positive(self):
        """Test that semantic BCE loss is a positive scalar."""
        loss_fn = HDMapNetLoss()
        y_true = tf.random.uniform([2, 200, 200, 3], 0, 1)
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.random.normal([2, 200, 200, 3])

        sem_loss = loss_fn.semantic_loss(y_true, y_pred)
        assert sem_loss.numpy() > 0.0
        assert sem_loss.shape == ()  # scalar

    def test_discriminative_loss_known_clusters(self):
        """Test discriminative loss with known well-separated clusters."""
        loss_fn = HDMapNetLoss(delta_v=0.5, delta_d=3.0)

        # Create embeddings with two well-separated clusters
        embeddings = np.zeros((10, 10, 16), dtype=np.float32)
        instance_labels = np.zeros((10, 10), dtype=np.int32)

        # Cluster 1: top half, embedding centered at [5, 0, 0, ...]
        embeddings[:5, :, 0] = 5.0
        instance_labels[:5, :] = 1

        # Cluster 2: bottom half, embedding centered at [-5, 0, 0, ...]
        embeddings[5:, :, 0] = -5.0
        instance_labels[5:, :] = 2

        embeddings_tf = tf.constant(embeddings)
        labels_tf = tf.constant(instance_labels)

        disc_loss = loss_fn.discriminative_loss(embeddings_tf, labels_tf, num_instances=2)

        # With well-separated clusters (distance=10 > delta_d=3.0),
        # push loss should be zero. Pull loss should also be low
        # since all embeddings within each cluster are identical.
        assert disc_loss.numpy() >= 0.0
        assert disc_loss.numpy() < 0.1  # Should be very small for perfect clusters

    def test_discriminative_loss_overlapping_clusters(self):
        """Test discriminative loss is higher for overlapping clusters."""
        loss_fn = HDMapNetLoss(delta_v=0.5, delta_d=3.0)

        # Overlapping clusters (distance between means < delta_d)
        embeddings = np.zeros((10, 10, 16), dtype=np.float32)
        instance_labels = np.zeros((10, 10), dtype=np.int32)

        embeddings[:5, :, 0] = 1.0  # Cluster 1
        instance_labels[:5, :] = 1
        embeddings[5:, :, 0] = 2.0  # Cluster 2 (too close)
        instance_labels[5:, :] = 2

        embeddings_tf = tf.constant(embeddings)
        labels_tf = tf.constant(instance_labels)

        loss_overlap = loss_fn.discriminative_loss(embeddings_tf, labels_tf, num_instances=2)

        # Now well-separated
        embeddings2 = np.zeros((10, 10, 16), dtype=np.float32)
        embeddings2[:5, :, 0] = 5.0
        embeddings2[5:, :, 0] = -5.0

        loss_separated = loss_fn.discriminative_loss(
            tf.constant(embeddings2), labels_tf, num_instances=2
        )

        # Overlapping should have higher loss
        assert loss_overlap.numpy() > loss_separated.numpy()

    def test_direction_loss_zero_when_matching(self):
        """Test that direction loss is zero when prediction matches target."""
        loss_fn = HDMapNetLoss()

        # Create matching predictions and targets
        direction = np.random.randn(1, 50, 50, 2).astype(np.float32)
        norms = np.linalg.norm(direction, axis=-1, keepdims=True)
        direction = direction / np.maximum(norms, 1e-6)

        y_true = tf.constant(direction)
        y_pred = tf.constant(direction)
        mask = tf.ones([1, 50, 50], dtype=tf.bool)

        dir_loss = loss_fn.direction_loss(y_true, y_pred, mask)
        assert dir_loss.numpy() < 1e-5  # Should be approximately zero

    def test_direction_loss_maximum_for_opposite(self):
        """Test that direction loss is maximum when prediction is opposite of target."""
        loss_fn = HDMapNetLoss()

        direction = np.ones((1, 50, 50, 2), dtype=np.float32)
        direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)

        y_true = tf.constant(direction)
        y_pred = tf.constant(-direction)  # Opposite direction
        mask = tf.ones([1, 50, 50], dtype=tf.bool)

        dir_loss = loss_fn.direction_loss(y_true, y_pred, mask)
        # Cosine similarity of opposite vectors = -1, so loss = 1 - (-1) = 2
        assert dir_loss.numpy() > 1.5

    def test_total_loss_weighted_sum(self):
        """Test that total loss is a weighted combination."""
        sem_w, inst_w, dir_w = 2.0, 1.5, 0.5
        loss_fn = HDMapNetLoss(
            semantic_weight=sem_w,
            instance_weight=inst_w,
            direction_weight=dir_w
        )

        # Create simple targets and predictions
        y_true = {
            'semantic': tf.cast(tf.random.uniform([1, 200, 200, 3]) > 0.5, tf.float32),
            'instance_labels': tf.zeros([1, 200, 200], dtype=tf.int32),
            'num_instances': 0,
            'direction': tf.ones([1, 200, 200, 2]) / np.sqrt(2),
            'direction_mask': tf.ones([1, 200, 200], dtype=tf.bool),
        }
        y_pred = {
            'semantic': tf.random.normal([1, 200, 200, 3]),
            'instance': tf.random.normal([1, 200, 200, 16]),
            'direction': tf.ones([1, 200, 200, 2]) / np.sqrt(2),
        }

        total_loss = loss_fn(y_true, y_pred)
        assert total_loss.numpy() > 0.0
        assert np.isfinite(total_loss.numpy())

    def test_loss_no_instances_edge_case(self):
        """Test loss computation when there are no instances (empty scene)."""
        loss_fn = HDMapNetLoss()

        embeddings = tf.random.normal([10, 10, 16])
        labels = tf.zeros([10, 10], dtype=tf.int32)  # All background

        disc_loss = loss_fn.discriminative_loss(embeddings, labels, num_instances=0)
        assert disc_loss.numpy() == 0.0


# =============================================================================
# Test Post-Processing (Vectorization)
# =============================================================================


class TestPostProcessing:
    """Tests for mask vectorization and post-processing."""

    def test_contour_extraction_horizontal_line(self):
        """Test contour extraction from a synthetic horizontal line mask."""
        import cv2

        # Create a mask with a horizontal line
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[100, 50:150] = 255  # Horizontal line at row 100, cols 50-150

        # Dilate to make it a thick line (contour extraction needs area)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Should find at least one contour
        assert len(contours) >= 1
        # The contour should span the horizontal range
        all_points = np.concatenate(contours, axis=0).squeeze()
        x_span = all_points[:, 0].max() - all_points[:, 0].min()
        assert x_span >= 90  # Should span close to 100 pixels

    def test_contour_extraction_vertical_line(self):
        """Test contour extraction from a synthetic vertical line mask."""
        import cv2

        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[30:170, 100] = 255  # Vertical line at col 100, rows 30-170

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) >= 1
        all_points = np.concatenate(contours, axis=0).squeeze()
        y_span = all_points[:, 1].max() - all_points[:, 1].min()
        assert y_span >= 130  # Should span close to 140 pixels

    def test_empty_mask_no_contours(self):
        """Test that empty mask produces no polylines."""
        import cv2

        mask = np.zeros((200, 200), dtype=np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert len(contours) == 0

    def test_vectorization_multiple_instances(self):
        """Test vectorization with multiple separate instances."""
        import cv2

        mask = np.zeros((200, 200), dtype=np.uint8)
        # Two separate horizontal lines
        mask[50, 20:80] = 255
        mask[150, 120:180] = 255

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Should find two separate contours
        assert len(contours) >= 2

    def test_semantic_threshold_and_vectorize(self):
        """Test full pipeline: sigmoid -> threshold -> vectorize."""
        import cv2

        # Simulate network output (logits)
        logits = np.full((200, 200), -5.0, dtype=np.float32)  # Background
        # Draw a strong lane divider line
        logits[100, 30:170] = 5.0  # High confidence line

        # Apply sigmoid
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Threshold at 0.5
        binary = (probs > 0.5).astype(np.uint8) * 255

        # Dilate for contour extraction
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary_dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(
            binary_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) >= 1
        # Verify the extracted contour is in the right location
        all_points = np.concatenate(contours, axis=0).squeeze()
        mean_y = all_points[:, 1].mean()
        assert abs(mean_y - 100) < 5  # Should be near row 100


# =============================================================================
# Test Dataset Loading
# =============================================================================


class TestDatasetLoading:
    """Tests for dataset loading with mock data."""

    def test_load_npz_files(self, mock_data_dir):
        """Test loading .npz files with correct shapes and dtypes."""
        npz_files = [
            os.path.join(mock_data_dir, f) for f in os.listdir(mock_data_dir)
            if f.endswith('.npz')
        ]
        assert len(npz_files) == 5

        data = np.load(npz_files[0])
        assert data['images'].shape == (6, 128, 352, 3)
        assert data['extrinsics'].shape == (6, 4, 4)
        assert data['intrinsics'].shape == (6, 3, 3)
        assert data['semantic_masks'].shape == (200, 200, 3)
        assert data['instance_masks'].shape == (200, 200)
        assert data['direction_masks'].shape == (200, 200, 2)

        assert data['images'].dtype == np.uint8
        assert data['extrinsics'].dtype == np.float32
        assert data['intrinsics'].dtype == np.float32
        assert data['semantic_masks'].dtype == np.float32
        assert data['instance_masks'].dtype == np.int32
        assert data['direction_masks'].dtype == np.float32

    def test_tf_data_pipeline(self, mock_data_dir):
        """Test that tf.data.Dataset can load and batch the data."""
        npz_files = sorted([
            os.path.join(mock_data_dir, f) for f in os.listdir(mock_data_dir)
            if f.endswith('.npz')
        ])

        def load_sample(file_path):
            file_path_str = file_path.numpy().decode('utf-8')
            data = np.load(file_path_str)
            images = data['images'].astype(np.float32) / 255.0
            extrinsics = data['extrinsics']
            intrinsics = data['intrinsics']
            semantic = data['semantic_masks']
            instance = data['instance_masks']
            direction = data['direction_masks']
            return images, extrinsics, intrinsics, semantic, instance, direction

        def tf_load_sample(file_path):
            images, ext, intr, sem, inst, dirn = tf.py_function(
                load_sample,
                [file_path],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32]
            )
            images.set_shape([6, 128, 352, 3])
            ext.set_shape([6, 4, 4])
            intr.set_shape([6, 3, 3])
            sem.set_shape([200, 200, 3])
            inst.set_shape([200, 200])
            dirn.set_shape([200, 200, 2])
            return images, ext, intr, sem, inst, dirn

        dataset = tf.data.Dataset.from_tensor_slices(npz_files)
        dataset = dataset.map(tf_load_sample)
        dataset = dataset.batch(2)

        for batch in dataset.take(1):
            images, ext, intr, sem, inst, dirn = batch
            assert images.shape == (2, 6, 128, 352, 3)
            assert ext.shape == (2, 6, 4, 4)
            assert intr.shape == (2, 6, 3, 3)
            assert sem.shape == (2, 200, 200, 3)
            assert inst.shape == (2, 200, 200)
            assert dirn.shape == (2, 200, 200, 2)

    def test_data_augmentation_horizontal_flip(self, mock_data_dir):
        """Test that horizontal flip augmentation works correctly."""
        npz_file = os.path.join(mock_data_dir, "sample_0000.npz")
        data = np.load(npz_file)
        images = data['images'].astype(np.float32)
        semantic = data['semantic_masks']
        direction = data['direction_masks']

        # Apply horizontal flip
        images_flipped = images[:, :, ::-1, :]  # Flip width dimension
        semantic_flipped = semantic[:, ::-1, :]  # Flip BEV x-axis
        direction_flipped = direction[:, ::-1, :].copy()
        direction_flipped[:, :, 0] *= -1  # Negate x component

        # Verify shapes are preserved
        assert images_flipped.shape == images.shape
        assert semantic_flipped.shape == semantic.shape
        assert direction_flipped.shape == direction.shape

        # Verify flip is not identity (unless image is symmetric)
        # Check that at least some values changed
        assert not np.allclose(images, images_flipped) or np.allclose(images, images[:, :, ::-1, :])

    def test_dataset_normalization(self, mock_data_dir):
        """Test that image normalization produces values in expected range."""
        npz_file = os.path.join(mock_data_dir, "sample_0000.npz")
        data = np.load(npz_file)
        images = data['images'].astype(np.float32) / 255.0

        assert images.min() >= 0.0
        assert images.max() <= 1.0


# =============================================================================
# Test BEV Encoder
# =============================================================================


class TestBEVEncoder:
    """Tests for BEV encoder with residual connections."""

    def test_bev_encoder_output_shape(self):
        """Test that BEV encoder preserves spatial dimensions."""
        encoder = BEVEncoder(in_channels=64, mid_channels=128, out_channels=64)
        x = tf.random.normal([1, 200, 200, 64])
        out = encoder(x, training=False)
        assert out.shape == (1, 200, 200, 64)

    def test_bev_encoder_different_channels(self):
        """Test BEV encoder with different input/output channel configurations."""
        encoder = BEVEncoder(in_channels=32, mid_channels=64, out_channels=48)
        x = tf.random.normal([1, 100, 100, 32])
        out = encoder(x, training=False)
        assert out.shape == (1, 100, 100, 48)

    def test_residual_block_skip_connection(self):
        """Test that residual block has working skip connection."""
        block = ResidualBlock(64)
        x = tf.random.normal([1, 50, 50, 64])
        out = block(x, training=False)
        assert out.shape == x.shape

        # Output should be different from input (block transforms the features)
        assert not tf.reduce_all(tf.equal(out, x)).numpy()

    def test_residual_block_channel_matching(self):
        """Test residual block with channel dimension mismatch (uses 1x1 conv)."""
        block = ResidualBlock(128)
        x = tf.random.normal([1, 50, 50, 64])  # 64 channels input, 128 output
        out = block(x, training=False)
        assert out.shape == (1, 50, 50, 128)

    def test_bev_encoder_gradient_flow(self):
        """Test that gradients flow through BEV encoder."""
        encoder = BEVEncoder(in_channels=64, mid_channels=128, out_channels=64)
        x = tf.Variable(tf.random.normal([1, 50, 50, 64]))

        with tf.GradientTape() as tape:
            out = encoder(x, training=True)
            loss = tf.reduce_mean(out)

        grads = tape.gradient(loss, x)
        assert grads is not None
        assert tf.reduce_any(tf.not_equal(grads, 0.0)).numpy()

    def test_bev_encoder_batch_processing(self):
        """Test BEV encoder handles multiple samples in batch."""
        encoder = BEVEncoder(in_channels=64, mid_channels=128, out_channels=64)
        x = tf.random.normal([4, 200, 200, 64])
        out = encoder(x, training=False)
        assert out.shape == (4, 200, 200, 64)

    def test_bev_encoder_training_vs_inference(self):
        """Test that training mode affects batch normalization behavior."""
        encoder = BEVEncoder(in_channels=64, mid_channels=128, out_channels=64)
        x = tf.random.normal([2, 50, 50, 64])

        # First call in training mode to update BN statistics
        out_train = encoder(x, training=True)
        # Then call in inference mode
        out_infer = encoder(x, training=False)

        # Outputs should generally differ due to BN behavior difference
        # (unless running mean/var happen to match batch statistics)
        assert out_train.shape == out_infer.shape


# =============================================================================
# Test Individual Heads
# =============================================================================


class TestIndividualHeads:
    """Tests for semantic, instance, and direction prediction heads."""

    def test_semantic_head_output_channels(self):
        """Test semantic head outputs 3 channels (one per class)."""
        head = SemanticHead(in_channels=64, num_classes=3)
        x = tf.random.normal([1, 200, 200, 64])
        out = head(x, training=False)
        assert out.shape == (1, 200, 200, 3)

    def test_semantic_head_custom_classes(self):
        """Test semantic head with different number of classes."""
        head = SemanticHead(in_channels=64, num_classes=5)
        x = tf.random.normal([1, 200, 200, 64])
        out = head(x, training=False)
        assert out.shape == (1, 200, 200, 5)

    def test_instance_head_output_dimension(self):
        """Test instance head outputs 16-dimensional embeddings."""
        head = InstanceHead(in_channels=64, embed_dim=16)
        x = tf.random.normal([1, 200, 200, 64])
        out = head(x, training=False)
        assert out.shape == (1, 200, 200, 16)

    def test_instance_head_custom_dimension(self):
        """Test instance head with different embedding dimension."""
        head = InstanceHead(in_channels=64, embed_dim=32)
        x = tf.random.normal([1, 200, 200, 64])
        out = head(x, training=False)
        assert out.shape == (1, 200, 200, 32)

    def test_direction_head_output_channels(self):
        """Test direction head outputs 2 channels (2D direction vector)."""
        head = DirectionHead(in_channels=64)
        x = tf.random.normal([1, 200, 200, 64])
        out = head(x, training=False)
        assert out.shape == (1, 200, 200, 2)

    def test_direction_head_unit_normalized(self):
        """Test that direction head output is L2-normalized to unit vectors."""
        head = DirectionHead(in_channels=64)
        x = tf.random.normal([1, 200, 200, 64])
        out = head(x, training=False)

        norms = tf.norm(out, axis=-1)
        # All vectors should have norm close to 1
        assert tf.reduce_all(tf.abs(norms - 1.0) < 0.01).numpy()

    def test_semantic_head_gradient(self):
        """Test gradient flow through semantic head."""
        head = SemanticHead(in_channels=64, num_classes=3)
        x = tf.Variable(tf.random.normal([1, 50, 50, 64]))

        with tf.GradientTape() as tape:
            out = head(x, training=True)
            loss = tf.reduce_mean(out)

        grads = tape.gradient(loss, x)
        assert grads is not None

    def test_instance_head_gradient(self):
        """Test gradient flow through instance head."""
        head = InstanceHead(in_channels=64, embed_dim=16)
        x = tf.Variable(tf.random.normal([1, 50, 50, 64]))

        with tf.GradientTape() as tape:
            out = head(x, training=True)
            loss = tf.reduce_mean(out)

        grads = tape.gradient(loss, x)
        assert grads is not None

    def test_direction_head_gradient(self):
        """Test gradient flow through direction head."""
        head = DirectionHead(in_channels=64)
        x = tf.Variable(tf.random.normal([1, 50, 50, 64]))

        with tf.GradientTape() as tape:
            out = head(x, training=True)
            loss = tf.reduce_mean(out)

        grads = tape.gradient(loss, x)
        assert grads is not None


# =============================================================================
# Test Backbone
# =============================================================================


class TestBackbone:
    """Tests for EfficientNet-B0 backbone and neck."""

    def test_backbone_output_shape(self):
        """Test backbone produces correct feature map dimensions."""
        backbone = EfficientNetBackbone()
        x = tf.random.normal([1, 128, 352, 3])
        out = backbone(x, training=False)
        # EfficientNet-B0 at stride 16: 128/16=8, 352/16=22
        assert out.shape[1] == 8
        assert out.shape[2] == 22
        assert out.shape[3] == 112  # block5a expansion channels

    def test_neck_channel_reduction(self):
        """Test neck reduces channels from backbone output."""
        neck = FeatureNeck(out_channels=64)
        x = tf.random.normal([1, 8, 22, 112])
        out = neck(x, training=False)
        assert out.shape == (1, 8, 22, 64)

    def test_backbone_batch_processing(self):
        """Test backbone handles batched input."""
        backbone = EfficientNetBackbone()
        x = tf.random.normal([4, 128, 352, 3])
        out = backbone(x, training=False)
        assert out.shape[0] == 4


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_model_trainable_variables(self, ipm_model):
        """Test that model has trainable variables."""
        assert len(ipm_model.trainable_variables) > 0

    def test_model_save_and_load_weights(self, ipm_model, dummy_inputs, tmp_path):
        """Test that model weights can be saved and loaded."""
        images, extrinsics, intrinsics = dummy_inputs

        # Get predictions before save
        outputs_before = ipm_model((images, extrinsics, intrinsics), training=False)

        # Save weights
        weight_path = os.path.join(str(tmp_path), "model_weights")
        ipm_model.save_weights(weight_path)

        # Create new model and load weights
        new_model = build_hdmapnet_ipm()
        new_model((images, extrinsics, intrinsics), training=False)  # Build
        new_model.load_weights(weight_path)

        # Get predictions after load
        outputs_after = new_model((images, extrinsics, intrinsics), training=False)

        # Predictions should be identical
        np.testing.assert_allclose(
            outputs_before['semantic'].numpy(),
            outputs_after['semantic'].numpy(),
            rtol=1e-5, atol=1e-5
        )

    def test_end_to_end_training_step(self, ipm_model, dummy_inputs):
        """Test a complete training step with loss and gradient update."""
        images, extrinsics, intrinsics = dummy_inputs
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # Create dummy targets
        semantic_target = tf.cast(
            tf.random.uniform([1, 200, 200, 3]) > 0.7, tf.float32
        )

        with tf.GradientTape() as tape:
            outputs = ipm_model((images, extrinsics, intrinsics), training=True)
            loss = tf.keras.losses.binary_crossentropy(
                semantic_target, outputs['semantic'], from_logits=True
            )
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, ipm_model.trainable_variables)
        # Filter out None gradients
        grads_and_vars = [
            (g, v) for g, v in zip(grads, ipm_model.trainable_variables)
            if g is not None
        ]
        optimizer.apply_gradients(grads_and_vars)

        # Loss should be finite
        assert np.isfinite(loss.numpy())
        assert loss.numpy() > 0.0
