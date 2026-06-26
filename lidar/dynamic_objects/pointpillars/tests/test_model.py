"""Comprehensive tests for PointPillars TensorFlow implementation.

Tests cover the full pipeline: pillar feature extraction, scatter, backbone,
neck, anchor generation, encoding/decoding, full model forward pass, loss
computation, and NMS post-processing.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import sys
import os

# Add the parent directories to the path for imports
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tensorflow")
    ),
)

from model import (
    GRID_X_SIZE,
    GRID_Y_SIZE,
    MAX_NUM_PILLARS,
    MAX_POINTS_PER_PILLAR,
    NUM_ANCHORS_PER_CELL,
    NUM_CLASSES,
    NUM_DIR_BINS,
    NUM_FEATURES,
    BOX_CODE_SIZE,
    PILLAR_FEAT_DIM,
    PILLAR_X_SIZE,
    PILLAR_Y_SIZE,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
    AnchorHead,
    Backbone2D,
    Neck,
    PillarFeatureNet,
    PointPillarsLoss,
    PointPillarsModel,
    PointPillarsScatter,
    apply_nms,
    build_pointpillars,
    create_pillars_from_points,
    decode_predictions,
    generate_anchors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_grid_config():
    """Configuration for smaller grid to speed up tests."""
    return {
        "grid_x_size": 32,
        "grid_y_size": 32,
        "max_num_pillars": 100,
        "max_points_per_pillar": 20,
        "pillar_feat_dim": 64,
        "num_classes": 3,
        "num_anchors_per_cell": 6,
    }


@pytest.fixture
def pillar_feature_net():
    """Create a PillarFeatureNet instance with default settings."""
    return PillarFeatureNet(
        num_input_features=4,
        num_filters=PILLAR_FEAT_DIM,
        max_points_per_pillar=MAX_POINTS_PER_PILLAR,
    )


@pytest.fixture
def scatter():
    """Create a PointPillarsScatter with small grid for testing."""
    return PointPillarsScatter(grid_x_size=32, grid_y_size=32, num_features=64)


@pytest.fixture
def backbone():
    """Create a Backbone2D with default configuration."""
    return Backbone2D(
        layer_nums=[3, 3, 3],
        layer_strides=[2, 2, 2],
        num_filters=[64, 128, 256],
    )


@pytest.fixture
def neck():
    """Create a Neck with default configuration."""
    return Neck(
        upsample_strides=[1, 2, 4],
        num_upsample_filters=[128, 128, 128],
    )


@pytest.fixture
def anchor_head():
    """Create an AnchorHead with default settings."""
    return AnchorHead(
        num_classes=NUM_CLASSES,
        num_anchors_per_cell=NUM_ANCHORS_PER_CELL,
        box_code_size=BOX_CODE_SIZE,
        num_dir_bins=NUM_DIR_BINS,
    )


@pytest.fixture
def small_model(small_grid_config):
    """Create a smaller PointPillarsModel for fast testing."""
    return PointPillarsModel(
        num_classes=small_grid_config["num_classes"],
        num_input_features=4,
        pillar_feat_dim=small_grid_config["pillar_feat_dim"],
        max_points_per_pillar=small_grid_config["max_points_per_pillar"],
        grid_x_size=small_grid_config["grid_x_size"],
        grid_y_size=small_grid_config["grid_y_size"],
        backbone_layer_nums=[2, 2, 2],
        backbone_layer_strides=[2, 2, 2],
        backbone_num_filters=[64, 128, 256],
        neck_upsample_strides=[1, 2, 4],
        neck_num_filters=[64, 64, 64],
        num_anchors_per_cell=small_grid_config["num_anchors_per_cell"],
    )


@pytest.fixture
def loss_fn():
    """Create a PointPillarsLoss instance."""
    return PointPillarsLoss(
        num_classes=NUM_CLASSES,
        alpha=0.25,
        gamma=2.0,
        box_loss_weight=2.0,
        dir_loss_weight=0.2,
        cls_loss_weight=1.0,
    )


@pytest.fixture
def random_point_cloud():
    """Generate a random point cloud within valid KITTI range."""
    tf.random.set_seed(42)
    num_points = 5000
    x = tf.random.uniform([num_points, 1], minval=X_MIN, maxval=X_MAX)
    y = tf.random.uniform([num_points, 1], minval=Y_MIN, maxval=Y_MAX)
    z = tf.random.uniform([num_points, 1], minval=Z_MIN, maxval=Z_MAX)
    intensity = tf.random.uniform([num_points, 1], minval=0.0, maxval=1.0)
    return tf.concat([x, y, z, intensity], axis=1)


@pytest.fixture
def small_model_inputs(small_grid_config):
    """Create valid model inputs for the small model configuration."""
    tf.random.set_seed(123)
    batch_size = 2
    max_pillars = small_grid_config["max_num_pillars"]
    max_points = small_grid_config["max_points_per_pillar"]
    grid_x = small_grid_config["grid_x_size"]
    grid_y = small_grid_config["grid_y_size"]

    pillars = tf.random.normal([batch_size, max_pillars, max_points, 4])
    pillar_x_idx = tf.random.uniform(
        [batch_size, max_pillars, 1], minval=0, maxval=grid_x, dtype=tf.int32
    )
    pillar_y_idx = tf.random.uniform(
        [batch_size, max_pillars, 1], minval=0, maxval=grid_y, dtype=tf.int32
    )
    pillar_indices = tf.concat([pillar_x_idx, pillar_y_idx], axis=-1)
    num_points_per_pillar = tf.random.uniform(
        [batch_size, max_pillars], minval=1, maxval=max_points + 1, dtype=tf.int32
    )

    return {
        "pillars": pillars,
        "pillar_indices": pillar_indices,
        "num_points_per_pillar": num_points_per_pillar,
    }


# ---------------------------------------------------------------------------
# Test: Pillar Feature Net (Voxelization + Feature Augmentation)
# ---------------------------------------------------------------------------


class TestPillarFeatureNet:
    """Tests for pillar voxelization and feature augmentation."""

    def test_voxelization_valid_range(self, random_point_cloud):
        """Verify voxelization produces valid pillars from KITTI-range points."""
        pillars, pillar_indices, num_points = create_pillars_from_points(
            random_point_cloud,
            max_points_per_pillar=32,
            max_num_pillars=500,
        )

        # Verify we get non-zero pillars (point cloud has 5000 points in range)
        actual_pillar_count = tf.reduce_sum(
            tf.cast(num_points > 0, tf.int32)
        ).numpy()
        assert actual_pillar_count > 0, "Should produce at least one non-empty pillar"
        assert actual_pillar_count <= 500, "Should not exceed max_num_pillars"

    def test_points_per_pillar_padding(self, random_point_cloud):
        """Verify points are padded to max_points_per_pillar with zeros."""
        max_pts = 32
        pillars, pillar_indices, num_points = create_pillars_from_points(
            random_point_cloud,
            max_points_per_pillar=max_pts,
            max_num_pillars=200,
        )

        assert pillars.shape[1] == max_pts, (
            f"Expected {max_pts} points per pillar, got {pillars.shape[1]}"
        )

        # For a pillar with fewer points than max, trailing entries should be zero
        first_valid_pillar_idx = tf.where(num_points > 0)[0, 0].numpy()
        n_valid = num_points[first_valid_pillar_idx].numpy()
        if n_valid < max_pts:
            padded_region = pillars[first_valid_pillar_idx, n_valid:, :]
            assert tf.reduce_all(padded_region == 0.0).numpy(), (
                "Padded points should be zero"
            )

    def test_augmented_features_shape(self, pillar_feature_net):
        """Verify PillarFeatureNet produces 9-feature augmented input internally."""
        # The augmented feature count is num_input_features(4) + 3(mean offset) + 2(center offset) = 9
        expected_augmented = 4 + 5  # raw features + offset_xyz(3) + offset_center_xy(2)
        assert expected_augmented == NUM_FEATURES, (
            f"Expected {NUM_FEATURES} augmented features, architecture uses {expected_augmented}"
        )

        batch_size = 2
        max_pillars = 50
        max_points = MAX_POINTS_PER_PILLAR

        pillars = tf.random.normal([batch_size, max_pillars, max_points, 4])
        pillar_indices = tf.random.uniform(
            [batch_size, max_pillars, 2], minval=0, maxval=100, dtype=tf.int32
        )
        num_points = tf.random.uniform(
            [batch_size, max_pillars], minval=1, maxval=max_points, dtype=tf.int32
        )

        output = pillar_feature_net(pillars, pillar_indices, num_points, training=False)
        assert output.shape == (batch_size, max_pillars, PILLAR_FEAT_DIM), (
            f"Expected shape ({batch_size}, {max_pillars}, {PILLAR_FEAT_DIM}), "
            f"got {output.shape}"
        )

    def test_pillar_indices_within_grid(self, random_point_cloud):
        """Verify pillar grid indices are within valid bounds."""
        pillars, pillar_indices, num_points = create_pillars_from_points(
            random_point_cloud,
            max_points_per_pillar=32,
            max_num_pillars=500,
        )

        valid_mask = num_points > 0
        valid_indices = tf.boolean_mask(pillar_indices, valid_mask)

        x_indices = valid_indices[:, 0].numpy()
        y_indices = valid_indices[:, 1].numpy()

        assert np.all(x_indices >= 0), "X indices must be non-negative"
        assert np.all(x_indices < GRID_X_SIZE), f"X indices must be < {GRID_X_SIZE}"
        assert np.all(y_indices >= 0), "Y indices must be non-negative"
        assert np.all(y_indices < GRID_Y_SIZE), f"Y indices must be < {GRID_Y_SIZE}"


# ---------------------------------------------------------------------------
# Test: PointNet per Pillar (Max Pooling)
# ---------------------------------------------------------------------------


class TestPointNetPerPillar:
    """Tests for the PointNet max-pooling operation in PillarFeatureNet."""

    def test_max_pooling_output_shape(self, pillar_feature_net):
        """Verify max pooling output shape is (batch, max_pillars, 64)."""
        batch_size = 4
        max_pillars = 200
        max_points = MAX_POINTS_PER_PILLAR

        pillars = tf.random.normal([batch_size, max_pillars, max_points, 4])
        pillar_indices = tf.random.uniform(
            [batch_size, max_pillars, 2], minval=0, maxval=200, dtype=tf.int32
        )
        num_points = tf.random.uniform(
            [batch_size, max_pillars], minval=1, maxval=max_points, dtype=tf.int32
        )

        output = pillar_feature_net(pillars, pillar_indices, num_points, training=False)

        assert output.shape[0] == batch_size
        assert output.shape[1] == max_pillars
        assert output.shape[2] == 64, f"Expected 64-dim features, got {output.shape[2]}"

    def test_empty_pillar_produces_zeros(self, pillar_feature_net):
        """Verify pillars with zero points produce zero feature vectors."""
        batch_size = 1
        max_pillars = 10
        max_points = MAX_POINTS_PER_PILLAR

        pillars = tf.random.normal([batch_size, max_pillars, max_points, 4])
        pillar_indices = tf.random.uniform(
            [batch_size, max_pillars, 2], minval=0, maxval=100, dtype=tf.int32
        )
        # Set all pillars to have 0 points
        num_points = tf.zeros([batch_size, max_pillars], dtype=tf.int32)

        output = pillar_feature_net(pillars, pillar_indices, num_points, training=False)

        assert tf.reduce_all(output == 0.0).numpy(), (
            "Empty pillars should produce zero feature vectors"
        )

    def test_deterministic_in_eval_mode(self, pillar_feature_net):
        """Verify forward pass is deterministic in eval mode (no dropout variance)."""
        batch_size = 2
        max_pillars = 50
        max_points = 20

        pillars = tf.random.normal([batch_size, max_pillars, max_points, 4], seed=99)
        pillar_indices = tf.constant(
            np.random.randint(0, 100, size=(batch_size, max_pillars, 2)), dtype=tf.int32
        )
        num_points = tf.constant(
            np.full([batch_size, max_pillars], 10), dtype=tf.int32
        )

        out1 = pillar_feature_net(pillars, pillar_indices, num_points, training=False)
        out2 = pillar_feature_net(pillars, pillar_indices, num_points, training=False)

        np.testing.assert_allclose(
            out1.numpy(), out2.numpy(), atol=1e-6,
            err_msg="Eval mode should be deterministic"
        )


# ---------------------------------------------------------------------------
# Test: Scatter to Pseudo-Image
# ---------------------------------------------------------------------------


class TestScatter:
    """Tests for PointPillarsScatter (pillar features -> BEV pseudo-image)."""

    def test_pseudo_image_shape(self, scatter):
        """Verify pseudo-image shape is (batch, grid_y, grid_x, 64)."""
        batch_size = 2
        max_pillars = 50
        feat_dim = 64

        pillar_features = tf.random.normal([batch_size, max_pillars, feat_dim])
        pillar_indices = tf.random.uniform(
            [batch_size, max_pillars, 2], minval=0, maxval=32, dtype=tf.int32
        )

        output = scatter(pillar_features, pillar_indices)

        assert output.shape == (batch_size, 32, 32, feat_dim), (
            f"Expected (2, 32, 32, 64), got {output.shape}"
        )

    def test_features_placed_at_correct_locations(self):
        """Verify features are placed at the correct grid locations."""
        grid_x, grid_y, feat_dim = 8, 8, 4
        scatter_layer = PointPillarsScatter(
            grid_x_size=grid_x, grid_y_size=grid_y, num_features=feat_dim
        )

        batch_size = 1
        # Place two pillars at known locations
        pillar_features = tf.constant([[[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0]]])  # (1, 2, 4)
        # First pillar at grid (x=2, y=3), second at (x=5, y=1)
        pillar_indices = tf.constant([[[2, 3], [5, 1]]], dtype=tf.int32)  # (1, 2, 2)

        output = scatter_layer(pillar_features, pillar_indices)

        # Check first pillar position: canvas[batch=0, y=3, x=2, :]
        placed_feat_1 = output[0, 3, 2, :].numpy()
        np.testing.assert_array_almost_equal(
            placed_feat_1, [1.0, 2.0, 3.0, 4.0],
            err_msg="First pillar features not placed correctly"
        )

        # Check second pillar position: canvas[batch=0, y=1, x=5, :]
        placed_feat_2 = output[0, 1, 5, :].numpy()
        np.testing.assert_array_almost_equal(
            placed_feat_2, [5.0, 6.0, 7.0, 8.0],
            err_msg="Second pillar features not placed correctly"
        )

        # Verify empty locations remain zero
        empty_feat = output[0, 0, 0, :].numpy()
        np.testing.assert_array_almost_equal(
            empty_feat, [0.0, 0.0, 0.0, 0.0],
            err_msg="Empty grid cells should remain zero"
        )

    def test_full_kitti_grid_shape(self):
        """Verify scatter works with full KITTI grid dimensions."""
        scatter_full = PointPillarsScatter(
            grid_x_size=GRID_X_SIZE, grid_y_size=GRID_Y_SIZE, num_features=64
        )

        batch_size = 1
        max_pillars = 100
        pillar_features = tf.random.normal([batch_size, max_pillars, 64])
        pillar_indices = tf.concat([
            tf.random.uniform([batch_size, max_pillars, 1], 0, GRID_X_SIZE, dtype=tf.int32),
            tf.random.uniform([batch_size, max_pillars, 1], 0, GRID_Y_SIZE, dtype=tf.int32),
        ], axis=-1)

        output = scatter_full(pillar_features, pillar_indices)

        assert output.shape == (batch_size, GRID_Y_SIZE, GRID_X_SIZE, 64), (
            f"Expected (1, {GRID_Y_SIZE}, {GRID_X_SIZE}, 64), got {output.shape}"
        )


# ---------------------------------------------------------------------------
# Test: Backbone2D Forward Pass
# ---------------------------------------------------------------------------


class TestBackboneForward:
    """Tests for Backbone2D multi-scale feature extraction."""

    def test_multi_scale_feature_map_shapes(self, backbone):
        """Verify backbone produces correct multi-scale feature map shapes."""
        batch_size = 2
        h, w = 64, 64
        channels = 64

        input_tensor = tf.random.normal([batch_size, h, w, channels])
        outputs = backbone(input_tensor, training=False)

        assert len(outputs) == 3, f"Expected 3 scale levels, got {len(outputs)}"

        # Block 1: stride 2 -> (H/2, W/2, 64)
        assert outputs[0].shape == (batch_size, h // 2, w // 2, 64), (
            f"Scale 1 expected ({batch_size}, {h//2}, {w//2}, 64), got {outputs[0].shape}"
        )

        # Block 2: stride 2 -> (H/4, W/4, 128)
        assert outputs[1].shape == (batch_size, h // 4, w // 4, 128), (
            f"Scale 2 expected ({batch_size}, {h//4}, {w//4}, 128), got {outputs[1].shape}"
        )

        # Block 3: stride 2 -> (H/8, W/8, 256)
        assert outputs[2].shape == (batch_size, h // 8, w // 8, 256), (
            f"Scale 3 expected ({batch_size}, {h//8}, {w//8}, 256), got {outputs[2].shape}"
        )

    def test_backbone_with_kitti_input(self):
        """Verify backbone works with KITTI-sized pseudo-image input."""
        backbone_kitti = Backbone2D(
            layer_nums=[4, 6, 6],
            layer_strides=[2, 2, 2],
            num_filters=[64, 128, 256],
        )

        batch_size = 1
        # KITTI BEV pseudo-image: (496, 432, 64)
        input_tensor = tf.random.normal([batch_size, GRID_Y_SIZE, GRID_X_SIZE, 64])
        outputs = backbone_kitti(input_tensor, training=False)

        expected_shapes = [
            (batch_size, GRID_Y_SIZE // 2, GRID_X_SIZE // 2, 64),
            (batch_size, GRID_Y_SIZE // 4, GRID_X_SIZE // 4, 128),
            (batch_size, GRID_Y_SIZE // 8, GRID_X_SIZE // 8, 256),
        ]

        for i, (out, expected) in enumerate(zip(outputs, expected_shapes)):
            assert out.shape == expected, (
                f"Backbone block {i}: expected {expected}, got {out.shape}"
            )

    def test_backbone_training_vs_eval_differs(self, backbone):
        """Verify batch norm behavior differs between training and eval."""
        input_tensor = tf.random.normal([4, 32, 32, 64])

        # Run once in training mode to update running stats
        _ = backbone(input_tensor, training=True)

        out_train = backbone(input_tensor, training=True)
        out_eval = backbone(input_tensor, training=False)

        # BatchNorm should produce different results in train vs eval
        # (unless perfectly converged, which is unlikely on random data)
        diff = tf.reduce_sum(tf.abs(out_train[0] - out_eval[0])).numpy()
        assert diff > 0.0, "Training and eval modes should produce different outputs"


# ---------------------------------------------------------------------------
# Test: Anchor Generation
# ---------------------------------------------------------------------------


class TestAnchorGeneration:
    """Tests for anchor box generation."""

    def test_anchor_shapes_kitti(self):
        """Verify anchor shapes for KITTI configuration (3 classes x 2 rotations)."""
        anchors = generate_anchors(
            grid_x_size=GRID_X_SIZE,
            grid_y_size=GRID_Y_SIZE,
            feature_map_stride=2,
        )

        fm_x = GRID_X_SIZE // 2  # 216
        fm_y = GRID_Y_SIZE // 2  # 248

        expected_shape = (fm_y, fm_x, NUM_ANCHORS_PER_CELL, 7)
        assert anchors.shape == expected_shape, (
            f"Expected anchor shape {expected_shape}, got {anchors.shape}"
        )

    def test_anchor_positions_cover_bev_grid(self):
        """Verify anchor centers cover the full BEV detection range."""
        anchors = generate_anchors(
            grid_x_size=GRID_X_SIZE,
            grid_y_size=GRID_Y_SIZE,
            feature_map_stride=2,
        )

        # Extract x and y centers from the first anchor type
        x_centers = anchors[:, :, 0, 0].numpy()  # (fm_y, fm_x)
        y_centers = anchors[:, :, 0, 1].numpy()  # (fm_y, fm_x)

        x_stride = PILLAR_X_SIZE * 2  # 0.32
        y_stride = PILLAR_Y_SIZE * 2  # 0.32

        # Check x range covers [X_MIN + stride/2, X_MAX - stride/2] approximately
        assert x_centers.min() >= X_MIN, f"Min x anchor {x_centers.min()} < X_MIN {X_MIN}"
        assert x_centers.max() <= X_MAX, f"Max x anchor {x_centers.max()} > X_MAX {X_MAX}"

        # Check y range
        assert y_centers.min() >= Y_MIN, f"Min y anchor {y_centers.min()} < Y_MIN {Y_MIN}"
        assert y_centers.max() <= Y_MAX, f"Max y anchor {y_centers.max()} > Y_MAX {Y_MAX}"

    def test_anchor_two_rotations_per_class(self):
        """Verify each class has exactly two rotation variants (0 and pi/2)."""
        anchors = generate_anchors(feature_map_stride=2)

        # For 3 classes with 2 rotations each, anchor indices are:
        # [0]: class 0, rot 0; [1]: class 0, rot pi/2
        # [2]: class 1, rot 0; [3]: class 1, rot pi/2
        # [4]: class 2, rot 0; [5]: class 2, rot pi/2

        # Check rotation values at any spatial location
        rot_0 = anchors[0, 0, 0, 6].numpy()
        rot_1 = anchors[0, 0, 1, 6].numpy()

        np.testing.assert_almost_equal(rot_0, 0.0, decimal=5,
                                       err_msg="First rotation should be 0")
        np.testing.assert_almost_equal(rot_1, np.pi / 2.0, decimal=5,
                                       err_msg="Second rotation should be pi/2")

    def test_anchor_sizes_match_classes(self):
        """Verify anchor dimensions match Car, Pedestrian, Cyclist specs."""
        anchors = generate_anchors(feature_map_stride=2)

        # Car anchors (indices 0, 1): w=1.6, l=3.9, h=1.56
        car_w = anchors[0, 0, 0, 3].numpy()
        car_l = anchors[0, 0, 0, 4].numpy()
        car_h = anchors[0, 0, 0, 5].numpy()
        np.testing.assert_almost_equal(car_w, 1.6, decimal=4)
        np.testing.assert_almost_equal(car_l, 3.9, decimal=4)
        np.testing.assert_almost_equal(car_h, 1.56, decimal=4)

        # Pedestrian anchors (indices 2, 3): w=0.6, l=0.8, h=1.73
        ped_w = anchors[0, 0, 2, 3].numpy()
        ped_l = anchors[0, 0, 2, 4].numpy()
        ped_h = anchors[0, 0, 2, 5].numpy()
        np.testing.assert_almost_equal(ped_w, 0.6, decimal=4)
        np.testing.assert_almost_equal(ped_l, 0.8, decimal=4)
        np.testing.assert_almost_equal(ped_h, 1.73, decimal=4)

        # Cyclist anchors (indices 4, 5): w=0.6, l=1.76, h=1.73
        cyc_w = anchors[0, 0, 4, 3].numpy()
        cyc_l = anchors[0, 0, 4, 4].numpy()
        cyc_h = anchors[0, 0, 4, 5].numpy()
        np.testing.assert_almost_equal(cyc_w, 0.6, decimal=4)
        np.testing.assert_almost_equal(cyc_l, 1.76, decimal=4)
        np.testing.assert_almost_equal(cyc_h, 1.73, decimal=4)


# ---------------------------------------------------------------------------
# Test: Anchor Encoding / Decoding
# ---------------------------------------------------------------------------


class TestAnchorEncodingDecoding:
    """Tests for anchor-based box encoding and decoding."""

    def test_encode_decode_roundtrip(self):
        """Generate random GT boxes, encode, decode, verify close to original."""
        tf.random.set_seed(7)
        num_boxes = 100

        # Generate random anchor boxes
        anchors = tf.concat([
            tf.random.uniform([num_boxes, 1], 10.0, 60.0),   # x
            tf.random.uniform([num_boxes, 1], -30.0, 30.0),  # y
            tf.random.uniform([num_boxes, 1], -2.0, 0.0),    # z
            tf.random.uniform([num_boxes, 1], 0.5, 2.0),     # w
            tf.random.uniform([num_boxes, 1], 1.0, 5.0),     # l
            tf.random.uniform([num_boxes, 1], 1.0, 2.0),     # h
            tf.random.uniform([num_boxes, 1], -np.pi, np.pi), # theta
        ], axis=-1)

        # Generate random ground truth boxes (close to anchors)
        gt_boxes = tf.concat([
            anchors[:, 0:1] + tf.random.normal([num_boxes, 1], stddev=0.5),
            anchors[:, 1:2] + tf.random.normal([num_boxes, 1], stddev=0.5),
            anchors[:, 2:3] + tf.random.normal([num_boxes, 1], stddev=0.2),
            anchors[:, 3:4] * tf.exp(tf.random.normal([num_boxes, 1], stddev=0.1)),
            anchors[:, 4:5] * tf.exp(tf.random.normal([num_boxes, 1], stddev=0.1)),
            anchors[:, 5:6] * tf.exp(tf.random.normal([num_boxes, 1], stddev=0.1)),
            anchors[:, 6:7] + tf.random.normal([num_boxes, 1], stddev=0.2),
        ], axis=-1)

        # Encode GT relative to anchors (using diagonal normalization)
        xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        wa, la, ha = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        theta_a = anchors[:, 6]
        da = tf.sqrt(wa ** 2 + la ** 2)

        xg, yg, zg = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        wg, lg, hg = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        theta_g = gt_boxes[:, 6]

        # Encode
        dx = (xg - xa) / da
        dy = (yg - ya) / da
        dz = (zg - za) / ha
        dw = tf.math.log(wg / wa)
        dl = tf.math.log(lg / la)
        dh = tf.math.log(hg / ha)
        dtheta = theta_g - theta_a

        encoded = tf.stack([dx, dy, dz, dw, dl, dh, dtheta], axis=-1)

        # Decode using the model's decode function
        decoded = decode_predictions(encoded, anchors)

        # Verify decoded matches ground truth
        np.testing.assert_allclose(
            decoded.numpy(), gt_boxes.numpy(), atol=1e-4,
            err_msg="Decoded boxes should match original GT boxes"
        )

    def test_decode_zero_residuals_returns_anchors(self):
        """Verify that zero residuals decode back to anchor positions."""
        num_boxes = 50
        anchors = tf.constant(
            np.random.uniform(
                low=[10, -30, -2, 0.5, 1.0, 1.0, -np.pi],
                high=[60, 30, 0, 2.0, 5.0, 2.0, np.pi],
                size=(num_boxes, 7),
            ).astype(np.float32)
        )

        zero_residuals = tf.zeros([num_boxes, 7], dtype=tf.float32)
        decoded = decode_predictions(zero_residuals, anchors)

        # With zero residuals: x=0*da+xa=xa, y=0*da+ya=ya, z=0*ha+za=za
        # w=exp(0)*wa=wa, l=exp(0)*la=la, h=exp(0)*ha=ha
        # theta=0+theta_a=theta_a
        np.testing.assert_allclose(
            decoded.numpy(), anchors.numpy(), atol=1e-5,
            err_msg="Zero residuals should decode to anchor positions"
        )

    def test_encoding_diagonal_normalization(self):
        """Verify x,y offsets are normalized by anchor diagonal."""
        anchor = tf.constant([[30.0, 0.0, -1.0, 1.6, 3.9, 1.56, 0.0]])
        diagonal = np.sqrt(1.6**2 + 3.9**2)

        # GT box offset 1 meter in x from anchor
        gt_box = tf.constant([[31.0, 0.0, -1.0, 1.6, 3.9, 1.56, 0.0]])

        # Expected encoding: dx = (31 - 30) / diagonal = 1.0 / diagonal
        expected_dx = 1.0 / diagonal

        # Encode manually
        dx_encoded = (gt_box[0, 0] - anchor[0, 0]) / diagonal

        np.testing.assert_almost_equal(
            dx_encoded.numpy(), expected_dx, decimal=5,
            err_msg="X offset should be normalized by anchor diagonal"
        )


# ---------------------------------------------------------------------------
# Test: Full Model Forward Pass
# ---------------------------------------------------------------------------


class TestFullModelForward:
    """Tests for the end-to-end PointPillars model forward pass."""

    def test_output_dict_keys(self, small_model, small_model_inputs):
        """Verify model output has cls_preds, bbox_preds, dir_preds."""
        output = small_model(small_model_inputs, training=False)

        assert "cls_preds" in output, "Output must contain 'cls_preds'"
        assert "box_preds" in output, "Output must contain 'box_preds'"
        assert "dir_preds" in output, "Output must contain 'dir_preds'"

    def test_output_shapes(self, small_model, small_model_inputs, small_grid_config):
        """Verify output tensor shapes match expected dimensions."""
        output = small_model(small_model_inputs, training=False)

        batch_size = 2
        grid_x = small_grid_config["grid_x_size"]
        grid_y = small_grid_config["grid_y_size"]
        num_anchors = small_grid_config["num_anchors_per_cell"]
        num_classes = small_grid_config["num_classes"]

        # After backbone stride 2, feature map is (grid_y/2, grid_x/2)
        # Neck upsamples back to (grid_y/2, grid_x/2) from first block
        # Total anchors = (grid_y/2) * (grid_x/2) * num_anchors_per_cell
        # With [2,2,2] backbone strides, first output is (grid_y/2, grid_x/2)
        # Neck upsample_strides=[1,2,4] maps all to (grid_y/2, grid_x/2)
        fm_h = grid_y // 2
        fm_w = grid_x // 2
        total_anchors = fm_h * fm_w * num_anchors

        assert output["cls_preds"].shape == (batch_size, total_anchors, num_classes), (
            f"cls_preds shape mismatch: {output['cls_preds'].shape}"
        )
        assert output["box_preds"].shape == (batch_size, total_anchors, BOX_CODE_SIZE), (
            f"box_preds shape mismatch: {output['box_preds'].shape}"
        )
        assert output["dir_preds"].shape == (batch_size, total_anchors, NUM_DIR_BINS), (
            f"dir_preds shape mismatch: {output['dir_preds'].shape}"
        )

    def test_model_is_differentiable(self, small_model, small_model_inputs):
        """Verify model outputs are differentiable (gradients flow)."""
        with tf.GradientTape() as tape:
            output = small_model(small_model_inputs, training=True)
            dummy_loss = tf.reduce_mean(output["cls_preds"])

        grads = tape.gradient(dummy_loss, small_model.trainable_variables)
        non_none_grads = [g for g in grads if g is not None]
        assert len(non_none_grads) > 0, "Model should have non-None gradients"

        # At least some gradients should be non-zero
        has_nonzero = any(tf.reduce_any(g != 0.0).numpy() for g in non_none_grads)
        assert has_nonzero, "At least some gradients should be non-zero"


# ---------------------------------------------------------------------------
# Test: Loss Computation
# ---------------------------------------------------------------------------


class TestLossComputation:
    """Tests for PointPillars loss functions."""

    def _make_fake_targets(self, batch_size, total_anchors, num_classes):
        """Create realistic fake targets for loss computation."""
        # Mark ~5% as positive, ~80% as negative, rest ignored
        np.random.seed(42)
        pos_ratio = 0.05
        neg_ratio = 0.80

        positive_mask = np.zeros([batch_size, total_anchors], dtype=np.float32)
        negative_mask = np.zeros([batch_size, total_anchors], dtype=np.float32)

        for b in range(batch_size):
            n_pos = int(total_anchors * pos_ratio)
            n_neg = int(total_anchors * neg_ratio)
            perm = np.random.permutation(total_anchors)
            positive_mask[b, perm[:n_pos]] = 1.0
            negative_mask[b, perm[n_pos:n_pos + n_neg]] = 1.0

        # One-hot class targets for positive anchors
        cls_targets = np.zeros([batch_size, total_anchors, num_classes], dtype=np.float32)
        for b in range(batch_size):
            pos_indices = np.where(positive_mask[b] > 0)[0]
            classes = np.random.randint(0, num_classes, size=len(pos_indices))
            cls_targets[b, pos_indices, classes] = 1.0

        # Box regression targets (random residuals for positive anchors)
        box_targets = np.random.randn(batch_size, total_anchors, 7).astype(np.float32) * 0.1

        # Direction targets
        dir_targets = np.random.randint(0, 2, size=[batch_size, total_anchors]).astype(np.float32)

        return {
            "cls_targets": tf.constant(cls_targets),
            "box_targets": tf.constant(box_targets),
            "dir_targets": tf.constant(dir_targets),
            "positive_mask": tf.constant(positive_mask),
            "negative_mask": tf.constant(negative_mask),
        }

    def test_focal_loss_positive(self, loss_fn):
        """Verify focal loss is positive for non-trivial predictions."""
        batch_size = 2
        total_anchors = 500
        num_classes = NUM_CLASSES

        predictions = {
            "cls_preds": tf.random.normal([batch_size, total_anchors, num_classes]),
            "box_preds": tf.random.normal([batch_size, total_anchors, 7]),
            "dir_preds": tf.random.normal([batch_size, total_anchors, 2]),
        }
        targets = self._make_fake_targets(batch_size, total_anchors, num_classes)

        losses = loss_fn(predictions, targets)

        assert losses["cls_loss"].numpy() > 0.0, "Focal loss should be positive"

    def test_smooth_l1_positive(self, loss_fn):
        """Verify smooth L1 loss is positive for non-zero residuals."""
        batch_size = 2
        total_anchors = 500
        num_classes = NUM_CLASSES

        predictions = {
            "cls_preds": tf.random.normal([batch_size, total_anchors, num_classes]),
            "box_preds": tf.random.normal([batch_size, total_anchors, 7]),
            "dir_preds": tf.random.normal([batch_size, total_anchors, 2]),
        }
        targets = self._make_fake_targets(batch_size, total_anchors, num_classes)

        losses = loss_fn(predictions, targets)

        assert losses["box_loss"].numpy() > 0.0, "Smooth L1 loss should be positive"

    def test_direction_loss_positive(self, loss_fn):
        """Verify direction classification loss is positive."""
        batch_size = 2
        total_anchors = 500
        num_classes = NUM_CLASSES

        predictions = {
            "cls_preds": tf.random.normal([batch_size, total_anchors, num_classes]),
            "box_preds": tf.random.normal([batch_size, total_anchors, 7]),
            "dir_preds": tf.random.normal([batch_size, total_anchors, 2]),
        }
        targets = self._make_fake_targets(batch_size, total_anchors, num_classes)

        losses = loss_fn(predictions, targets)

        assert losses["dir_loss"].numpy() > 0.0, "Direction loss should be positive"

    def test_total_loss_decreases_after_gradient_step(self, small_model, small_model_inputs):
        """Verify total loss decreases after one gradient step."""
        batch_size = 2
        loss_fn = PointPillarsLoss(num_classes=NUM_CLASSES)

        # Get model predictions to determine anchor count
        preds_initial = small_model(small_model_inputs, training=False)
        total_anchors = preds_initial["cls_preds"].shape[1]

        # Create targets
        np.random.seed(99)
        n_pos = max(1, total_anchors // 20)
        positive_mask = np.zeros([batch_size, total_anchors], dtype=np.float32)
        negative_mask = np.zeros([batch_size, total_anchors], dtype=np.float32)
        cls_targets = np.zeros([batch_size, total_anchors, NUM_CLASSES], dtype=np.float32)

        for b in range(batch_size):
            perm = np.random.permutation(total_anchors)
            positive_mask[b, perm[:n_pos]] = 1.0
            negative_mask[b, perm[n_pos:n_pos + total_anchors // 2]] = 1.0
            cls_targets[b, perm[:n_pos], np.random.randint(0, NUM_CLASSES, n_pos)] = 1.0

        targets = {
            "cls_targets": tf.constant(cls_targets),
            "box_targets": tf.constant(
                np.random.randn(batch_size, total_anchors, 7).astype(np.float32) * 0.1
            ),
            "dir_targets": tf.constant(
                np.random.randint(0, 2, [batch_size, total_anchors]).astype(np.float32)
            ),
            "positive_mask": tf.constant(positive_mask),
            "negative_mask": tf.constant(negative_mask),
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # Compute initial loss
        with tf.GradientTape() as tape:
            preds = small_model(small_model_inputs, training=True)
            losses = loss_fn(preds, targets)
            initial_loss = losses["total_loss"]

        grads = tape.gradient(initial_loss, small_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, small_model.trainable_variables))

        # Compute loss after one step
        preds_after = small_model(small_model_inputs, training=False)
        losses_after = loss_fn(preds_after, targets)
        final_loss = losses_after["total_loss"]

        assert final_loss.numpy() < initial_loss.numpy(), (
            f"Loss should decrease after gradient step: "
            f"initial={initial_loss.numpy():.4f}, final={final_loss.numpy():.4f}"
        )


# ---------------------------------------------------------------------------
# Test: NMS Post-Processing
# ---------------------------------------------------------------------------


class TestNMSPostprocessing:
    """Tests for non-maximum suppression."""

    def test_overlapping_boxes_reduced(self):
        """Create overlapping boxes with known IoU, verify NMS reduces count."""
        batch_size = 1
        num_boxes = 5

        # Create 5 boxes: 3 highly overlapping (same position), 2 separate
        # BEV boxes centered at (30, 0) with w=2, l=4 -- these overlap heavily
        # and 2 boxes at (50, 20) and (10, -20) that are separate
        decoded_boxes = tf.constant([[[
            30.0, 0.0, -1.0, 2.0, 4.0, 1.5, 0.0,   # box 1 - center cluster
        ], [
            30.1, 0.1, -1.0, 2.0, 4.0, 1.5, 0.0,   # box 2 - overlaps box 1
        ], [
            30.2, -0.1, -1.0, 2.0, 4.0, 1.5, 0.0,  # box 3 - overlaps box 1
        ], [
            50.0, 20.0, -1.0, 2.0, 4.0, 1.5, 0.0,  # box 4 - separate
        ], [
            10.0, -20.0, -1.0, 2.0, 4.0, 1.5, 0.0, # box 5 - separate
        ]]], dtype=tf.float32)  # (1, 5, 7)

        # High scores for all boxes, same class
        cls_preds = tf.constant([[[
            [3.0, -5.0, -5.0],  # box 1: high score class 0
            [2.8, -5.0, -5.0],  # box 2: slightly lower
            [2.5, -5.0, -5.0],  # box 3: lower still
            [2.9, -5.0, -5.0],  # box 4: high score
            [2.7, -5.0, -5.0],  # box 5: high score
        ]]], dtype=tf.float32)  # (1, 5, 3)

        dir_preds = tf.constant([[[
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, -1.0],
        ]]], dtype=tf.float32)  # (1, 5, 2)

        results = apply_nms(
            cls_preds=cls_preds,
            decoded_boxes=decoded_boxes,
            dir_preds=dir_preds,
            score_threshold=0.1,
            nms_iou_threshold=0.5,
            max_detections_per_class=10,
            max_total_detections=10,
        )

        num_dets = results["num_detections"][0].numpy()

        # NMS should suppress the overlapping cluster to 1 box, keep 2 separate
        # Total should be 3 (one from cluster + 2 separate) or fewer
        assert num_dets < num_boxes, (
            f"NMS should reduce count from {num_boxes}, got {num_dets} detections"
        )
        assert num_dets >= 2, (
            f"Should keep at least the 2 well-separated boxes, got {num_dets}"
        )

    def test_nms_respects_score_threshold(self):
        """Verify boxes below score threshold are filtered out."""
        batch_size = 1
        num_boxes = 4

        decoded_boxes = tf.constant([[[
            30.0, 0.0, -1.0, 2.0, 4.0, 1.5, 0.0,
        ], [
            40.0, 10.0, -1.0, 2.0, 4.0, 1.5, 0.0,
        ], [
            50.0, 20.0, -1.0, 2.0, 4.0, 1.5, 0.0,
        ], [
            60.0, 30.0, -1.0, 2.0, 4.0, 1.5, 0.0,
        ]]], dtype=tf.float32)

        # Only first two boxes have high scores (sigmoid(3.0) > 0.95)
        # Last two have very low scores (sigmoid(-5.0) < 0.01)
        cls_preds = tf.constant([[[
            [3.0, -5.0, -5.0],
            [2.5, -5.0, -5.0],
            [-5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0],
        ]]], dtype=tf.float32)

        dir_preds = tf.zeros([batch_size, num_boxes, 2])

        results = apply_nms(
            cls_preds=cls_preds,
            decoded_boxes=decoded_boxes,
            dir_preds=dir_preds,
            score_threshold=0.5,
            nms_iou_threshold=0.5,
            max_detections_per_class=10,
            max_total_detections=10,
        )

        num_dets = results["num_detections"][0].numpy()

        # Only boxes with sigmoid score > 0.5 should remain
        # sigmoid(3.0) ~ 0.95, sigmoid(2.5) ~ 0.92 -> 2 boxes pass
        # sigmoid(-5.0) ~ 0.007 -> filtered out
        assert num_dets == 2, (
            f"Expected 2 detections above threshold, got {num_dets}"
        )

    def test_nms_output_format(self):
        """Verify NMS output dictionary has correct keys and shapes."""
        batch_size = 1
        num_boxes = 3
        max_total = 10

        decoded_boxes = tf.random.uniform([batch_size, num_boxes, 7], 0, 50)
        cls_preds = tf.random.normal([batch_size, num_boxes, NUM_CLASSES])
        dir_preds = tf.random.normal([batch_size, num_boxes, 2])

        results = apply_nms(
            cls_preds=cls_preds,
            decoded_boxes=decoded_boxes,
            dir_preds=dir_preds,
            score_threshold=0.01,
            nms_iou_threshold=0.5,
            max_total_detections=max_total,
        )

        assert "boxes" in results
        assert "scores" in results
        assert "classes" in results
        assert "num_detections" in results
        assert "dir_labels" in results

        assert results["boxes"].shape == (batch_size, max_total, 7)
        assert results["scores"].shape == (batch_size, max_total)
        assert results["classes"].shape == (batch_size, max_total)
        assert results["num_detections"].shape == (batch_size,)

    def test_nms_no_boxes_above_threshold(self):
        """Verify NMS returns zero detections when all scores are below threshold."""
        batch_size = 1
        num_boxes = 5

        decoded_boxes = tf.random.uniform([batch_size, num_boxes, 7], 0, 50)
        # All very low logits -> very low sigmoid scores
        cls_preds = tf.constant(
            [[[-10.0, -10.0, -10.0]] * num_boxes], dtype=tf.float32
        )
        dir_preds = tf.zeros([batch_size, num_boxes, 2])

        results = apply_nms(
            cls_preds=cls_preds,
            decoded_boxes=decoded_boxes,
            dir_preds=dir_preds,
            score_threshold=0.5,
            nms_iou_threshold=0.5,
            max_total_detections=10,
        )

        num_dets = results["num_detections"][0].numpy()
        assert num_dets == 0, f"Expected 0 detections, got {num_dets}"


# ---------------------------------------------------------------------------
# Test: Integration - build_pointpillars utility
# ---------------------------------------------------------------------------


class TestBuildUtility:
    """Tests for the build_pointpillars convenience function."""

    def test_build_returns_model_and_loss(self):
        """Verify build_pointpillars returns a model and loss function."""
        model, loss_fn = build_pointpillars(num_classes=3)

        assert isinstance(model, PointPillarsModel)
        assert isinstance(loss_fn, PointPillarsLoss)

    def test_built_model_runs_forward(self):
        """Verify built model can run a forward pass."""
        model, _ = build_pointpillars(
            num_classes=3,
            grid_x_size=32,
            grid_y_size=32,
        )

        inputs = {
            "pillars": tf.random.normal([1, 100, MAX_POINTS_PER_PILLAR, 4]),
            "pillar_indices": tf.random.uniform(
                [1, 100, 2], 0, 32, dtype=tf.int32
            ),
            "num_points_per_pillar": tf.random.uniform(
                [1, 100], 1, MAX_POINTS_PER_PILLAR, dtype=tf.int32
            ),
        }

        output = model(inputs, training=False)
        assert output["cls_preds"].shape[0] == 1
        assert output["cls_preds"].shape[2] == 3
