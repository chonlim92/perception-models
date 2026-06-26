"""Comprehensive pytest unit tests for the TensorFlow BEVFormer model.

Tests cover all major components: backbone, FPN, spatial/temporal attention,
BEV encoder, DETR decoder, detection heads, loss computation, and end-to-end
forward/backward passes.

All tests use dummy/random data with appropriate shapes to verify correct
tensor dimensions and basic functionality without requiring real data.
"""

import sys
import os

import numpy as np
import pytest
import tensorflow as tf

# Add project root to path so the model module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow.model import (
    BEVFormer,
    BEVEncoder,
    DETRDecoder,
    DetectionHead,
    FeaturePyramidNetwork,
    SpatialCrossAttention,
    TemporalSelfAttention,
    HungarianMatcher,
    focal_loss,
    l1_loss,
    build_bevformer,
    DEFAULT_CONFIG,
)


# =============================================================================
# Test Configuration (smaller than production for fast testing)
# =============================================================================

TEST_BATCH_SIZE = 2
TEST_IMG_H = 224
TEST_IMG_W = 400
TEST_BEV_H = 50
TEST_BEV_W = 50
TEST_EMBED_DIMS = 256
TEST_NUM_CAMERAS = 6
TEST_NUM_CLASSES = 10
TEST_CODE_SIZE = 10
TEST_NUM_QUERIES = 900
TEST_NUM_HEADS = 8
TEST_NUM_LEVELS = 4
TEST_PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

TEST_CONFIG = {
    "bev_h": TEST_BEV_H,
    "bev_w": TEST_BEV_W,
    "embed_dims": TEST_EMBED_DIMS,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "num_heads": TEST_NUM_HEADS,
    "num_queries": TEST_NUM_QUERIES,
    "num_cameras": TEST_NUM_CAMERAS,
    "pc_range": TEST_PC_RANGE,
    "num_points_spatial": 4,
    "num_points_temporal": 4,
    "num_levels": TEST_NUM_LEVELS,
    "fpn_in_channels": [512, 1024, 2048],
    "fpn_out_channels": 256,
    "fpn_num_outs": 4,
    "num_classes": TEST_NUM_CLASSES,
    "code_size": TEST_CODE_SIZE,
    "dropout_rate": 0.1,
    "ffn_dim": 512,
}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def bevformer_model():
    """Create a BEVFormer model instance for testing."""
    model = BEVFormer(config=TEST_CONFIG)
    # Build the model with a dummy forward pass
    dummy_inputs = {
        "images": tf.random.normal([1, TEST_NUM_CAMERAS, TEST_IMG_H, TEST_IMG_W, 3]),
        "lidar2img": tf.random.normal([1, TEST_NUM_CAMERAS, 4, 4]),
        "ego_motion": tf.eye(4, batch_shape=[1]),
        "prev_bev": None,
    }
    _ = model(dummy_inputs, training=False)
    return model


@pytest.fixture
def dummy_images():
    """Generate dummy multi-camera images."""
    return tf.random.normal([TEST_BATCH_SIZE, TEST_NUM_CAMERAS, TEST_IMG_H, TEST_IMG_W, 3])


@pytest.fixture
def dummy_lidar2img():
    """Generate dummy lidar-to-image projection matrices."""
    return tf.random.normal([TEST_BATCH_SIZE, TEST_NUM_CAMERAS, 4, 4])


@pytest.fixture
def dummy_ego_motion():
    """Generate dummy ego-motion matrices (near-identity for testing)."""
    return tf.eye(4, batch_shape=[TEST_BATCH_SIZE])


@pytest.fixture
def dummy_bev_queries():
    """Generate dummy BEV queries."""
    return tf.random.normal([TEST_BATCH_SIZE, TEST_BEV_H * TEST_BEV_W, TEST_EMBED_DIMS])


@pytest.fixture
def dummy_prev_bev():
    """Generate dummy previous BEV features."""
    return tf.random.normal([TEST_BATCH_SIZE, TEST_BEV_H * TEST_BEV_W, TEST_EMBED_DIMS])


@pytest.fixture
def fpn_layer():
    """Create an FPN layer for testing."""
    return FeaturePyramidNetwork(
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        name="test_fpn",
    )


@pytest.fixture
def spatial_cross_attention_layer():
    """Create a SpatialCrossAttention layer for testing."""
    return SpatialCrossAttention(
        embed_dims=TEST_EMBED_DIMS,
        num_heads=TEST_NUM_HEADS,
        num_points=4,
        num_levels=TEST_NUM_LEVELS,
        num_cameras=TEST_NUM_CAMERAS,
        pc_range=TEST_PC_RANGE,
        dropout_rate=0.1,
        name="test_spatial_attn",
    )


@pytest.fixture
def temporal_self_attention_layer():
    """Create a TemporalSelfAttention layer for testing."""
    return TemporalSelfAttention(
        embed_dims=TEST_EMBED_DIMS,
        num_heads=TEST_NUM_HEADS,
        num_points=4,
        dropout_rate=0.1,
        name="test_temporal_attn",
    )


@pytest.fixture
def detection_head_layer():
    """Create a DetectionHead layer for testing."""
    return DetectionHead(
        num_classes=TEST_NUM_CLASSES,
        code_size=TEST_CODE_SIZE,
        embed_dims=TEST_EMBED_DIMS,
        name="test_det_head",
    )


@pytest.fixture
def decoder_layer():
    """Create a DETRDecoder layer for testing."""
    return DETRDecoder(
        num_layers=6,
        embed_dims=TEST_EMBED_DIMS,
        num_heads=TEST_NUM_HEADS,
        num_queries=TEST_NUM_QUERIES,
        ffn_dim=512,
        dropout_rate=0.1,
        name="test_decoder",
    )


@pytest.fixture
def bev_encoder_layer():
    """Create a BEVEncoder for testing."""
    return BEVEncoder(
        num_layers=6,
        embed_dims=TEST_EMBED_DIMS,
        num_heads=TEST_NUM_HEADS,
        num_points_spatial=4,
        num_points_temporal=4,
        num_levels=TEST_NUM_LEVELS,
        num_cameras=TEST_NUM_CAMERAS,
        pc_range=TEST_PC_RANGE,
        ffn_dim=512,
        dropout_rate=0.1,
        name="test_bev_encoder",
    )


# =============================================================================
# Tests
# =============================================================================


class TestBackboneForwardPass:
    """Test ResNet101 backbone outputs correct shapes for stages C3, C4, C5."""

    def test_backbone_forward_pass(self, bevformer_model):
        """Verify backbone produces 3 feature maps at expected resolutions."""
        batch_size = TEST_BATCH_SIZE
        images_flat = tf.random.normal(
            [batch_size * TEST_NUM_CAMERAS, TEST_IMG_H, TEST_IMG_W, 3]
        )

        backbone_outputs = bevformer_model.backbone(images_flat, training=False)

        # Should produce 3 outputs (C3, C4, C5)
        assert len(backbone_outputs) == 3, (
            f"Expected 3 backbone outputs, got {len(backbone_outputs)}"
        )

        # C3: 1/8 resolution, 512 channels
        c3 = backbone_outputs[0]
        expected_h_c3 = TEST_IMG_H // 8
        expected_w_c3 = TEST_IMG_W // 8
        assert c3.shape[0] == batch_size * TEST_NUM_CAMERAS
        assert c3.shape[1] == expected_h_c3
        assert c3.shape[2] == expected_w_c3
        assert c3.shape[3] == 512

        # C4: 1/16 resolution, 1024 channels
        c4 = backbone_outputs[1]
        expected_h_c4 = TEST_IMG_H // 16
        expected_w_c4 = TEST_IMG_W // 16
        assert c4.shape[0] == batch_size * TEST_NUM_CAMERAS
        assert c4.shape[1] == expected_h_c4
        assert c4.shape[2] == expected_w_c4
        assert c4.shape[3] == 1024

        # C5: 1/32 resolution, 2048 channels
        c5 = backbone_outputs[2]
        expected_h_c5 = TEST_IMG_H // 32
        expected_w_c5 = TEST_IMG_W // 32
        assert c5.shape[0] == batch_size * TEST_NUM_CAMERAS
        assert c5.shape[1] == expected_h_c5
        assert c5.shape[2] == expected_w_c5
        assert c5.shape[3] == 2048


class TestFPNForwardPass:
    """Test FPN produces 4 feature maps with 256 channels at correct resolutions."""

    def test_fpn_forward_pass(self, fpn_layer):
        """Verify FPN output shapes and channel dimensions."""
        batch_cameras = TEST_BATCH_SIZE * TEST_NUM_CAMERAS

        # Simulate backbone outputs at different scales
        c3 = tf.random.normal([batch_cameras, TEST_IMG_H // 8, TEST_IMG_W // 8, 512])
        c4 = tf.random.normal([batch_cameras, TEST_IMG_H // 16, TEST_IMG_W // 16, 1024])
        c5 = tf.random.normal([batch_cameras, TEST_IMG_H // 32, TEST_IMG_W // 32, 2048])

        fpn_outputs = fpn_layer([c3, c4, c5], training=False)

        # FPN should produce 4 output levels
        assert len(fpn_outputs) == 4, f"Expected 4 FPN outputs, got {len(fpn_outputs)}"

        # All outputs should have 256 channels
        for i, feat in enumerate(fpn_outputs):
            assert feat.shape[0] == batch_cameras, (
                f"Level {i}: batch mismatch, expected {batch_cameras}, got {feat.shape[0]}"
            )
            assert feat.shape[3] == 256, (
                f"Level {i}: expected 256 channels, got {feat.shape[3]}"
            )

        # First 3 levels should match backbone spatial resolutions
        assert fpn_outputs[0].shape[1] == TEST_IMG_H // 8
        assert fpn_outputs[0].shape[2] == TEST_IMG_W // 8
        assert fpn_outputs[1].shape[1] == TEST_IMG_H // 16
        assert fpn_outputs[1].shape[2] == TEST_IMG_W // 16
        assert fpn_outputs[2].shape[1] == TEST_IMG_H // 32
        assert fpn_outputs[2].shape[2] == TEST_IMG_W // 32

        # Level 4 is extra conv (stride 2 from C5)
        assert fpn_outputs[3].shape[1] == (TEST_IMG_H // 32 + 1) // 2 or \
               fpn_outputs[3].shape[1] == TEST_IMG_H // 64


class TestSpatialCrossAttention:
    """Test spatial cross-attention with dummy BEV queries and image features."""

    def test_spatial_cross_attention(self, spatial_cross_attention_layer, dummy_bev_queries,
                                     dummy_lidar2img):
        """Verify spatial cross-attention output shape (B, H*W, C)."""
        bev_h = TEST_BEV_H
        bev_w = TEST_BEV_W
        batch_cameras = TEST_BATCH_SIZE * TEST_NUM_CAMERAS

        # Create multi-scale features simulating FPN outputs
        spatial_shapes = [
            (TEST_IMG_H // 8, TEST_IMG_W // 8),
            (TEST_IMG_H // 16, TEST_IMG_W // 16),
            (TEST_IMG_H // 32, TEST_IMG_W // 32),
            (TEST_IMG_H // 64, TEST_IMG_W // 64),
        ]

        multi_scale_features = [
            tf.random.normal([batch_cameras, h, w, TEST_EMBED_DIMS])
            for h, w in spatial_shapes
        ]

        output = spatial_cross_attention_layer(
            bev_queries=dummy_bev_queries,
            multi_scale_features=multi_scale_features,
            lidar2img=dummy_lidar2img,
            bev_h=bev_h,
            bev_w=bev_w,
            spatial_shapes=spatial_shapes,
            training=False,
        )

        # Output should be (B, bev_h*bev_w, embed_dims)
        expected_shape = (TEST_BATCH_SIZE, bev_h * bev_w, TEST_EMBED_DIMS)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

        # Output should not be all zeros (attention should produce meaningful values)
        assert not tf.reduce_all(output == 0.0).numpy(), (
            "Spatial cross-attention output is all zeros"
        )


class TestTemporalSelfAttention:
    """Test temporal self-attention with current and previous BEV features."""

    def test_temporal_self_attention(self, temporal_self_attention_layer,
                                     dummy_bev_queries, dummy_prev_bev, dummy_ego_motion):
        """Verify temporal self-attention output shape with ego-motion."""
        output = temporal_self_attention_layer(
            bev_queries=dummy_bev_queries,
            prev_bev=dummy_prev_bev,
            ego_motion=dummy_ego_motion,
            bev_h=TEST_BEV_H,
            bev_w=TEST_BEV_W,
            training=False,
        )

        # Output should be (B, bev_h*bev_w, embed_dims)
        expected_shape = (TEST_BATCH_SIZE, TEST_BEV_H * TEST_BEV_W, TEST_EMBED_DIMS)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_temporal_self_attention_no_prev(self, temporal_self_attention_layer,
                                             dummy_bev_queries, dummy_ego_motion):
        """Verify temporal self-attention works when prev_bev is None (first frame)."""
        output = temporal_self_attention_layer(
            bev_queries=dummy_bev_queries,
            prev_bev=None,
            ego_motion=dummy_ego_motion,
            bev_h=TEST_BEV_H,
            bev_w=TEST_BEV_W,
            training=False,
        )

        expected_shape = (TEST_BATCH_SIZE, TEST_BEV_H * TEST_BEV_W, TEST_EMBED_DIMS)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )


class TestBEVEncoderFull:
    """Test full BEV encoder (6 layers) with proper inputs."""

    def test_bev_encoder_full(self, bev_encoder_layer, dummy_bev_queries, dummy_prev_bev,
                              dummy_ego_motion, dummy_lidar2img):
        """Verify BEV encoder output shape (B, bev_h*bev_w, 256)."""
        batch_cameras = TEST_BATCH_SIZE * TEST_NUM_CAMERAS

        spatial_shapes = [
            (TEST_IMG_H // 8, TEST_IMG_W // 8),
            (TEST_IMG_H // 16, TEST_IMG_W // 16),
            (TEST_IMG_H // 32, TEST_IMG_W // 32),
            (TEST_IMG_H // 64, TEST_IMG_W // 64),
        ]

        multi_scale_features = [
            tf.random.normal([batch_cameras, h, w, TEST_EMBED_DIMS])
            for h, w in spatial_shapes
        ]

        output = bev_encoder_layer(
            bev_queries=dummy_bev_queries,
            prev_bev=dummy_prev_bev,
            ego_motion=dummy_ego_motion,
            multi_scale_features=multi_scale_features,
            lidar2img=dummy_lidar2img,
            bev_h=TEST_BEV_H,
            bev_w=TEST_BEV_W,
            spatial_shapes=spatial_shapes,
            training=False,
        )

        expected_shape = (TEST_BATCH_SIZE, TEST_BEV_H * TEST_BEV_W, TEST_EMBED_DIMS)
        assert output.shape == expected_shape, (
            f"Expected BEV encoder output shape {expected_shape}, got {output.shape}"
        )


class TestDecoderForward:
    """Test DETR decoder with object queries and BEV memory."""

    def test_decoder_forward(self, decoder_layer):
        """Verify decoder output shape (B, num_queries, embed_dims)."""
        # BEV features as memory for cross-attention
        bev_features = tf.random.normal(
            [TEST_BATCH_SIZE, TEST_BEV_H * TEST_BEV_W, TEST_EMBED_DIMS]
        )

        output = decoder_layer(bev_features, training=False)

        expected_shape = (TEST_BATCH_SIZE, TEST_NUM_QUERIES, TEST_EMBED_DIMS)
        assert output.shape == expected_shape, (
            f"Expected decoder output shape {expected_shape}, got {output.shape}"
        )


class TestDetectionHead:
    """Test classification and regression heads."""

    def test_detection_head(self, detection_head_layer):
        """Verify classification (10 classes) and regression (10 code values) output shapes."""
        query_features = tf.random.normal(
            [TEST_BATCH_SIZE, TEST_NUM_QUERIES, TEST_EMBED_DIMS]
        )

        cls_logits, bbox_preds = detection_head_layer(query_features, training=False)

        # Classification: (B, num_queries, num_classes)
        expected_cls_shape = (TEST_BATCH_SIZE, TEST_NUM_QUERIES, TEST_NUM_CLASSES)
        assert cls_logits.shape == expected_cls_shape, (
            f"Expected cls shape {expected_cls_shape}, got {cls_logits.shape}"
        )

        # Regression: (B, num_queries, code_size)
        expected_reg_shape = (TEST_BATCH_SIZE, TEST_NUM_QUERIES, TEST_CODE_SIZE)
        assert bbox_preds.shape == expected_reg_shape, (
            f"Expected reg shape {expected_reg_shape}, got {bbox_preds.shape}"
        )


class TestLossComputation:
    """Test Hungarian matching + focal loss + L1 loss."""

    def test_loss_computation(self):
        """Verify loss computation produces scalar > 0 with dummy predictions and GT."""
        num_queries = 100  # Smaller for speed
        num_gt = 5

        # Dummy predictions
        cls_logits = tf.random.normal([1, num_queries, TEST_NUM_CLASSES])
        bbox_preds = tf.random.uniform([1, num_queries, TEST_CODE_SIZE])

        # Dummy ground truth
        gt_labels = tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)
        gt_bboxes = tf.random.uniform([num_gt, TEST_CODE_SIZE])

        gt_labels_list = [gt_labels]
        gt_bboxes_list = [gt_bboxes]

        # Create a minimal model instance for loss computation
        config = TEST_CONFIG.copy()
        config["num_queries"] = num_queries
        config["num_encoder_layers"] = 1
        config["num_decoder_layers"] = 1

        # Use matcher and loss functions directly
        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0)

        # Run matching
        pred_indices, gt_indices = matcher.match(
            cls_logits[0], bbox_preds[0], gt_labels, gt_bboxes
        )

        # Verify matching produced valid indices
        assert len(pred_indices) == num_gt, (
            f"Expected {num_gt} matches, got {len(pred_indices)}"
        )
        assert len(gt_indices) == num_gt

        # Compute focal loss
        target_classes = tf.zeros([num_queries], dtype=tf.int32)
        cls_loss = focal_loss(cls_logits[0], target_classes, TEST_NUM_CLASSES)
        assert cls_loss.shape == (), f"Expected scalar loss, got shape {cls_loss.shape}"
        assert cls_loss.numpy() > 0, "Focal loss should be > 0"

        # Compute L1 loss on matched predictions
        matched_preds = tf.gather(bbox_preds[0], pred_indices)
        matched_gt = tf.gather(gt_bboxes, gt_indices)
        bbox_loss = l1_loss(matched_preds, matched_gt)
        assert bbox_loss.shape == (), f"Expected scalar loss, got shape {bbox_loss.shape}"
        assert bbox_loss.numpy() > 0, "L1 loss should be > 0"


class TestFullModelForward:
    """Test end-to-end forward pass with all inputs."""

    def test_full_model_forward(self, bevformer_model):
        """Verify all output shapes from end-to-end forward pass."""
        inputs = {
            "images": tf.random.normal(
                [TEST_BATCH_SIZE, TEST_NUM_CAMERAS, TEST_IMG_H, TEST_IMG_W, 3]
            ),
            "lidar2img": tf.random.normal([TEST_BATCH_SIZE, TEST_NUM_CAMERAS, 4, 4]),
            "ego_motion": tf.eye(4, batch_shape=[TEST_BATCH_SIZE]),
            "prev_bev": tf.random.normal(
                [TEST_BATCH_SIZE, TEST_BEV_H * TEST_BEV_W, TEST_EMBED_DIMS]
            ),
        }

        outputs = bevformer_model(inputs, training=False)

        # Check all output keys exist
        assert "cls_logits" in outputs
        assert "bbox_preds" in outputs
        assert "bev_features" in outputs

        # Classification logits: (B, num_queries, num_classes)
        assert outputs["cls_logits"].shape == (
            TEST_BATCH_SIZE, TEST_NUM_QUERIES, TEST_NUM_CLASSES
        ), f"cls_logits shape: {outputs['cls_logits'].shape}"

        # Bounding box predictions: (B, num_queries, code_size)
        assert outputs["bbox_preds"].shape == (
            TEST_BATCH_SIZE, TEST_NUM_QUERIES, TEST_CODE_SIZE
        ), f"bbox_preds shape: {outputs['bbox_preds'].shape}"

        # BEV features: (B, bev_h*bev_w, embed_dims)
        assert outputs["bev_features"].shape == (
            TEST_BATCH_SIZE, TEST_BEV_H * TEST_BEV_W, TEST_EMBED_DIMS
        ), f"bev_features shape: {outputs['bev_features'].shape}"

    def test_full_model_forward_no_prev_bev(self, bevformer_model):
        """Verify forward pass works without previous BEV features (first frame)."""
        inputs = {
            "images": tf.random.normal(
                [TEST_BATCH_SIZE, TEST_NUM_CAMERAS, TEST_IMG_H, TEST_IMG_W, 3]
            ),
            "lidar2img": tf.random.normal([TEST_BATCH_SIZE, TEST_NUM_CAMERAS, 4, 4]),
            "ego_motion": tf.eye(4, batch_shape=[TEST_BATCH_SIZE]),
            "prev_bev": None,
        }

        outputs = bevformer_model(inputs, training=False)

        assert outputs["cls_logits"].shape == (
            TEST_BATCH_SIZE, TEST_NUM_QUERIES, TEST_NUM_CLASSES
        )
        assert outputs["bbox_preds"].shape == (
            TEST_BATCH_SIZE, TEST_NUM_QUERIES, TEST_CODE_SIZE
        )
        assert outputs["bev_features"].shape == (
            TEST_BATCH_SIZE, TEST_BEV_H * TEST_BEV_W, TEST_EMBED_DIMS
        )


class TestGradientFlow:
    """Test that gradients flow through the model without NaN values."""

    def test_gradient_flow(self, bevformer_model):
        """Run forward + backward pass, verify no NaN gradients."""
        inputs = {
            "images": tf.random.normal(
                [1, TEST_NUM_CAMERAS, TEST_IMG_H, TEST_IMG_W, 3]
            ),
            "lidar2img": tf.random.normal([1, TEST_NUM_CAMERAS, 4, 4]),
            "ego_motion": tf.eye(4, batch_shape=[1]),
            "prev_bev": None,
        }

        with tf.GradientTape() as tape:
            outputs = bevformer_model(inputs, training=True)
            # Use a simple loss: mean of cls_logits
            loss = tf.reduce_mean(outputs["cls_logits"])

        gradients = tape.gradient(loss, bevformer_model.trainable_variables)

        # Check that we got gradients (at least some should be non-None)
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > 0, "No gradients were computed"

        # Verify no NaN gradients in any trainable variable
        for i, grad in enumerate(gradients):
            if grad is not None:
                has_nan = tf.reduce_any(tf.math.is_nan(grad)).numpy()
                assert not has_nan, (
                    f"NaN gradient found in variable "
                    f"{bevformer_model.trainable_variables[i].name}"
                )

        # Verify no Inf gradients
        for i, grad in enumerate(gradients):
            if grad is not None:
                has_inf = tf.reduce_any(tf.math.is_inf(grad)).numpy()
                assert not has_inf, (
                    f"Inf gradient found in variable "
                    f"{bevformer_model.trainable_variables[i].name}"
                )


class TestModelTrainableParams:
    """Test model has expected number of trainable parameters."""

    def test_model_trainable_params(self, bevformer_model):
        """Verify model has expected order of magnitude of trainable parameters.

        BEVFormer-Base with ResNet101 backbone should have roughly 60-80M parameters.
        The test checks that parameter count is in a reasonable range.
        """
        total_params = sum(
            tf.size(var).numpy() for var in bevformer_model.trainable_variables
        )

        # BEVFormer should have tens of millions of parameters
        # ResNet101 alone has ~44.5M, plus FPN, encoder, decoder, heads
        # Expect roughly 40M-200M total trainable params
        assert total_params > 10_000_000, (
            f"Model has too few parameters: {total_params:,}. "
            f"Expected at least 10M for BEVFormer."
        )
        assert total_params < 500_000_000, (
            f"Model has too many parameters: {total_params:,}. "
            f"Expected less than 500M for BEVFormer."
        )

        # Verify specific component parameter counts are reasonable
        backbone_params = sum(
            tf.size(var).numpy() for var in bevformer_model.backbone.trainable_variables
        )
        assert backbone_params > 30_000_000, (
            f"Backbone has too few parameters: {backbone_params:,}. "
            f"ResNet101 should have ~44.5M."
        )


class TestBEVGridGeneration:
    """Test BEV grid coordinate generation."""

    def test_bev_grid_generation(self):
        """Verify BEV grid coordinates cover the expected range.

        The spatial cross-attention generates 3D reference points for the BEV grid.
        These should be uniformly distributed in [0, 1] range (normalized).
        """
        spatial_attn = SpatialCrossAttention(
            embed_dims=TEST_EMBED_DIMS,
            num_heads=TEST_NUM_HEADS,
            num_points=4,
            num_levels=TEST_NUM_LEVELS,
            num_cameras=TEST_NUM_CAMERAS,
            pc_range=TEST_PC_RANGE,
            dropout_rate=0.1,
            name="test_grid_gen",
        )

        bev_h = TEST_BEV_H
        bev_w = TEST_BEV_W
        num_z_anchors = 4

        ref_points = spatial_attn._get_reference_points_3d(
            bev_h, bev_w, num_z_anchors=num_z_anchors
        )

        # Shape: (1, bev_h*bev_w, num_z_anchors, 3)
        expected_shape = (1, bev_h * bev_w, num_z_anchors, 3)
        assert ref_points.shape == expected_shape, (
            f"Expected ref points shape {expected_shape}, got {ref_points.shape}"
        )

        # All coordinates should be in [0, 1] (normalized)
        min_val = tf.reduce_min(ref_points).numpy()
        max_val = tf.reduce_max(ref_points).numpy()
        assert min_val >= 0.0, f"Min reference point value {min_val} < 0"
        assert max_val <= 1.0, f"Max reference point value {max_val} > 1"

        # X and Y coordinates should span approximately [0, 1] with uniform spacing
        x_coords = ref_points[0, :, 0, 0].numpy()  # First z-anchor, x component
        y_coords = ref_points[0, :, 0, 1].numpy()  # First z-anchor, y component
        z_coords = ref_points[0, 0, :, 2].numpy()  # First spatial pos, all z-anchors

        # X should have bev_w unique values approximately
        unique_x = np.unique(np.round(x_coords, decimals=5))
        assert len(unique_x) == bev_w, (
            f"Expected {bev_w} unique x values, got {len(unique_x)}"
        )

        # Z should span from 0 to 1
        assert z_coords[0] == pytest.approx(0.0, abs=1e-5), (
            f"Z min should be ~0.0, got {z_coords[0]}"
        )
        assert z_coords[-1] == pytest.approx(1.0, abs=1e-5), (
            f"Z max should be ~1.0, got {z_coords[-1]}"
        )

        # Verify the grid covers the full spatial range
        assert np.min(x_coords) > 0.0, "X should not start at exactly 0 (half-cell offset)"
        assert np.max(x_coords) < 1.0, "X should not end at exactly 1 (half-cell offset)"


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestHungarianMatcher:
    """Test Hungarian matching edge cases."""

    def test_matcher_empty_gt(self):
        """Verify matcher handles empty ground truth gracefully."""
        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0)

        cls_logits = tf.random.normal([100, TEST_NUM_CLASSES])
        bbox_preds = tf.random.uniform([100, TEST_CODE_SIZE])
        gt_labels = tf.constant([], dtype=tf.int32)
        gt_bboxes = tf.zeros([0, TEST_CODE_SIZE])

        pred_indices, gt_indices = matcher.match(
            cls_logits, bbox_preds, gt_labels, gt_bboxes
        )

        assert len(pred_indices) == 0
        assert len(gt_indices) == 0

    def test_matcher_single_gt(self):
        """Verify matcher works with a single ground truth object."""
        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0)

        cls_logits = tf.random.normal([50, TEST_NUM_CLASSES])
        bbox_preds = tf.random.uniform([50, TEST_CODE_SIZE])
        gt_labels = tf.constant([3], dtype=tf.int32)
        gt_bboxes = tf.random.uniform([1, TEST_CODE_SIZE])

        pred_indices, gt_indices = matcher.match(
            cls_logits, bbox_preds, gt_labels, gt_bboxes
        )

        assert len(pred_indices) == 1
        assert len(gt_indices) == 1
        assert gt_indices[0] == 0


class TestFocalLoss:
    """Test focal loss computation."""

    def test_focal_loss_scalar(self):
        """Verify focal loss returns a scalar."""
        logits = tf.random.normal([50, TEST_NUM_CLASSES])
        targets = tf.random.uniform([50], minval=0, maxval=TEST_NUM_CLASSES, dtype=tf.int32)

        loss = focal_loss(logits, targets, TEST_NUM_CLASSES)

        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert loss.numpy() > 0, "Focal loss should be positive"
        assert not tf.math.is_nan(loss).numpy(), "Focal loss should not be NaN"

    def test_l1_loss_scalar(self):
        """Verify L1 loss returns a scalar."""
        preds = tf.random.uniform([10, TEST_CODE_SIZE])
        targets = tf.random.uniform([10, TEST_CODE_SIZE])

        loss = l1_loss(preds, targets)

        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert loss.numpy() > 0, "L1 loss should be positive"
        assert not tf.math.is_nan(loss).numpy(), "L1 loss should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
