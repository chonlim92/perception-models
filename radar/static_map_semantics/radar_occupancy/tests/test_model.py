# [IMPLEMENTED BY CLAUDE - was missing]
"""
Unit tests for radar occupancy PyTorch models and losses.

Tests cover:
- PillarOccNet forward pass shape validation
- TemporalPillarOccNet forward pass with multi-frame fusion
- ClassicalISM Bayesian update and probability output
- FocalLoss computation with ignore_index
- WCELoss computation with multi-class targets
- RadarOccupancyLoss combined loss computation
"""

import sys
import os

# Add parent directories to path so we can import the pytorch package
_this_dir = os.path.dirname(os.path.abspath(__file__))
_radar_occ_dir = os.path.dirname(_this_dir)
_static_map_dir = os.path.dirname(_radar_occ_dir)
_radar_dir = os.path.dirname(_static_map_dir)
_repo_root = os.path.dirname(_radar_dir)

sys.path.insert(0, _radar_occ_dir)
sys.path.insert(0, _repo_root)

import numpy as np
import pytest
import torch

from pytorch.model import ClassicalISM, PillarOccNet, TemporalPillarOccNet
from pytorch.losses import FocalLoss, WCELoss, RadarOccupancyLoss


@pytest.fixture
def base_config():
    """Base configuration dictionary for model instantiation."""
    return {
        "grid": {
            "grid_size": [200, 200],
            "cell_size": 0.5,
            "x_range": [-50, 50],
            "y_range": [-50, 50],
        },
        "model": {
            "type": "pillar_occ_net",
            "pillar": {
                "input_features": 9,
                "pillar_features": 64,
                "max_points_per_pillar": 20,
            },
            "backbone": {
                "encoder_channels": [64, 128, 256, 512],
                "decoder_channels": [256, 128, 64],
            },
            "heads": {
                "semantics": {"enabled": True, "num_classes": 5},
            },
            "temporal": {
                "num_frames": 3,
                "fusion_method": "concat_conv",
                "temporal_conv_channels": 64,
            },
        },
        "classical_ism": {
            "free_log_odds": -0.4,
            "occ_log_odds": 0.85,
            "clamp_range": [-5, 5],
            "range_sigma": 1.0,
            "angle_sigma": 3.0,
            "rcs_weight": True,
        },
    }


@pytest.fixture
def temporal_config(base_config):
    """Configuration with temporal settings for TemporalPillarOccNet."""
    config = base_config.copy()
    config["model"] = dict(base_config["model"])
    config["model"]["type"] = "temporal_pillar_occ_net"
    return config


class TestPillarOccNet:
    """Tests for the single-frame PillarOccNet model."""

    def test_pillar_occ_net_forward_shape(self, base_config):
        """Test that PillarOccNet produces correct output shapes."""
        model = PillarOccNet(base_config)
        model.eval()

        batch_size = 2
        max_pillars = 100
        max_points = 20
        input_features = 9

        # Create dummy inputs
        pillar_features = torch.randn(batch_size, max_pillars, max_points, input_features)
        pillar_indices = torch.randint(0, 200, (batch_size, max_pillars, 2))
        num_pillars = torch.tensor([50, 60])

        with torch.no_grad():
            output = model(pillar_features, pillar_indices, num_pillars)

        assert output["occupancy"].shape == (2, 1, 200, 200), (
            f"Expected occupancy shape (2, 1, 200, 200), got {output['occupancy'].shape}"
        )
        assert output["semantics"].shape == (2, 5, 200, 200), (
            f"Expected semantics shape (2, 5, 200, 200), got {output['semantics'].shape}"
        )


class TestTemporalPillarOccNet:
    """Tests for the multi-frame TemporalPillarOccNet model."""

    def test_temporal_pillar_occ_net_forward_shape(self, temporal_config):
        """Test that TemporalPillarOccNet produces correct output shapes with temporal fusion."""
        model = TemporalPillarOccNet(temporal_config)
        model.eval()

        batch_size = 2
        max_pillars = 100
        max_points = 20
        input_features = 9
        num_frames = 3

        # Create dummy sequence inputs (one per frame)
        pillar_features_seq = [
            torch.randn(batch_size, max_pillars, max_points, input_features)
            for _ in range(num_frames)
        ]
        pillar_indices_seq = [
            torch.randint(0, 200, (batch_size, max_pillars, 2))
            for _ in range(num_frames)
        ]
        num_pillars_seq = [
            torch.tensor([50, 60])
            for _ in range(num_frames)
        ]

        # Ego transforms: (B, T-1, 4, 4) - identity matrices (no motion)
        ego_transforms = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames - 1, 4, 4).clone()

        with torch.no_grad():
            output = model(pillar_features_seq, pillar_indices_seq, num_pillars_seq, ego_transforms)

        assert output["occupancy"].shape == (2, 1, 200, 200), (
            f"Expected occupancy shape (2, 1, 200, 200), got {output['occupancy'].shape}"
        )
        assert output["semantics"].shape == (2, 5, 200, 200), (
            f"Expected semantics shape (2, 5, 200, 200), got {output['semantics'].shape}"
        )


class TestClassicalISM:
    """Tests for the classical Inverse Sensor Model."""

    def test_classical_ism_update_and_output(self, base_config):
        """Test that ClassicalISM updates occupancy grid correctly."""
        ism = ClassicalISM(base_config)

        # Create some radar points: (N, 6) = [x, y, z, rcs, vr_comp, dt]
        radar_points = np.array([
            [10.0, 5.0, 0.5, 15.0, -2.0, 0.0],
            [20.0, -3.0, 0.3, 10.0, 1.5, 0.0],
            [5.0, 15.0, 0.8, 20.0, -1.0, 0.0],
            [-8.0, 12.0, 0.4, 5.0, 0.5, 0.0],
            [30.0, 0.0, 0.6, 25.0, -3.0, 0.0],
        ], dtype=np.float32)

        # Update the grid
        ism.update(radar_points)

        # Get occupancy probability
        occ_prob = ism.get_occupancy_probability()

        # Assert shape matches grid_size
        assert occ_prob.shape == tuple(base_config["grid"]["grid_size"]), (
            f"Expected shape {tuple(base_config['grid']['grid_size'])}, got {occ_prob.shape}"
        )

        # Assert values are in [0, 1]
        assert np.all(occ_prob >= 0.0), "Occupancy probabilities must be >= 0"
        assert np.all(occ_prob <= 1.0), "Occupancy probabilities must be <= 1"

        # Assert some cells have been updated (not all remain at prior 0.5)
        assert not np.allclose(occ_prob, 0.5), (
            "After update, occupancy grid should not remain all 0.5 (prior)"
        )


class TestFocalLoss:
    """Tests for the binary focal loss."""

    def test_focal_loss_computation(self):
        """Test FocalLoss is scalar, positive, finite, and respects ignore_index."""
        loss_fn = FocalLoss(alpha=0.75, gamma=2.0, ignore_index=255)

        # Create dummy predictions and targets
        pred = torch.randn(2, 1, 50, 50)
        target = torch.zeros(2, 50, 50, dtype=torch.long)

        # Set some cells to occupied (1)
        target[:, 10:20, 10:20] = 1

        # Set some cells to ignore (255)
        target[:, 0:5, 0:5] = 255

        loss = loss_fn(pred, target)

        # Assert loss is scalar
        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"

        # Assert loss is positive
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

        # Assert loss is finite
        assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"

        # Verify ignore_index cells do not contribute:
        # Compute with all cells being ignore_index -> loss should be 0
        target_all_ignore = torch.full((2, 50, 50), 255, dtype=torch.long)
        loss_ignored = loss_fn(pred, target_all_ignore)
        assert loss_ignored.item() == 0.0, (
            f"Expected zero loss when all cells are ignore_index, got {loss_ignored.item()}"
        )


class TestWCELoss:
    """Tests for the weighted cross-entropy loss."""

    def test_wce_loss_computation(self):
        """Test WCELoss is scalar, positive, finite for multi-class targets."""
        num_classes = 5
        class_weights = torch.ones(num_classes)
        loss_fn = WCELoss(class_weights=class_weights, ignore_index=255)

        # Create dummy predictions and targets
        pred = torch.randn(2, num_classes, 50, 50)
        target = torch.randint(0, num_classes, (2, 50, 50), dtype=torch.long)

        # Set some cells to ignore (255)
        target[:, 45:50, 45:50] = 255

        loss = loss_fn(pred, target)

        # Assert loss is scalar
        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"

        # Assert loss is positive
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

        # Assert loss is finite
        assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"


class TestRadarOccupancyLoss:
    """Tests for the combined radar occupancy loss."""

    def test_radar_occupancy_loss_combined(self):
        """Test combined loss returns correct dict and weighted total."""
        occ_weight = 1.0
        sem_weight = 0.5

        loss_fn = RadarOccupancyLoss(
            occ_weight=occ_weight,
            sem_weight=sem_weight,
            focal_alpha=0.75,
            focal_gamma=2.0,
            ignore_index=255,
        )

        # Create dummy occupancy predictions and targets
        occ_pred = torch.randn(2, 1, 50, 50)
        occ_target = torch.zeros(2, 50, 50, dtype=torch.long)
        occ_target[:, 10:20, 10:20] = 1

        # Create dummy semantic predictions and targets
        num_classes = 5
        sem_pred = torch.randn(2, num_classes, 50, 50)
        sem_target = torch.randint(0, num_classes, (2, 50, 50), dtype=torch.long)

        loss_dict = loss_fn(occ_pred, occ_target, sem_pred, sem_target)

        # Assert returns dict with expected keys
        assert "total" in loss_dict, "Loss dict must contain 'total' key"
        assert "occupancy" in loss_dict, "Loss dict must contain 'occupancy' key"
        assert "semantic" in loss_dict, "Loss dict must contain 'semantic' key"

        # Assert all values are scalar, positive, finite
        for key in ["total", "occupancy", "semantic"]:
            val = loss_dict[key]
            assert val.dim() == 0, f"Expected scalar for '{key}', got shape {val.shape}"
            assert val.item() > 0, f"Expected positive value for '{key}', got {val.item()}"
            assert torch.isfinite(val), f"Expected finite value for '{key}', got {val.item()}"

        # Assert total = occ_weight * occupancy + sem_weight * semantic
        expected_total = occ_weight * loss_dict["occupancy"] + sem_weight * loss_dict["semantic"]
        assert torch.allclose(loss_dict["total"], expected_total, atol=1e-6), (
            f"Total loss {loss_dict['total'].item():.6f} != "
            f"occ_weight * occupancy + sem_weight * semantic = {expected_total.item():.6f}"
        )
