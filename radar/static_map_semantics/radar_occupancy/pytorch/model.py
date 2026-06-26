"""
Radar Occupancy Grid Mapping — PyTorch Implementation

Three approaches:
1. ClassicalISM: Bayesian inverse sensor model (no learning)
2. PillarOccNet: Neural single-frame occupancy prediction
3. TemporalPillarOccNet: Multi-frame temporal fusion
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalISM:
    """Classical Inverse Sensor Model with Bayesian log-odds fusion.

    No neural network — pure probabilistic occupancy mapping.
    """

    def __init__(self, config):
        self.grid_size = config["grid"]["grid_size"]
        self.cell_size = config["grid"]["cell_size"]
        self.x_range = config["grid"]["x_range"]
        self.y_range = config["grid"]["y_range"]

        ism_cfg = config["classical_ism"]
        self.free_log_odds = ism_cfg["free_log_odds"]
        self.occ_log_odds = ism_cfg["occ_log_odds"]
        self.clamp_range = ism_cfg["clamp_range"]
        self.range_sigma = ism_cfg["range_sigma"]
        self.angle_sigma = math.radians(ism_cfg["angle_sigma"])
        self.rcs_weight = ism_cfg["rcs_weight"]

        self.log_odds_map = np.zeros(self.grid_size, dtype=np.float32)

    def reset(self):
        self.log_odds_map = np.zeros(self.grid_size, dtype=np.float32)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        gx = int((x - self.x_range[0]) / self.cell_size)
        gy = int((y - self.y_range[0]) / self.cell_size)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Convert grid indices to world coordinates (cell center)."""
        x = self.x_range[0] + (gx + 0.5) * self.cell_size
        y = self.y_range[0] + (gy + 0.5) * self.cell_size
        return x, y

    def update(self, radar_points, sensor_origin=None):
        """Update occupancy grid with a set of radar detections.

        Args:
            radar_points: (N, 6) array [x, y, z, rcs, vr_comp, dt]
            sensor_origin: (2,) sensor position in world frame [x, y]
        """
        if sensor_origin is None:
            sensor_origin = np.array([0.0, 0.0])

        for i in range(len(radar_points)):
            x, y, z, rcs, vr, dt = radar_points[i]

            det_range = math.sqrt((x - sensor_origin[0]) ** 2 +
                                  (y - sensor_origin[1]) ** 2)
            det_angle = math.atan2(y - sensor_origin[1], x - sensor_origin[0])

            rcs_factor = 1.0
            if self.rcs_weight:
                rcs_factor = np.clip((rcs + 10.0) / 30.0, 0.3, 1.5)

            self._update_ray(sensor_origin, det_range, det_angle, rcs_factor)

        self.log_odds_map = np.clip(
            self.log_odds_map, self.clamp_range[0], self.clamp_range[1]
        )

    def _update_ray(self, origin, det_range, det_angle, rcs_factor):
        """Cast ray and update cells along it."""
        num_steps = int(det_range / self.cell_size) + 1
        step_size = self.cell_size * 0.5

        for step in range(int(det_range / step_size) + 1):
            r = step * step_size
            x = origin[0] + r * math.cos(det_angle)
            y = origin[1] + r * math.sin(det_angle)

            gx, gy = self.world_to_grid(x, y)
            if not (0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1]):
                continue

            dist_to_det = abs(r - det_range)

            if dist_to_det < self.range_sigma * 2:
                gauss_weight = math.exp(-0.5 * (dist_to_det / self.range_sigma) ** 2)
                update = self.occ_log_odds * gauss_weight * rcs_factor
                self.log_odds_map[gx, gy] += update
            elif r < det_range - self.range_sigma * 2:
                self.log_odds_map[gx, gy] += self.free_log_odds * 0.5

    def get_occupancy_probability(self):
        """Convert log-odds map to probability map."""
        return 1.0 / (1.0 + np.exp(-self.log_odds_map))

    def get_binary_occupancy(self, threshold=0.5):
        """Get binary occupancy map."""
        prob = self.get_occupancy_probability()
        return (prob > threshold).astype(np.uint8)


class PillarFeatureNet(nn.Module):
    """Encode radar points into pillar features using PointNet."""

    def __init__(self, in_channels=9, out_channels=64, max_points_per_pillar=20):
        super().__init__()
        self.max_points = max_points_per_pillar
        self.out_channels = out_channels

        self.net = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, pillar_features, pillar_indices, num_pillars):
        """
        Args:
            pillar_features: (B, max_pillars, max_points, C_in)
            pillar_indices: (B, max_pillars, 2) grid indices for each pillar
            num_pillars: (B,) actual number of pillars per sample

        Returns:
            pillar_encodings: (B, max_pillars, C_out)
        """
        B, P, N, C = pillar_features.shape

        features_flat = pillar_features.reshape(B * P * N, C)
        encoded = self.net(features_flat)
        encoded = encoded.reshape(B, P, N, self.out_channels)

        pillar_encodings = encoded.max(dim=2)[0]

        return pillar_encodings


class ScatterBEV(nn.Module):
    """Scatter pillar features to BEV pseudo-image."""

    def __init__(self, grid_size, channels):
        super().__init__()
        self.grid_size = grid_size
        self.channels = channels

    def forward(self, pillar_features, pillar_indices, num_pillars):
        """
        Args:
            pillar_features: (B, max_pillars, C)
            pillar_indices: (B, max_pillars, 2) [grid_x, grid_y]
            num_pillars: (B,)

        Returns:
            bev: (B, C, H, W)
        """
        B = pillar_features.shape[0]
        H, W = self.grid_size
        C = self.channels

        bev = torch.zeros(B, C, H, W, device=pillar_features.device,
                         dtype=pillar_features.dtype)

        for b in range(B):
            n = num_pillars[b]
            if n == 0:
                continue
            indices = pillar_indices[b, :n]  # (n, 2)
            features = pillar_features[b, :n]  # (n, C)

            valid = (indices[:, 0] >= 0) & (indices[:, 0] < H) & \
                    (indices[:, 1] >= 0) & (indices[:, 1] < W)
            indices = indices[valid]
            features = features[valid]

            bev[b, :, indices[:, 0], indices[:, 1]] = features.T

        return bev


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU with optional stride."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample → Concat skip → Conv block."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.bn_up = nn.BatchNorm2d(out_ch)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.relu(self.bn_up(self.up(x)), inplace=True)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                            align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetBackbone(nn.Module):
    """U-Net encoder-decoder for BEV feature processing."""

    def __init__(self, in_channels=64, encoder_channels=None, decoder_channels=None):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
        if decoder_channels is None:
            decoder_channels = [256, 128, 64]

        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for ch_out in encoder_channels:
            stride = 2 if ch_out != encoder_channels[0] else 1
            self.encoders.append(ConvBlock(ch_in, ch_out, stride=stride))
            ch_in = ch_out

        self.decoders = nn.ModuleList()
        enc_channels_rev = list(reversed(encoder_channels[:-1]))
        for i, ch_out in enumerate(decoder_channels):
            skip_ch = enc_channels_rev[i] if i < len(enc_channels_rev) else 0
            self.decoders.append(UpBlock(ch_in, skip_ch, ch_out))
            ch_in = ch_out

        self.out_channels = decoder_channels[-1]

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.encoders) - 1:
                skips.append(x)

        skips = list(reversed(skips))
        for i, dec in enumerate(self.decoders):
            skip = skips[i] if i < len(skips) else torch.zeros_like(x)
            x = dec(x, skip)

        return x


class OccupancyHead(nn.Module):
    """Predict binary occupancy probability."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        return self.conv(x)


class SemanticHead(nn.Module):
    """Predict semantic class for each cell."""

    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, 1),
        )

    def forward(self, x):
        return self.conv(x)


class PillarOccNet(nn.Module):
    """Neural radar occupancy prediction from single frame."""

    def __init__(self, config):
        super().__init__()
        pillar_cfg = config["model"]["pillar"]
        backbone_cfg = config["model"]["backbone"]
        heads_cfg = config["model"]["heads"]
        grid_cfg = config["grid"]

        self.grid_size = grid_cfg["grid_size"]

        self.pillar_net = PillarFeatureNet(
            in_channels=pillar_cfg["input_features"],
            out_channels=pillar_cfg["pillar_features"],
            max_points_per_pillar=pillar_cfg["max_points_per_pillar"],
        )

        self.scatter = ScatterBEV(
            grid_size=self.grid_size,
            channels=pillar_cfg["pillar_features"],
        )

        self.backbone = UNetBackbone(
            in_channels=pillar_cfg["pillar_features"],
            encoder_channels=backbone_cfg["encoder_channels"],
            decoder_channels=backbone_cfg["decoder_channels"],
        )

        out_ch = self.backbone.out_channels
        self.occ_head = OccupancyHead(out_ch)

        if heads_cfg["semantics"]["enabled"]:
            self.sem_head = SemanticHead(out_ch, heads_cfg["semantics"]["num_classes"])
        else:
            self.sem_head = None

    def forward(self, pillar_features, pillar_indices, num_pillars):
        """
        Args:
            pillar_features: (B, max_pillars, max_points, 9)
            pillar_indices: (B, max_pillars, 2)
            num_pillars: (B,)

        Returns:
            dict with 'occupancy' (B, 1, H, W) and optionally 'semantics' (B, K, H, W)
        """
        pillar_enc = self.pillar_net(pillar_features, pillar_indices, num_pillars)
        bev = self.scatter(pillar_enc, pillar_indices, num_pillars)
        features = self.backbone(bev)

        outputs = {"occupancy": self.occ_head(features)}
        if self.sem_head is not None:
            outputs["semantics"] = self.sem_head(features)

        return outputs


class TemporalPillarOccNet(nn.Module):
    """Multi-frame temporal radar occupancy prediction.

    Fuses BEV features from current + past T frames using ego-motion compensation.
    """

    def __init__(self, config):
        super().__init__()
        pillar_cfg = config["model"]["pillar"]
        backbone_cfg = config["model"]["backbone"]
        heads_cfg = config["model"]["heads"]
        temporal_cfg = config["model"]["temporal"]
        grid_cfg = config["grid"]

        self.grid_size = grid_cfg["grid_size"]
        self.cell_size = grid_cfg["cell_size"]
        self.x_range = grid_cfg["x_range"]
        self.y_range = grid_cfg["y_range"]
        self.num_frames = temporal_cfg["num_frames"]
        self.fusion_method = temporal_cfg["fusion_method"]

        self.pillar_net = PillarFeatureNet(
            in_channels=pillar_cfg["input_features"],
            out_channels=pillar_cfg["pillar_features"],
            max_points_per_pillar=pillar_cfg["max_points_per_pillar"],
        )

        self.scatter = ScatterBEV(
            grid_size=self.grid_size,
            channels=pillar_cfg["pillar_features"],
        )

        feat_ch = pillar_cfg["pillar_features"]

        if self.fusion_method == "concat_conv":
            self.temporal_conv = nn.Sequential(
                nn.Conv2d(feat_ch * self.num_frames, temporal_cfg["temporal_conv_channels"],
                         3, padding=1, bias=False),
                nn.BatchNorm2d(temporal_cfg["temporal_conv_channels"]),
                nn.ReLU(inplace=True),
            )
            backbone_in = temporal_cfg["temporal_conv_channels"]
        elif self.fusion_method == "attention":
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=feat_ch, num_heads=4, batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(feat_ch)
            backbone_in = feat_ch
        elif self.fusion_method == "gru":
            self.temporal_gru = nn.GRU(
                input_size=feat_ch, hidden_size=feat_ch, batch_first=True
            )
            backbone_in = feat_ch
        else:
            backbone_in = feat_ch

        self.backbone = UNetBackbone(
            in_channels=backbone_in,
            encoder_channels=backbone_cfg["encoder_channels"],
            decoder_channels=backbone_cfg["decoder_channels"],
        )

        out_ch = self.backbone.out_channels
        self.occ_head = OccupancyHead(out_ch)

        if heads_cfg["semantics"]["enabled"]:
            self.sem_head = SemanticHead(out_ch, heads_cfg["semantics"]["num_classes"])
        else:
            self.sem_head = None

    def warp_bev(self, bev_features, ego_transform):
        """Warp past BEV features to current ego frame.

        Args:
            bev_features: (B, C, H, W) past frame BEV features
            ego_transform: (B, 4, 4) transform from past to current frame

        Returns:
            warped: (B, C, H, W) aligned BEV features
        """
        B, C, H, W = bev_features.shape

        ys = torch.linspace(self.y_range[0], self.y_range[1], H,
                           device=bev_features.device)
        xs = torch.linspace(self.x_range[0], self.x_range[1], W,
                           device=bev_features.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        ones = torch.ones_like(grid_x)
        zeros = torch.zeros_like(grid_x)
        coords = torch.stack([grid_x, grid_y, zeros, ones], dim=-1)  # (H, W, 4)
        coords = coords.reshape(-1, 4).T  # (4, H*W)

        warped_list = []
        for b in range(B):
            T = ego_transform[b]  # (4, 4)
            T_inv = torch.inverse(T)

            past_coords = T_inv @ coords  # (4, H*W)
            past_x = past_coords[0]
            past_y = past_coords[1]

            norm_x = 2.0 * (past_x - self.x_range[0]) / \
                     (self.x_range[1] - self.x_range[0]) - 1.0
            norm_y = 2.0 * (past_y - self.y_range[0]) / \
                     (self.y_range[1] - self.y_range[0]) - 1.0

            sample_grid = torch.stack([norm_x, norm_y], dim=-1)
            sample_grid = sample_grid.reshape(1, H, W, 2)

            warped = F.grid_sample(
                bev_features[b:b+1], sample_grid,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            warped_list.append(warped)

        return torch.cat(warped_list, dim=0)

    def forward(self, pillar_features_seq, pillar_indices_seq, num_pillars_seq,
                ego_transforms):
        """
        Args:
            pillar_features_seq: list of T tensors, each (B, max_pillars, max_points, C)
            pillar_indices_seq: list of T tensors, each (B, max_pillars, 2)
            num_pillars_seq: list of T tensors, each (B,)
            ego_transforms: (B, T-1, 4, 4) transforms from past frames to current

        Returns:
            dict with 'occupancy' and optionally 'semantics'
        """
        bev_features_list = []

        for t in range(self.num_frames):
            pillar_enc = self.pillar_net(
                pillar_features_seq[t], pillar_indices_seq[t], num_pillars_seq[t]
            )
            bev = self.scatter(pillar_enc, pillar_indices_seq[t], num_pillars_seq[t])

            if t < self.num_frames - 1:
                ego_t = ego_transforms[:, t]  # (B, 4, 4)
                bev = self.warp_bev(bev, ego_t)

            bev_features_list.append(bev)

        if self.fusion_method == "concat_conv":
            fused = torch.cat(bev_features_list, dim=1)  # (B, C*T, H, W)
            fused = self.temporal_conv(fused)
        elif self.fusion_method == "attention":
            B, C, H, W = bev_features_list[0].shape
            stacked = torch.stack(bev_features_list, dim=2)  # (B, C, T, H, W)
            stacked = stacked.permute(0, 3, 4, 2, 1).reshape(B * H * W, self.num_frames, C)

            current = stacked[:, -1:, :]
            attn_out, _ = self.temporal_attn(current, stacked, stacked)
            attn_out = self.temporal_norm(attn_out + current)

            fused = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif self.fusion_method == "gru":
            B, C, H, W = bev_features_list[0].shape
            stacked = torch.stack(bev_features_list, dim=2)  # (B, C, T, H, W)
            stacked = stacked.permute(0, 3, 4, 2, 1).reshape(B * H * W, self.num_frames, C)

            output, _ = self.temporal_gru(stacked)
            fused = output[:, -1, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            fused = bev_features_list[-1]

        features = self.backbone(fused)

        outputs = {"occupancy": self.occ_head(features)}
        if self.sem_head is not None:
            outputs["semantics"] = self.sem_head(features)

        return outputs


def build_model(config):
    """Factory function to build the appropriate model."""
    model_type = config["model"]["type"]

    if model_type == "classical_ism":
        return ClassicalISM(config)
    elif model_type == "pillar_occ_net":
        return PillarOccNet(config)
    elif model_type == "temporal_pillar_occ_net":
        return TemporalPillarOccNet(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
