"""
Temporal Fusion Module for StreamMapNet.

This module implements the key innovation of StreamMapNet: streaming temporal BEV
(Bird's Eye View) feature fusion. Instead of re-computing BEV features for all
historical frames at each timestep, it maintains a streaming temporal buffer and
propagates information forward using ego-motion warping and temporal attention.

Pipeline at each timestep:
    1. Warp previous BEV features using ego-motion transformation matrices
    2. Fuse warped previous features with current BEV features via temporal attention
    3. Store fused features in temporal buffer for next frame

Reference:
    Yuan, T., et al. "StreamMapNet: Streaming Mapping Network for Vectorized Online
    HD Map Construction." WACV 2024.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EgoMotionWarper(nn.Module):
    """Warps BEV features from a previous frame to the current frame using ego-motion.

    Given the 4x4 ego-motion transformation matrix T_prev2curr (which transforms points
    from the previous ego frame to the current ego frame), this module:
      1. Creates a 2D coordinate grid on the BEV plane (z=0 in ego coordinates)
      2. Applies the inverse transformation (T_curr2prev) to find where each current
         BEV cell was located in the previous frame's coordinate system
      3. Uses bilinear grid_sample to warp the previous BEV features into alignment

    The BEV coordinate system:
      - x-axis: [-x_bound, +x_bound] meters (lateral, left-right)
      - y-axis: [-y_bound, +y_bound] meters (longitudinal, front-back)

    Args:
        bev_height: Number of grid cells along the y-axis (longitudinal).
        bev_width: Number of grid cells along the x-axis (lateral).
        x_bound: Half-range of BEV in x-direction in meters. Default: 30.0.
        y_bound: Half-range of BEV in y-direction in meters. Default: 15.0.
        align_corners: Whether to align corners in grid_sample. Default: False.
    """

    def __init__(
        self,
        bev_height: int,
        bev_width: int,
        x_bound: float = 30.0,
        y_bound: float = 15.0,
        align_corners: bool = False,
    ):
        super().__init__()
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.align_corners = align_corners

        # Pre-compute the BEV coordinate grid (in meters)
        # grid_x covers [-x_bound, x_bound], grid_y covers [-y_bound, y_bound]
        # Shape: (bev_height, bev_width) each
        xs = torch.linspace(-x_bound, x_bound, bev_width)
        ys = torch.linspace(-y_bound, y_bound, bev_height)
        # grid_y: (H, W), grid_x: (H, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # Homogeneous coordinates on the z=0 plane: (H, W, 4)
        # Each point is [x, y, 0, 1] in the current ego frame
        ones = torch.ones_like(grid_x)
        zeros = torch.zeros_like(grid_x)
        # Shape: (H*W, 4)
        bev_coords = torch.stack([grid_x, grid_y, zeros, ones], dim=-1)
        bev_coords = bev_coords.reshape(-1, 4)  # (H*W, 4)

        # Register as buffer (not a parameter, moves with device)
        self.register_buffer("bev_coords", bev_coords, persistent=False)

    def forward(
        self,
        prev_bev: Tensor,
        ego_motion: Tensor,
    ) -> Tensor:
        """Warp previous BEV features to align with the current frame.

        Args:
            prev_bev: Previous frame's BEV features.
                Shape: (B, C, H, W) where H=bev_height, W=bev_width.
            ego_motion: Transformation matrix from previous frame to current frame,
                i.e., T_prev2curr that satisfies: p_curr = T_prev2curr @ p_prev.
                Shape: (B, 4, 4).

        Returns:
            Warped BEV features aligned to the current frame.
            Shape: (B, C, H, W).
        """
        B = prev_bev.shape[0]
        device = prev_bev.device

        # We want to find, for each cell in the CURRENT BEV grid, where it
        # corresponds to in the PREVIOUS BEV grid. This requires T_curr2prev.
        # T_curr2prev = inv(T_prev2curr)
        ego_motion_inv = torch.inverse(ego_motion)  # (B, 4, 4)

        # Transform current BEV coordinates to previous frame
        # bev_coords: (H*W, 4) -> expand to (B, H*W, 4)
        coords = self.bev_coords.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 4)

        # Apply inverse ego-motion: p_prev = T_curr2prev @ p_curr
        # coords: (B, H*W, 4) -> transpose for matmul -> (B, 4, H*W)
        coords_t = coords.permute(0, 2, 1)  # (B, 4, H*W)
        prev_coords = torch.bmm(ego_motion_inv, coords_t)  # (B, 4, H*W)
        prev_coords = prev_coords.permute(0, 2, 1)  # (B, H*W, 4)

        # Extract x, y coordinates in previous frame (z and w are not needed)
        prev_x = prev_coords[:, :, 0]  # (B, H*W)
        prev_y = prev_coords[:, :, 1]  # (B, H*W)

        # Normalize to [-1, 1] for grid_sample
        # x ranges from -x_bound to x_bound -> normalize to [-1, 1]
        # y ranges from -y_bound to y_bound -> normalize to [-1, 1]
        norm_x = prev_x / self.x_bound  # (B, H*W)
        norm_y = prev_y / self.y_bound  # (B, H*W)

        # Reshape to (B, H, W, 2) for grid_sample
        # grid_sample expects grid in (B, H_out, W_out, 2) with (x, y) in [-1, 1]
        grid = torch.stack([norm_x, norm_y], dim=-1)  # (B, H*W, 2)
        grid = grid.reshape(B, self.bev_height, self.bev_width, 2)  # (B, H, W, 2)

        # Warp using bilinear interpolation; zero-pad for out-of-bounds regions
        warped = F.grid_sample(
            prev_bev,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )  # (B, C, H, W)

        return warped


class TemporalAttention(nn.Module):
    """Multi-head cross-attention for fusing current and warped previous BEV features.

    Performs cross-attention where:
      - Query: current BEV features (what information does the current frame need?)
      - Key/Value: warped previous BEV features (what can the past provide?)

    Optionally includes a self-attention pathway and a gating mechanism to control
    how much temporal information to incorporate.

    Args:
        embed_dim: Dimension of BEV feature channels (C).
        num_heads: Number of attention heads. Default: 8.
        dropout: Dropout rate for attention weights. Default: 0.1.
        use_gate: Whether to use a learned gating mechanism. Default: True.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gate: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_gate = use_gate

        # Cross-attention: current queries, previous keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms for pre-norm attention
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_key = nn.LayerNorm(embed_dim)

        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # Gating mechanism: learn to blend current and temporal features
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )

    def forward(
        self,
        current_bev: Tensor,
        warped_prev_bev: Tensor,
    ) -> Tensor:
        """Fuse current and warped previous BEV features via temporal attention.

        Args:
            current_bev: Current frame BEV features. Shape: (B, C, H, W).
            warped_prev_bev: Warped previous BEV features. Shape: (B, C, H, W).

        Returns:
            Fused BEV features. Shape: (B, C, H, W).
        """
        B, C, H, W = current_bev.shape

        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        current_flat = current_bev.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        prev_flat = warped_prev_bev.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Pre-norm
        query = self.norm_query(current_flat)  # (B, H*W, C)
        key_value = self.norm_key(prev_flat)  # (B, H*W, C)

        # Cross-attention: query from current, key/value from warped previous
        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
        )  # (B, H*W, C)

        # Residual connection
        fused = current_flat + attn_out  # (B, H*W, C)

        # Feed-forward with residual
        fused = fused + self.ffn(self.norm_ffn(fused))  # (B, H*W, C)

        # Gating: learn to blend temporal information with current
        if self.use_gate:
            gate_input = torch.cat([current_flat, fused], dim=-1)  # (B, H*W, 2C)
            gate_weight = self.gate(gate_input)  # (B, H*W, C) in [0, 1]
            fused = gate_weight * fused + (1.0 - gate_weight) * current_flat

        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        fused = fused.permute(0, 2, 1).reshape(B, C, H, W)

        return fused


class TemporalFusion(nn.Module):
    """Streaming Temporal BEV Fusion module for StreamMapNet.

    Maintains a temporal buffer of previous BEV features and performs ego-motion
    warping + temporal attention fusion at each timestep. This is the core innovation
    that enables StreamMapNet to leverage temporal information without re-encoding
    previous frames.

    The streaming pipeline at each timestep:
        1. Retrieve previous BEV state from temporal buffer
        2. Warp each previous state to the current frame using ego-motion matrices
        3. Fuse warped previous features with current features via temporal attention
        4. Store the fused output back into the temporal buffer for future use

    Multi-frame support:
        When temporal_window > 1, multiple previous frames are maintained. Each is
        independently warped to the current frame, then fused sequentially (from oldest
        to newest) with the current BEV features using shared or separate attention layers.

    Args:
        embed_dim: Dimension of BEV feature channels.
        bev_height: Number of grid cells along the y-axis (longitudinal).
        bev_width: Number of grid cells along the x-axis (lateral).
        x_bound: Half-range of BEV in x-direction in meters. Default: 30.0.
        y_bound: Half-range of BEV in y-direction in meters. Default: 15.0.
        temporal_window: Number of previous frames to maintain. Default: 1.
        num_heads: Number of attention heads. Default: 8.
        dropout: Dropout rate. Default: 0.1.
        use_gate: Whether to use gating in temporal attention. Default: True.
        share_attention: Whether all temporal steps share the same attention layer.
            Default: True.

    Example:
        >>> temporal_fusion = TemporalFusion(
        ...     embed_dim=256, bev_height=200, bev_width=400,
        ...     x_bound=30.0, y_bound=15.0, temporal_window=3,
        ... )
        >>> # First frame (no previous state)
        >>> current_bev = torch.randn(2, 256, 200, 400)
        >>> ego_matrices = torch.eye(4).unsqueeze(0).expand(2, -1, -1).unsqueeze(1).expand(-1, 3, -1, -1)
        >>> fused, state = temporal_fusion(current_bev, ego_matrices)
        >>> # Subsequent frames
        >>> fused, state = temporal_fusion(next_bev, ego_matrices, prev_bev_state=state)
    """

    def __init__(
        self,
        embed_dim: int,
        bev_height: int,
        bev_width: int,
        x_bound: float = 30.0,
        y_bound: float = 15.0,
        temporal_window: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gate: bool = True,
        share_attention: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.temporal_window = temporal_window
        self.share_attention = share_attention

        # Ego-motion warping module
        self.warper = EgoMotionWarper(
            bev_height=bev_height,
            bev_width=bev_width,
            x_bound=x_bound,
            y_bound=y_bound,
        )

        # Temporal attention layers
        if share_attention:
            self.temporal_attn = TemporalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_gate=use_gate,
            )
        else:
            # Separate attention layer for each temporal step
            self.temporal_attns = nn.ModuleList([
                TemporalAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_gate=use_gate,
                )
                for _ in range(temporal_window)
            ])

        # Projection layer to combine multi-frame features when temporal_window > 1
        if temporal_window > 1:
            self.temporal_proj = nn.Sequential(
                nn.Conv2d(embed_dim * (temporal_window + 1), embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def _get_attention_layer(self, index: int) -> TemporalAttention:
        """Get the temporal attention layer for a given temporal index.

        Args:
            index: Temporal index (0 = oldest frame in buffer).

        Returns:
            The attention module to use.
        """
        if self.share_attention:
            return self.temporal_attn
        return self.temporal_attns[index]

    def forward(
        self,
        current_bev: Tensor,
        ego_motion_matrices: Tensor,
        prev_bev_state: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Perform streaming temporal BEV fusion.

        Args:
            current_bev: Current frame's BEV features.
                Shape: (B, C, H, W).
            ego_motion_matrices: Ego-motion transformation matrices from each previous
                frame to the current frame. T_prev_i_to_curr for each frame i in the
                temporal window.
                Shape: (B, temporal_window, 4, 4).
                ego_motion_matrices[:, i] transforms points from frame (t - temporal_window + i)
                to the current frame t.
            prev_bev_state: List of previous BEV feature tensors from the temporal buffer.
                Each element has shape (B, C, H, W). Length <= temporal_window.
                None or empty list for the first frame.

        Returns:
            Tuple of:
                - fused_bev: Fused BEV features. Shape: (B, C, H, W).
                - new_state: Updated temporal buffer (list of BEV tensors to pass to
                  next timestep). Length = min(len(prev_state) + 1, temporal_window).
        """
        B, C, H, W = current_bev.shape

        # Handle first frame: no previous state available
        if prev_bev_state is None or len(prev_bev_state) == 0:
            fused_bev = self.output_proj(current_bev)
            # Initialize state buffer with the current (unfused) features
            new_state = [current_bev.detach()]
            return fused_bev, new_state

        num_prev_frames = len(prev_bev_state)

        # Single previous frame: simple warp + fuse
        if self.temporal_window == 1 or num_prev_frames == 1:
            # Use the most recent ego-motion matrix
            # ego_motion_matrices[:, -1] is T from most recent prev frame to current
            ego_matrix = ego_motion_matrices[:, -num_prev_frames]  # (B, 4, 4)
            warped = self.warper(prev_bev_state[-1], ego_matrix)  # (B, C, H, W)

            attn_layer = self._get_attention_layer(0)
            fused_bev = attn_layer(current_bev, warped)  # (B, C, H, W)
            fused_bev = self.output_proj(fused_bev)

        else:
            # Multi-frame fusion: warp and fuse each previous frame
            warped_features = []

            for i, prev_feat in enumerate(prev_bev_state):
                # Index into ego_motion_matrices:
                # prev_bev_state[0] is the oldest, prev_bev_state[-1] is most recent
                # ego_motion_matrices[:, i] corresponds to the transform for frame i
                matrix_idx = self.temporal_window - num_prev_frames + i
                ego_matrix = ego_motion_matrices[:, matrix_idx]  # (B, 4, 4)

                warped = self.warper(prev_feat, ego_matrix)  # (B, C, H, W)
                warped_features.append(warped)

            # Sequential attention fusion: fuse from oldest to newest
            fused_bev = current_bev
            for i, warped in enumerate(warped_features):
                attn_layer = self._get_attention_layer(
                    min(i, self.temporal_window - 1)
                )
                fused_bev = attn_layer(fused_bev, warped)

            # Additionally, combine via channel concatenation + projection
            # for richer multi-frame aggregation
            all_features = warped_features + [current_bev]
            # Pad to temporal_window + 1 channels if fewer frames available
            while len(all_features) < self.temporal_window + 1:
                all_features.insert(0, torch.zeros_like(current_bev))

            concat_features = torch.cat(all_features, dim=1)  # (B, C*(T+1), H, W)
            projected = self.temporal_proj(concat_features)  # (B, C, H, W)

            # Residual blend between attention-fused and projection-fused
            fused_bev = fused_bev + projected
            fused_bev = self.output_proj(fused_bev)

        # Update temporal buffer: append current fused features, maintain window size
        new_state = list(prev_bev_state) + [fused_bev.detach()]
        if len(new_state) > self.temporal_window:
            new_state = new_state[-self.temporal_window:]

        return fused_bev, new_state

    @torch.no_grad()
    def reset_state(self) -> List[Tensor]:
        """Reset the temporal buffer (e.g., at the start of a new sequence).

        Returns:
            Empty list representing a cleared temporal buffer.
        """
        return []

    def propagate_sequence(
        self,
        bev_sequence: Tensor,
        ego_motion_sequence: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Process an entire sequence of BEV features with streaming temporal fusion.

        Convenience method for processing a full temporal sequence. Iterates through
        each frame, maintaining the streaming state internally.

        Args:
            bev_sequence: Sequence of BEV features.
                Shape: (B, T, C, H, W) where T is the sequence length.
            ego_motion_sequence: Ego-motion matrices for the full sequence.
                Shape: (B, T, temporal_window, 4, 4).
                ego_motion_sequence[:, t] contains the T_prev_to_t matrices for timestep t.

        Returns:
            Tuple of:
                - fused_sequence: Fused BEV features for all timesteps.
                  Shape: (B, T, C, H, W).
                - final_state: Temporal buffer state after processing the last frame.
        """
        B, T, C, H, W = bev_sequence.shape
        fused_outputs = []
        state: Optional[List[Tensor]] = None

        for t in range(T):
            current_bev = bev_sequence[:, t]  # (B, C, H, W)
            ego_matrices = ego_motion_sequence[:, t]  # (B, temporal_window, 4, 4)

            fused_bev, state = self.forward(current_bev, ego_matrices, state)
            fused_outputs.append(fused_bev)

        # Stack along time dimension
        fused_sequence = torch.stack(fused_outputs, dim=1)  # (B, T, C, H, W)
        return fused_sequence, state


def build_temporal_fusion(
    embed_dim: int = 256,
    bev_height: int = 200,
    bev_width: int = 400,
    x_bound: float = 30.0,
    y_bound: float = 15.0,
    temporal_window: int = 1,
    num_heads: int = 8,
    dropout: float = 0.1,
    use_gate: bool = True,
    share_attention: bool = True,
) -> TemporalFusion:
    """Factory function to construct a TemporalFusion module.

    Args:
        embed_dim: BEV feature channel dimension.
        bev_height: BEV grid height (y-axis cells).
        bev_width: BEV grid width (x-axis cells).
        x_bound: BEV x-range in meters (half-width).
        y_bound: BEV y-range in meters (half-height).
        temporal_window: Number of previous frames to retain.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        use_gate: Enable gating mechanism in attention.
        share_attention: Share attention weights across temporal steps.

    Returns:
        Configured TemporalFusion module.
    """
    return TemporalFusion(
        embed_dim=embed_dim,
        bev_height=bev_height,
        bev_width=bev_width,
        x_bound=x_bound,
        y_bound=y_bound,
        temporal_window=temporal_window,
        num_heads=num_heads,
        dropout=dropout,
        use_gate=use_gate,
        share_attention=share_attention,
    )


if __name__ == "__main__":
    # Demonstration / smoke test
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    batch_size = 2
    embed_dim = 64  # Small for demo; production uses 256
    bev_h, bev_w = 50, 100  # Small for demo; production uses 200x400
    x_bound, y_bound = 30.0, 15.0
    temporal_window = 3
    num_frames = 5

    print("=" * 70)
    print("StreamMapNet Temporal Fusion Module - Smoke Test")
    print("=" * 70)
    print(f"  Batch size: {batch_size}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  BEV size: {bev_h} x {bev_w}")
    print(f"  BEV range: x=[{-x_bound}, {x_bound}], y=[{-y_bound}, {y_bound}] m")
    print(f"  Temporal window: {temporal_window}")
    print(f"  Num frames: {num_frames}")
    print(f"  Device: {device}")
    print()

    # Build module
    temporal_fusion = build_temporal_fusion(
        embed_dim=embed_dim,
        bev_height=bev_h,
        bev_width=bev_w,
        x_bound=x_bound,
        y_bound=y_bound,
        temporal_window=temporal_window,
        num_heads=4,
    ).to(device)

    num_params = sum(p.numel() for p in temporal_fusion.parameters())
    print(f"  Total parameters: {num_params:,}")
    print()

    # Simulate a driving sequence
    state = None
    for t in range(num_frames):
        # Simulate current BEV features (from backbone)
        current_bev = torch.randn(batch_size, embed_dim, bev_h, bev_w, device=device)

        # Simulate ego-motion: small translation + rotation per frame
        ego_matrices = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
        ego_matrices = ego_matrices.expand(batch_size, temporal_window, -1, -1).clone()

        # Add a small forward translation (0.5m per frame along y)
        for i in range(temporal_window):
            frames_back = temporal_window - i
            ego_matrices[:, i, 1, 3] = 0.5 * frames_back  # y-translation

            # Add a small rotation (~1 degree per frame)
            angle = 0.017 * frames_back  # radians
            cos_a = torch.cos(torch.tensor(angle))
            sin_a = torch.sin(torch.tensor(angle))
            ego_matrices[:, i, 0, 0] = cos_a
            ego_matrices[:, i, 0, 1] = -sin_a
            ego_matrices[:, i, 1, 0] = sin_a
            ego_matrices[:, i, 1, 1] = cos_a

        # Forward pass
        fused_bev, state = temporal_fusion(current_bev, ego_matrices, state)

        print(f"  Frame {t}: input={current_bev.shape}, "
              f"output={fused_bev.shape}, "
              f"state_len={len(state)}")

    print()
    print("  All frames processed successfully.")

    # Test sequence processing mode
    print()
    print("-" * 70)
    print("  Testing propagate_sequence()...")
    bev_seq = torch.randn(batch_size, num_frames, embed_dim, bev_h, bev_w, device=device)
    ego_seq = torch.eye(4, device=device).reshape(1, 1, 1, 4, 4)
    ego_seq = ego_seq.expand(batch_size, num_frames, temporal_window, -1, -1).clone()

    fused_seq, final_state = temporal_fusion.propagate_sequence(bev_seq, ego_seq)
    print(f"  Input sequence: {bev_seq.shape}")
    print(f"  Output sequence: {fused_seq.shape}")
    print(f"  Final state length: {len(final_state)}")
    print()
    print("  SUCCESS - Temporal fusion module is fully operational.")
    print("=" * 70)
