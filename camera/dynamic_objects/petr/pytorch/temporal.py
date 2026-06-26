"""
Temporal modeling components for StreamPETR.

Implements query propagation across frames, ego-motion compensation,
motion-aware layer normalization, and temporal memory bank management
for streaming 3D object detection.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class EgoMotionCompensation(nn.Module):
    """Compensate for ego-vehicle motion between frames.

    Transforms 3D positions/features from a previous frame's coordinate
    system to the current frame's coordinate system using the relative
    ego-motion transformation matrix.

    Args:
        pc_range: Point cloud range (x_min, y_min, z_min, x_max, y_max, z_max)
            used for normalizing/denormalizing reference points.
    """

    def __init__(
        self,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        super().__init__()
        self.pc_range = pc_range

    def forward(
        self,
        reference_points: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> torch.Tensor:
        """Transform reference points from previous frame to current frame.

        Args:
            reference_points: Normalized 3D reference points (B, Q, 3) in [0,1].
            ego_motion: Transformation matrix from previous ego frame to
                current ego frame (B, 4, 4). This is T_cur_prev, such that
                p_cur = T_cur_prev @ p_prev.

        Returns:
            Transformed reference points in current frame coords (B, Q, 3),
            still normalized to [0, 1].
        """
        B, Q, _ = reference_points.shape
        device = reference_points.device

        # Denormalize reference points to world coordinates
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        mins = torch.tensor(
            [x_min, y_min, z_min], device=device, dtype=reference_points.dtype
        )
        maxs = torch.tensor(
            [x_max, y_max, z_max], device=device, dtype=reference_points.dtype
        )
        points_world = reference_points * (maxs - mins) + mins  # (B, Q, 3)

        # Convert to homogeneous coordinates
        ones = torch.ones(B, Q, 1, device=device, dtype=points_world.dtype)
        points_homo = torch.cat([points_world, ones], dim=-1)  # (B, Q, 4)

        # Apply ego-motion transformation
        # ego_motion: (B, 4, 4), points_homo: (B, Q, 4)
        points_transformed = torch.einsum(
            "bij,bqj->bqi", ego_motion, points_homo
        )  # (B, Q, 4)

        # Extract xyz and re-normalize
        points_xyz = points_transformed[..., :3]  # (B, Q, 3)
        points_norm = (points_xyz - mins) / (maxs - mins)
        points_norm = points_norm.clamp(0.0, 1.0)

        return points_norm


class MotionAwareLayerNorm(nn.Module):
    """Layer normalization conditioned on ego-motion.

    Modulates standard LayerNorm parameters (gamma, beta) using encoded
    ego-motion information (velocity and angular velocity). This allows
    the model to adapt its feature normalization based on how the
    ego-vehicle is moving.

    Args:
        embed_dims: Feature dimension to normalize.
        motion_dims: Dimension of the ego-motion encoding.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        motion_dims: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims

        # Standard LayerNorm (learnable base parameters)
        self.norm = nn.LayerNorm(embed_dims)

        # Motion encoder: encode 6-DoF motion (vx, vy, vz, wx, wy, wz)
        # into a latent representation
        self.motion_encoder = nn.Sequential(
            nn.Linear(6, motion_dims),
            nn.ReLU(inplace=True),
            nn.Linear(motion_dims, motion_dims),
            nn.ReLU(inplace=True),
        )

        # Modulation layers: produce scale (gamma) and shift (beta)
        # adjustments conditioned on motion
        self.gamma_fc = nn.Linear(motion_dims, embed_dims)
        self.beta_fc = nn.Linear(motion_dims, embed_dims)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize modulation layers to produce near-identity transform."""
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        ego_motion_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Apply motion-conditioned layer normalization.

        Args:
            x: Input features (B, Q, C) or (B, N, Q, C).
            ego_motion_vec: Ego-motion vector (B, 6) containing
                [vx, vy, vz, angular_vx, angular_vy, angular_vz].

        Returns:
            Normalized and motion-modulated features, same shape as input.
        """
        # Apply base layer normalization
        x_norm = self.norm(x)

        # Encode motion
        motion_feat = self.motion_encoder(ego_motion_vec)  # (B, motion_dims)

        # Generate modulation parameters
        gamma = self.gamma_fc(motion_feat)  # (B, embed_dims)
        beta = self.beta_fc(motion_feat)  # (B, embed_dims)

        # Expand to match input dimensions
        if x.dim() == 3:
            # (B, Q, C) case
            gamma = gamma.unsqueeze(1)  # (B, 1, C)
            beta = beta.unsqueeze(1)  # (B, 1, C)
        elif x.dim() == 4:
            # (B, N, Q, C) case
            gamma = gamma.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, C)
            beta = beta.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, C)

        # Apply modulation: y = gamma * x_norm + beta
        out = gamma * x_norm + beta

        return out


class TemporalMemory(nn.Module):
    """Memory bank for maintaining previous frame queries and positions.

    Stores object queries and their associated 3D reference points from
    previous frames, enabling temporal reasoning in StreamPETR.

    Args:
        embed_dims: Dimension of stored query embeddings.
        max_memory_length: Maximum number of previous frames to store.
        num_propagated_queries: Number of queries to propagate per frame.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        max_memory_length: int = 1,
        num_propagated_queries: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.max_memory_length = max_memory_length
        self.num_propagated_queries = num_propagated_queries

        # Score projection to select top-k queries to propagate
        self.score_proj = nn.Linear(embed_dims, 1)

        # Memory buffers (not parameters, managed manually)
        self._memory_queries: Optional[torch.Tensor] = None
        self._memory_reference_points: Optional[torch.Tensor] = None
        self._memory_scores: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear the temporal memory (e.g., at start of new sequence)."""
        self._memory_queries = None
        self._memory_reference_points = None
        self._memory_scores = None

    def has_memory(self) -> bool:
        """Check if memory contains any stored queries."""
        return self._memory_queries is not None

    @torch.no_grad()
    def update(
        self,
        queries: torch.Tensor,
        reference_points: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        """Store current frame's queries in memory for next frame.

        Args:
            queries: Object queries (B, Q, C) from current frame decoder output.
            reference_points: Associated 3D reference points (B, Q, 3).
            scores: Optional confidence scores (B, Q) for query selection.
        """
        # Compute scores if not provided
        if scores is None:
            scores = self.score_proj(queries.detach()).squeeze(-1)  # (B, Q)

        # Select top-k queries by score
        B, Q, C = queries.shape
        k = min(self.num_propagated_queries, Q)
        topk_scores, topk_indices = torch.topk(scores, k, dim=-1)  # (B, k)

        # Gather top-k queries and reference points
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        selected_queries = torch.gather(
            queries.detach(), 1, topk_indices_expanded
        )  # (B, k, C)

        topk_indices_3d = topk_indices.unsqueeze(-1).expand(-1, -1, 3)
        selected_ref_pts = torch.gather(
            reference_points.detach(), 1, topk_indices_3d
        )  # (B, k, 3)

        self._memory_queries = selected_queries
        self._memory_reference_points = selected_ref_pts
        self._memory_scores = topk_scores.detach()

    def get_memory(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve stored queries and reference points.

        Returns:
            Tuple of (queries, reference_points) or (None, None) if empty.
            queries: (B, num_propagated, C)
            reference_points: (B, num_propagated, 3)
        """
        return self._memory_queries, self._memory_reference_points


class QueryPropagation(nn.Module):
    """Propagate object queries from previous frames with ego-motion compensation.

    Core component of StreamPETR that enables temporal reasoning by
    propagating high-confidence queries across frames, compensating for
    ego-vehicle motion, and combining them with fresh learnable queries.

    Args:
        embed_dims: Dimension of query embeddings.
        num_learnable_queries: Number of fresh learnable queries per frame.
        num_propagated_queries: Number of queries propagated from previous frame.
        pc_range: Point cloud range for coordinate normalization.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_learnable_queries: int = 644,
        num_propagated_queries: int = 256,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_learnable_queries = num_learnable_queries
        self.num_propagated_queries = num_propagated_queries

        # Ego-motion compensation
        self.ego_motion_comp = EgoMotionCompensation(pc_range=pc_range)

        # Temporal memory
        self.memory = TemporalMemory(
            embed_dims=embed_dims,
            num_propagated_queries=num_propagated_queries,
        )

        # Motion-aware layer norm for propagated queries
        self.motion_ln = MotionAwareLayerNorm(embed_dims=embed_dims)

        # Projection to align propagated queries with current frame
        self.propagation_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        # Learnable initial queries (for first frame or new detections)
        self.learnable_queries = nn.Embedding(num_learnable_queries, embed_dims)
        self.learnable_reference_points = nn.Embedding(num_learnable_queries, 3)

        # Initialize reference points to cover the perception range
        nn.init.uniform_(self.learnable_reference_points.weight, 0.0, 1.0)

    def forward(
        self,
        ego_motion: Optional[torch.Tensor] = None,
        ego_motion_vec: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate queries for current frame by combining propagated and learnable queries.

        Args:
            ego_motion: Ego-motion matrix from prev to current frame (B, 4, 4).
                None for first frame.
            ego_motion_vec: 6-DoF ego-motion vector (B, 6) for motion-aware LN.
                None for first frame.
            batch_size: Batch size (needed for first frame).

        Returns:
            Tuple of:
                - Combined queries (B, Q_total, C) where Q_total =
                  num_propagated + num_learnable (or just num_learnable for first frame).
                - Combined reference points (B, Q_total, 3).
        """
        device = self.learnable_queries.weight.device

        # Get learnable queries (always included)
        learn_queries = self.learnable_queries.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, Q_learn, C)
        learn_ref_pts = self.learnable_reference_points.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, Q_learn, 3)
        learn_ref_pts = learn_ref_pts.sigmoid()  # Normalize to [0, 1]

        # Check if we have memory from previous frame
        if not self.memory.has_memory() or ego_motion is None:
            # First frame: only use learnable queries
            return learn_queries, learn_ref_pts

        # Get stored queries from previous frame
        prev_queries, prev_ref_pts = self.memory.get_memory()
        if prev_queries is None:
            return learn_queries, learn_ref_pts

        # Move to current device if needed
        prev_queries = prev_queries.to(device)
        prev_ref_pts = prev_ref_pts.to(device)

        # Apply ego-motion compensation to reference points
        compensated_ref_pts = self.ego_motion_comp(
            prev_ref_pts, ego_motion
        )  # (B, Q_prop, 3)

        # Apply motion-aware layer norm to propagated queries
        if ego_motion_vec is not None:
            prop_queries = self.motion_ln(prev_queries, ego_motion_vec)
        else:
            prop_queries = prev_queries

        # Project propagated queries
        prop_queries = self.propagation_proj(prop_queries)  # (B, Q_prop, C)

        # Combine propagated and learnable queries
        combined_queries = torch.cat(
            [prop_queries, learn_queries], dim=1
        )  # (B, Q_prop + Q_learn, C)
        combined_ref_pts = torch.cat(
            [compensated_ref_pts, learn_ref_pts], dim=1
        )  # (B, Q_prop + Q_learn, 3)

        return combined_queries, combined_ref_pts

    def update_memory(
        self,
        queries: torch.Tensor,
        reference_points: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> None:
        """Update temporal memory with current frame results.

        Should be called after decoder produces output for current frame.

        Args:
            queries: Decoder output queries (B, Q, C).
            reference_points: Refined reference points (B, Q, 3).
            scores: Detection confidence scores (B, Q).
        """
        self.memory.update(queries, reference_points, scores)

    def reset_memory(self) -> None:
        """Reset temporal memory (call at start of new sequence)."""
        self.memory.reset()
