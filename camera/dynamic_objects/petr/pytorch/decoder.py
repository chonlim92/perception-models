"""
Transformer decoder for PETR.

Standard (non-deformable) transformer decoder with multi-head self-attention
among object queries, multi-head cross-attention from queries to position-aware
image features (global attention), FFN blocks, and support for iterative
bounding box refinement.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional key/value projections.

    Args:
        embed_dims: Total dimension of the model.
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        assert (
            self.head_dim * num_heads == embed_dims
        ), "embed_dims must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            query: Query tensor (B, Q, C).
            key: Key tensor (B, K, C).
            value: Value tensor (B, K, C).
            key_padding_mask: Bool mask (B, K) where True means ignore.
            attn_mask: Additive attention mask (Q, K) or (B*H, Q, K).

        Returns:
            Attention output (B, Q, C).
        """
        B, Q, _ = query.shape
        _, K, _ = key.shape
        H = self.num_heads

        # Project queries, keys, values
        q = self.q_proj(query).reshape(B, Q, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, K, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, K, H, self.head_dim).transpose(1, 2)
        # q, k, v: (B, H, Q/K, head_dim)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Q, K)

        # Apply attention mask (additive)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn = attn + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, K) -> (B, 1, 1, K)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, Q, head_dim)
        out = out.transpose(1, 2).reshape(B, Q, self.embed_dims)
        out = self.out_proj(out)

        return out


class FFN(nn.Module):
    """Feed-Forward Network with two linear layers and activation.

    Args:
        embed_dims: Input and output dimension.
        feedforward_dims: Hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, feedforward_dims)
        self.fc2 = nn.Linear(feedforward_dims, embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Linear -> ReLU -> Dropout -> Linear -> Dropout.

        Args:
            x: Input tensor (B, Q, C).

        Returns:
            Output tensor (B, Q, C).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Single layer of the PETR transformer decoder.

    Consists of:
    1. Multi-head self-attention among object queries
    2. Multi-head cross-attention: queries attend to ALL position-aware
       image features (global attention, not deformable/local)
    3. Feed-forward network
    Each with residual connection and layer normalization (pre-norm style).

    Args:
        embed_dims: Feature dimension.
        num_heads: Number of attention heads.
        feedforward_dims: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (global attention to all image features)
        self.cross_attn = MultiHeadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_dims=feedforward_dims,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through one decoder layer.

        Args:
            query: Object queries (B, Q, C).
            key: Image features / memory (B, K, C) where K = N_cams * D * H * W.
            value: Same as key (for standard attention).
            query_pos: Position embedding for queries (B, Q, C).
            key_pos: Position embedding for keys (B, K, C). If None, assumed
                already added to key.
            self_attn_mask: Mask for self-attention (Q, Q).
            cross_attn_mask: Mask for cross-attention (Q, K).
            key_padding_mask: Padding mask for keys (B, K).

        Returns:
            Updated query features (B, Q, C).
        """
        # --- Self-attention ---
        residual = query
        query_with_pos = query + query_pos
        query = self.self_attn(
            query=query_with_pos,
            key=query_with_pos,
            value=query,
            attn_mask=self_attn_mask,
        )
        query = residual + self.dropout1(query)
        query = self.norm1(query)

        # --- Cross-attention (global attention to all image features) ---
        residual = query
        query_with_pos = query + query_pos
        if key_pos is not None:
            key_with_pos = key + key_pos
        else:
            key_with_pos = key
        query = self.cross_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=cross_attn_mask,
        )
        query = residual + self.dropout2(query)
        query = self.norm2(query)

        # --- FFN ---
        residual = query
        query = self.ffn(query)
        query = residual + self.dropout3(query)
        query = self.norm3(query)

        return query


class PETRTransformerDecoder(nn.Module):
    """PETR Transformer Decoder with iterative bounding box refinement.

    Stack of TransformerDecoderLayers. Each layer's output produces
    intermediate predictions that are refined by subsequent layers.

    Args:
        num_layers: Number of decoder layers (default 6).
        embed_dims: Feature dimension.
        num_heads: Number of attention heads.
        feedforward_dims: FFN hidden dimension.
        dropout: Dropout probability.
        return_intermediate: Whether to return intermediate layer outputs
            (needed for auxiliary losses and iterative refinement).
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.return_intermediate = return_intermediate

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_dims=feedforward_dims,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dims)

        # Post-initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        reg_branches: Optional[nn.ModuleList] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through all decoder layers with iterative refinement.

        Args:
            query: Initial object queries (B, Q, C).
            key: Position-aware image features (B, K, C).
            value: Image features (B, K, C).
            query_pos: Position embedding for queries (B, Q, C).
            key_pos: Position embedding for keys (B, K, C).
            reference_points: Initial 3D reference points (B, Q, 3), normalized.
            reg_branches: Optional list of regression heads (one per layer)
                for iterative refinement. Each takes (B, Q, C) -> (B, Q, code_size).
            self_attn_mask: Mask for self-attention.
            key_padding_mask: Padding mask for cross-attention keys.

        Returns:
            Tuple of:
                - Final query output (B, Q, C).
                - List of intermediate query outputs (one per layer if
                  return_intermediate, else just the last).
                - List of reference points after each layer's refinement.
        """
        intermediate_outputs = []
        intermediate_ref_pts = []

        output = query

        for layer_idx, layer in enumerate(self.layers):
            # Update query position embedding based on current reference points
            # (The position embedding can be recomputed if reference points change)
            output = layer(
                query=output,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                self_attn_mask=self_attn_mask,
                key_padding_mask=key_padding_mask,
            )

            # Iterative refinement of reference points
            if reg_branches is not None and reference_points is not None:
                # Get regression prediction for this layer
                reg_output = reg_branches[layer_idx](output)  # (B, Q, code_size)
                # Update reference points with predicted offsets
                # Only use the first 3 values (cx, cy, cz offsets)
                new_reference_points = reference_points + reg_output[..., :3]
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate_outputs.append(self.final_norm(output))
                if reference_points is not None:
                    intermediate_ref_pts.append(reference_points)

        if not self.return_intermediate:
            output = self.final_norm(output)
            intermediate_outputs.append(output)
            if reference_points is not None:
                intermediate_ref_pts.append(reference_points)

        # Stack intermediate outputs: (num_layers, B, Q, C)
        intermediate_outputs_stacked = torch.stack(intermediate_outputs, dim=0)

        return output, intermediate_outputs, intermediate_ref_pts
