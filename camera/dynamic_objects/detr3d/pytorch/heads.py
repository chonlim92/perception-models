"""
DETR3D Detection Heads: Classification and 3D bounding box regression.

Produces per-query class predictions and 3D bounding box parameters
(center_x, center_y, center_z, w, l, h, sin(yaw), cos(yaw), vx, vy).
Supports auxiliary losses from intermediate decoder layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import copy


class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ):
        """
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.
            num_layers: Total number of linear layers (including output).
        """
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


class DETR3DClassificationHead(nn.Module):
    """Classification head producing per-query class logits."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_classes: int = 10,
        num_layers: int = 2,
        hidden_dim: int = 256,
    ):
        """
        Args:
            embed_dims: Input query feature dimension.
            num_classes: Number of object classes (excluding background).
            num_layers: Number of layers in the MLP.
            hidden_dim: Hidden dimension of the MLP.
        """
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = MLP(
            input_dim=embed_dims,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
        )
        # Initialize final layer bias for focal loss (prior probability)
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_head.layers[-1].bias, bias_value.item())

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_features: Decoder output, shape (B, N, embed_dims).

        Returns:
            Class logits, shape (B, N, num_classes).
        """
        return self.cls_head(query_features)


class DETR3DRegressionHead(nn.Module):
    """Regression head producing 3D bounding box parameters.

    Predicts 10 values per query:
    - center_x, center_y, center_z: 3D center (sigmoid-normalized to detection range)
    - w, l, h: width, length, height (log-scale)
    - sin(yaw), cos(yaw): orientation
    - vx, vy: velocity
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_layers: int = 2,
        hidden_dim: int = 256,
        num_output: int = 10,
    ):
        """
        Args:
            embed_dims: Input query feature dimension.
            num_layers: Number of layers in the MLP.
            hidden_dim: Hidden dimension.
            num_output: Number of regression targets (default 10).
        """
        super().__init__()
        self.num_output = num_output
        self.reg_head = MLP(
            input_dim=embed_dims,
            hidden_dim=hidden_dim,
            output_dim=num_output,
            num_layers=num_layers,
        )

    def forward(
        self,
        query_features: torch.Tensor,
        reference_points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query_features: Decoder output, shape (B, N, embed_dims).
            reference_points: Normalized reference points from decoder,
                              shape (B, N, 3). If provided, center predictions
                              are added as refinements to reference points.

        Returns:
            Bounding box predictions, shape (B, N, 10).
            Format: [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
            - cx, cy, cz are in [0, 1] (normalized to detection range)
            - w, l, h are positive (exp applied)
            - sin(yaw), cos(yaw) are unconstrained
            - vx, vy are unconstrained
        """
        raw_pred = self.reg_head(query_features)  # (B, N, 10)

        # Split predictions
        center_pred = raw_pred[..., :3]  # cx, cy, cz
        size_pred = raw_pred[..., 3:6]   # w, l, h
        rot_pred = raw_pred[..., 6:8]    # sin(yaw), cos(yaw)
        vel_pred = raw_pred[..., 8:10]   # vx, vy

        # Apply sigmoid to center for normalization to [0, 1]
        if reference_points is not None:
            # Refine reference points: sigmoid(inverse_sigmoid(ref) + delta)
            ref_clamped = reference_points.clamp(1e-5, 1 - 1e-5)
            inv_sigmoid_ref = torch.log(ref_clamped / (1 - ref_clamped))
            center = torch.sigmoid(inv_sigmoid_ref + center_pred)
        else:
            center = torch.sigmoid(center_pred)

        # Apply exp to size predictions for positivity
        size = size_pred.exp()

        # Concatenate all predictions
        bbox_pred = torch.cat([center, size, rot_pred, vel_pred], dim=-1)

        return bbox_pred


class DETR3DHead(nn.Module):
    """Combined DETR3D detection head with classification and regression.

    Supports auxiliary losses from intermediate decoder layers by applying
    shared heads to each intermediate output.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_classes: int = 10,
        num_reg_layers: int = 2,
        num_cls_layers: int = 2,
        hidden_dim: int = 256,
        num_decoder_layers: int = 6,
        share_heads: bool = True,
    ):
        """
        Args:
            embed_dims: Query feature dimension from decoder.
            num_classes: Number of object classes.
            num_reg_layers: Number of MLP layers for regression.
            num_cls_layers: Number of MLP layers for classification.
            hidden_dim: Hidden dimension for head MLPs.
            num_decoder_layers: Number of decoder layers (for auxiliary heads).
            share_heads: If True, share head weights across all decoder layers.
                         If False, use separate heads per layer.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers
        self.share_heads = share_heads

        # Create classification and regression heads
        cls_head = DETR3DClassificationHead(
            embed_dims=embed_dims,
            num_classes=num_classes,
            num_layers=num_cls_layers,
            hidden_dim=hidden_dim,
        )
        reg_head = DETR3DRegressionHead(
            embed_dims=embed_dims,
            num_layers=num_reg_layers,
            hidden_dim=hidden_dim,
        )

        if share_heads:
            # Share weights across all decoder layers
            self.cls_heads = nn.ModuleList([cls_head] * num_decoder_layers)
            self.reg_heads = nn.ModuleList([reg_head] * num_decoder_layers)
        else:
            # Independent heads per layer
            self.cls_heads = nn.ModuleList([
                copy.deepcopy(cls_head) for _ in range(num_decoder_layers)
            ])
            self.reg_heads = nn.ModuleList([
                copy.deepcopy(reg_head) for _ in range(num_decoder_layers)
            ])

    def forward(
        self,
        intermediate_outputs: List[torch.Tensor],
        intermediate_ref_points: List[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            intermediate_outputs: List of decoder layer outputs,
                                  each (B, N, embed_dims).
            intermediate_ref_points: List of reference points from each layer,
                                     each (B, N, 3) in normalized [0, 1].

        Returns:
            Dictionary with:
                'cls_scores': List of classification logits per layer,
                              each (B, N, num_classes).
                'bbox_preds': List of bbox predictions per layer,
                              each (B, N, 10).
        """
        cls_scores = []
        bbox_preds = []

        for layer_idx, (output, ref_points) in enumerate(
            zip(intermediate_outputs, intermediate_ref_points)
        ):
            cls_score = self.cls_heads[layer_idx](output)
            bbox_pred = self.reg_heads[layer_idx](output, ref_points)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
        }

    def forward_single(
        self,
        query_features: torch.Tensor,
        reference_points: torch.Tensor,
        layer_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward for a single decoder layer output.

        Args:
            query_features: (B, N, embed_dims).
            reference_points: (B, N, 3) normalized.
            layer_idx: Which layer's head to use (default -1 = last).

        Returns:
            cls_scores: (B, N, num_classes).
            bbox_preds: (B, N, 10).
        """
        cls_score = self.cls_heads[layer_idx](query_features)
        bbox_pred = self.reg_heads[layer_idx](query_features, reference_points)
        return cls_score, bbox_pred
