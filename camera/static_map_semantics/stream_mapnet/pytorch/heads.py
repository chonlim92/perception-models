"""StreamMapNet Prediction Heads for Map Element Detection.

This module implements the classification and regression heads that operate on
top of the transformer decoder output embeddings to predict map elements.

Heads:
    - Classification head: predicts element class + no_object
    - Point regression head: predicts K ordered polyline points per element
    - Direction head (optional): predicts polyline direction/orientation

The heads support iterative refinement across decoder layers, where each layer
progressively refines the point predictions from the previous layer.

Reference:
    Yuan et al., "StreamMapNet: Streaming Mapping Network for Vectorized Online
    HD Map Construction", WACV 2024.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activation and optional dropout.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        num_layers: Number of linear layers (minimum 2).
        dropout: Dropout rate applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 2, "MLP must have at least 2 layers"

        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Output tensor of shape (..., output_dim).
        """
        return self.net(x)


class ClassificationHead(nn.Module):
    """Classification head for map element type prediction.

    Predicts the class of each map element query including a no_object class
    for queries that do not correspond to any real map element.

    Classes:
        0: lane_divider
        1: road_boundary
        2: ped_crossing
        3: no_object (background)

    Args:
        d_model: Input embedding dimension from the decoder.
        num_classes: Number of foreground map element classes (default 3).
        hidden_dim: Hidden dimension of the classification MLP.
        num_layers: Number of layers in the classification MLP.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        # +1 for the no_object class
        self.cls_head = MLP(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=num_classes + 1,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict class logits for each query.

        Args:
            query_embeddings: Decoder output embeddings of shape
                (B, N_queries, d_model) or (num_layers, B, N_queries, d_model)
                for intermediate outputs.

        Returns:
            Class logits of shape (B, N_queries, num_classes+1) or
            (num_layers, B, N_queries, num_classes+1) for intermediate outputs.
            Raw logits (no softmax applied) suitable for cross-entropy loss.
        """
        return self.cls_head(query_embeddings)


class PointRegressionHead(nn.Module):
    """Point regression head for predicting ordered polyline points.

    Each map element is represented as an ordered sequence of K points in
    normalized BEV space [0, 1]. The head predicts 2D coordinates for each
    point using sigmoid activation to constrain outputs to valid range.

    Supports iterative refinement: each decoder layer refines point predictions
    from the previous layer by predicting residual offsets.

    Args:
        d_model: Input embedding dimension from the decoder.
        num_points: Number of ordered points per map element (K).
        hidden_dim: Hidden dimension of the regression MLP.
        num_layers: Number of layers in the regression MLP.
        dropout: Dropout rate.
        iterative_refinement: Whether to use iterative refinement across layers.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_points: int = 20,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        iterative_refinement: bool = True,
    ):
        super().__init__()
        self.num_points = num_points
        self.iterative_refinement = iterative_refinement

        # Predict K points x 2 coordinates
        self.reg_head = MLP(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=num_points * 2,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        prev_points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict point coordinates for each query.

        Args:
            query_embeddings: Decoder output embeddings of shape
                (B, N_queries, d_model).
            prev_points: Previous layer's point predictions of shape
                (B, N_queries, K, 2) for iterative refinement. If None,
                predicts absolute coordinates.

        Returns:
            Predicted points of shape (B, N_queries, K, 2) with coordinates
            in normalized BEV space [0, 1] via sigmoid activation.
        """
        B, N_q, _ = query_embeddings.shape

        # Raw prediction: (B, N_queries, K*2)
        raw_pred = self.reg_head(query_embeddings)

        # Reshape to (B, N_queries, K, 2)
        raw_pred = raw_pred.view(B, N_q, self.num_points, 2)

        if self.iterative_refinement and prev_points is not None:
            # Predict residual offsets and add to previous predictions
            # Use inverse sigmoid on prev_points to work in logit space
            prev_logits = self._inverse_sigmoid(prev_points)
            points = torch.sigmoid(prev_logits + raw_pred)
        else:
            # Direct prediction with sigmoid
            points = torch.sigmoid(raw_pred)

        return points

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute inverse sigmoid (logit) function.

        Args:
            x: Input tensor with values in (0, 1).
            eps: Small constant for numerical stability.

        Returns:
            Inverse sigmoid of x.
        """
        x = x.clamp(min=eps, max=1 - eps)
        return torch.log(x / (1 - x))


class DirectionHead(nn.Module):
    """Direction prediction head for polyline orientation.

    Predicts a direction vector (unit vector) indicating the primary
    orientation of each polyline map element. This helps distinguish
    the directionality of lane dividers and road boundaries.

    The output is a 2D unit vector represented as (cos(theta), sin(theta)).

    Args:
        d_model: Input embedding dimension from the decoder.
        hidden_dim: Hidden dimension of the direction MLP.
        num_layers: Number of layers in the direction MLP.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dir_head = MLP(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict direction vectors for each query.

        Args:
            query_embeddings: Decoder output embeddings of shape
                (B, N_queries, d_model) or
                (num_layers, B, N_queries, d_model).

        Returns:
            Normalized direction vectors of shape (B, N_queries, 2) or
            (num_layers, B, N_queries, 2). Each vector is L2-normalized
            to unit length.
        """
        raw_dir = self.dir_head(query_embeddings)  # (..., 2)
        # L2 normalize to get unit direction vector
        direction = F.normalize(raw_dir, p=2, dim=-1)
        return direction


class MapElementHeads(nn.Module):
    """Combined prediction heads for StreamMapNet map element detection.

    Wraps classification, point regression, and optional direction heads into
    a single module. Supports iterative refinement across transformer decoder
    layers where each layer progressively improves predictions.

    Args:
        d_model: Model dimension from the transformer decoder.
        num_classes: Number of foreground map element classes.
        num_points: Number of ordered points per map element (K).
        hidden_dim: Hidden dimension for MLP heads.
        cls_num_layers: Number of MLP layers in classification head.
        reg_num_layers: Number of MLP layers in regression head.
        dir_num_layers: Number of MLP layers in direction head.
        dropout: Dropout rate for MLP layers.
        iterative_refinement: Whether to refine points across decoder layers.
        use_direction_head: Whether to include the direction prediction head.
        share_head_weights: Whether to share head weights across decoder layers
            (if False, each layer gets its own set of heads).
        num_decoder_layers: Number of decoder layers (used when not sharing weights).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 3,
        num_points: int = 20,
        hidden_dim: int = 256,
        cls_num_layers: int = 2,
        reg_num_layers: int = 3,
        dir_num_layers: int = 2,
        dropout: float = 0.0,
        iterative_refinement: bool = True,
        use_direction_head: bool = True,
        share_head_weights: bool = True,
        num_decoder_layers: int = 6,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_points = num_points
        self.iterative_refinement = iterative_refinement
        self.use_direction_head = use_direction_head
        self.share_head_weights = share_head_weights
        self.num_decoder_layers = num_decoder_layers

        if share_head_weights:
            # Single set of heads shared across all decoder layers
            self.cls_head = ClassificationHead(
                d_model=d_model,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=cls_num_layers,
                dropout=dropout,
            )
            self.reg_head = PointRegressionHead(
                d_model=d_model,
                num_points=num_points,
                hidden_dim=hidden_dim,
                num_layers=reg_num_layers,
                dropout=dropout,
                iterative_refinement=iterative_refinement,
            )
            if use_direction_head:
                self.dir_head = DirectionHead(
                    d_model=d_model,
                    hidden_dim=hidden_dim // 2,
                    num_layers=dir_num_layers,
                    dropout=dropout,
                )
        else:
            # Separate heads for each decoder layer
            self.cls_heads = nn.ModuleList(
                [
                    ClassificationHead(
                        d_model=d_model,
                        num_classes=num_classes,
                        hidden_dim=hidden_dim,
                        num_layers=cls_num_layers,
                        dropout=dropout,
                    )
                    for _ in range(num_decoder_layers)
                ]
            )
            self.reg_heads = nn.ModuleList(
                [
                    PointRegressionHead(
                        d_model=d_model,
                        num_points=num_points,
                        hidden_dim=hidden_dim,
                        num_layers=reg_num_layers,
                        dropout=dropout,
                        iterative_refinement=iterative_refinement,
                    )
                    for _ in range(num_decoder_layers)
                ]
            )
            if use_direction_head:
                self.dir_heads = nn.ModuleList(
                    [
                        DirectionHead(
                            d_model=d_model,
                            hidden_dim=hidden_dim // 2,
                            num_layers=dir_num_layers,
                            dropout=dropout,
                        )
                        for _ in range(num_decoder_layers)
                    ]
                )

    def forward(
        self,
        decoder_outputs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run prediction heads on decoder outputs with iterative refinement.

        Args:
            decoder_outputs: Transformer decoder output embeddings of shape
                (num_layers, B, N_queries, d_model) when using intermediate
                outputs, or (B, N_queries, d_model) for single-layer output.

        Returns:
            Dictionary containing:
                - 'cls_logits': Classification logits of shape
                  (num_layers, B, N_queries, num_classes+1) or
                  (B, N_queries, num_classes+1).
                - 'points': Predicted polyline points of shape
                  (num_layers, B, N_queries, K, 2) or
                  (B, N_queries, K, 2). Coordinates in [0, 1].
                - 'directions' (optional): Direction vectors of shape
                  (num_layers, B, N_queries, 2) or (B, N_queries, 2).
        """
        # Handle both intermediate (multi-layer) and single-layer outputs
        if decoder_outputs.dim() == 4:
            # Multi-layer: (num_layers, B, N_queries, d_model)
            return self._forward_intermediate(decoder_outputs)
        else:
            # Single layer: (B, N_queries, d_model)
            return self._forward_single(decoder_outputs)

    def _forward_single(
        self, query_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for single-layer decoder output.

        Args:
            query_embeddings: Shape (B, N_queries, d_model).

        Returns:
            Prediction dictionary with shapes:
                - 'cls_logits': (B, N_queries, num_classes+1)
                - 'points': (B, N_queries, K, 2)
                - 'directions': (B, N_queries, 2) [optional]
        """
        cls_head = self.cls_head if self.share_head_weights else self.cls_heads[-1]
        reg_head = self.reg_head if self.share_head_weights else self.reg_heads[-1]

        results: Dict[str, torch.Tensor] = {}
        results["cls_logits"] = cls_head(query_embeddings)
        results["points"] = reg_head(query_embeddings, prev_points=None)

        if self.use_direction_head:
            dir_head = self.dir_head if self.share_head_weights else self.dir_heads[-1]
            results["directions"] = dir_head(query_embeddings)

        return results

    def _forward_intermediate(
        self, decoder_outputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with iterative refinement across decoder layers.

        Each decoder layer's output is passed through the heads. For point
        regression, each layer refines the previous layer's predictions by
        predicting residual offsets in logit space.

        Args:
            decoder_outputs: Shape (num_layers, B, N_queries, d_model).

        Returns:
            Prediction dictionary with shapes:
                - 'cls_logits': (num_layers, B, N_queries, num_classes+1)
                - 'points': (num_layers, B, N_queries, K, 2)
                - 'directions': (num_layers, B, N_queries, 2) [optional]
        """
        num_layers = decoder_outputs.shape[0]

        all_cls_logits: List[torch.Tensor] = []
        all_points: List[torch.Tensor] = []
        all_directions: List[torch.Tensor] = []

        prev_points: Optional[torch.Tensor] = None

        for layer_idx in range(num_layers):
            layer_query = decoder_outputs[layer_idx]  # (B, N_queries, d_model)

            # Get appropriate heads for this layer
            if self.share_head_weights:
                cls_head = self.cls_head
                reg_head = self.reg_head
                dir_head = self.dir_head if self.use_direction_head else None
            else:
                cls_head = self.cls_heads[layer_idx]
                reg_head = self.reg_heads[layer_idx]
                dir_head = self.dir_heads[layer_idx] if self.use_direction_head else None

            # Classification
            cls_logits = cls_head(layer_query)  # (B, N_queries, num_classes+1)
            all_cls_logits.append(cls_logits)

            # Point regression with iterative refinement
            points = reg_head(layer_query, prev_points=prev_points)  # (B, N_queries, K, 2)
            all_points.append(points)

            # Detach for next layer's refinement to prevent gradient explosion
            if self.iterative_refinement:
                prev_points = points.detach()

            # Direction prediction
            if dir_head is not None:
                direction = dir_head(layer_query)  # (B, N_queries, 2)
                all_directions.append(direction)

        results: Dict[str, torch.Tensor] = {}
        results["cls_logits"] = torch.stack(all_cls_logits, dim=0)
        results["points"] = torch.stack(all_points, dim=0)

        if self.use_direction_head and all_directions:
            results["directions"] = torch.stack(all_directions, dim=0)

        return results


def build_map_heads(
    d_model: int = 256,
    num_classes: int = 3,
    num_points: int = 20,
    num_decoder_layers: int = 6,
    iterative_refinement: bool = True,
    use_direction_head: bool = True,
) -> MapElementHeads:
    """Factory function to build map prediction heads with default config.

    Args:
        d_model: Model dimension from the transformer decoder.
        num_classes: Number of foreground map element classes.
        num_points: Number of ordered points per polyline (K).
        num_decoder_layers: Number of decoder layers for iterative refinement.
        iterative_refinement: Whether to use iterative point refinement.
        use_direction_head: Whether to include direction prediction.

    Returns:
        Configured MapElementHeads module.

    Example:
        >>> heads = build_map_heads(d_model=256, num_classes=3, num_points=20)
        >>> # Simulate decoder output: 6 layers, batch=2, 150 queries, 256 dim
        >>> decoder_out = torch.randn(6, 2, 150, 256)
        >>> predictions = heads(decoder_out)
        >>> predictions['cls_logits'].shape
        torch.Size([6, 2, 150, 4])
        >>> predictions['points'].shape
        torch.Size([6, 2, 150, 20, 2])
        >>> predictions['directions'].shape
        torch.Size([6, 2, 150, 2])
    """
    return MapElementHeads(
        d_model=d_model,
        num_classes=num_classes,
        num_points=num_points,
        hidden_dim=d_model,
        cls_num_layers=2,
        reg_num_layers=3,
        dir_num_layers=2,
        dropout=0.0,
        iterative_refinement=iterative_refinement,
        use_direction_head=use_direction_head,
        share_head_weights=True,
        num_decoder_layers=num_decoder_layers,
    )
