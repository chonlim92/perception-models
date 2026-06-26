"""MapTR prediction heads for classification and point regression.

This module implements the prediction heads that operate on the decoded query
features from the MapTR decoder to produce:
- Per-instance class predictions (which type of map element)
- Per-point coordinate predictions (ordered vertices of each map element)

Reference: MapTR: Structured Modeling and Learning for Online Vectorized HD Map
Construction (Liao et al., ICLR 2023)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """MLP head for predicting map element class logits.

    Takes instance-level features (aggregated from point features) and predicts
    which class of map element each query represents (e.g., lane divider,
    road boundary, pedestrian crossing, or background/no-object).

    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear (logits)

    Args:
        embed_dims: Input feature dimension.
        num_classes: Number of map element classes (including background if applicable).
        hidden_dims: Hidden layer dimension. Defaults to embed_dims.
        num_hidden_layers: Number of hidden layers in the MLP.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_classes: int = 3,
        hidden_dims: Optional[int] = None,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        hidden_dims = hidden_dims or embed_dims

        layers = []
        in_dims = embed_dims
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dims, hidden_dims))
            layers.append(nn.ReLU(inplace=True))
            in_dims = hidden_dims
        layers.append(nn.Linear(in_dims, num_classes))

        self.mlp = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize with Xavier uniform for linear layers."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Initialize the final classification layer with smaller weights
        # to produce near-uniform predictions at initialization
        final_layer = self.mlp[-1]
        nn.init.xavier_uniform_(final_layer.weight, gain=0.01)
        nn.init.constant_(final_layer.bias, 0.0)

    def forward(self, instance_features: torch.Tensor) -> torch.Tensor:
        """Predict class logits for each instance query.

        Args:
            instance_features: Instance-level features [B, num_queries, embed_dims].

        Returns:
            Class logits [B, num_queries, num_classes].
        """
        return self.mlp(instance_features)


class PointRegressionHead(nn.Module):
    """MLP head for predicting normalized point coordinates.

    Takes point-level features and predicts 2D coordinates for each point
    in each map element instance. Coordinates are normalized to [0, 1] via
    sigmoid activation.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid

    Args:
        embed_dims: Input feature dimension.
        hidden_dims: Hidden layer dimension. Defaults to embed_dims.
        num_hidden_layers: Number of hidden layers in the MLP.
        output_dims: Output coordinate dimensions (2 for x, y).
    """

    def __init__(
        self,
        embed_dims: int = 256,
        hidden_dims: Optional[int] = None,
        num_hidden_layers: int = 2,
        output_dims: int = 2,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.output_dims = output_dims
        hidden_dims = hidden_dims or embed_dims

        layers = []
        in_dims = embed_dims
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dims, hidden_dims))
            layers.append(nn.ReLU(inplace=True))
            in_dims = hidden_dims
        layers.append(nn.Linear(in_dims, output_dims))

        self.mlp = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize with Xavier uniform for linear layers."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """Predict normalized point coordinates.

        Args:
            point_features: Point-level features [B, num_queries, num_points, embed_dims].

        Returns:
            Predicted coordinates [B, num_queries, num_points, 2] normalized to [0, 1].
        """
        coords = self.mlp(point_features)
        coords = coords.sigmoid()
        return coords


class MapTRHead(nn.Module):
    """Combined prediction head for MapTR.

    Takes decoder outputs (per-layer query features and reference points) and
    produces classification and regression predictions for each decoder layer.
    This supports auxiliary losses on intermediate decoder layer outputs.

    The head aggregates point features to instance features for classification
    (via mean pooling across points) while using full point-level features for
    coordinate regression.

    For iterative refinement, the point regression head predicts offsets relative
    to the reference points from the decoder, which are then added to produce
    final coordinates.

    Args:
        embed_dims: Feature embedding dimension.
        num_classes: Number of map element classes.
        num_queries: Number of instance queries.
        num_points: Number of points per instance.
        num_decoder_layers: Number of decoder layers (for shared/per-layer heads).
        cls_hidden_dims: Hidden dims for classification MLP.
        reg_hidden_dims: Hidden dims for regression MLP.
        num_cls_layers: Number of hidden layers in classification MLP.
        num_reg_layers: Number of hidden layers in regression MLP.
        share_head_across_layers: Whether to share prediction heads across decoder
            layers. If False, each layer gets its own head (more parameters but
            can specialize).
        use_iterative_refinement: Whether to use reference points from decoder
            for iterative refinement of coordinates.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_classes: int = 3,
        num_queries: int = 50,
        num_points: int = 20,
        num_decoder_layers: int = 6,
        cls_hidden_dims: Optional[int] = None,
        reg_hidden_dims: Optional[int] = None,
        num_cls_layers: int = 2,
        num_reg_layers: int = 2,
        share_head_across_layers: bool = True,
        use_iterative_refinement: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_points = num_points
        self.num_decoder_layers = num_decoder_layers
        self.share_head_across_layers = share_head_across_layers
        self.use_iterative_refinement = use_iterative_refinement

        if share_head_across_layers:
            # Single shared head used for all decoder layer outputs
            self.cls_head = ClassificationHead(
                embed_dims=embed_dims,
                num_classes=num_classes,
                hidden_dims=cls_hidden_dims,
                num_hidden_layers=num_cls_layers,
            )
            self.reg_head = PointRegressionHead(
                embed_dims=embed_dims,
                hidden_dims=reg_hidden_dims,
                num_hidden_layers=num_reg_layers,
                output_dims=2,
            )
        else:
            # Per-layer heads
            self.cls_heads = nn.ModuleList(
                [
                    ClassificationHead(
                        embed_dims=embed_dims,
                        num_classes=num_classes,
                        hidden_dims=cls_hidden_dims,
                        num_hidden_layers=num_cls_layers,
                    )
                    for _ in range(num_decoder_layers)
                ]
            )
            self.reg_heads = nn.ModuleList(
                [
                    PointRegressionHead(
                        embed_dims=embed_dims,
                        hidden_dims=reg_hidden_dims,
                        num_hidden_layers=num_reg_layers,
                        output_dims=2,
                    )
                    for _ in range(num_decoder_layers)
                ]
            )

    def _get_cls_head(self, layer_idx: int) -> ClassificationHead:
        """Get classification head for a given layer."""
        if self.share_head_across_layers:
            return self.cls_head
        return self.cls_heads[layer_idx]

    def _get_reg_head(self, layer_idx: int) -> PointRegressionHead:
        """Get regression head for a given layer."""
        if self.share_head_across_layers:
            return self.reg_head
        return self.reg_heads[layer_idx]

    def forward(
        self,
        decoder_outputs: List[torch.Tensor],
        reference_points: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Forward pass producing predictions for each decoder layer.

        Args:
            decoder_outputs: List of query features from each decoder layer,
                each of shape [B, num_queries, num_points, embed_dims].
            reference_points: Optional list of reference points from decoder,
                each of shape [B, num_queries, num_points, 2]. Used for
                iterative refinement when use_iterative_refinement=True.

        Returns:
            Dictionary containing:
                - "cls_scores": List of class logits [B, num_queries, num_classes]
                  for each decoder layer.
                - "point_coords": List of point coordinates [B, num_queries, num_points, 2]
                  for each decoder layer (normalized to [0, 1]).
        """
        all_cls_scores = []
        all_point_coords = []

        for layer_idx, layer_output in enumerate(decoder_outputs):
            # layer_output: [B, num_queries, num_points, embed_dims]
            batch_size = layer_output.shape[0]

            # --- Classification ---
            # Aggregate point features to instance features via mean pooling
            instance_features = layer_output.mean(dim=2)  # [B, num_queries, embed_dims]
            cls_head = self._get_cls_head(layer_idx)
            cls_scores = cls_head(instance_features)  # [B, num_queries, num_classes]
            all_cls_scores.append(cls_scores)

            # --- Point Regression ---
            reg_head = self._get_reg_head(layer_idx)

            if self.use_iterative_refinement and reference_points is not None:
                # In iterative refinement mode, the decoder already produces
                # refined reference points. The regression head predicts a
                # residual offset to further refine coordinates.
                ref_pts = reference_points[layer_idx]  # [B, num_queries, num_points, 2]
                # Predict offset in logit space
                reg_offset = reg_head.mlp(layer_output)  # [B, num_queries, num_points, 2]
                # Add offset to reference points in inverse-sigmoid space
                ref_logits = torch.special.logit(ref_pts.clamp(1e-5, 1 - 1e-5))
                point_coords = (ref_logits + reg_offset).sigmoid()
            else:
                # Direct regression without iterative refinement
                point_coords = reg_head(layer_output)  # [B, num_queries, num_points, 2]

            all_point_coords.append(point_coords)

        return {
            "cls_scores": all_cls_scores,
            "point_coords": all_point_coords,
        }

    def predict(
        self,
        decoder_outputs: List[torch.Tensor],
        reference_points: Optional[List[torch.Tensor]] = None,
        score_threshold: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """Run inference and produce final predictions from the last decoder layer.

        This is a convenience method for inference that returns only the final
        layer predictions with optional score thresholding.

        Args:
            decoder_outputs: List of query features from each decoder layer.
            reference_points: Optional list of reference points from decoder.
            score_threshold: Minimum confidence score to keep a prediction.

        Returns:
            Dictionary containing:
                - "scores": Confidence scores [B, num_queries] (max class probability).
                - "labels": Predicted class labels [B, num_queries].
                - "points": Predicted point coordinates [B, num_queries, num_points, 2].
                - "mask": Boolean mask [B, num_queries] indicating valid predictions
                  (above score threshold).
        """
        # Get predictions from all layers
        outputs = self.forward(decoder_outputs, reference_points)

        # Use only the last layer's predictions
        cls_scores = outputs["cls_scores"][-1]  # [B, num_queries, num_classes]
        point_coords = outputs["point_coords"][-1]  # [B, num_queries, num_points, 2]

        # Convert logits to probabilities and get best class
        cls_probs = F.softmax(cls_scores, dim=-1)
        scores, labels = cls_probs.max(dim=-1)  # [B, num_queries]

        # Create validity mask based on score threshold
        mask = scores > score_threshold

        return {
            "scores": scores,
            "labels": labels,
            "points": point_coords,
            "mask": mask,
        }
