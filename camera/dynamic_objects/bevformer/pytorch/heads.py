"""BEVFormer Detection Head.

Implements the task-specific prediction heads for 3D object detection from
decoded object query features. Includes classification and 3D bounding box
regression heads, with support for auxiliary losses from intermediate decoder
layers. Uses NMS-free inference via top-k confidence selection.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["BEVFormerHead"]


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable hidden layers.

    A simple feed-forward network used as the prediction head for both
    classification and bounding box regression.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        output_dims: int,
        num_hidden_layers: int = 2,
    ) -> None:
        """Initialize MLP.

        Args:
            input_dims: Input feature dimension.
            hidden_dims: Hidden layer dimension.
            output_dims: Output prediction dimension.
            num_hidden_layers: Number of hidden layers before the output layer.
        """
        super().__init__()
        layers: List[nn.Module] = []

        # First hidden layer
        layers.append(nn.Linear(input_dims, hidden_dims))
        layers.append(nn.ReLU(inplace=True))

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.ReLU(inplace=True))

        # Output layer (no activation - applied externally if needed)
        layers.append(nn.Linear(hidden_dims, output_dims))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with Xavier uniform for weights, zeros for biases."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (..., input_dims).

        Returns:
            Predictions of shape (..., output_dims).
        """
        return self.mlp(x)


class BEVFormerHead(nn.Module):
    """BEVFormer 3D Object Detection Head.

    Takes decoded object query features from each decoder layer and predicts:
      - Classification scores via focal loss (sigmoid activation)
      - 3D bounding box parameters: center (cx, cy, cz), size (w, l, h),
        yaw (sin, cos), and velocity (vx, vy) -- 10 parameters total

    Supports auxiliary losses from intermediate decoder layers by applying
    shared prediction heads to outputs from each layer.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_classes: int = 10,
        num_queries: int = 900,
        num_reg_params: int = 10,
        hidden_dims: int = 256,
        num_hidden_layers: int = 2,
        top_k_inference: int = 300,
    ) -> None:
        """Initialize BEVFormer detection head.

        Args:
            embed_dims: Input feature dimension from decoder.
            num_classes: Number of object classes for classification.
            num_queries: Number of object queries (for reference only).
            num_reg_params: Number of bounding box regression parameters.
                Default 10: cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy.
            hidden_dims: Hidden dimension for MLP prediction heads.
            num_hidden_layers: Number of hidden layers in each MLP head.
            top_k_inference: Number of top predictions to keep at inference time.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_reg_params = num_reg_params
        self.top_k_inference = top_k_inference

        # Classification head: predicts class logits (sigmoid applied at loss/inference)
        self.cls_head = MLP(
            input_dims=embed_dims,
            hidden_dims=hidden_dims,
            output_dims=num_classes,
            num_hidden_layers=num_hidden_layers,
        )

        # Regression head: predicts 10 bbox parameters (no final activation)
        self.reg_head = MLP(
            input_dims=embed_dims,
            hidden_dims=hidden_dims,
            output_dims=num_reg_params,
            num_hidden_layers=num_hidden_layers,
        )

        # Initialize classification bias for focal loss (prior probability ~0.01)
        self._init_cls_bias()

    def _init_cls_bias(self) -> None:
        """Initialize classification head output bias for focal loss.

        Sets the bias so that the initial predicted probability is approximately
        0.01 (prior_prob), which stabilizes early training with focal loss.
        """
        prior_prob = 0.01
        bias_value = -torch.log(
            torch.tensor((1.0 - prior_prob) / prior_prob)
        ).item()
        # The last linear layer in the MLP
        final_cls_layer = self.cls_head.mlp[-1]
        assert isinstance(final_cls_layer, nn.Linear)
        nn.init.constant_(final_cls_layer.bias, bias_value)

    def forward(
        self,
        decoder_outputs: List[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """Forward pass: predict classifications and bounding boxes.

        Applies shared prediction heads to outputs from each decoder layer,
        enabling auxiliary loss computation during training.

        Args:
            decoder_outputs: List of decoded features from each decoder layer.
                Each tensor has shape (B, num_queries, embed_dims).

        Returns:
            Dictionary with keys:
                - "cls_scores": List of classification logits per layer,
                    each (B, num_queries, num_classes).
                - "bbox_preds": List of bbox predictions per layer,
                    each (B, num_queries, num_reg_params).
        """
        all_cls_scores: List[torch.Tensor] = []
        all_bbox_preds: List[torch.Tensor] = []

        for layer_output in decoder_outputs:
            # Classification: (B, num_queries, num_classes)
            cls_scores = self.cls_head(layer_output)
            all_cls_scores.append(cls_scores)

            # Regression: (B, num_queries, num_reg_params)
            bbox_preds = self.reg_head(layer_output)
            all_bbox_preds.append(bbox_preds)

        return {
            "cls_scores": all_cls_scores,
            "bbox_preds": all_bbox_preds,
        }

    @torch.no_grad()
    def inference(
        self,
        decoder_outputs: List[torch.Tensor],
        score_threshold: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """NMS-free inference: select top-k predictions by confidence.

        Uses only the final decoder layer output for inference. Applies sigmoid
        to classification logits, takes the maximum class score per query, and
        returns the top-k detections.

        Args:
            decoder_outputs: List of decoded features from each decoder layer.
                Only the last one is used for inference.
            score_threshold: Minimum confidence score to keep a detection.

        Returns:
            Dictionary with keys:
                - "scores": Confidence scores, shape (B, top_k).
                - "labels": Predicted class indices, shape (B, top_k).
                - "bboxes": Predicted 3D bounding boxes, shape (B, top_k, num_reg_params).
        """
        # Use final decoder layer output
        final_output = decoder_outputs[-1]
        batch_size = final_output.shape[0]

        # Get predictions
        cls_logits = self.cls_head(final_output)  # (B, num_queries, num_classes)
        bbox_preds = self.reg_head(final_output)  # (B, num_queries, num_reg_params)

        # Apply sigmoid for classification probabilities
        cls_probs = torch.sigmoid(cls_logits)  # (B, num_queries, num_classes)

        # Get max class score per query
        max_scores, max_labels = cls_probs.max(dim=-1)  # (B, num_queries)

        # Select top-k per batch
        k = min(self.top_k_inference, max_scores.shape[1])
        topk_scores, topk_indices = max_scores.topk(k, dim=1)  # (B, k)

        # Gather corresponding labels and bboxes
        topk_labels = torch.gather(max_labels, 1, topk_indices)  # (B, k)
        topk_bboxes = torch.gather(
            bbox_preds,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.num_reg_params),
        )  # (B, k, num_reg_params)

        # Apply score threshold mask
        valid_mask = topk_scores > score_threshold  # (B, k)
        topk_scores = topk_scores * valid_mask.float()
        topk_labels = topk_labels * valid_mask.long()

        return {
            "scores": topk_scores,
            "labels": topk_labels,
            "bboxes": topk_bboxes,
        }
