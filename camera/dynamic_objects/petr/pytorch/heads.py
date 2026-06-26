"""
Detection heads for PETR.

Classification and regression branches with iterative refinement support.
Predicts 3D bounding boxes (center, size, rotation) and velocities from
decoder query features.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """Multi-layer perceptron head with configurable depth and hidden dims.

    Args:
        input_dims: Input feature dimension.
        hidden_dims: Hidden layer dimension.
        output_dims: Output dimension.
        num_layers: Number of layers (minimum 2: input -> hidden -> output).
    """

    def __init__(
        self,
        input_dims: int = 256,
        hidden_dims: int = 256,
        output_dims: int = 10,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "MLPHead requires at least 2 layers"

        layers: List[nn.Module] = []
        # First layer
        layers.append(nn.Linear(input_dims, hidden_dims))
        layers.append(nn.ReLU(inplace=True))
        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.ReLU(inplace=True))
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dims, output_dims))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            x: Input features (..., input_dims).

        Returns:
            Output predictions (..., output_dims).
        """
        return self.mlp(x)


class PETRDetectionHead(nn.Module):
    """Detection head for PETR 3D object detection.

    Produces classification scores, bounding box parameters, and velocity
    predictions. Supports iterative refinement across decoder layers.

    The regression target is a 10-dimensional vector:
        [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
    where (cx, cy, cz) are center offsets relative to reference points,
    (w, l, h) are dimensions, (sin, cos) encode yaw angle, and
    (vx, vy) are velocities.

    Args:
        embed_dims: Input feature dimension from decoder.
        num_classes: Number of object classes.
        code_size: Dimension of the bounding box code (default 10).
        num_layers: Number of decoder layers (for iterative refinement heads).
        num_mlp_layers: Number of layers in each MLP head.
        shared_head: If True, share classification/regression heads across
            decoder layers. If False, use separate heads per layer.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_classes: int = 10,
        code_size: int = 10,
        num_layers: int = 6,
        num_mlp_layers: int = 3,
        shared_head: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.code_size = code_size
        self.num_layers = num_layers
        self.shared_head = shared_head

        if shared_head:
            # Single shared classification head
            self.cls_head = MLPHead(
                input_dims=embed_dims,
                hidden_dims=embed_dims,
                output_dims=num_classes,
                num_layers=num_mlp_layers,
            )
            # Single shared regression head
            self.reg_head = MLPHead(
                input_dims=embed_dims,
                hidden_dims=embed_dims,
                output_dims=code_size,
                num_layers=num_mlp_layers,
            )
        else:
            # Separate heads for each decoder layer (iterative refinement)
            self.cls_heads = nn.ModuleList(
                [
                    MLPHead(
                        input_dims=embed_dims,
                        hidden_dims=embed_dims,
                        output_dims=num_classes,
                        num_layers=num_mlp_layers,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.reg_heads = nn.ModuleList(
                [
                    MLPHead(
                        input_dims=embed_dims,
                        hidden_dims=embed_dims,
                        output_dims=code_size,
                        num_layers=num_mlp_layers,
                    )
                    for _ in range(num_layers)
                ]
            )

        self._init_bias()

    def _init_bias(self) -> None:
        """Initialize classification bias for focal loss stability.

        Sets the bias of the final classification layer so that the initial
        predicted probability is ~0.01 (prior probability).
        """
        import math

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        if self.shared_head:
            nn.init.constant_(self.cls_head.mlp[-1].bias, bias_value)
        else:
            for cls_head in self.cls_heads:
                nn.init.constant_(cls_head.mlp[-1].bias, bias_value)

    def get_reg_branches(self) -> nn.ModuleList:
        """Return regression branches for iterative refinement in decoder.

        Returns:
            ModuleList of regression heads (one per decoder layer).
        """
        if self.shared_head:
            return nn.ModuleList([self.reg_head] * self.num_layers)
        else:
            return self.reg_heads

    def forward(
        self,
        decoder_outputs: List[torch.Tensor],
        reference_points: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Compute predictions from decoder layer outputs.

        Args:
            decoder_outputs: List of decoder outputs, one per layer.
                Each tensor has shape (B, Q, C).
            reference_points: Optional list of reference points per layer
                (B, Q, 3). Used for decoding absolute positions.

        Returns:
            Dictionary with keys:
                'cls_scores': List of (B, Q, num_classes) per layer.
                'bbox_preds': List of (B, Q, code_size) per layer.
                'reference_points': List of (B, Q, 3) per layer (if provided).
        """
        all_cls_scores = []
        all_bbox_preds = []

        for layer_idx, dec_out in enumerate(decoder_outputs):
            if self.shared_head:
                cls_score = self.cls_head(dec_out)  # (B, Q, num_classes)
                bbox_pred = self.reg_head(dec_out)  # (B, Q, code_size)
            else:
                cls_score = self.cls_heads[layer_idx](dec_out)
                bbox_pred = self.reg_heads[layer_idx](dec_out)

            all_cls_scores.append(cls_score)
            all_bbox_preds.append(bbox_pred)

        results: Dict[str, List[torch.Tensor]] = {
            "cls_scores": all_cls_scores,
            "bbox_preds": all_bbox_preds,
        }
        if reference_points is not None:
            results["reference_points"] = reference_points

        return results

    def decode_bbox(
        self,
        bbox_pred: torch.Tensor,
        reference_points: torch.Tensor,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> torch.Tensor:
        """Decode bounding box predictions to absolute coordinates.

        The network predicts offsets relative to reference points for the
        center (cx, cy, cz). Other parameters (w, l, h, sin, cos, vx, vy)
        are predicted directly.

        Args:
            bbox_pred: Raw predictions (B, Q, code_size).
            reference_points: Normalized reference points (B, Q, 3) in [0,1].
            pc_range: Point cloud range for denormalization.

        Returns:
            Decoded bounding boxes (B, Q, code_size) in world coordinates.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        device = bbox_pred.device
        dtype = bbox_pred.dtype

        # Decode center position: reference_point + offset
        # Reference points are in [0,1], denormalize to world coords
        cx = reference_points[..., 0:1] * (x_max - x_min) + x_min + bbox_pred[..., 0:1]
        cy = reference_points[..., 1:2] * (y_max - y_min) + y_min + bbox_pred[..., 1:2]
        cz = reference_points[..., 2:3] * (z_max - z_min) + z_min + bbox_pred[..., 2:3]

        # Dimensions (w, l, h) - predicted as log scale
        w = bbox_pred[..., 3:4].exp()
        l = bbox_pred[..., 4:5].exp()
        h = bbox_pred[..., 5:6].exp()

        # Rotation (sin, cos of yaw)
        sin_yaw = bbox_pred[..., 6:7]
        cos_yaw = bbox_pred[..., 7:8]

        # Velocity (vx, vy)
        vx = bbox_pred[..., 8:9]
        vy = bbox_pred[..., 9:10]

        decoded = torch.cat([cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy], dim=-1)

        return decoded


class VelocityHead(nn.Module):
    """Dedicated velocity prediction head.

    Optionally used as a separate branch for velocity estimation,
    which can be trained with a different loss weight.

    Args:
        embed_dims: Input feature dimension.
        hidden_dims: Hidden layer dimension.
        num_layers: Number of MLP layers.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        hidden_dims: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.velocity_head = MLPHead(
            input_dims=embed_dims,
            hidden_dims=hidden_dims,
            output_dims=2,  # vx, vy
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict velocity from query features.

        Args:
            x: Query features (B, Q, C).

        Returns:
            Velocity predictions (B, Q, 2) as (vx, vy) in m/s.
        """
        return self.velocity_head(x)
