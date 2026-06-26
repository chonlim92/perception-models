# CRAFT: Training Guide

## Joint Training Strategy for Camera-Radar 3D Object Detection

---

## 1. Training Overview

### Training Philosophy

CRAFT employs a joint training strategy that simultaneously optimizes both the camera and radar branches along with the fusion module. This end-to-end approach ensures that:

1. Each modality learns representations complementary to the other
2. The fusion module learns to weight reliable information from each sensor
3. Gradients flow through both branches, enabling co-adaptation
4. The detection head learns to exploit fused features effectively

### Training Configuration Summary

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch 1.10+ with CUDA 11.3+ |
| Distributed training | 8x NVIDIA A100 (40GB or 80GB) |
| Total training epochs | 20 (or 36 with extended schedule) |
| Effective batch size | 32 (4 per GPU x 8 GPUs) |
| Training time | ~24-48 hours (8x A100) |
| Mixed precision | FP16 (automatic mixed precision) |
| Gradient accumulation | 1 (or 2 for larger effective batch) |
| Random seed | 42 (for reproducibility) |

---

## 2. Loss Functions

### 2.1 Multi-Task Loss Formulation

The total training loss is a weighted combination of multiple task-specific losses:

```
L_total = λ_heat * L_heatmap + λ_off * L_offset + λ_ht * L_height + 
          λ_size * L_size + λ_rot * L_rotation + λ_vel * L_velocity
```

### 2.2 Heatmap Loss (Classification)

The heatmap loss uses a modified focal loss for handling class imbalance:

```python
class GaussianFocalLoss(nn.Module):
    """
    Focal loss for heatmap classification with Gaussian targets.
    
    Positive targets are Gaussian-smoothed around GT centers.
    Negative targets use focal weight to reduce easy-negative contribution.
    """
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha  # Focusing parameter for hard examples
        self.beta = beta    # Weight reduction for easy negatives near GT
    
    def forward(self, pred_heatmap, gt_heatmap):
        """
        Args:
            pred_heatmap: (B, C, H, W) - predicted class probabilities [0, 1]
            gt_heatmap: (B, C, H, W) - Gaussian target heatmap [0, 1]
        
        Returns:
            loss: Scalar focal loss value
        """
        pos_mask = gt_heatmap.eq(1).float()
        neg_mask = gt_heatmap.lt(1).float()
        
        # Clamp predictions for numerical stability
        pred = torch.clamp(pred_heatmap, min=1e-6, max=1-1e-6)
        
        # Positive loss (at GT center pixels)
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        
        # Negative loss (everywhere else, weighted by distance to GT)
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.alpha) * \
                   torch.pow(1 - gt_heatmap, self.beta) * neg_mask
        
        # Normalize by number of positive pixels
        num_pos = pos_mask.sum()
        if num_pos == 0:
            return neg_loss.sum()
        
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss
```

**Gaussian Target Generation:**

```python
def generate_heatmap_target(gt_boxes, gt_labels, bev_size, bev_range, num_classes):
    """
    Generate Gaussian heatmap targets for training.
    
    Each GT box center generates a 2D Gaussian on the BEV heatmap,
    with radius proportional to the box size.
    """
    H, W = bev_size
    heatmap = torch.zeros(num_classes, H, W)
    
    for box, label in zip(gt_boxes, gt_labels):
        # Convert box center to BEV pixel coordinates
        cx_pixel = (box[0] - bev_range[0]) / (bev_range[1] - bev_range[0]) * W
        cy_pixel = (box[1] - bev_range[2]) / (bev_range[3] - bev_range[2]) * H
        
        # Compute Gaussian radius based on box size
        width_pixels = box[3] / (bev_range[1] - bev_range[0]) * W
        length_pixels = box[4] / (bev_range[3] - bev_range[2]) * H
        radius = gaussian_radius((length_pixels, width_pixels), min_overlap=0.1)
        radius = max(0, int(radius))
        
        # Draw Gaussian
        draw_gaussian(heatmap[label], (int(cx_pixel), int(cy_pixel)), radius)
    
    return heatmap
```

### 2.3 Regression Losses

```python
class RegressionLosses(nn.Module):
    """
    Regression losses for box parameters.
    Only computed at positive (GT center) locations.
    """
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def offset_loss(self, pred_offset, gt_offset, mask):
        """
        L1 loss for sub-pixel center offset.
        
        Args:
            pred_offset: (B, 2, H, W)
            gt_offset: (B, 2, H, W)
            mask: (B, 1, H, W) - positive locations only
        """
        loss = self.l1_loss(pred_offset, gt_offset) * mask
        return loss.sum() / (mask.sum() + 1e-8)
    
    def size_loss(self, pred_size, gt_size, mask):
        """
        L1 loss for log-scale size prediction.
        
        Args:
            pred_size: (B, 2, H, W) - predicted log(w), log(l)
            gt_size: (B, 2, H, W) - target log(w), log(l)
            mask: (B, 1, H, W)
        """
        loss = self.l1_loss(pred_size, gt_size) * mask
        return loss.sum() / (mask.sum() + 1e-8)
    
    def height_loss(self, pred_height, gt_height, mask):
        """L1 loss for z-center and height."""
        loss = self.l1_loss(pred_height, gt_height) * mask
        return loss.sum() / (mask.sum() + 1e-8)
    
    def rotation_loss(self, pred_rot, gt_rot, mask):
        """
        L1 loss for sin/cos yaw prediction.
        Alternative: bin-based rotation loss for better convergence.
        """
        loss = self.l1_loss(pred_rot, gt_rot) * mask
        return loss.sum() / (mask.sum() + 1e-8)
    
    def velocity_loss(self, pred_vel, gt_vel, mask):
        """L1 loss for velocity prediction."""
        loss = self.l1_loss(pred_vel, gt_vel) * mask
        return loss.sum() / (mask.sum() + 1e-8)
```

### 2.4 Loss Weight Configuration

| Loss Component | Weight (λ) | Notes |
|---------------|------------|-------|
| Heatmap (focal) | 1.0 | Primary classification signal |
| Center offset | 1.0 | Sub-pixel refinement |
| Height (z + h) | 0.25 | Lower weight due to radar z-uncertainty |
| Size (w, l) | 0.25 | Log-scale makes this numerically smaller |
| Rotation (sin, cos) | 1.0 | Critical for orientation accuracy |
| Velocity (vx, vy) | 0.25 | Radar provides strong supervision signal |

---

## 3. Loss Balancing Between Modalities

### 3.1 Multi-Branch Loss Strategy

To train both branches effectively, CRAFT optionally applies auxiliary losses to each branch independently:

```python
class CRAFTLoss(nn.Module):
    """
    Complete loss function for CRAFT training.
    
    Includes:
    - Main detection loss on fused features
    - Auxiliary camera-only detection loss
    - Auxiliary radar-only detection loss
    """
    def __init__(self, config):
        super().__init__()
        self.main_loss = DetectionLoss(config)
        self.camera_aux_loss = DetectionLoss(config)
        self.radar_aux_loss = DetectionLoss(config)
        
        # Balancing weights
        self.w_main = 1.0      # Main fusion loss
        self.w_camera = 0.25   # Camera auxiliary loss
        self.w_radar = 0.25    # Radar auxiliary loss
    
    def forward(self, predictions, gt_targets):
        """
        Args:
            predictions: Dict containing:
                'main': Predictions from fused features
                'camera_aux': Predictions from camera-only branch
                'radar_aux': Predictions from radar-only branch
            gt_targets: Ground truth annotations
        
        Returns:
            total_loss: Scalar
            loss_dict: Per-component losses for logging
        """
        # Main fusion loss
        main_loss, main_dict = self.main_loss(predictions['main'], gt_targets)
        
        # Camera auxiliary loss (optional, for branch pre-training)
        cam_loss, cam_dict = self.camera_aux_loss(
            predictions['camera_aux'], gt_targets)
        
        # Radar auxiliary loss
        radar_loss, radar_dict = self.radar_aux_loss(
            predictions['radar_aux'], gt_targets)
        
        # Total loss with balancing
        total = self.w_main * main_loss + \
                self.w_camera * cam_loss + \
                self.w_radar * radar_loss
        
        loss_dict = {
            'loss_total': total,
            'loss_main': main_loss,
            'loss_camera_aux': cam_loss,
            'loss_radar_aux': radar_loss,
            **{f'main_{k}': v for k, v in main_dict.items()},
        }
        
        return total, loss_dict
```

### 3.2 Gradient Balancing

To prevent one modality from dominating training:

```python
class GradientBalancer:
    """
    Balances gradient magnitudes between camera and radar branches.
    Prevents one modality from dominating the fusion module gradients.
    """
    def __init__(self, alpha=0.9, threshold_ratio=5.0):
        self.alpha = alpha
        self.threshold_ratio = threshold_ratio
        self.camera_grad_ema = None
        self.radar_grad_ema = None
    
    def balance(self, camera_grad_norm, radar_grad_norm):
        """
        Compute scaling factors to balance gradient magnitudes.
        
        If one branch has >threshold_ratio times the gradient of the other,
        scale it down to prevent dominance.
        """
        # Update EMA
        if self.camera_grad_ema is None:
            self.camera_grad_ema = camera_grad_norm
            self.radar_grad_ema = radar_grad_norm
        else:
            self.camera_grad_ema = self.alpha * self.camera_grad_ema + \
                                   (1 - self.alpha) * camera_grad_norm
            self.radar_grad_ema = self.alpha * self.radar_grad_ema + \
                                  (1 - self.alpha) * radar_grad_norm
        
        ratio = self.camera_grad_ema / (self.radar_grad_ema + 1e-8)
        
        if ratio > self.threshold_ratio:
            camera_scale = self.threshold_ratio / ratio
            radar_scale = 1.0
        elif ratio < 1.0 / self.threshold_ratio:
            camera_scale = 1.0
            radar_scale = ratio * self.threshold_ratio
        else:
            camera_scale = 1.0
            radar_scale = 1.0
        
        return camera_scale, radar_scale
```

---

## 4. Learning Rate Schedules

### 4.1 Optimizer Configuration

```python
def build_optimizer(model, config):
    """
    Build optimizer with per-component learning rate groups.
    
    Different learning rates for:
    - Pre-trained backbone (lower LR to preserve features)
    - Randomly initialized modules (higher LR for fast convergence)
    """
    # Parameter groups with different learning rates
    param_groups = [
        # Camera backbone (pre-trained, fine-tune with lower LR)
        {
            'params': model.camera_backbone.parameters(),
            'lr': config.backbone_lr,            # 2e-5
            'weight_decay': config.weight_decay,  # 0.01
            'name': 'camera_backbone'
        },
        # Camera FPN (randomly initialized)
        {
            'params': model.camera_fpn.parameters(),
            'lr': config.base_lr,                # 2e-4
            'weight_decay': config.weight_decay,
            'name': 'camera_fpn'
        },
        # Radar branch (randomly initialized)
        {
            'params': model.radar_branch.parameters(),
            'lr': config.base_lr,                # 2e-4
            'weight_decay': config.weight_decay,
            'name': 'radar_branch'
        },
        # Fusion transformer (randomly initialized)
        {
            'params': model.scft.parameters(),
            'lr': config.base_lr,                # 2e-4
            'weight_decay': config.weight_decay,
            'name': 'scft'
        },
        # Detection head (randomly initialized)
        {
            'params': model.detection_head.parameters(),
            'lr': config.base_lr,                # 2e-4
            'weight_decay': config.weight_decay,
            'name': 'detection_head'
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    return optimizer
```

### 4.2 Learning Rate Schedule

CRAFT uses a cosine annealing schedule with warm-up:

```python
def build_lr_scheduler(optimizer, config):
    """
    Cosine annealing with linear warm-up.
    
    Schedule:
    1. Linear warm-up: 0 -> base_lr over first 500 iterations
    2. Cosine decay: base_lr -> min_lr over remaining training
    """
    warmup_iters = config.warmup_iters        # 500
    total_iters = config.total_epochs * config.iters_per_epoch
    min_lr_ratio = config.min_lr_ratio        # 1e-3 (final LR = base_lr * 1e-3)
    
    def lr_lambda(iter):
        if iter < warmup_iters:
            # Linear warm-up
            return iter / warmup_iters
        else:
            # Cosine annealing
            progress = (iter - warmup_iters) / (total_iters - warmup_iters)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * \
                   (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
```

### 4.3 Learning Rate Summary

| Phase | Epochs | Camera Backbone LR | Other Modules LR |
|-------|--------|-------------------|------------------|
| Warm-up | 0 - 0.5 | 0 -> 2e-5 | 0 -> 2e-4 |
| Cosine decay | 0.5 - 20 | 2e-5 -> 2e-8 | 2e-4 -> 2e-7 |

**Alternative: Step Schedule (simpler):**

| Phase | Epochs | Learning Rate |
|-------|--------|--------------|
| Initial | 0 - 12 | 2e-4 |
| Decay 1 | 12 - 16 | 2e-5 |
| Decay 2 | 16 - 20 | 2e-6 |

---

## 5. Data Augmentation

### 5.1 Camera Augmentation

```python
class CameraAugmentation:
    """
    Data augmentation pipeline for camera images.
    
    Note: Augmentations that change geometry (flip, resize) must also
    update the camera calibration matrices accordingly.
    """
    def __init__(self, config):
        self.config = config
    
    def __call__(self, images, intrinsics, extrinsics):
        """
        Args:
            images: (6, 3, H, W) multi-view images
            intrinsics: (6, 3, 3) camera matrices
            extrinsics: (6, 4, 4) camera poses
        
        Returns:
            Augmented images and updated calibration
        """
        augmented_images = []
        augmented_intrinsics = []
        
        for i in range(6):
            img = images[i]
            K = intrinsics[i].copy()
            
            # 1. Random resize (scale augmentation)
            if self.config.random_resize:
                scale = np.random.uniform(0.9, 1.1)
                img = F.interpolate(img, scale_factor=scale)
                K[0, 0] *= scale  # fx
                K[1, 1] *= scale  # fy
                K[0, 2] *= scale  # cx
                K[1, 2] *= scale  # cy
            
            # 2. Random crop to target size
            if self.config.random_crop:
                crop_h, crop_w = self.config.target_size  # (256, 704)
                max_y = img.shape[1] - crop_h
                max_x = img.shape[2] - crop_w
                y_off = np.random.randint(0, max(max_y, 1))
                x_off = np.random.randint(0, max(max_x, 1))
                img = img[:, y_off:y_off+crop_h, x_off:x_off+crop_w]
                K[0, 2] -= x_off  # Update principal point
                K[1, 2] -= y_off
            
            # 3. Color jitter (does not affect calibration)
            if self.config.color_jitter:
                img = self._apply_color_jitter(img, 
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            
            # 4. Random horizontal flip
            if self.config.random_flip and np.random.random() < 0.5:
                img = torch.flip(img, dims=[2])
                K[0, 2] = img.shape[2] - K[0, 2]  # Mirror principal point
                # Also update extrinsic to reflect the flip
            
            # 5. Normalization (always applied)
            img = self._normalize(img, 
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            
            augmented_images.append(img)
            augmented_intrinsics.append(K)
        
        return augmented_images, augmented_intrinsics, extrinsics
```

### 5.2 Radar Augmentation

```python
class RadarAugmentation:
    """
    Data augmentation for radar point cloud.
    
    Must maintain physical plausibility of radar measurements.
    """
    def __init__(self, config):
        self.config = config
    
    def __call__(self, radar_points, gt_boxes):
        """
        Args:
            radar_points: (N, 7) - [x, y, z, vx, vy, RCS, dt]
            gt_boxes: (M, 9) - ground truth boxes
        
        Returns:
            Augmented radar points and GT boxes
        """
        # 1. Global rotation (around z-axis)
        if self.config.random_rotation:
            angle = np.random.uniform(-np.pi/4, np.pi/4)  # ±45 degrees
            rot_matrix = self._rotation_z(angle)
            radar_points[:, :3] = (rot_matrix @ radar_points[:, :3].T).T
            radar_points[:, 3:5] = (rot_matrix[:2, :2] @ radar_points[:, 3:5].T).T
            gt_boxes = self._rotate_boxes(gt_boxes, angle)
        
        # 2. Global scaling
        if self.config.random_scale:
            scale = np.random.uniform(0.95, 1.05)
            radar_points[:, :3] *= scale
            radar_points[:, 3:5] *= scale  # Scale velocity too
            gt_boxes[:, :3] *= scale       # Scale positions
            gt_boxes[:, 3:6] *= scale      # Scale dimensions
            gt_boxes[:, 7:9] *= scale      # Scale velocities
        
        # 3. Global translation
        if self.config.random_translation:
            tx = np.random.uniform(-0.5, 0.5)
            ty = np.random.uniform(-0.5, 0.5)
            radar_points[:, 0] += tx
            radar_points[:, 1] += ty
            gt_boxes[:, 0] += tx
            gt_boxes[:, 1] += ty
        
        # 4. Random point dropout (simulate radar failures)
        if self.config.random_dropout:
            keep_prob = np.random.uniform(0.8, 1.0)
            mask = np.random.random(len(radar_points)) < keep_prob
            radar_points = radar_points[mask]
        
        # 5. Gaussian noise on measurements
        if self.config.point_noise:
            # Position noise
            radar_points[:, :2] += np.random.normal(0, 0.1, radar_points[:, :2].shape)
            # Velocity noise
            radar_points[:, 3:5] += np.random.normal(0, 0.05, radar_points[:, 3:5].shape)
            # RCS noise
            radar_points[:, 5] += np.random.normal(0, 1.0, radar_points[:, 5].shape)
        
        # 6. GT-paste (copy-paste augmentation for rare classes)
        if self.config.gt_paste:
            radar_points, gt_boxes = self._gt_paste(
                radar_points, gt_boxes,
                paste_classes=['bicycle', 'motorcycle', 'construction_vehicle'],
                max_paste=3
            )
        
        return radar_points, gt_boxes
```

### 5.3 Synchronized Multi-Modal Augmentation

```python
class SynchronizedAugmentation:
    """
    Augmentations that must be applied consistently across both modalities.
    
    When transforming the 3D scene (rotation, flip, scale), both camera
    and radar data must be transformed together.
    """
    def __init__(self, config):
        self.config = config
    
    def __call__(self, sample):
        """
        Apply synchronized augmentations to the full multi-modal sample.
        """
        # BEV-level augmentations (applied to both modalities)
        if self.config.bev_rotation:
            angle = np.random.uniform(-22.5, 22.5) * np.pi / 180
            sample = self._rotate_sample(sample, angle)
        
        if self.config.bev_flip:
            if np.random.random() < 0.5:
                sample = self._flip_sample_x(sample)  # Left-right flip
        
        if self.config.bev_scale:
            scale = np.random.uniform(0.9, 1.1)
            sample = self._scale_sample(sample, scale)
        
        return sample
```

### 5.4 Augmentation Configuration

| Augmentation | Probability | Range | Applied To |
|-------------|-------------|-------|------------|
| Random resize | 1.0 | [0.9, 1.1] | Camera |
| Random crop | 1.0 | Target size | Camera |
| Color jitter | 0.5 | brightness=0.2 | Camera |
| Horizontal flip | 0.5 | - | Both (synchronized) |
| Global rotation | 1.0 | [-22.5, 22.5]° | Both (synchronized) |
| Global scale | 1.0 | [0.95, 1.05] | Both (synchronized) |
| Global translation | 0.5 | [-0.5, 0.5]m | Radar + GT |
| Point dropout | 0.3 | keep_prob=[0.8, 1.0] | Radar |
| Gaussian noise | 0.5 | σ_pos=0.1m, σ_vel=0.05m/s | Radar |
| GT paste | 0.5 | max 3 objects | Radar + GT |
| PhotoMetric distortion | 0.5 | Auto | Camera |

---

## 6. Training Curriculum and Warm-Up Strategies

### 6.1 Training Phases

CRAFT training follows a multi-phase curriculum:

```
Phase 1: Backbone Warm-Up (Epochs 0-2)
├── Camera backbone: Frozen (pre-trained weights preserved)
├── Radar branch: Training with higher LR
├── SCFT: Training
└── Detection head: Training

Phase 2: Joint Fine-Tuning (Epochs 2-15)
├── Camera backbone: Unfrozen, low LR (2e-5)
├── Radar branch: Normal LR (2e-4)
├── SCFT: Normal LR (2e-4)
└── Detection head: Normal LR (2e-4)

Phase 3: Convergence (Epochs 15-20)
├── All modules: Decaying LR (cosine schedule)
├── Augmentation: Reduced (disable color jitter, reduce rotation range)
└── EMA: Applied for final model
```

### 6.2 Backbone Pre-Training Strategy

```python
class BackboneWarmUpScheduler:
    """
    Manages the backbone unfreezing schedule.
    
    Strategy:
    1. First N epochs: backbone frozen (only train new modules)
    2. Gradual unfreezing: unfreeze from deeper layers to shallow
    3. Lower learning rate for backbone throughout
    """
    def __init__(self, model, freeze_epochs=2, unfreeze_schedule=None):
        self.model = model
        self.freeze_epochs = freeze_epochs
        self.unfreeze_schedule = unfreeze_schedule or {
            2: ['layer4'],      # Unfreeze deepest first
            3: ['layer3'],
            4: ['layer2'],
            5: ['layer1', 'conv1'],  # Unfreeze shallowest last
        }
        
        # Initially freeze entire backbone
        self._freeze_backbone()
    
    def _freeze_backbone(self):
        for param in self.model.camera_backbone.parameters():
            param.requires_grad = False
    
    def step(self, epoch):
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            for layer_name in layers_to_unfreeze:
                layer = getattr(self.model.camera_backbone, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"Epoch {epoch}: Unfroze {layer_name}")
```

### 6.3 Exponential Moving Average (EMA)

```python
class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model weights that is updated
    as an exponential moving average of the training weights.
    Used for evaluation (more stable than final training weights).
    """
    def __init__(self, model, decay=0.9999, warmup_steps=2000):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def update(self, model):
        self.step_count += 1
        
        # Ramp up decay during warm-up
        decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        
        with torch.no_grad():
            for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)
```

### 6.4 Curriculum for Radar Sparsity

```python
class RadarSweepCurriculum:
    """
    Gradually increases the number of accumulated radar sweeps during training.
    
    Rationale: Starting with fewer sweeps forces the model to handle extreme
    sparsity early, then more sweeps provide denser supervision later.
    """
    def __init__(self, start_sweeps=1, max_sweeps=6, ramp_epochs=5):
        self.start_sweeps = start_sweeps
        self.max_sweeps = max_sweeps
        self.ramp_epochs = ramp_epochs
    
    def get_num_sweeps(self, epoch):
        if epoch >= self.ramp_epochs:
            return self.max_sweeps
        
        progress = epoch / self.ramp_epochs
        sweeps = int(self.start_sweeps + 
                    (self.max_sweeps - self.start_sweeps) * progress)
        return max(self.start_sweeps, sweeps)
```

---

## 7. Training Script Configuration

### 7.1 Complete Training Configuration

```yaml
# config/craft_nusc_default.yaml

# Model
model:
  type: "CRAFT"
  camera_backbone: "resnet50"
  camera_pretrained: "imagenet"
  fpn_channels: 256
  radar_pillar_channels: 64
  radar_sparse_layers: [3, 5, 5]
  scft_layers: 6
  scft_heads: 8
  scft_dim: 256
  num_classes: 10
  bev_size: [256, 256]
  bev_range: [-51.2, -51.2, 51.2, 51.2]

# Data
data:
  dataset: "nuScenes"
  version: "v1.0-trainval"
  root: "/data/nuscenes/"
  train_split: "train"
  val_split: "val"
  image_size: [256, 704]
  radar_sweeps: 6
  radar_features: ["x", "y", "z", "vx_comp", "vy_comp", "rcs", "timestamp"]
  max_radar_points: 2048

# Training
training:
  epochs: 20
  batch_size: 4         # Per GPU
  num_workers: 4        # DataLoader workers per GPU
  gpus: 8
  sync_bn: true         # Synchronized batch norm across GPUs
  amp: true             # Automatic mixed precision
  clip_grad_norm: 35.0  # Gradient clipping

# Optimizer
optimizer:
  type: "AdamW"
  base_lr: 2.0e-4
  backbone_lr: 2.0e-5
  weight_decay: 0.01
  betas: [0.9, 0.999]

# LR Schedule
lr_schedule:
  type: "cosine"
  warmup_iters: 500
  warmup_type: "linear"
  min_lr_ratio: 1.0e-3

# Loss
loss:
  heatmap_weight: 1.0
  offset_weight: 1.0
  height_weight: 0.25
  size_weight: 0.25
  rotation_weight: 1.0
  velocity_weight: 0.25
  camera_aux_weight: 0.25
  radar_aux_weight: 0.25

# Augmentation
augmentation:
  # Camera
  random_resize: [0.9, 1.1]
  random_crop: true
  color_jitter: true
  photo_metric_distortion: true
  # Radar
  random_rotation: [-22.5, 22.5]  # degrees
  random_scale: [0.95, 1.05]
  random_flip: true
  radar_dropout: 0.3
  point_noise: true
  gt_paste: true

# EMA
ema:
  enabled: true
  decay: 0.9999
  warmup_steps: 2000

# Logging
logging:
  log_interval: 50      # Log every N iterations
  eval_interval: 1      # Evaluate every N epochs
  save_interval: 1      # Save checkpoint every N epochs
  tensorboard: true
  wandb: false
```

### 7.2 Launch Command

```bash
# Single-node multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    tools/train.py \
    --config config/craft_nusc_default.yaml \
    --work_dir work_dirs/craft_r50_20e \
    --seed 42

# Multi-node training (2 nodes x 8 GPUs)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    tools/train.py \
    --config config/craft_nusc_default.yaml \
    --work_dir work_dirs/craft_r50_20e_2node
```

### 7.3 Monitoring and Debugging

**Key metrics to monitor during training:**

| Metric | Healthy Range | Action if Abnormal |
|--------|--------------|-------------------|
| Total loss | Decreasing, 5.0 -> 1.0 | Check LR, data loading |
| Heatmap loss | 1.0 -> 0.1 | Verify target generation |
| Camera gradient norm | 0.1 - 10.0 | Adjust backbone LR |
| Radar gradient norm | 0.1 - 10.0 | Check pillar encoding |
| SCFT gradient norm | 0.1 - 5.0 | Reduce LR if exploding |
| Learning rate | Per schedule | Verify scheduler |
| GPU memory | < 90% capacity | Reduce batch/image size |
| Training speed | > 2 iter/s (8xA100) | Check data pipeline |
| Val mAP | Increasing after epoch 3 | Patient; check augmentation |

**Common Training Issues and Solutions:**

| Issue | Symptom | Solution |
|-------|---------|----------|
| NaN loss | Loss becomes NaN | Reduce LR, increase grad clip, check data |
| Mode collapse | Heatmap predicts uniform values | Increase focal loss alpha, verify targets |
| Camera dominance | Radar branch gradients vanish | Increase radar aux loss weight |
| Slow convergence | mAP not improving after 5 epochs | Verify augmentation, check data loading |
| OOM | CUDA out of memory | Reduce BEV size, use gradient checkpointing |
| Poor velocity | High velocity error | Verify radar ego-compensation, check vel loss |
