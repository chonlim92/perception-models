# Model Architecture: CenterPoint

## Architecture Overview

CenterPoint follows a single-stage (or optionally two-stage) detection pipeline that processes raw LiDAR point clouds and outputs 3D bounding boxes with tracking information.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CenterPoint Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Point Cloud ──→ Voxelization ──→ 3D Backbone ──→ BEV Collapse     │
│                                                                     │
│  BEV Collapse ──→ 2D Backbone ──→ Center Head ──→ Detections       │
│                                                                     │
│  [Optional] Detections ──→ Two-Stage Refinement ──→ Final Boxes    │
│                                                                     │
│  Final Boxes ──→ Tracker ──→ Tracked Objects                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Voxelization

### Dynamic Voxelization

Unlike hard voxelization (which caps points per voxel), dynamic voxelization processes all points without truncation:

```python
class DynamicVoxelization:
    """
    Dynamic voxelization: assigns each point to a voxel without 
    limiting the number of points per voxel. All points contribute
    to voxel features via scatter-mean operations.
    """
    def __init__(self, voxel_size, point_cloud_range):
        self.voxel_size = voxel_size      # [0.075, 0.075, 0.2]
        self.pc_range = point_cloud_range  # [-54, -54, -5, 54, 54, 3]
        
        # Grid dimensions
        self.grid_size = [
            int((pc_range[3] - pc_range[0]) / voxel_size[0]),  # 1440
            int((pc_range[4] - pc_range[1]) / voxel_size[1]),  # 1440
            int((pc_range[5] - pc_range[2]) / voxel_size[2]),  # 40
        ]
    
    def forward(self, points):
        """
        Args:
            points: [N, 5] (x, y, z, intensity, time_lag)
        Returns:
            voxel_features: [M, C] mean features per voxel
            voxel_coords: [M, 4] (batch_id, z, y, x) integer coordinates
        """
        # Compute voxel indices for each point
        coords = torch.floor((points[:, :3] - self.pc_range[:3]) / self.voxel_size)
        coords = coords.long()  # [N, 3] integer voxel coordinates
        
        # Filter out-of-range points
        valid = (coords >= 0).all(dim=1) & (coords < self.grid_size).all(dim=1)
        points = points[valid]
        coords = coords[valid]
        
        # Unique voxel identification
        voxel_ids = coords[:, 0] * (self.grid_size[1] * self.grid_size[2]) + \
                    coords[:, 1] * self.grid_size[2] + coords[:, 2]
        
        # Scatter mean: average features of all points in same voxel
        unique_ids, inverse = torch.unique(voxel_ids, return_inverse=True)
        voxel_features = scatter_mean(points, inverse, dim=0)
        
        # Get unique coordinates
        voxel_coords = scatter_mean(coords.float(), inverse, dim=0).long()
        
        return voxel_features, voxel_coords
```

### Mean Feature Encoding

For each occupied voxel, the feature is the mean of all point features within it:

- Input features per point: [x, y, z, intensity, time_lag] (5 channels)
- Additional features: [x - x_mean, y - y_mean, z - z_mean] (offset from voxel center)
- Final voxel feature dimension: 5 or 8 channels (depending on variant)

---

## 2. 3D Sparse Convolutional Backbone

### Architecture

The 3D backbone processes the sparse voxel representation using spconv (Sparse Convolution library):

```
Input: Sparse Tensor [M voxels, C_in channels, grid (1440, 1440, 40)]
       ↓
Stage 1: SubMConv3d(C_in, 16, kernel=3) × 2, stride=1
       Output: [M₁, 16, (1440, 1440, 40)]
       ↓
Stage 2: SparseConv3d(16, 32, kernel=3, stride=2) + SubMConv3d(32, 32, k=3) × 2
       Output: [M₂, 32, (720, 720, 20)]
       ↓
Stage 3: SparseConv3d(32, 64, kernel=3, stride=2) + SubMConv3d(64, 64, k=3) × 2
       Output: [M₃, 64, (360, 360, 10)]
       ↓
Stage 4: SparseConv3d(64, 128, kernel=3, stride=2) + SubMConv3d(128, 128, k=3) × 2
       Output: [M₄, 128, (180, 180, 5)]
       ↓
Final: SparseConv3d(128, 128, kernel=(3,1,1), stride=(2,1,1))
       Output: [M₅, 128, (180, 180, 2)]
```

### Sparse Convolution Types

#### SubMConv3d (Submanifold Sparse Convolution)

- Only computes output at locations where the input is active (non-zero).
- Preserves the sparsity pattern: output has the same set of active locations as input.
- Efficient for processing sparse data without "dilating" the active set.

```python
# SubMConv3d: output active sites = input active sites
# No new voxels are activated
SubMConv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
```

#### SparseConv3d (Regular Sparse Convolution)

- Computes output at all locations within the kernel's receptive field of any active input.
- Can "grow" the active set (new voxels become active).
- Used with stride > 1 for downsampling.

```python
# SparseConv3d with stride: downsamples spatial dimensions
SparseConv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
```

### Detailed Layer Configuration

```python
class SparseBEVBackbone(nn.Module):
    """3D Sparse Convolutional Backbone for CenterPoint."""
    
    def __init__(self):
        # Stage 1: No downsampling
        self.stage1 = spconv.SparseSequential(
            SubMConv3d(5, 16, 3, padding=1, bias=False),
            BatchNorm1d(16), ReLU(),
            SubMConv3d(16, 16, 3, padding=1, bias=False),
            BatchNorm1d(16), ReLU(),
        )
        
        # Stage 2: 2x downsampling
        self.stage2 = spconv.SparseSequential(
            SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False),
            BatchNorm1d(32), ReLU(),
            SubMConv3d(32, 32, 3, padding=1, bias=False),
            BatchNorm1d(32), ReLU(),
            SubMConv3d(32, 32, 3, padding=1, bias=False),
            BatchNorm1d(32), ReLU(),
        )
        
        # Stage 3: 4x downsampling (cumulative)
        self.stage3 = spconv.SparseSequential(
            SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False),
            BatchNorm1d(64), ReLU(),
            SubMConv3d(64, 64, 3, padding=1, bias=False),
            BatchNorm1d(64), ReLU(),
            SubMConv3d(64, 64, 3, padding=1, bias=False),
            BatchNorm1d(64), ReLU(),
        )
        
        # Stage 4: 8x downsampling (cumulative)
        self.stage4 = spconv.SparseSequential(
            SparseConv3d(64, 128, 3, stride=2, padding=1, bias=False),
            BatchNorm1d(128), ReLU(),
            SubMConv3d(128, 128, 3, padding=1, bias=False),
            BatchNorm1d(128), ReLU(),
            SubMConv3d(128, 128, 3, padding=1, bias=False),
            BatchNorm1d(128), ReLU(),
        )
        
        # Final Z-compression layer
        self.z_compress = spconv.SparseSequential(
            SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            BatchNorm1d(128), ReLU(),
        )
```

---

## 3. BEV Feature Collapse

### Reshaping 3D Features to 2D

After the 3D backbone, the sparse 3D feature volume is converted to a dense 2D BEV feature map:

```python
class BEVCollapse(nn.Module):
    """
    Collapse the Z dimension by concatenating features along height.
    
    Input: Sparse 3D tensor [M, 128, (180, 180, 2)]
    Output: Dense 2D tensor [B, 256, 180, 180]
    """
    def forward(self, sparse_3d):
        # Convert sparse to dense: [B, C, Z, Y, X] = [B, 128, 2, 180, 180]
        dense_3d = sparse_3d.dense()
        
        B, C, Z, H, W = dense_3d.shape
        # Reshape: concatenate Z slices along channel dimension
        # [B, 128, 2, 180, 180] -> [B, 256, 180, 180]
        bev_features = dense_3d.reshape(B, C * Z, H, W)
        
        return bev_features  # [B, 256, 180, 180]
```

### Spatial Dimensions After Collapse

| Stage | Spatial Dimensions | Channels | Notes |
|-------|-------------------|----------|-------|
| Input voxel grid | 1440 x 1440 x 40 | 5 | Raw voxel features |
| After Stage 4 | 180 x 180 x 5 | 128 | 8x spatial downsample |
| After Z-compress | 180 x 180 x 2 | 128 | Additional Z downsample |
| After BEV collapse | 180 x 180 | 256 | Z concatenated to channels |

---

## 4. 2D Backbone (BEV Feature Refinement)

### Architecture: ResNet-like with Deconvolution

```python
class BEV2DBackbone(nn.Module):
    """
    2D backbone for BEV feature refinement.
    Two stages with deconvolution for upsampling.
    """
    def __init__(self):
        # Stage 1: stride 1, maintain resolution
        self.stage1 = nn.Sequential(
            # 5 residual blocks at 180x180
            ConvBlock(256, 128, stride=1),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
        )
        
        # Stage 2: stride 2, downsample then upsample
        self.stage2 = nn.Sequential(
            ConvBlock(128, 256, stride=2),   # 180x180 -> 90x90
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        
        # Deconvolution (transposed conv) for upsampling
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=1, stride=1),  # 180x180
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 90x90 -> 180x180
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
    
    def forward(self, bev_features):
        """
        Args:
            bev_features: [B, 256, 180, 180]
        Returns:
            fused_features: [B, 256, 180, 180]
        """
        x1 = self.stage1(bev_features)   # [B, 128, 180, 180]
        x2 = self.stage2(x1)             # [B, 256, 90, 90]
        
        up1 = self.deconv1(x1)           # [B, 128, 180, 180]
        up2 = self.deconv2(x2)           # [B, 128, 180, 180]
        
        # Concatenate multi-scale features
        fused = torch.cat([up1, up2], dim=1)  # [B, 256, 180, 180]
        
        return fused
```

### Feature Map Resolution

The final BEV feature map has resolution 180 x 180, corresponding to:
- Physical coverage: 108m x 108m ([-54, 54] range)
- Each pixel: 0.6m x 0.6m (= 0.075m voxel * 8x downsample)

---

## 5. Center Head (Detection Head)

### Separate Heads per Class Group

CenterPoint uses task-specific heads for different class groups to avoid negative transfer:

```python
# nuScenes class groups (6 heads)
CLASS_GROUPS = [
    ['car'],                                    # Head 0
    ['truck', 'construction_vehicle'],          # Head 1
    ['bus', 'trailer'],                         # Head 2
    ['barrier'],                                # Head 3
    ['motorcycle', 'bicycle'],                  # Head 4
    ['pedestrian', 'traffic_cone'],             # Head 5
]
```

### Head Architecture

Each class group has an independent detection head:

```python
class CenterHead(nn.Module):
    """Detection head for one class group."""
    
    def __init__(self, in_channels=256, num_classes=1, head_channels=64):
        # Shared feature extraction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(),
        )
        
        # Heatmap head (num_classes channels)
        self.heatmap = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(),
            nn.Conv2d(head_channels, num_classes, 1),  # Final prediction
        )
        
        # Regression heads
        self.offset = self._make_head(head_channels, 2)    # dx, dy
        self.height = self._make_head(head_channels, 1)    # z
        self.size = self._make_head(head_channels, 3)      # log(w), log(l), log(h)
        self.rotation = self._make_head(head_channels, 2)  # sin(yaw), cos(yaw)
        self.velocity = self._make_head(head_channels, 2)  # vx, vy
    
    def _make_head(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
        )
    
    def forward(self, bev_features):
        shared = self.shared_conv(bev_features)
        
        return {
            'heatmap': torch.sigmoid(self.heatmap(shared)),
            'offset': self.offset(shared),
            'height': self.height(shared),
            'size': self.size(shared),
            'rotation': self.rotation(shared),
            'velocity': self.velocity(shared),
        }
```

### Heatmap: Gaussian Focal Loss

```python
def gaussian_focal_loss(pred, target, alpha=2.0, beta=4.0):
    """
    Modified focal loss for Gaussian heatmap targets.
    
    Unlike standard focal loss where targets are 0/1, here targets are
    continuous Gaussians [0, 1]. The loss reduces penalty for predictions
    near (but not exactly at) ground truth centers.
    
    Args:
        pred: [B, C, H, W] predicted heatmap (after sigmoid)
        target: [B, C, H, W] ground truth Gaussian heatmap
        alpha: focusing parameter for positive locations
        beta: focusing parameter for negative locations (near-center suppression)
    """
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    
    # Positive loss (at exact center)
    pos_loss = -torch.log(pred + 1e-12) * torch.pow(1 - pred, alpha) * pos_mask
    
    # Negative loss (everywhere else, reduced near centers)
    neg_loss = -torch.log(1 - pred + 1e-12) * torch.pow(pred, alpha) * \
               torch.pow(1 - target, beta) * neg_mask
    
    loss = (pos_loss.sum() + neg_loss.sum()) / max(pos_mask.sum(), 1)
    return loss
```

### Regression Outputs Detail

| Output | Channels | Encoding | Loss | Weight |
|--------|----------|----------|------|--------|
| Heatmap | num_classes | sigmoid -> [0,1] | Gaussian Focal Loss | 1.0 |
| Offset | 2 | (dx, dy) sub-pixel | L1 Loss | 2.0 |
| Height | 1 | absolute z (meters) | L1 Loss | 2.0 |
| Size | 3 | (log w, log l, log h) | L1 Loss | 0.2 |
| Rotation | 2 | (sin θ, cos θ) | L1 Loss | 1.0 |
| Velocity | 2 | (vx, vy) m/s | L1 Loss | 0.2 |

**Note:** Regression losses are only computed at ground truth center locations (positive positions in the heatmap).

---

## 6. Two-Stage Refinement

### Motivation

The first stage predicts boxes from a single BEV location (the center). For large objects, this may miss important contextual features from the object's extent. The second stage extracts features from multiple locations on the predicted box.

### Feature Extraction

```python
class SecondStageFeatureExtractor(nn.Module):
    """Extract BEV features at predicted box locations for refinement."""
    
    def __init__(self, num_point_features=5):
        # Feature points: center + 4 face centers
        self.num_points = num_point_features  # center, front, back, left, right
    
    def get_sample_points(self, boxes):
        """
        Get feature sampling locations from predicted boxes.
        
        Args:
            boxes: [N, 7] (x, y, z, w, l, h, yaw)
        Returns:
            points: [N, 5, 2] BEV coordinates for bilinear sampling
        """
        cx, cy = boxes[:, 0], boxes[:, 1]
        w, l = boxes[:, 3], boxes[:, 4]
        yaw = boxes[:, 6]
        
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        # 5 sampling points in local box frame
        # Center: (0, 0)
        # Front:  (l/2, 0)
        # Back:   (-l/2, 0)
        # Left:   (0, w/2)
        # Right:  (0, -w/2)
        offsets_local = torch.stack([
            torch.zeros_like(cx), torch.zeros_like(cy),      # center
            l/2, torch.zeros_like(cy),                       # front
            -l/2, torch.zeros_like(cy),                      # back
            torch.zeros_like(cx), w/2,                       # left
            torch.zeros_like(cx), -w/2,                      # right
        ]).reshape(-1, 5, 2)
        
        # Rotate to global frame
        rot_matrix = torch.stack([cos_yaw, -sin_yaw, sin_yaw, cos_yaw], dim=1).reshape(-1, 2, 2)
        offsets_global = torch.bmm(offsets_local, rot_matrix.transpose(1, 2))
        
        # Add center offset
        points = offsets_global + torch.stack([cx, cy], dim=1).unsqueeze(1)
        
        return points
    
    def forward(self, bev_features, boxes):
        """
        Args:
            bev_features: [B, C, H, W] BEV feature map
            boxes: [N, 7] predicted boxes from first stage
        Returns:
            point_features: [N, 5*C] concatenated features
        """
        sample_points = self.get_sample_points(boxes)  # [N, 5, 2]
        
        # Bilinear interpolation at sample points
        # Convert world coordinates to normalized grid coordinates
        grid = self.world_to_grid(sample_points)  # [N, 5, 2] normalized to [-1, 1]
        
        # Sample features
        features = F.grid_sample(bev_features, grid)  # [B, C, N, 5]
        
        # Reshape to [N, 5*C]
        point_features = features.permute(2, 3, 1).reshape(len(boxes), -1)
        
        return point_features
```

### MLP Refinement Network

```python
class SecondStageMLP(nn.Module):
    """MLP for refining first-stage predictions."""
    
    def __init__(self, in_features=5*256, hidden_dim=256):
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Output heads
        self.cls_head = nn.Linear(hidden_dim, 1)   # confidence adjustment
        self.reg_head = nn.Linear(hidden_dim, 7)   # box residuals (dx,dy,dz,dw,dl,dh,dyaw)
    
    def forward(self, point_features):
        """
        Args:
            point_features: [N, 5*C] features from extraction
        Returns:
            cls_scores: [N, 1] refined confidence scores
            box_residuals: [N, 7] refinement deltas
        """
        hidden = self.mlp(point_features)
        cls_scores = torch.sigmoid(self.cls_head(hidden))
        box_residuals = self.reg_head(hidden)
        return cls_scores, box_residuals
```

### Two-Stage Training

- **IoU target:** Compute IoU between first-stage predictions and ground truth. Assign positive (IoU > 0.5) and negative (IoU < 0.25) for classification.
- **Regression target:** Residuals between first-stage predictions and matched ground truth boxes.
- **Loss:** Binary cross-entropy for classification, smooth-L1 for regression.

---

## 7. Tracker: Greedy Center-Distance Matching

### Algorithm

```python
class CenterPointTracker:
    """
    Simple online tracker using greedy closest-point matching
    with velocity-based motion prediction.
    """
    
    def __init__(self, max_age=3, dist_threshold=4.0):
        self.max_age = max_age           # Frames before deleting lost track
        self.dist_threshold = dist_threshold  # Max matching distance (meters)
        self.tracks = []                 # Active tracks
        self.next_id = 0                 # Next available track ID
    
    def update(self, detections, dt):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of dicts with keys:
                'center': [x, y] BEV center
                'velocity': [vx, vy] predicted velocity
                'box': full box parameters
            dt: time since last frame (seconds)
        
        Returns:
            tracked_objects: List of detections with assigned track IDs
        """
        if len(self.tracks) == 0:
            # First frame: initialize all as new tracks
            return self._init_tracks(detections)
        
        # Step 1: Predict current positions of existing tracks using velocity
        predicted_centers = []
        for track in self.tracks:
            pred_x = track['center'][0] + track['velocity'][0] * dt
            pred_y = track['center'][1] + track['velocity'][1] * dt
            predicted_centers.append([pred_x, pred_y])
        predicted_centers = np.array(predicted_centers)
        
        # Step 2: Compute distance matrix
        det_centers = np.array([d['center'] for d in detections])
        
        # L2 distance between predicted track positions and new detections
        dist_matrix = cdist(predicted_centers, det_centers)  # [num_tracks, num_dets]
        
        # Step 3: Greedy matching (closest first)
        matched_tracks = set()
        matched_dets = set()
        assignments = {}
        
        # Sort all pairs by distance
        pairs = []
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                if dist_matrix[i, j] < self.dist_threshold:
                    pairs.append((dist_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0])
        
        for dist, track_idx, det_idx in pairs:
            if track_idx in matched_tracks or det_idx in matched_dets:
                continue
            assignments[det_idx] = track_idx
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
        
        # Step 4: Update matched tracks
        tracked_objects = []
        for det_idx, track_idx in assignments.items():
            track = self.tracks[track_idx]
            track['center'] = detections[det_idx]['center']
            track['velocity'] = detections[det_idx]['velocity']
            track['box'] = detections[det_idx]['box']
            track['age'] = 0
            tracked_objects.append({**detections[det_idx], 'track_id': track['id']})
        
        # Step 5: Create new tracks for unmatched detections
        for det_idx in range(len(detections)):
            if det_idx not in matched_dets:
                new_track = {
                    'id': self.next_id,
                    'center': detections[det_idx]['center'],
                    'velocity': detections[det_idx]['velocity'],
                    'box': detections[det_idx]['box'],
                    'age': 0,
                }
                self.tracks.append(new_track)
                tracked_objects.append({**detections[det_idx], 'track_id': self.next_id})
                self.next_id += 1
        
        # Step 6: Age and remove lost tracks
        for track_idx in range(len(self.tracks)):
            if track_idx not in matched_tracks:
                self.tracks[track_idx]['age'] += 1
        
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        
        return tracked_objects
```

### Tracker Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_age | 3 | Delete track after 3 frames without match |
| dist_threshold | 4.0 m | Maximum association distance |
| velocity_weight | 1.0 | Weight for velocity prediction in center extrapolation |
| min_hits | 1 | Minimum detections before track is confirmed |

### Key Design Choices

1. **No appearance features:** Unlike vision-based trackers, LiDAR tracking relies purely on geometric proximity and motion prediction.
2. **Greedy vs. Hungarian:** Greedy matching is simpler and works well because velocity prediction makes the assignment problem nearly unambiguous.
3. **Velocity from detection head:** The velocity predicted by the detection network (from multi-sweep input) provides strong motion cues without requiring separate Kalman filtering.

---

## Full Model Summary

### Parameter Count (nuScenes voxel variant)

| Component | Parameters | Notes |
|-----------|-----------|-------|
| 3D Backbone | ~2.5M | Sparse conv is parameter-efficient |
| 2D Backbone | ~5.2M | ResNet-style with deconv |
| Center Heads (x6) | ~1.8M | Lightweight per-group heads |
| Two-Stage MLP | ~0.5M | Optional refinement |
| **Total** | **~10M** | Relatively lightweight |

### Inference Speed

| Configuration | Device | FPS | Latency |
|--------------|--------|-----|---------|
| Voxel (no 2nd stage) | A100 | ~16 | 62 ms |
| Voxel (with 2nd stage) | A100 | ~11 | 91 ms |
| Pillar (no 2nd stage) | A100 | ~25 | 40 ms |
| Voxel (no 2nd stage) | RTX 3090 | ~12 | 83 ms |

### Memory Usage (Training)

| Batch Size | GPU Memory | Notes |
|-----------|------------|-------|
| 1 | ~8 GB | Minimum for debugging |
| 4 | ~24 GB | Standard training (A100 40GB) |
| 8 | ~45 GB | Large batch (A100 80GB) |
