# Evaluation Guide: CenterPoint Metrics and Benchmarks

## Overview

CenterPoint is evaluated on two primary benchmarks: nuScenes and Waymo Open Dataset. Each has distinct evaluation protocols, metrics, and submission requirements. This guide details the full evaluation pipeline for both detection and tracking.

---

## nuScenes Detection Metrics

### nuScenes Detection Score (NDS)

NDS is the primary ranking metric on the nuScenes leaderboard. It combines mean Average Precision (mAP) with five True Positive (TP) metrics that measure localization quality:

```
NDS = (1/10) * [5 * mAP + sum(TP_metrics)]

NDS = 0.5 * mAP + 0.1 * (mATE + mASE + mAOE + mAVE + mAAE)
     where each TP metric is (1 - min(metric, 1.0))
```

**Note:** Each TP metric is inverted and capped so that lower error = higher score:
```python
TP_score = max(0, 1.0 - TP_error)
NDS = (5 * mAP + sum([TP_score_i for i in range(5)])) / 10
```

### Mean Average Precision (mAP)

nuScenes uses a **BEV center distance** matching criterion instead of 3D IoU:

```python
def compute_nuscenes_mAP(predictions, ground_truths, dist_thresholds=[0.5, 1.0, 2.0, 4.0]):
    """
    Compute mAP using BEV center distance matching.
    
    A prediction is matched to a ground truth if the BEV (x,y) center 
    distance is below the threshold. This is different from KITTI/Waymo
    which use 3D IoU matching.
    
    Args:
        predictions: list of predicted boxes per frame
        ground_truths: list of GT boxes per frame  
        dist_thresholds: [0.5, 1.0, 2.0, 4.0] meters
    
    Returns:
        mAP: mean AP across all classes and thresholds
    """
    APs = {}
    
    for class_name in CLASSES:
        class_preds = filter_by_class(predictions, class_name)
        class_gts = filter_by_class(ground_truths, class_name)
        
        ap_per_threshold = []
        for dist_thresh in dist_thresholds:
            # Match predictions to GT by BEV center distance
            matches = match_by_center_distance(class_preds, class_gts, dist_thresh)
            
            # Compute precision-recall curve
            precision, recall = compute_pr_curve(matches, class_preds, class_gts)
            
            # AP = area under PR curve (40-point interpolation)
            ap = compute_ap(precision, recall, num_points=40)
            ap_per_threshold.append(ap)
        
        # Average across thresholds for this class
        APs[class_name] = np.mean(ap_per_threshold)
    
    # Mean across all classes
    mAP = np.mean(list(APs.values()))
    return mAP
```

### Distance Thresholds

| Threshold | Strictness | Primarily affects |
|-----------|-----------|-------------------|
| 0.5 m | Very strict | Nearby, well-visible objects |
| 1.0 m | Strict | Standard objects |
| 2.0 m | Moderate | Farther objects, large vehicles |
| 4.0 m | Lenient | Distant or partially occluded objects |

### True Positive (TP) Metrics

TP metrics are computed only on matched (true positive) detections, measuring various aspects of localization quality:

| Metric | Abbreviation | Unit | Description |
|--------|-------------|------|-------------|
| Mean Translation Error | mATE | meters | Euclidean center distance (2D BEV) |
| Mean Scale Error | mASE | 1 - IoU | 1 minus 3D IoU after alignment |
| Mean Orientation Error | mAOE | radians | Smallest yaw angle difference |
| Mean Velocity Error | mAVE | m/s | Absolute velocity error |
| Mean Attribute Error | mAAE | 1 - accuracy | 1 minus attribute classification accuracy |

```python
def compute_tp_metrics(matched_predictions, matched_ground_truths):
    """Compute the 5 TP metrics on matched detection pairs."""
    
    # ATE: Average Translation Error (BEV center distance)
    ate = np.mean([
        np.linalg.norm(pred[:2] - gt[:2]) 
        for pred, gt in zip(matched_predictions, matched_ground_truths)
    ])
    
    # ASE: Average Scale Error (1 - IoU after center alignment)
    ase = np.mean([
        1.0 - compute_3d_iou_aligned(pred, gt)
        for pred, gt in zip(matched_predictions, matched_ground_truths)
    ])
    
    # AOE: Average Orientation Error (smallest angle difference)
    aoe = np.mean([
        min(abs(pred_yaw - gt_yaw), 2*np.pi - abs(pred_yaw - gt_yaw))
        for pred_yaw, gt_yaw in zip(pred_yaws, gt_yaws)
    ])
    
    # AVE: Average Velocity Error (L2 distance of velocity vectors)
    ave = np.mean([
        np.linalg.norm(pred_vel - gt_vel)
        for pred_vel, gt_vel in zip(pred_velocities, gt_velocities)
    ])
    
    # AAE: Average Attribute Error (1 - attribute accuracy)
    aae = 1.0 - np.mean([
        pred_attr == gt_attr
        for pred_attr, gt_attr in zip(pred_attributes, gt_attributes)
    ])
    
    return {'mATE': ate, 'mASE': ase, 'mAOE': aoe, 'mAVE': ave, 'mAAE': aae}
```

---

## nuScenes Tracking Metrics

### AMOTA (Average Multi-Object Tracking Accuracy)

AMOTA is the primary tracking metric on nuScenes, extending MOTA to average over multiple recall thresholds:

```python
def compute_amota(tracking_results, ground_truths, recall_thresholds=np.linspace(0, 1, 40)):
    """
    AMOTA: Average MOTA across recall thresholds.
    
    This addresses the issue where MOTA is dominated by false positives
    at high-recall operating points.
    """
    mota_values = []
    
    for recall_thresh in recall_thresholds:
        # Filter predictions to achieve this recall level
        filtered_preds = filter_to_recall(tracking_results, recall_thresh)
        
        # Compute MOTA at this recall
        fp, fn, ids = count_errors(filtered_preds, ground_truths)
        num_gt = count_ground_truths(ground_truths)
        
        mota = 1.0 - (fp + fn + ids) / max(num_gt, 1)
        mota_values.append(max(0, mota))  # Clamp to [0, 1]
    
    amota = np.mean(mota_values)
    return amota
```

### Tracking Metrics Summary

| Metric | Description | Better |
|--------|-------------|--------|
| AMOTA | Average MOTA across recall thresholds | Higher |
| AMOTP | Average MOTP (localization accuracy of tracked objects) | Lower |
| IDS | Identity Switches (track ID changes for same object) | Lower |
| FRAG | Fragmentations (track interrupted and restarted) | Lower |
| Recall | Fraction of GT objects tracked at least once | Higher |
| MOTA | Multi-Object Tracking Accuracy (at best recall) | Higher |
| MOTP | Multi-Object Tracking Precision (localization of TPs) | Lower |

### AMOTP (Average Multi-Object Tracking Precision)

```python
def compute_amotp(tracking_results, ground_truths):
    """
    AMOTP: Average localization precision of tracked objects.
    Computed as the average BEV distance between matched track 
    predictions and ground truths.
    """
    distances = []
    for frame in frames:
        matches = match_tracks_to_gt(tracking_results[frame], ground_truths[frame])
        for pred, gt in matches:
            dist = np.linalg.norm(pred['center'][:2] - gt['center'][:2])
            distances.append(dist)
    
    amotp = np.mean(distances) if distances else float('inf')
    return amotp
```

### Identity Switches (IDS)

An identity switch occurs when a ground truth object is tracked with one ID in frame t and a different ID in frame t+1:

```python
def count_identity_switches(track_assignments):
    """
    Count identity switches across all frames.
    
    An IDS occurs when gt_object_i is matched to track_a in frame t
    but matched to track_b (b != a) in frame t+1.
    """
    ids_count = 0
    prev_assignment = {}  # gt_id -> track_id
    
    for frame in sorted(frames):
        curr_assignment = get_assignments(frame)  # gt_id -> track_id
        
        for gt_id, track_id in curr_assignment.items():
            if gt_id in prev_assignment:
                if prev_assignment[gt_id] != track_id:
                    ids_count += 1
        
        prev_assignment = curr_assignment
    
    return ids_count
```

### Fragmentation (FRAG)

A fragmentation occurs when a ground truth track is interrupted (not matched for one or more frames) and then resumes:

```python
def count_fragmentations(track_assignments):
    """
    Count fragmentations: gaps in tracking of a GT object.
    
    A FRAG occurs when gt_object_i is tracked, then untracked for 
    one or more frames, then tracked again (possibly with same or different ID).
    """
    frag_count = 0
    gt_tracking_state = {}  # gt_id -> 'tracked' | 'lost'
    
    for frame in sorted(frames):
        curr_tracked = set(get_matched_gt_ids(frame))
        
        for gt_id in all_gt_ids:
            prev_state = gt_tracking_state.get(gt_id, 'lost')
            curr_state = 'tracked' if gt_id in curr_tracked else 'lost'
            
            if prev_state == 'lost' and curr_state == 'tracked':
                if gt_id in gt_tracking_state:  # Was tracked before
                    frag_count += 1
            
            gt_tracking_state[gt_id] = curr_state
    
    return frag_count
```

---

## Waymo Open Dataset Metrics

### Average Precision with Heading (APH)

Waymo uses APH as its primary metric, which is AP weighted by heading accuracy:

```python
def compute_aph(predictions, ground_truths, iou_threshold):
    """
    APH = AP * heading_accuracy_weight
    
    A prediction is a true positive if:
    1. 3D IoU with a GT box exceeds the threshold
    2. The heading similarity is used to weight the TP
    
    Heading similarity: s = (1 + cos(angle_diff)) / 2
    """
    matches = match_by_3d_iou(predictions, ground_truths, iou_threshold)
    
    # Weight each TP by heading accuracy
    for match in matches:
        angle_diff = abs(match.pred_yaw - match.gt_yaw)
        heading_sim = (1 + np.cos(angle_diff)) / 2
        match.weight = heading_sim
    
    # Compute weighted AP
    aph = compute_weighted_ap(matches, predictions, ground_truths)
    return aph
```

### Difficulty Levels

| Level | Criteria | Description |
|-------|----------|-------------|
| L1 (LEVEL_1) | >= 5 LiDAR points | Well-observed objects |
| L2 (LEVEL_2) | >= 1 LiDAR point (all labeled) | Includes harder cases |

### IoU Thresholds (Waymo)

| Class | 3D IoU Threshold | BEV IoU Threshold |
|-------|------------------|-------------------|
| Vehicle | 0.7 | 0.7 |
| Pedestrian | 0.5 | 0.5 |
| Cyclist | 0.5 | 0.5 |

### Waymo Metrics Table

| Metric | Description |
|--------|-------------|
| AP (3D) | Average Precision with 3D IoU matching |
| AP (BEV) | Average Precision with BEV IoU matching |
| APH (3D) | AP weighted by heading accuracy |
| APH (BEV) | BEV AP weighted by heading accuracy |
| AP (L1) | AP on LEVEL_1 difficulty objects |
| AP (L2) | AP on LEVEL_2 difficulty objects (more inclusive) |

---

## Evaluation Pipeline

### nuScenes Evaluation

```python
def evaluate_nuscenes(model, dataloader, eval_config):
    """Full nuScenes evaluation pipeline."""
    
    # Step 1: Run inference on all validation frames
    all_predictions = []
    for batch in dataloader:
        predictions = model(batch)
        predictions = decode_predictions(predictions)  # Convert to boxes
        predictions = transform_to_global(predictions, batch['ego_pose'])
        all_predictions.extend(predictions)
    
    # Step 2: Format predictions for nuScenes eval
    submission = format_nuscenes_submission(all_predictions)
    
    # Step 3: Run official nuScenes evaluation
    from nuscenes.eval.detection.evaluate import DetectionEval
    
    eval_result = DetectionEval(
        nusc=nusc,
        config=eval_config,
        result_path=submission_path,
        eval_set='val',
        output_dir=output_dir,
    )
    metrics = eval_result.main()
    
    return metrics  # Contains NDS, mAP, per-class APs, TP metrics
```

### Waymo Evaluation

```python
def evaluate_waymo(model, dataloader):
    """Full Waymo evaluation pipeline."""
    
    # Step 1: Run inference
    all_predictions = []
    for batch in dataloader:
        predictions = model(batch)
        predictions = decode_predictions(predictions)
        all_predictions.extend(predictions)
    
    # Step 2: Convert to Waymo format (protobuf)
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.metrics.python import detection_metrics
    
    # Step 3: Compute metrics using official eval tool
    metrics = detection_metrics.compute_detection_metrics(
        predictions_proto,
        ground_truths_proto,
        config=waymo_eval_config,
    )
    
    return metrics  # Contains AP, APH at L1 and L2 for each class
```

---

## nuScenes Submission Format

### Detection Submission

```json
{
    "meta": {
        "use_camera": false,
        "use_lidar": true,
        "use_radar": false,
        "use_map": false,
        "use_external": false
    },
    "results": {
        "sample_token_1": [
            {
                "sample_token": "sample_token_1",
                "translation": [100.5, 200.3, 1.2],
                "size": [1.8, 4.5, 1.5],
                "rotation": [0.707, 0.0, 0.0, 0.707],
                "velocity": [5.2, -1.3],
                "detection_name": "car",
                "detection_score": 0.95,
                "attribute_name": "vehicle.moving"
            }
        ]
    }
}
```

### Tracking Submission

```json
{
    "meta": {
        "use_camera": false,
        "use_lidar": true,
        "use_radar": false,
        "use_map": false,
        "use_external": false
    },
    "results": {
        "sample_token_1": [
            {
                "sample_token": "sample_token_1",
                "translation": [100.5, 200.3, 1.2],
                "size": [1.8, 4.5, 1.5],
                "rotation": [0.707, 0.0, 0.0, 0.707],
                "velocity": [5.2, -1.3],
                "tracking_name": "car",
                "tracking_score": 0.95,
                "tracking_id": "track_001",
                "attribute_name": "vehicle.moving"
            }
        ]
    }
}
```

---

## Expected Performance

### CenterPoint-Voxel on nuScenes Val

| Class | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m | Mean AP |
|-------|---------|---------|---------|---------|---------|
| car | 86.2 | 89.1 | 90.5 | 91.2 | 89.3 |
| truck | 38.5 | 52.1 | 58.7 | 61.3 | 52.7 |
| bus | 47.3 | 68.2 | 72.1 | 73.5 | 65.3 |
| trailer | 15.2 | 33.8 | 44.1 | 51.2 | 36.1 |
| construction_vehicle | 7.8 | 16.5 | 22.3 | 28.1 | 18.7 |
| pedestrian | 83.5 | 86.2 | 87.1 | 87.8 | 86.2 |
| motorcycle | 52.3 | 58.7 | 60.1 | 61.5 | 58.2 |
| bicycle | 28.5 | 35.2 | 37.8 | 39.1 | 35.2 |
| traffic_cone | 68.5 | 72.1 | 73.2 | 74.1 | 72.0 |
| barrier | 60.2 | 68.5 | 72.3 | 74.8 | 68.9 |
| **Overall** | | | | | **58.3** |

### CenterPoint-Voxel on nuScenes Val (Full Metrics)

| Metric | Value |
|--------|-------|
| NDS | 66.8 |
| mAP | 58.3 |
| mATE | 0.262 m |
| mASE | 0.254 |
| mAOE | 0.358 rad |
| mAVE | 0.278 m/s |
| mAAE | 0.192 |

### CenterPoint Tracking on nuScenes Val

| Metric | Value |
|--------|-------|
| AMOTA | 63.5 |
| AMOTP | 0.555 m |
| IDS | 760 |
| FRAG | 512 |
| Recall | 71.2% |

### CenterPoint on Waymo Val

| Class | APH (L1) | APH (L2) | AP (L1) | AP (L2) |
|-------|-----------|-----------|---------|---------|
| Vehicle | 72.8 | 66.2 | 73.5 | 66.8 |
| Pedestrian | 74.1 | 62.6 | 78.3 | 66.2 |
| Cyclist | 71.3 | 65.0 | 72.8 | 66.4 |

---

## Evaluation Best Practices

### Pre-Evaluation Checklist

1. **Coordinate frame:** Ensure all predictions are in the correct frame (global for nuScenes, ego for Waymo).
2. **Score calibration:** Verify detection scores are in [0, 1] range.
3. **Box format:** Confirm dimensions order matches evaluation convention (w, l, h for nuScenes; l, w, h for Waymo).
4. **Velocity:** Ensure velocity is in global frame for nuScenes evaluation.
5. **Tracking IDs:** Verify IDs are consistent strings/integers, unique within each scene.

### Common Evaluation Pitfalls

| Issue | Impact | Solution |
|-------|--------|----------|
| Wrong coordinate frame | All metrics severely degraded | Verify ego-to-global transform |
| Swapped width/length | mASE degraded, mAOE degraded | Check box dimension order |
| Missing velocity | mAVE = 1.0 (worst) | Ensure velocity head is enabled |
| Score threshold too high | Low recall, high precision | Use score_threshold = 0.1 for eval |
| Duplicate detections | mAP inflated/deflated | Apply NMS if peaks overlap |
| Wrong yaw convention | mAOE degraded | Verify yaw=0 direction |

### Post-Processing for Evaluation

```python
def post_process_for_eval(raw_detections, score_threshold=0.1):
    """
    Post-process raw model outputs for evaluation submission.
    """
    results = []
    
    for det in raw_detections:
        # Filter by score
        if det['score'] < score_threshold:
            continue
        
        # Decode box from predictions
        box = decode_box(
            center=det['center'] + det['offset'],
            height=det['height'],
            size=torch.exp(det['size']),  # log -> linear
            rotation=torch.atan2(det['sin_yaw'], det['cos_yaw']),
            velocity=det['velocity'],
        )
        
        # Apply optional NMS (usually not needed for CenterPoint)
        # boxes = nms_bev(boxes, scores, iou_threshold=0.2)
        
        results.append(box)
    
    return results
```
