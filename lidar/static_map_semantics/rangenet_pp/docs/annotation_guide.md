# RangeNet++: Annotation Guide

## SemanticKITTI Labeling Scheme

### Overview

SemanticKITTI uses a hierarchical labeling scheme with 28 fine-grained classes that are mapped to 19 evaluation classes plus 1 unlabeled/ignored class. The mapping follows the Cityscapes convention where applicable.

---

## Class Definitions (19 Evaluation Classes + Unlabeled)

### Class 0: Unlabeled
- **Description:** Points that are not assigned to any semantic class, or classes that are ignored during evaluation.
- **Includes:** Outliers, noise, moving objects in static map context, classes with insufficient samples.
- **Training:** Ignored in loss computation (masked out).

### Class 1: Car
- **Description:** Standard passenger vehicles including sedans, SUVs, coupes, station wagons, minivans.
- **Criteria:** Four-wheeled motorized vehicles designed primarily for passenger transport.
- **Excludes:** Trucks, buses, construction vehicles.
- **Typical size:** 3.5-5.5m length, 1.5-2.2m width, 1.4-1.8m height.

### Class 2: Bicycle
- **Description:** Non-motorized two-wheeled vehicles (pedal-powered).
- **Criteria:** Includes parked bicycles and bicycle racks. Does not include the rider.
- **Note:** When a person is riding a bicycle, the bicycle is labeled separately from the person (person becomes "bicyclist").

### Class 3: Motorcycle
- **Description:** Motorized two-wheeled vehicles including scooters and mopeds.
- **Criteria:** Engine-powered two-wheeled vehicles. Does not include the rider.
- **Note:** When ridden, the rider is labeled "motorcyclist" and the vehicle remains "motorcycle."

### Class 4: Truck
- **Description:** Large motorized vehicles for cargo transport.
- **Criteria:** Vehicles larger than standard cars used for goods transport (box trucks, semi-trucks, delivery vans larger than passenger vans).
- **Typical size:** >6m length.

### Class 5: Other-Vehicle
- **Description:** Motorized vehicles not fitting other vehicle categories.
- **Includes:** Buses, trailers, construction vehicles (excavators, bulldozers), forklifts, trains, boats on trailers.
- **Criteria:** Catch-all for uncommon vehicle types.

### Class 6: Person
- **Description:** Pedestrians and standing/sitting humans.
- **Criteria:** People who are walking, standing, sitting, or in wheelchairs. Not actively riding a bicycle or motorcycle.
- **Note:** Only the person's body; carried objects (backpacks, bags) are included.

### Class 7: Bicyclist
- **Description:** Person actively riding or sitting on a bicycle.
- **Criteria:** Combined label for person + bicycle when the person is mounted on the bike.
- **Note:** Labeled as a single object (person + bike together) for practical segmentation.

### Class 8: Motorcyclist
- **Description:** Person actively riding a motorcycle or scooter.
- **Criteria:** Combined label for person + motorcycle when the person is mounted.
- **Note:** Labeled as a single object (person + motorcycle together).

### Class 9: Road
- **Description:** Drivable road surface intended for vehicles.
- **Criteria:** Paved or unpaved surfaces where cars normally drive, including lane markings.
- **Includes:** Highway lanes, city streets, road intersections.
- **Excludes:** Parking areas, sidewalks, bike lanes.

### Class 10: Parking
- **Description:** Designated parking areas and driveways.
- **Criteria:** Paved areas intended for vehicle parking (parking lots, parallel parking spots, driveways).
- **Boundary:** Where the parking surface meets the road or sidewalk.

### Class 11: Sidewalk
- **Description:** Paved pedestrian walkways alongside roads.
- **Criteria:** Elevated or at-grade paths designated for pedestrian use.
- **Includes:** Curbs (as part of sidewalk boundary).

### Class 12: Other-Ground
- **Description:** Ground surfaces not fitting road, parking, sidewalk, or terrain categories.
- **Includes:** Railroad tracks, loading docks, plazas, pedestrian islands, unmarked ground.
- **Criteria:** Man-made ground surfaces that are neither road nor sidewalk.

### Class 13: Building
- **Description:** Permanent structures with walls and roofs.
- **Criteria:** Residential houses, commercial buildings, garages, sheds, bridges (structural parts).
- **Includes:** Walls, facades, architectural elements.
- **Excludes:** Fences (separate class), temporary structures.

### Class 14: Fence
- **Description:** Boundary structures that are not solid walls.
- **Criteria:** Wire fences, chain-link fences, wooden fences, metal railings, guardrails, barriers.
- **Distinguishing feature:** Partially transparent/permeable structures vs. solid buildings.

### Class 15: Vegetation
- **Description:** Plants, trees, bushes, hedges above ground level.
- **Criteria:** All plant matter that is above the ground surface.
- **Includes:** Tree canopies, bushes, hedges, climbing plants on walls.
- **Excludes:** Tree trunks (separate class), grass on ground (terrain).

### Class 16: Trunk
- **Description:** Tree trunks and large woody stems.
- **Criteria:** The main structural stem of trees from ground to first major branch.
- **Distinguishing feature:** Vertical, cylindrical, woody structure connecting ground to canopy.
- **Rationale:** Separated from vegetation because trunks have very different geometry (thin, vertical poles vs. volumetric canopies).

### Class 17: Terrain
- **Description:** Natural ground surfaces (unpaved).
- **Criteria:** Grass, soil, gravel, sand, natural ground cover.
- **Includes:** Lawns, garden beds, natural earth surfaces.
- **Excludes:** Vegetation above ground level (bushes, trees).

### Class 18: Pole
- **Description:** Thin vertical structures.
- **Criteria:** Street light poles, traffic light poles, sign posts, utility poles, bollards.
- **Characteristic:** Thin, tall, vertical objects with small cross-section.
- **Excludes:** Tree trunks (separate class), fence posts (part of fence).

### Class 19: Traffic-Sign
- **Description:** Traffic signs, street signs, and signal heads.
- **Criteria:** Sign boards (stop signs, speed limit signs, directional signs), traffic light housings.
- **Includes:** The sign face and mounting bracket (not the pole below).
- **Boundary:** Where the sign/signal connects to the pole.

---

## Label Mapping

### Original to Evaluation Mapping

SemanticKITTI has 28 original classes mapped to 20 (19 + unlabeled) for evaluation:

```yaml
# Original label -> Evaluation label
0: 0      # unlabeled -> unlabeled
1: 0      # outlier -> unlabeled
10: 1     # car -> car
11: 2     # bicycle -> bicycle
13: 5     # bus -> other-vehicle
15: 3     # motorcycle -> motorcycle
16: 5     # on-rails -> other-vehicle
18: 4     # truck -> truck
20: 5     # other-vehicle -> other-vehicle
30: 6     # person -> person
31: 7     # bicyclist -> bicyclist
32: 8     # motorcyclist -> motorcyclist
40: 9     # road -> road
44: 10    # parking -> parking
48: 11    # sidewalk -> sidewalk
49: 12    # other-ground -> other-ground
50: 13    # building -> building
51: 14    # fence -> fence
52: 0     # other-structure -> unlabeled
60: 9     # lane-marking -> road
70: 15    # vegetation -> vegetation
71: 16    # trunk -> trunk
72: 17    # terrain -> terrain
80: 18    # pole -> pole
81: 19    # traffic-sign -> traffic-sign
99: 0     # other-object -> unlabeled
252: 1    # moving-car -> car
253: 7    # moving-bicyclist -> bicyclist
254: 6    # moving-person -> person
255: 8    # moving-motorcyclist -> motorcyclist
256: 5    # moving-on-rails -> other-vehicle
257: 5    # moving-bus -> other-vehicle
258: 4    # moving-truck -> truck
259: 5    # moving-other-vehicle -> other-vehicle
```

---

## Scan Unfolding for Range Image

### Problem

The Velodyne HDL-64E scans continuously while the vehicle moves. Due to ego-motion during a single rotation (~100ms at 10Hz), the resulting point cloud is slightly distorted. Points captured at the beginning of the rotation are in a different ego-vehicle frame than points at the end.

### Unfolding Process

1. **Timestamp assignment:** Each point has an implicit timestamp based on its firing order within the rotation.
2. **Ego-motion compensation:** Using the vehicle's motion (from IMU/odometry), transform all points to a common reference frame (typically the frame at the middle or end of the rotation).
3. **Range image projection:** After unfolding, project the motion-compensated points to the range image.

### Impact on Range Image Quality

Without unfolding:
- Moving objects appear smeared or doubled.
- Straight structures (walls, fences) appear slightly curved.
- Ground plane may show rippling artifacts.

With unfolding:
- Cleaner object boundaries.
- More accurate geometric relationships.
- Better correspondence between range image pixels and true 3D positions.

### Implementation Note

For RangeNet++, scan unfolding is optional but recommended. The model can learn to handle minor distortions, but unfolded scans generally yield better results (0.5-1.0 mIoU improvement).

---

## Handling Unlabeled and Ignored Points

### Categories of Unlabeled Points

1. **True unlabeled (class 0):** Points that annotators could not assign to any class (ambiguous, too far, noise).
2. **Outlier points (original class 1):** Sensor artifacts, multi-path reflections, points clearly caused by noise.
3. **Merged classes:** Some original classes (other-structure, other-object) are merged into unlabeled for evaluation.

### Treatment During Training

```python
# Ignore mask for loss computation
ignore_mask = (labels == 0)  # Unlabeled points

# In range image space
loss = cross_entropy(predictions, labels)
loss = loss * (~ignore_mask).float()  # Zero out loss for unlabeled pixels
loss = loss.sum() / (~ignore_mask).float().sum()  # Normalize by valid pixels only
```

### Treatment During Evaluation

- Predictions for points with ground-truth label 0 (unlabeled) are **not** counted in mIoU.
- The model may still predict class 0, but the evaluator skips these points entirely.
- Per-class IoU is computed only over the 19 semantic classes.

### Empty Pixels in Range Image

Empty pixels (no point projected) are handled separately:
- **Representation:** All channels set to 0.0.
- **Loss:** Masked out (not included in loss computation).
- **Prediction:** Network may output any label; it is ignored during back-projection to 3D.
- **Masking strategy:** Binary mask channel indicates valid/invalid pixels.

---

## Annotation Quality Considerations

### Known Challenges

1. **Boundary ambiguity:** At object boundaries, LiDAR points may be mixed between foreground and background due to beam divergence.
2. **Distant objects:** Points far from the sensor are sparse, making class assignment difficult.
3. **Thin structures:** Poles, signs, and fences receive very few points, making consistent annotation challenging.
4. **Dynamic vs. static:** Moving objects (cars, people) are annotated with their semantic class, but some datasets distinguish moving vs. static instances.

### Label Noise Handling

RangeNet++ handles label noise through:
- Class weighting that reduces the influence of noisy/ambiguous classes.
- The KNN post-processing smooths predictions spatially, reducing the impact of isolated labeling errors.
- Training with augmentation improves robustness to minor annotation inconsistencies.

---

## Annotation Color Scheme

Standard visualization colors for SemanticKITTI classes:

| Class ID | Class Name | RGB Color |
|----------|-----------|-----------|
| 0 | unlabeled | (0, 0, 0) |
| 1 | car | (0, 0, 255) |
| 2 | bicycle | (245, 150, 100) |
| 3 | motorcycle | (150, 60, 30) |
| 4 | truck | (0, 0, 200) |
| 5 | other-vehicle | (200, 40, 255) |
| 6 | person | (255, 30, 30) |
| 7 | bicyclist | (255, 0, 255) |
| 8 | motorcyclist | (255, 150, 255) |
| 9 | road | (255, 0, 255) |
| 10 | parking | (255, 150, 255) |
| 11 | sidewalk | (75, 0, 75) |
| 12 | other-ground | (75, 0, 175) |
| 13 | building | (0, 200, 255) |
| 14 | fence | (50, 120, 255) |
| 15 | vegetation | (0, 175, 0) |
| 16 | trunk | (0, 60, 135) |
| 17 | terrain | (80, 240, 150) |
| 18 | pole | (150, 240, 255) |
| 19 | traffic-sign | (0, 0, 255) |
