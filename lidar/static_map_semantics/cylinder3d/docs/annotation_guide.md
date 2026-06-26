# Cylinder3D: Annotation Guide

## Overview

This document details the semantic class definitions, label formats, and annotation conventions used in the two primary datasets for Cylinder3D: SemanticKITTI and nuScenes-lidarseg.

---

## SemanticKITTI Class Definitions

### 19 Evaluation Classes

SemanticKITTI defines 28 raw classes that are mapped to 19 evaluation classes (plus "unlabeled"). The evaluation mapping collapses fine-grained distinctions into broader categories.

| Eval ID | Class Name | Color (RGB) | Description |
|---------|-----------|-------------|-------------|
| 0 | unlabeled | (0, 0, 0) | Unlabeled, noise, or outlier points |
| 1 | car | (0, 0, 142) | Passenger vehicles: sedans, SUVs, minivans |
| 2 | bicycle | (119, 11, 32) | Non-motorized two-wheeled vehicles |
| 3 | motorcycle | (0, 0, 230) | Motorized two-wheeled vehicles (scooters, motorbikes) |
| 4 | truck | (0, 0, 70) | Heavy goods vehicles, delivery trucks |
| 5 | other-vehicle | (0, 0, 90) | Buses, trailers, construction vehicles, unclassified vehicles |
| 6 | person | (220, 20, 60) | Standing, walking, or sitting pedestrians |
| 7 | bicyclist | (255, 0, 0) | Person actively riding a bicycle |
| 8 | motorcyclist | (255, 0, 100) | Person actively riding a motorcycle |
| 9 | road | (128, 64, 128) | Drivable road surface (asphalt, concrete) |
| 10 | parking | (244, 35, 232) | Parking areas (lots, garages, on-street spaces) |
| 11 | sidewalk | (152, 251, 152) | Pedestrian walkways, curbs |
| 12 | other-ground | (70, 70, 70) | Ground surfaces not road/parking/sidewalk (dirt, gravel) |
| 13 | building | (70, 70, 70) | Permanent structures (houses, offices, walls) |
| 14 | fence | (190, 153, 153) | Barriers: fences, guardrails, retaining walls |
| 15 | vegetation | (107, 142, 35) | Trees, bushes, hedges, grass >20 cm |
| 16 | trunk | (150, 100, 100) | Tree trunks and major branches |
| 17 | terrain | (70, 130, 180) | Low grass (<20 cm), soil, gravel patches |
| 18 | pole | (153, 153, 153) | Vertical poles: traffic signs, lamp posts, bollards |
| 19 | traffic-sign | (250, 170, 30) | Sign faces, traffic lights, signal heads |

### Raw-to-Evaluation Class Mapping

```yaml
# From semantic-kitti.yaml
learning_map:
  0:  0    # unlabeled → unlabeled
  1:  0    # outlier → unlabeled
  10: 1    # car → car
  11: 2    # bicycle → bicycle
  13: 5    # bus → other-vehicle
  15: 3    # motorcycle → motorcycle
  16: 5    # on-rails → other-vehicle
  18: 4    # truck → truck
  20: 5    # other-vehicle → other-vehicle
  30: 6    # person → person
  31: 7    # bicyclist → bicyclist
  32: 8    # motorcyclist → motorcyclist
  40: 9    # road → road
  44: 10   # parking → parking
  48: 11   # sidewalk → sidewalk
  49: 12   # other-ground → other-ground
  50: 13   # building → building
  51: 14   # fence → fence
  52: 0    # other-structure → unlabeled
  60: 9    # lane-marking → road
  70: 15   # vegetation → vegetation
  71: 16   # trunk → trunk
  72: 17   # terrain → terrain
  80: 18   # pole → pole
  81: 19   # traffic-sign → traffic-sign
  99: 0    # other-object → unlabeled
  252: 1   # moving-car → car
  253: 7   # moving-bicyclist → bicyclist
  254: 6   # moving-person → person
  255: 8   # moving-motorcyclist → motorcyclist
  256: 5   # moving-on-rails → other-vehicle
  257: 5   # moving-bus → other-vehicle
  258: 4   # moving-truck → truck
  259: 5   # moving-other-vehicle → other-vehicle
```

### Annotation Rules (SemanticKITTI)

1. **Boundary points:** Points on object boundaries are assigned to the object they primarily belong to (>50% of the local surface)
2. **Moving vs. static:** Moving objects (cars in motion, walking people) receive special raw labels (252-259) but map to the same evaluation classes
3. **Occlusion:** Partially occluded objects are labeled with their true class, not as unlabeled
4. **Minimum size:** Objects smaller than ~5 points are typically labeled as unlabeled
5. **Ground plane:** The road label extends to lane markings; curbs belong to sidewalk
6. **Vegetation vs. terrain:** Grass shorter than 20 cm is terrain; taller vegetation is vegetation
7. **Trunk vs. vegetation:** Tree trunks up to the first major branching point are trunk; canopy is vegetation

---

## nuScenes-lidarseg Class Definitions

### 16 Evaluation Classes

| ID | Class Name | Description |
|----|-----------|-------------|
| 0 | noise | Points with no valid return or clearly erroneous |
| 1 | barrier | Temporary barriers, jersey barriers, traffic cones, delineators |
| 2 | bicycle | Bicycles without riders |
| 3 | bus | Large passenger vehicles, shuttles |
| 4 | car | Passenger vehicles, sedans, SUVs, pickup trucks |
| 5 | construction_vehicle | Excavators, cranes, bulldozers, cement mixers |
| 6 | motorcycle | Motorcycles, scooters without riders |
| 7 | pedestrian | Standing, walking, sitting adults and children |
| 8 | traffic_cone | Individual traffic cones |
| 9 | trailer | Semi-trailers, cargo trailers |
| 10 | truck | Rigid trucks, box trucks, flatbeds |
| 11 | driveable_surface | Road surface where the ego vehicle can legally drive |
| 12 | other_flat | Other flat surfaces: sidewalks, pedestrian crossings, shoulders |
| 13 | sidewalk | Pedestrian walkways, raised paths |
| 14 | terrain | Natural ground: grass, soil, sand, gravel |
| 15 | manmade | Buildings, walls, fences, poles, signs |
| 16 | vegetation | Trees, bushes, hedges, large plants |

### Raw-to-Evaluation Mapping (nuScenes)

nuScenes has 32 fine-grained categories that collapse to 16 evaluation classes:

```python
# 32 → 16 mapping
general_to_lidarseg = {
    'noise': 0,
    'animal': 0,                      # rare, mapped to noise
    'human.pedestrian.adult': 7,
    'human.pedestrian.child': 7,
    'human.pedestrian.construction_worker': 7,
    'human.pedestrian.police_officer': 7,
    'human.pedestrian.personal_mobility': 0,  # segways → noise
    'human.pedestrian.stroller': 0,           # → noise
    'human.pedestrian.wheelchair': 0,         # → noise
    'movable_object.barrier': 1,
    'movable_object.debris': 0,               # → noise
    'movable_object.pushable_pullable': 0,    # → noise
    'movable_object.trafficcone': 8,
    'static_object.bicycle_rack': 0,          # → noise
    'vehicle.bicycle': 2,
    'vehicle.bus.bendy': 3,
    'vehicle.bus.rigid': 3,
    'vehicle.car': 4,
    'vehicle.construction': 5,
    'vehicle.emergency.ambulance': 4,         # → car
    'vehicle.emergency.police': 4,            # → car
    'vehicle.motorcycle': 6,
    'vehicle.trailer': 9,
    'vehicle.truck': 10,
    'flat.driveable_surface': 11,
    'flat.other': 12,
    'flat.sidewalk': 13,
    'flat.terrain': 14,
    'static.manmade': 15,
    'static.other': 15,                       # → manmade
    'static.vegetation': 16,
}
```

---

## Label Format Details

### SemanticKITTI Label Format (uint32)

```
┌─────────────────────────────────────────────┐
│         32-bit unsigned integer              │
├──────────────────┬──────────────────────────┤
│ Upper 16 bits    │ Lower 16 bits            │
│ Instance ID      │ Semantic Label ID        │
├──────────────────┼──────────────────────────┤
│ 0x0000 = none    │ 0x0000 = unlabeled       │
│ 0x0001 = inst 1  │ 0x000A = car (10)        │
│ 0x0002 = inst 2  │ 0x001E = person (30)     │
│ ...              │ ...                      │
└──────────────────┴──────────────────────────┘

Extraction:
  semantic_label = label & 0xFFFF          # mask lower 16 bits
  instance_id    = label >> 16             # shift upper 16 bits

Note: Instance IDs are only meaningful for "thing" classes (vehicles, people).
      "Stuff" classes (road, vegetation) have instance_id = 0.
```

### nuScenes Label Format (uint8)

```
┌─────────────────────┐
│  8-bit unsigned int │
│  Semantic class ID  │
│  (0–31)             │
└─────────────────────┘

Simple per-point class assignment. No instance encoding in the lidarseg labels.
Instance information is available separately via 3D bounding box annotations.
```

---

## Class Frequency and Imbalance Statistics

### SemanticKITTI Class Distribution (Training Set)

| Class | Points (millions) | Frequency (%) | Relative Weight |
|-------|------------------|---------------|-----------------|
| road | 580.1 | 28.4% | 1.0× (baseline) |
| vegetation | 395.2 | 19.4% | 1.5× |
| building | 265.8 | 13.0% | 2.2× |
| sidewalk | 175.4 | 8.6% | 3.3× |
| terrain | 169.3 | 8.3% | 3.4× |
| car | 138.7 | 6.8% | 4.2× |
| fence | 85.1 | 4.2% | 6.8× |
| pole | 45.2 | 2.2% | 12.8× |
| trunk | 39.8 | 2.0% | 14.2× |
| parking | 38.6 | 1.9% | 14.9× |
| other-ground | 28.3 | 1.4% | 20.3× |
| traffic-sign | 22.1 | 1.1% | 25.8× |
| other-vehicle | 19.4 | 1.0% | 28.4× |
| truck | 15.8 | 0.8% | 35.5× |
| person | 6.2 | 0.3% | 94.7× |
| bicycle | 3.1 | 0.15% | 189.4× |
| bicyclist | 5.8 | 0.28% | 101.4× |
| motorcycle | 2.4 | 0.12% | 236.8× |
| motorcyclist | 0.8 | 0.04% | 710.4× |

**Key Observations:**
- Class imbalance ratio (most to least frequent): ~710:1
- Road alone accounts for 28.4% of all points
- Vulnerable road users (person, bicyclist, motorcyclist) together represent <1%
- This extreme imbalance motivates the use of weighted cross-entropy and Lovasz-softmax loss

### nuScenes-lidarseg Class Distribution (Training Set)

| Class | Frequency (%) | Relative Weight |
|-------|---------------|-----------------|
| driveable_surface | 32.1% | 1.0× |
| vegetation | 18.7% | 1.7× |
| manmade | 17.3% | 1.9× |
| terrain | 10.2% | 3.1× |
| other_flat | 7.8% | 4.1× |
| sidewalk | 5.1% | 6.3× |
| car | 4.2% | 7.6× |
| truck | 1.4% | 22.9× |
| bus | 0.9% | 35.7× |
| barrier | 0.8% | 40.1× |
| trailer | 0.6% | 53.5× |
| pedestrian | 0.3% | 107.0× |
| construction_vehicle | 0.2% | 160.5× |
| motorcycle | 0.1% | 321.0× |
| bicycle | 0.1% | 321.0× |
| traffic_cone | 0.1% | 321.0× |

---

## Annotation Methodology

### SemanticKITTI Annotation Process

1. **Initial labeling:** Performed by trained annotators using 3D point cloud visualization tools
2. **Multi-scan accumulation:** Annotators viewed accumulated scans (using poses) for better context
3. **Consistency check:** Labels were cross-checked against camera images
4. **Quality assurance:** Multiple passes with inter-annotator agreement checking
5. **Tool:** Custom annotation tool based on 3D point cloud visualization with class painting

### nuScenes Annotation Process

1. **3D bounding boxes:** Primary annotation is 3D boxes for "thing" classes
2. **Point-level assignment:** Points within boxes inherit the box class
3. **Surface segmentation:** "Stuff" classes (road, vegetation, etc.) annotated via surface painting
4. **Multi-sensor fusion:** Annotators reference cameras and accumulated scans
5. **Tool:** Scale AI annotation platform with custom 3D labeling interface

---

## Cross-Dataset Class Correspondence

| SemanticKITTI | nuScenes Equivalent | Notes |
|---------------|-------------------|-------|
| car | car | Direct match |
| bicycle | bicycle | Direct match |
| motorcycle | motorcycle | Direct match |
| truck | truck | Direct match |
| other-vehicle | bus, trailer, construction_vehicle | Split into finer categories |
| person | pedestrian | Direct match |
| bicyclist | — | No separate class (bicycle + pedestrian) |
| motorcyclist | — | No separate class (motorcycle + pedestrian) |
| road | driveable_surface | Slightly different definition |
| parking | other_flat | Approximate |
| sidewalk | sidewalk | Direct match |
| other-ground | other_flat | Approximate |
| building | manmade | Broader category in nuScenes |
| fence | manmade | Merged into manmade |
| vegetation | vegetation | Direct match |
| trunk | vegetation | Merged into vegetation |
| terrain | terrain | Direct match |
| pole | manmade | Merged into manmade |
| traffic-sign | manmade | Merged into manmade |
| — | barrier | No direct SemanticKITTI equivalent |
| — | traffic_cone | No direct SemanticKITTI equivalent |

---

## Handling Label Noise and Edge Cases

### Common Annotation Issues

1. **Scan-line artifacts:** Points at object boundaries may be assigned to the wrong class due to LiDAR scan-line bleeding
2. **Glass surfaces:** Windows and glass often produce no returns or noisy points
3. **Moving objects at rest:** Parked vehicles share the same label as moving ones after mapping
4. **Overhanging vegetation:** Points from tree canopy directly above road may be mislabeled as road
5. **Thin structures:** Wires, antennas, and very thin poles may have inconsistent labels

### Recommended Preprocessing

```python
# Remove unlabeled/noise points for training
valid_mask = semantic_labels > 0  # or != 0 for SemanticKITTI

# Apply learning_map to convert raw → evaluation labels
mapped_labels = np.vectorize(learning_map.get)(raw_labels)

# Ignore index for loss computation
ignore_label = 0  # unlabeled class not included in loss or metrics
```
