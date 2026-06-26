# Scenario Trees System Guide — Complete Teaching Document

This guide teaches the Functional Scenario Trees system from scratch. It explains why scenario management matters, how automated tagging works, and how to use this system to improve your perception models.

---

## 1. Why Scenario Management Matters

### The Long-Tail Problem in Autonomous Driving

Your perception model trains on thousands of hours of driving data. It works great 99% of the time. But that 1% — the long tail of rare scenarios — is where accidents happen.

```
Scenario Frequency Distribution:

  Frequency
    |████████████████████████  Sunny + Highway + No traffic
    |██████████████████        Clear + Urban + Light traffic
    |███████████████           Overcast + Suburban
    |██████████                Rain + Urban + Moderate traffic
    |█████                     Night + Highway + Construction
    |███                       Fog + Rural + Pedestrian on road
    |█                         Snow + Night + Emergency vehicle
    |·                         Rain + Night + Intersection + Cyclist cut-in
    +──────────────────────────────────────────────────────→
    
    ↑ Your model is great here        ↑ Your model FAILS here
      (trained on many examples)         (seen only 2-3 times)
```

### The Real-World Workflow Problem

Consider this scenario:

> "Our perception model missed a cyclist at a rainy night intersection. We need more training data for this specific scenario to fix it."

Without scenario management:
1. Manually watch 10,000 hours of recordings looking for "rainy night intersection + cyclist"
2. Give up after 3 days
3. Hope the problem doesn't cause an accident

With Scenario Trees:
1. Query: `weather=rain AND lighting=night AND road_type=intersection AND has_cyclist=True`
2. System returns 47 matching recordings in 0.3 seconds
3. Add them to training set, retrain, verify the fix

```
┌─────────────────────────────────────────────────────────────┐
│                 SCENARIO-DRIVEN DEVELOPMENT CYCLE            │
│                                                             │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│    │  Model   │─────→│  Find    │─────→│  Analyze │       │
│    │ Failure  │      │ Scenario │      │  Root    │       │
│    │ Detected │      │ Pattern  │      │  Cause   │       │
│    └──────────┘      └──────────┘      └──────────┘       │
│         ↑                                    │              │
│         │                                    ↓              │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│    │  Verify  │←─────│  Retrain │←─────│   Mine   │       │
│    │  Fix on  │      │   with   │      │  Similar │       │
│    │  Scenario│      │  Balanced │      │  Data    │       │
│    │  Set     │      │  Data    │      │          │       │
│    └──────────┘      └──────────┘      └──────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. The 6-Layer Scenario Taxonomy

Based on PEGASUS and ASAM standards used in the automotive industry. Each recorded driving scene is tagged across all 6 layers.

### Layer 1: Road Topology (Where are you driving?)

```
L1 Road Topology
├── Highway
│   ├── lanes: 2, 3, 4, 5+
│   ├── features: on_ramp, off_ramp, merge, split
│   └── surface: asphalt, concrete
├── Urban
│   ├── intersection: signalized, unsignalized, roundabout
│   ├── lanes: 1, 2, 3
│   └── features: parking, bus_stop, crosswalk
├── Suburban
│   ├── residential_street
│   └── collector_road
├── Rural
│   ├── country_road
│   └── unpaved
└── Special
    ├── tunnel
    ├── bridge
    └── parking_garage
```

### Layer 2: Traffic Infrastructure (What's built into the road?)

| Element | Examples | Why It Matters |
|---------|----------|---------------|
| Traffic signals | Red/yellow/green lights, arrow signals | Must detect state for decision-making |
| Signs | Speed limit, stop, yield, construction | Must read and obey |
| Markings | Lane lines, arrows, crosswalks | Guide lane-keeping |
| Barriers | Guardrails, bollards, curbs | Define drivable boundaries |
| Lighting | Street lamps, overhead lights | Affects camera exposure |

### Layer 3: Temporary Modifications (What's changed from the map?)

- Construction zones (cones, barriers, lane shifts)
- Temporary signs overriding permanent ones
- Road closures and detours
- Emergency scenes (police/fire vehicles blocking)
- Special events (parades, markets)

These are critical because they invalidate the HD map — the car must rely on real-time perception.

### Layer 4: Dynamic Objects (Who's around you?)

```
L4 Dynamic Objects
├── Vehicles
│   ├── car, truck, bus, van, motorcycle
│   ├── emergency_vehicle (special behavior!)
│   └── behaviors: cut_in, cut_out, lane_change, U_turn,
│                  sudden_brake, running_red_light
├── Vulnerable Road Users (VRU)
│   ├── pedestrian (adult, child, wheelchair, stroller)
│   ├── cyclist
│   └── behaviors: jaywalking, entering_crosswalk,
│                  cycling_in_lane, group_crossing
└── Other
    ├── animal (dog, deer, bird)
    └── debris (tire, box, unknown_object)
```

### Layer 5: Environment (What are the conditions?)

| Dimension | Values | Impact on Perception |
|-----------|--------|---------------------|
| Weather | clear, rain, heavy_rain, snow, fog, hail | Camera blur, LiDAR absorption, radar clutter |
| Lighting | day, dawn, dusk, night, direct_sunlight | Camera saturation, shadow contrast |
| Road surface | dry, wet, icy, snow_covered, muddy | Affects radar reflections, tire tracks |
| Visibility | >200m, 100-200m, 50-100m, <50m | Detection range limitations |
| Wind | calm, moderate, strong | Affects cyclist/pedestrian trajectories |

### Layer 6: Digital Information (How's the sensor health?)

- Sensor degradation (dirty lens, rain drops on camera)
- GPS quality (urban canyon, tunnel)
- Map accuracy (outdated map, construction not in map)
- Communication (V2X availability)

---

## 3. CLIP-Based Scene Classification

### What Is CLIP?

CLIP (Contrastive Language-Image Pretraining) by OpenAI is a model trained on 400 million image-text pairs from the internet. It learns to align images and text in a shared embedding space.

```
CLIP Architecture:
                                                    
  Image ──→ [Image Encoder] ──→ Image Embedding (512-dim)
                                        │
                              cosine_similarity
                                        │
  Text  ──→ [Text Encoder]  ──→ Text Embedding (512-dim)
  
  "a photo of a highway"        →  similarity = 0.92 (match!)
  "a photo of a parking lot"    →  similarity = 0.23 (no match)
```

### Why CLIP Is Perfect for Scenario Classification

Traditional classification requires:
1. Collect labeled training data for each scenario type
2. Train a custom classifier
3. If you add a new category, collect more data and retrain

CLIP enables ZERO-SHOT classification:
1. Define text prompts for each scenario (no training data needed!)
2. Compute similarity between image and each prompt
3. Highest similarity = predicted class
4. Adding new categories = just write new text prompts

### How We Use CLIP for Scenario Tagging

```python
# Define prompts for road type classification
road_prompts = [
    "a photo taken while driving on a highway with multiple lanes",
    "a photo taken while driving in an urban city with buildings",
    "a photo taken while driving on a rural country road",
    "a photo taken while driving through a residential neighborhood",
]

# Define prompts for weather
weather_prompts = [
    "a photo taken while driving in clear sunny weather",
    "a photo taken while driving in the rain with wet road",
    "a photo taken while driving in fog with low visibility",
    "a photo taken while driving in snow",
]

# Classify: encode image, compute similarity with each prompt
image_embedding = clip_model.encode_image(camera_image)
for prompt_set in [road_prompts, weather_prompts, ...]:
    text_embeddings = clip_model.encode_text(prompt_set)
    similarities = cosine_similarity(image_embedding, text_embeddings)
    predicted_class = prompt_set[argmax(similarities)]
```

### Practical Tips for CLIP Prompts

- Be specific: "a highway with 3 lanes" works better than "highway"
- Include context: "a photo taken from a car" helps CLIP understand the viewpoint
- Use multiple prompts per class and average (ensemble)
- Validate on a small labeled subset before trusting at scale

---

## 4. Temporal Event Detection

Some scenarios are defined by EVENTS, not static properties. These require trajectory analysis over time.

### Cut-In Detection

A cut-in occurs when a vehicle from an adjacent lane moves into the ego vehicle's lane ahead.

```
Time t=0:           Time t=1:           Time t=2:
                                        
 Lane 1 | Lane 2    Lane 1 | Lane 2    Lane 1 | Lane 2
        |                  |                  |
   EGO  | [Car B]    EGO  |  [B→]      EGO  |
        |                  | /               |
        |             [B moving]        [Car B]
        |                  |           (now in Lane 1!)
        |                  |                  |
```

Detection algorithm:
1. Track lateral position of all vehicles relative to lane boundaries
2. If a vehicle's lateral position crosses a lane boundary
3. AND it ends up in the ego vehicle's lane
4. AND it's within a longitudinal distance threshold
5. → FLAG as cut-in event

### Time-to-Collision (TTC) Computation

```
TTC = distance_to_lead / relative_velocity

  If TTC < 3.0 seconds: CRITICAL (near-miss)
  If TTC < 1.5 seconds: EMERGENCY (near-collision)
  If TTC < 0.5 seconds: IMMINENT COLLISION
```

### Hard Braking Detection

```
Monitor ego vehicle longitudinal acceleration:
  If deceleration > 4.0 m/s²  → moderate braking
  If deceleration > 6.0 m/s²  → hard braking event
  If deceleration > 8.0 m/s²  → emergency braking
```

### Lane Change Detection

Track ego and other vehicle trajectories:
- Lateral displacement > 2.0m over 3-5 seconds = lane change
- Distinguish: planned (with indicator) vs sudden (no indicator)

---

## 5. Data Mining: Finding Corner Cases

### What Is a Corner Case?

A corner case is a scenario that is:
- **Rare** in the overall dataset (statistically unusual)
- **Potentially dangerous** if the model fails
- **Informative** for improving model robustness

Examples: a shopping cart in the road, a vehicle driving the wrong way, a pedestrian in a wheelchair crossing at night, a reflection causing a phantom detection.

### Isolation Forest for Novelty Detection

Isolation Forest works by randomly splitting data. Normal points require MANY splits to isolate. Anomalous points require FEW splits (they're far from the crowd).

```
Normal data point:                    Anomalous point:
Takes 8 random splits                Takes 2 random splits
to isolate this point                 to isolate this point

   ┌──────────────────┐                ┌──────────────────┐
   │     ┌────────┐   │                │                  │
   │     │  ┌──┐  │   │                │  ┌───────────┐  │
   │     │  │ ·│  │   │                │  │     ·     │  │  ← isolated
   │     │  └──┘  │   │                │  └───────────┘  │    quickly!
   │     └────────┘   │                │                  │
   │                   │                │  ··········      │
   │   ··········      │                │  ··········      │
   │   ··········      │                │  ··········      │
   └──────────────────┘                └──────────────────┘

  Anomaly score = 1/path_length (shorter path = more anomalous)
```

### Embedding-Based Novelty

1. Compute embedding for each recording (e.g., average CLIP embedding of all frames)
2. Cluster embeddings (HDBSCAN or K-means)
3. Points far from ALL cluster centers = novel scenarios

### Coverage Analysis: Finding Gaps

Cross-tabulate scenario attributes to find MISSING combinations:

```
Coverage Matrix (weather × road_type × lighting):

                    Highway    Urban    Rural
                    D   N      D   N    D   N     (D=day, N=night)
  Clear           342  89    567 234   45  12
  Rain             78  23    145  56    8   3
  Fog              12   4     23   7    2   0  ← GAP: fog+rural+night
  Snow              5   1      8   2    0   0  ← GAP: snow+rural+any
  
  If a cell has 0 or very few examples: COVERAGE GAP
  → Plan targeted data collection for these scenarios
```

### Difficulty Scoring for Curriculum Learning

Score each recording by perception difficulty:
- Number of objects (more = harder)
- Object density (cluttered = harder)
- Occlusion level (high occlusion = harder)
- Weather severity (rain/fog = harder)
- Speed (faster ego = harder for temporal reasoning)

Then train in order: easy → medium → hard (curriculum learning).

---

## 6. Integration with Model Training

### Balanced Dataset Creation

Problem: Training on raw data means the model sees "sunny highway" 1000x more than "rainy night intersection." It optimizes for the common case and ignores the rare.

Solution: Use scenario tags to create BALANCED training splits:

```python
from scenario_trees.scenario_manager.split_generator import SplitGenerator

generator = SplitGenerator(database)
splits = generator.generate(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    # Balance across these dimensions:
    balance_on=["weather", "road_type", "lighting", "has_vru"],
    # Ensure minimum samples per combination:
    min_per_combination=10,
    # Oversample rare scenarios:
    oversample_rare=True,
)
```

### Failure Analysis Pipeline

After evaluation, correlate errors with scenario attributes:

```
Model Evaluation Results + Scenario Tags → Failure Patterns

Example output:
  "mAP drops 23% in: weather=rain AND lighting=night"
  "Cyclist detection fails 45% in: road_type=intersection AND has_cyclist=True"
  "False positives increase 3x in: weather=fog"

→ These patterns tell you EXACTLY what data to collect/mine for retraining
```

### Active Learning Integration

When labeling budget is limited, prioritize:
1. Scenarios where model is UNCERTAIN (high entropy predictions)
2. Scenarios that are NOVEL (far from training distribution)
3. Scenarios that are DIFFICULT (high difficulty score)
4. Scenarios that FILL COVERAGE GAPS

---

## 7. SOTIF and Safety Validation

### What Is SOTIF?

SOTIF (Safety Of The Intended Functionality, ISO 21448) addresses situations where the system is working as designed but still causes harm due to:
- Sensor limitations (LiDAR can't see through fog)
- Algorithm limitations (model never trained on scenario X)
- Edge cases not considered during development

### How Scenario Trees Support SOTIF

```
SOTIF Argument Structure:

  "Our perception system is safe because:"
  
  1. We have identified ALL relevant scenario types (taxonomy)
  2. We have TESTED on representative samples of each (coverage)
  3. We have measured performance on each (per-scenario metrics)
  4. We have validated that residual risk is acceptable
  
  Scenario Trees provide:
  - Completeness: 6-layer taxonomy ensures nothing is forgotten
  - Coverage metrics: quantified % of scenario space tested
  - Per-scenario evaluation: metrics broken down by scenario type
  - Gap identification: where more testing is needed
```

### Safety-Critical Scenario Prioritization

Not all scenarios are equally important for safety. Prioritize based on:

| Factor | Weight | Example |
|--------|--------|---------|
| Severity if failure occurs | High | Pedestrian crossing at night |
| Frequency of scenario | Medium | Common urban driving |
| Model uncertainty | High | Scenarios where model is unsure |
| Controllability | Low | Can the driver take over? |

---

## 8. Practical Usage Examples

### Example 1: Find All Rainy Night Scenarios

```python
from scenario_trees.scenario_manager.query_engine import QueryEngine

engine = QueryEngine(database)
results = engine.query(
    weather="rain",
    lighting="night",
)
print(f"Found {len(results)} rainy night recordings")
# Export as file list for training
engine.export_filelist(results, "rainy_night_training.txt")
```

### Example 2: Analyze Dataset Coverage

```python
from scenario_trees.data_mining.coverage_analyzer import CoverageAnalyzer

analyzer = CoverageAnalyzer(database)
report = analyzer.full_report(
    dimensions=["weather", "lighting", "road_type", "has_vru"]
)
print(report.gaps)        # Missing combinations
print(report.imbalances)  # Severely under-represented combos
print(report.coverage_score)  # Overall coverage metric (0-1)
```

### Example 3: Find Corner Cases for a Specific Model Failure

```python
from scenario_trees.data_mining.novelty_detector import NoveltyDetector

# After model evaluation shows failures on certain scenarios:
detector = NoveltyDetector()
detector.fit(training_embeddings)

# Find scenarios most different from training distribution
novel_scenarios = detector.find_novel(
    test_embeddings,
    threshold=0.8,  # top 20% most unusual
)
# These are likely where the model struggles
```

---

## Summary

The Scenario Trees system turns unmanageable driving data into a structured, queryable, and actionable asset. Instead of manually searching through recordings, you can:

1. **Auto-tag** every recording with 6-layer metadata (CLIP + trajectory analysis)
2. **Query** recordings by any combination of attributes (instant results)
3. **Mine** for corner cases and novel scenarios (find what you didn't know existed)
4. **Analyze** coverage gaps (know what's missing from your data)
5. **Balance** training sets (ensure rare scenarios get proper representation)
6. **Validate** safety claims (SOTIF compliance with evidence)

This closes the loop between model failures and data-driven fixes — the most critical workflow in production autonomous driving development.
