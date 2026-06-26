"""
Functional Scenario Tree based on PEGASUS/ASAM taxonomy for autonomous driving.

Implements a 6-layer hierarchical scenario classification:
  Layer 1: Road Topology
  Layer 2: Traffic Infrastructure
  Layer 3: Temporary Modifications
  Layer 4: Dynamic Objects
  Layer 5: Environment
  Layer 6: Digital Information

Node IDs use a semantic dotted notation:
  L{layer}.{category}.{subcategory}
  Example: L5.weather.rain, L4.behavior.cut_in, L1.highway
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class ScenarioTreeNode:
    """A single node in the functional scenario tree."""

    id: str
    name: str
    layer: int
    description: str = ""
    detection_method: str = ""
    parent_id: Optional[str] = None
    children: list["ScenarioTreeNode"] = field(default_factory=list)

    def add_child(self, child: "ScenarioTreeNode") -> "ScenarioTreeNode":
        """Add a child node and set its parent_id."""
        child.parent_id = self.id
        self.children.append(child)
        return child

    def __iter__(self) -> Iterator["ScenarioTreeNode"]:
        """Pre-order traversal of the subtree rooted at this node."""
        yield self
        for child in self.children:
            yield from child

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        """Approximate depth based on ID components."""
        return len(self.id.split("."))


def _build_layer1() -> ScenarioTreeNode:
    """Build Layer 1: Road Topology."""
    root = ScenarioTreeNode(
        id="L1",
        name="Road Topology",
        layer=1,
        description="Classification of road types, geometries, and structural layouts",
        detection_method="map-based + perception",
    )

    # Road types
    road_types = root.add_child(ScenarioTreeNode(
        id="L1.road_type",
        name="Road Types",
        layer=1,
        description="Functional road classification",
        detection_method="HD-map lookup + CLIP scene classification",
    ))
    for rid, rname, desc, detect in [
        ("L1.highway", "Highway", "Multi-lane high-speed road with controlled access",
         "CLIP classification + map road_class attribute"),
        ("L1.urban", "Urban", "City road with mixed traffic and frequent intersections",
         "CLIP classification + map urban boundary"),
        ("L1.rural", "Rural", "Road outside urban areas with lower traffic density",
         "CLIP classification + map rural designation"),
    ]:
        road_types.add_child(ScenarioTreeNode(
            id=rid, name=rname, layer=1, description=desc, detection_method=detect,
        ))

    # Intersection types
    intersections = root.add_child(ScenarioTreeNode(
        id="L1.intersection",
        name="Intersection Types",
        layer=1,
        description="Road junction geometries",
        detection_method="HD-map topology + lane graph analysis",
    ))
    for iid, iname, desc, detect in [
        ("L1.intersection.t_junction", "T-Junction",
         "Three-way intersection forming a T shape",
         "Map node degree=3 + camera detection"),
        ("L1.intersection.crossroads", "Crossroads",
         "Four-way intersection with perpendicular roads",
         "Map node degree=4 + camera detection"),
        ("L1.intersection.roundabout", "Roundabout",
         "Circular intersection with rotary traffic flow",
         "Map circular lane geometry detection"),
    ]:
        intersections.add_child(ScenarioTreeNode(
            id=iid, name=iname, layer=1, description=desc, detection_method=detect,
        ))

    # Lane counts
    lanes = root.add_child(ScenarioTreeNode(
        id="L1.lanes",
        name="Lane Counts",
        layer=1,
        description="Number of lanes in driving direction",
        detection_method="Lane detection model + HD-map",
    ))
    for lid, lname, desc in [
        ("L1.lanes.1_lane", "1-Lane", "Single lane road"),
        ("L1.lanes.2_lane", "2-Lane", "Two-lane road"),
        ("L1.lanes.3_lane", "3-Lane", "Three-lane road"),
        ("L1.lanes.4plus_lane", "4+Lane", "Four or more lanes"),
    ]:
        lanes.add_child(ScenarioTreeNode(
            id=lid, name=lname, layer=1, description=desc,
            detection_method="Lane detection model count",
        ))

    # Road geometry
    geometry = root.add_child(ScenarioTreeNode(
        id="L1.geometry",
        name="Road Geometry",
        layer=1,
        description="Physical shape and gradient of road",
        detection_method="IMU + map + perception",
    ))
    for gid, gname, desc, detect in [
        ("L1.geometry.straight", "Straight",
         "Road with no significant curvature",
         "Curvature < 0.002 m^-1 over 100m window"),
        ("L1.geometry.curve", "Curve",
         "Road with lateral curvature",
         "Curvature >= 0.002 m^-1 over 100m window"),
        ("L1.geometry.hill", "Hill",
         "Road with significant gradient change",
         "IMU pitch > 3 degrees or map gradient > 5%"),
        ("L1.geometry.bridge", "Bridge",
         "Road elevated over another feature",
         "Map bridge attribute + no ground returns below"),
        ("L1.geometry.tunnel", "Tunnel",
         "Enclosed road passage",
         "Map tunnel attribute + sudden illumination drop"),
    ]:
        geometry.add_child(ScenarioTreeNode(
            id=gid, name=gname, layer=1, description=desc, detection_method=detect,
        ))

    return root


def _build_layer2() -> ScenarioTreeNode:
    """Build Layer 2: Traffic Infrastructure."""
    root = ScenarioTreeNode(
        id="L2",
        name="Traffic Infrastructure",
        layer=2,
        description="Permanent traffic control and guidance infrastructure",
        detection_method="perception + map",
    )

    # Traffic lights
    lights = root.add_child(ScenarioTreeNode(
        id="L2.traffic_light",
        name="Traffic Lights",
        layer=2,
        description="Signal-controlled traffic regulation",
        detection_method="Traffic light detection + color classification CNN",
    ))
    for tid, tname, desc in [
        ("L2.traffic_light.red", "Red", "Stop signal - no passage permitted"),
        ("L2.traffic_light.yellow", "Yellow", "Caution signal - prepare to stop"),
        ("L2.traffic_light.green", "Green", "Go signal - passage permitted"),
        ("L2.traffic_light.flashing", "Flashing", "Flashing signal - proceed with caution"),
    ]:
        lights.add_child(ScenarioTreeNode(
            id=tid, name=tname, layer=2, description=desc,
            detection_method="Traffic light detection + color classification CNN",
        ))

    # Signs
    signs = root.add_child(ScenarioTreeNode(
        id="L2.sign",
        name="Signs",
        layer=2,
        description="Regulatory and informational traffic signs",
        detection_method="Camera sign detection + classification model",
    ))
    for sid, sname, desc in [
        ("L2.sign.speed_limit", "Speed Limit", "Posted maximum speed regulation"),
        ("L2.sign.stop", "Stop", "Mandatory stop sign"),
        ("L2.sign.yield", "Yield", "Give way to other traffic"),
        ("L2.sign.no_entry", "No Entry", "Prohibited entry sign"),
        ("L2.sign.construction", "Construction", "Construction zone warning sign"),
    ]:
        signs.add_child(ScenarioTreeNode(
            id=sid, name=sname, layer=2, description=desc,
            detection_method="Camera sign detection + classification model",
        ))

    # Markings
    markings = root.add_child(ScenarioTreeNode(
        id="L2.marking",
        name="Markings",
        layer=2,
        description="Road surface markings and painted indicators",
        detection_method="Camera lane/marking segmentation",
    ))
    for mid, mname, desc in [
        ("L2.marking.solid_line", "Solid Line", "No-crossing boundary marking"),
        ("L2.marking.dashed_line", "Dashed Line", "Lane boundary permitting crossing"),
        ("L2.marking.crosswalk", "Crosswalk", "Pedestrian crossing marking"),
        ("L2.marking.stop_line", "Stop Line", "Line indicating where to stop"),
    ]:
        markings.add_child(ScenarioTreeNode(
            id=mid, name=mname, layer=2, description=desc,
            detection_method="Camera lane/marking segmentation",
        ))

    # Barriers
    barriers = root.add_child(ScenarioTreeNode(
        id="L2.barrier",
        name="Barriers and Guardrails",
        layer=2,
        description="Physical separation and protection structures",
        detection_method="LiDAR + camera object detection",
    ))
    for bid, bname, desc in [
        ("L2.barrier.metal_guardrail", "Metal Guardrail",
         "Steel barrier for road edge protection"),
        ("L2.barrier.concrete_barrier", "Concrete Barrier",
         "Concrete median or edge barrier"),
        ("L2.barrier.bollard", "Bollard", "Post-style access restriction"),
    ]:
        barriers.add_child(ScenarioTreeNode(
            id=bid, name=bname, layer=2, description=desc,
            detection_method="LiDAR + camera object detection",
        ))

    return root


def _build_layer3() -> ScenarioTreeNode:
    """Build Layer 3: Temporary Modifications."""
    root = ScenarioTreeNode(
        id="L3",
        name="Temporary Modifications",
        layer=3,
        description="Temporary changes to road layout and traffic conditions",
        detection_method="Perception + V2X + map updates",
    )

    # Construction zones
    construction = root.add_child(ScenarioTreeNode(
        id="L3.construction",
        name="Construction Zones",
        layer=3,
        description="Active construction or maintenance areas",
        detection_method="Construction sign/cone detection + map service alerts",
    ))
    for cid, cname, desc in [
        ("L3.construction.lane_closure", "Lane Closure",
         "One or more lanes closed for construction"),
        ("L3.construction.detour", "Detour",
         "Temporary rerouting of traffic"),
        ("L3.construction.speed_reduction", "Speed Reduction",
         "Temporary speed limit reduction in work zone"),
    ]:
        construction.add_child(ScenarioTreeNode(
            id=cid, name=cname, layer=3, description=desc,
            detection_method="Construction sign/cone detection",
        ))

    # Temporary signs
    temp_signs = root.add_child(ScenarioTreeNode(
        id="L3.temp_sign",
        name="Temporary Signs",
        layer=3,
        description="Non-permanent regulatory or warning signage",
        detection_method="Camera sign recognition (temporary class)",
    ))
    for tsid, tsname, desc in [
        ("L3.temp_sign.temp_speed_limit", "Temporary Speed Limit",
         "Reduced speed for temporary conditions"),
        ("L3.temp_sign.detour_sign", "Temporary Detour Sign",
         "Sign indicating temporary alternate route"),
        ("L3.temp_sign.cone_barrel", "Warning Cone/Barrel",
         "Portable delineation devices"),
    ]:
        temp_signs.add_child(ScenarioTreeNode(
            id=tsid, name=tsname, layer=3, description=desc,
            detection_method="Camera sign recognition (temporary class)",
        ))

    # Road closures
    closures = root.add_child(ScenarioTreeNode(
        id="L3.closure",
        name="Road Closures",
        layer=3,
        description="Complete road or lane blockages",
        detection_method="Map update + V2X + perception of barrier",
    ))
    for rcid, rcname, desc in [
        ("L3.closure.full", "Full Closure", "Road completely closed to traffic"),
        ("L3.closure.partial", "Partial Closure", "Some lanes or directions closed"),
    ]:
        closures.add_child(ScenarioTreeNode(
            id=rcid, name=rcname, layer=3, description=desc,
            detection_method="Map update + V2X + perception of barrier",
        ))

    # Events
    events = root.add_child(ScenarioTreeNode(
        id="L3.event",
        name="Events",
        layer=3,
        description="Incidents and activities causing temporary road changes",
        detection_method="V2X + traffic service API + perception",
    ))
    for eid, ename, desc in [
        ("L3.event.accident", "Accident", "Traffic collision causing obstruction"),
        ("L3.event.road_work", "Road Work", "Active maintenance or repair operations"),
    ]:
        events.add_child(ScenarioTreeNode(
            id=eid, name=ename, layer=3, description=desc,
            detection_method="V2X + traffic service API + perception",
        ))

    return root


def _build_layer4() -> ScenarioTreeNode:
    """Build Layer 4: Dynamic Objects."""
    root = ScenarioTreeNode(
        id="L4",
        name="Dynamic Objects",
        layer=4,
        description="Moving traffic participants and their behaviors",
        detection_method="Multi-sensor fusion detection + tracking",
    )

    # Vehicles
    vehicles = root.add_child(ScenarioTreeNode(
        id="L4.vehicle",
        name="Vehicles",
        layer=4,
        description="Motorized and non-motorized vehicle types",
        detection_method="LiDAR + camera 3D object detection",
    ))
    for vid, vname, desc in [
        ("L4.vehicle.car", "Car", "Passenger vehicle / sedan / SUV"),
        ("L4.vehicle.truck", "Truck", "Heavy goods vehicle or lorry"),
        ("L4.vehicle.bus", "Bus", "Public transit or coach vehicle"),
        ("L4.vehicle.motorcycle", "Motorcycle", "Two-wheeled motorized vehicle"),
        ("L4.vehicle.bicycle", "Bicycle", "Human-powered two-wheeled vehicle"),
        ("L4.vehicle.emergency", "Emergency Vehicle",
         "Police, ambulance, or fire vehicle with priority"),
    ]:
        vehicles.add_child(ScenarioTreeNode(
            id=vid, name=vname, layer=4, description=desc,
            detection_method="LiDAR + camera 3D object detection with class head",
        ))

    # Pedestrians
    pedestrians = root.add_child(ScenarioTreeNode(
        id="L4.pedestrian",
        name="Pedestrians",
        layer=4,
        description="Pedestrian types and groupings",
        detection_method="Camera pedestrian detection + attribute recognition",
    ))
    for pid, pname, desc in [
        ("L4.pedestrian.adult", "Adult", "Adult pedestrian"),
        ("L4.pedestrian.child", "Child", "Child pedestrian with unpredictable behavior"),
        ("L4.pedestrian.group", "Group", "Multiple pedestrians moving together"),
        ("L4.pedestrian.wheelchair", "Wheelchair", "Person using wheelchair or mobility aid"),
    ]:
        pedestrians.add_child(ScenarioTreeNode(
            id=pid, name=pname, layer=4, description=desc,
            detection_method="Camera pedestrian detection + attribute recognition",
        ))

    # Behaviors
    behaviors = root.add_child(ScenarioTreeNode(
        id="L4.behavior",
        name="Behaviors",
        layer=4,
        description="Dynamic behaviors of traffic participants",
        detection_method="Tracking + trajectory analysis + prediction models",
    ))
    for bid, bname, desc, detect in [
        ("L4.behavior.cut_in", "Cut-In",
         "Vehicle entering ego lane from adjacent lane",
         "Lateral velocity toward ego lane + proximity < 20m + heading alignment"),
        ("L4.behavior.lane_change", "Lane-Change",
         "Vehicle transitioning between lanes",
         "Track lateral displacement > lane_width within 3s window"),
        ("L4.behavior.jaywalking", "Jaywalking",
         "Pedestrian crossing outside designated area",
         "Pedestrian on road + no crosswalk marking within 10m"),
        ("L4.behavior.sudden_braking", "Sudden-Braking",
         "Abrupt deceleration of preceding vehicle",
         "Longitudinal deceleration > 4 m/s^2 of tracked vehicle"),
        ("L4.behavior.u_turn", "U-Turn",
         "Vehicle performing a 180-degree turn",
         "Heading change > 150 degrees within 5s window"),
        ("L4.behavior.overtaking", "Overtaking",
         "Vehicle passing another in the same direction",
         "Object passes ego with lateral offset then returns to lane"),
    ]:
        behaviors.add_child(ScenarioTreeNode(
            id=bid, name=bname, layer=4, description=desc, detection_method=detect,
        ))

    return root


def _build_layer5() -> ScenarioTreeNode:
    """Build Layer 5: Environment."""
    root = ScenarioTreeNode(
        id="L5",
        name="Environment",
        layer=5,
        description="Environmental and weather conditions affecting driving",
        detection_method="Multi-sensor analysis + weather service",
    )

    # Weather
    weather = root.add_child(ScenarioTreeNode(
        id="L5.weather",
        name="Weather",
        layer=5,
        description="Atmospheric weather conditions",
        detection_method="Rain sensor + camera analysis + LiDAR density + weather API",
    ))
    for wid, wname, desc, detect in [
        ("L5.weather.clear", "Clear",
         "No precipitation, good visibility",
         "High LiDAR density + no rain streaks + brightness normal"),
        ("L5.weather.rain", "Rain",
         "Light to moderate rainfall",
         "Rain sensor active + Gabor filter rain streak detection"),
        ("L5.weather.heavy_rain", "Heavy Rain",
         "Intense rainfall reducing visibility significantly",
         "Rain sensor high + visibility < 200m + high camera noise"),
        ("L5.weather.snow", "Snow",
         "Snowfall affecting traction and visibility",
         "Low temperature + white particle detection + surface reflectance change"),
        ("L5.weather.fog", "Fog",
         "Reduced visibility due to atmospheric moisture",
         "LiDAR range reduced + low contrast ratio + backscatter detection"),
        ("L5.weather.hail", "Hail",
         "Ice pellet precipitation",
         "Audio sensor + radar clutter pattern + weather service alert"),
    ]:
        weather.add_child(ScenarioTreeNode(
            id=wid, name=wname, layer=5, description=desc, detection_method=detect,
        ))

    # Lighting
    lighting = root.add_child(ScenarioTreeNode(
        id="L5.lighting",
        name="Lighting",
        layer=5,
        description="Ambient light conditions",
        detection_method="Camera exposure analysis + solar position",
    ))
    for lid, lname, desc, detect in [
        ("L5.lighting.daylight", "Daylight",
         "Full natural daylight illumination",
         "Mean image brightness > 120 + sun elevation > 10 degrees"),
        ("L5.lighting.dawn", "Dawn",
         "Transitional low-angle sunrise lighting",
         "Sun elevation 0-10 degrees + warm color temperature"),
        ("L5.lighting.dusk", "Dusk",
         "Transitional low-angle sunset lighting",
         "Sun elevation 0-10 degrees + reddish color cast + decreasing brightness"),
        ("L5.lighting.night", "Night",
         "Nighttime darkness with artificial lighting",
         "Mean brightness < 50 + sun elevation < 0 degrees"),
        ("L5.lighting.tunnel_dark", "Tunnel-Dark",
         "Sudden darkness transition entering tunnel",
         "Rapid brightness drop > 80% within 1s + tunnel map attribute"),
    ]:
        lighting.add_child(ScenarioTreeNode(
            id=lid, name=lname, layer=5, description=desc, detection_method=detect,
        ))

    # Road surface
    surface = root.add_child(ScenarioTreeNode(
        id="L5.surface",
        name="Road Surface",
        layer=5,
        description="Road surface conditions affecting traction",
        detection_method="Camera texture analysis + friction estimation model",
    ))
    for sid, sname, desc, detect in [
        ("L5.surface.dry", "Dry",
         "Normal dry road surface",
         "No specular reflections + normal texture contrast"),
        ("L5.surface.wet", "Wet",
         "Water film on road surface",
         "Specular reflections + reduced texture contrast + dark pavement"),
        ("L5.surface.icy", "Icy",
         "Ice-covered road surface",
         "Very low friction estimate + temperature < 0C + glossy appearance"),
        ("L5.surface.snow_covered", "Snow-Covered",
         "Snow-covered road surface",
         "High brightness + loss of lane markings + white surface"),
        ("L5.surface.gravel", "Gravel",
         "Loose gravel surface",
         "High texture variance + dust detection + map surface attribute"),
    ]:
        surface.add_child(ScenarioTreeNode(
            id=sid, name=sname, layer=5, description=desc, detection_method=detect,
        ))

    return root


def _build_layer6() -> ScenarioTreeNode:
    """Build Layer 6: Digital Information."""
    root = ScenarioTreeNode(
        id="L6",
        name="Digital Information",
        layer=6,
        description="Digital and connectivity aspects affecting perception quality",
        detection_method="System diagnostics + connectivity monitoring",
    )

    # Sensor degradation
    sensors = root.add_child(ScenarioTreeNode(
        id="L6.sensor",
        name="Sensor Degradation",
        layer=6,
        description="Sensor impairment or occlusion conditions",
        detection_method="Sensor health monitoring + output quality metrics",
    ))
    for sid, sname, desc, detect in [
        ("L6.sensor.lidar_blocked", "Lidar-Blocked",
         "Lidar sensor obstructed by dirt, snow, or object",
         "Point count drop > 50% + sector-specific void detection"),
        ("L6.sensor.camera_glare", "Camera-Glare",
         "Camera blinded by direct sunlight or headlights",
         "Saturated pixel ratio > 30% + bloom detection"),
        ("L6.sensor.radar_interference", "Radar-Interference",
         "Radar signal degraded by interference or clutter",
         "False target rate > threshold + SNR drop below nominal"),
    ]:
        sensors.add_child(ScenarioTreeNode(
            id=sid, name=sname, layer=6, description=desc, detection_method=detect,
        ))

    # Map accuracy
    map_acc = root.add_child(ScenarioTreeNode(
        id="L6.map_accuracy",
        name="Map Accuracy",
        layer=6,
        description="Quality and currency of HD map data",
        detection_method="Map confidence scoring + localization residual analysis",
    ))
    for mid, mname, desc, detect in [
        ("L6.map_accuracy.high", "High",
         "Map data current and high fidelity",
         "Localization residual < 0.1m + all features match"),
        ("L6.map_accuracy.medium", "Medium",
         "Map data with minor discrepancies",
         "Localization residual 0.1-0.5m + some features mismatch"),
        ("L6.map_accuracy.low", "Low",
         "Map data significantly diverges from reality",
         "Localization residual > 0.5m + major topology mismatch"),
        ("L6.map_accuracy.outdated", "Outdated",
         "Map data no longer reflects current road state",
         "New construction detected + map age > update threshold"),
    ]:
        map_acc.add_child(ScenarioTreeNode(
            id=mid, name=mname, layer=6, description=desc, detection_method=detect,
        ))

    # V2X availability
    v2x = root.add_child(ScenarioTreeNode(
        id="L6.v2x",
        name="V2X Availability",
        layer=6,
        description="Vehicle-to-everything communication status",
        detection_method="V2X stack diagnostics + message rate monitoring",
    ))
    for vid, vname, desc, detect in [
        ("L6.v2x.available", "Available",
         "Full V2X connectivity with low latency",
         "Message rate > 10 Hz + latency < 100ms"),
        ("L6.v2x.limited", "Limited",
         "Partial V2X connectivity or high latency",
         "Message rate 1-10 Hz or latency 100-500ms"),
        ("L6.v2x.unavailable", "Unavailable",
         "No V2X connectivity",
         "No messages received for > 5s"),
    ]:
        v2x.add_child(ScenarioTreeNode(
            id=vid, name=vname, layer=6, description=desc, detection_method=detect,
        ))

    return root


def build_default_tree() -> ScenarioTreeNode:
    """
    Build and return the complete 6-layer functional scenario tree.

    Returns the root node with all layers as children. Each layer contains
    categories and leaf-level scenario attributes following the PEGASUS/ASAM
    taxonomy structure.
    """
    root = ScenarioTreeNode(
        id="root",
        name="Functional Scenario Tree",
        layer=0,
        description="PEGASUS/ASAM-based 6-layer functional scenario taxonomy for autonomous driving",
        detection_method="",
    )

    for builder in [
        _build_layer1,
        _build_layer2,
        _build_layer3,
        _build_layer4,
        _build_layer5,
        _build_layer6,
    ]:
        layer_root = builder()
        layer_root.parent_id = root.id
        root.children.append(layer_root)

    return root


def get_node_by_id(tree: ScenarioTreeNode, node_id: str) -> Optional[ScenarioTreeNode]:
    """
    Find and return a node by its ID using depth-first traversal.

    Args:
        tree: The root node to search from.
        node_id: The ID to search for (case-sensitive).

    Returns:
        The matching ScenarioTreeNode, or None if not found.
    """
    for node in tree:
        if node.id == node_id:
            return node
    return None


def get_nodes_by_layer(tree: ScenarioTreeNode, layer: int) -> list[ScenarioTreeNode]:
    """
    Get all nodes at a given layer number.

    Args:
        tree: The root node to search from.
        layer: The layer number (1-6).

    Returns:
        List of all nodes at the specified layer.
    """
    return [node for node in tree if node.layer == layer]


def get_leaf_nodes(tree: ScenarioTreeNode) -> list[ScenarioTreeNode]:
    """
    Get all terminal (leaf) nodes in the tree.

    Args:
        tree: The root node to search from.

    Returns:
        List of leaf nodes with no children.
    """
    return [node for node in tree if node.is_leaf]
