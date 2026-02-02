import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import simplekml
import plotly.graph_objects as go
from matplotlib.ticker import MultipleLocator
from dataclasses import dataclass, asdict

# --- Data Structures ---

@dataclass
class MinefieldPoint:
    """Represents a point in the minefield (BM, TP, etc.)"""
    id: str
    description: str
    easting: float = 0.0
    northing: float = 0.0
    bearing_from_prev: float = 0.0 # Degrees
    distance_from_prev: float = 0.0 # Meters

@dataclass
class MineStrip:
    """Represents a strip/row of mines"""
    id: str
    start_reference: str # e.g., "TP1"
    end_reference: str   # e.g., "TP2"
    start_offset: float  # Distance from Start Ref along the path
    end_offset: float    # Distance from Start Ref along the path (or length)
    lateral_offset: float # distance Left (-) or Right (+) of the centerline
    mine_type: str
    density: float
    total_mines: int

@dataclass
class Landmark:
    """Represents a standalone landmark."""
    id: str
    description: str
    easting: float
    northing: float

@dataclass
class SafeLane:
    """Represents a safe lane or route."""
    id: str
    start_easting: float
    start_northing: float
    bearing: float
    distance: float
    width: float

# --- Strategic Planning Logic ---

# --- New Advanced Data ---
KNOWN_TANKS = {
    "Al Zarrar": {"width": 3.30, "track_width": 0.65},
    "Al Khalid": {"width": 3.50, "track_width": 0.7},
    "Haider": {"width": 3.45, "track_width": 0.8},
    "T 80": {"width": 3.4, "track_width": 0.58}
}

KNOWN_MINES = {
    "ND MK1": {"diameter": 0.09, "eff_rg_add": 0.06, "min_overlap": 0.015},
    "ND MK3": {"diameter": 0.76, "eff_rg_add": 0.56, "min_overlap": 0.01}
}

def calculate_advanced_stopping_power(tank_name: str, mine_name: str, density: float) -> float:
    """
    Calculates P_total (Probability of hitting at least one mine) for a single tank 
    traversing a minefield with the given density.
    Based on Flask app logic: P_CAS intersection method.
    """
    if tank_name not in KNOWN_TANKS or mine_name not in KNOWN_MINES:
        return 0.0
    
    if density <= 0:
        return 0.0

    T = KNOWN_TANKS[tank_name]["track_width"]
    W = KNOWN_TANKS[tank_name]["width"]
    N = W - 2 * T # Hull width between tracks
    
    R = KNOWN_MINES[mine_name]["diameter"] / 2
    O = KNOWN_MINES[mine_name]["min_overlap"]
    
    # Logic assumes density is for the *strip* the tank is crossing.
    # User's code iterates through 'densities' (plural), implying multiple rows.
    # For this single-value function, we assume 1 row of this density.
    # If the plan has 3 mine rows (Standard Mixed), we treat them as independent events?
    # User's code: P_total = 1 - product((1-p) for p in P_CAS_list).
    # We will compute P_CAS for ONE row here. The app can call this multiple times if needed,
    # or we assume 'density' implies the aggregate density?
    # No, typically density is per meter front for the whole field.
    # Let's align with the provided code: It takes a LIST of densities.
    # We will assume a single row calculation here for simplicity or treat 'density' as the row density.
    
    Q = 1 / density # Spacing
    D_eff = max(T + (R - O) - (Q - N - T - R + O), 0)
    
    P_A = (T + KNOWN_MINES[mine_name]["eff_rg_add"]) / Q
    P_B = P_A # Symmetric tracks
    
    P_A_intersection_B = max(D_eff / Q, 0)
    
    # Probability of Casualty for this row
    P_CAS = max(P_A + P_B - P_A_intersection_B, 0)
    
    return min(P_CAS, 1.0)
    
def calculate_cumulative_stopping_power(tank_name: str, mine_name: str, row_densities: list) -> float:
    """
    Calculates cumulative probability across multiple rows.
    """
    p_cas_list = []
    for d in row_densities:
        p = calculate_advanced_stopping_power(tank_name, mine_name, d)
        p_cas_list.append(p)
        
    # P_total = 1 - Product(1 - P_CAS_i)
    # If P_CAS is probability of HIT.
    # (1-P) is prob of MISS.
    # Product(Miss) is prob of Missing ALL.
    # 1 - Product(Miss) is prob of Hitting AT LEAST ONE.
    
    prob_miss_all = 1.0
    for p in p_cas_list:
        prob_miss_all *= (1.0 - p)
        
    return 1.0 - prob_miss_all


def calculate_stopping_power(density: float, mine_type: str) -> float:
    # ... (Keep existing as fallback or legacy wrapper) ...
    # Refactoring to allow string inputs for legacy calls
    width_map = {
        "Anti-Personnel": 0.5,
        "Anti-Tank": 3.5,
        "Fragmentation": 20.0
    }
    w = width_map.get(mine_type, 1.0)
    if density <= 0: return 0.0
    pk = 1 - math.exp(-1 * density * w)
    return max(0.0, min(1.0, pk))

def predict_density_for_pk(target_pk: float, mine_type: str) -> float:
    # ... (Keep existing) ...
    width_map = {
        "Anti-Personnel": 0.5,
        "Anti-Tank": 3.5,
        "Fragmentation": 20.0
    }
    w = width_map.get(mine_type, 1.0)
    if target_pk >= 0.99: target_pk = 0.99
    if target_pk <= 0: return 0.0
    density = -math.log(1 - target_pk) / w
    return density

def generate_tactical_layout(frontage: float, depth: float, density: float, mine_type: str) -> dict:
    """
    Generates a 'Superimposed Mixed Strip' Pattern based on Field Manuals.
    Reference: Mixed Strip with Super Imposed by Two Mine.
    
    Structure per Strip:
    1. Base Line (SSM-ESM):
       - AP Mines (Red): Spaced 1.5m (2 Paces).
    2. Second Line (+2m depth):
       - AT Mines (Blue): Spaced 4.5m (6 Paces). Start offset 3m (4 Paces).
    3. Third Line (+4m depth):
       - Frag/Special (Pink): Spaced 12m. Start offset 9m.
       
    Strips are repeated every 30m depth.
    """
    row_spacing = 30.0 
    available_depth = depth - 30 
    num_rows = max(1, int(available_depth / row_spacing) + 1)
    
    center_y = depth / 2
    total_row_span = (num_rows - 1) * row_spacing
    start_y = center_y - (total_row_span / 2)
    
    mine_points = []
    markers = []
    annotations = [] # New list for dimension arrows
    
    # Reference Markers
    markers.append({"id": "BM", "x": -10, "y": center_y, "type": "Benchmark", "color": "black", "marker": "s"}) 
    markers.append({"id": "TP1", "x": 0, "y": center_y, "type": "TP", "color": "blue", "marker": "D"})
    markers.append({"id": "TP2", "x": frontage, "y": center_y, "type": "TP", "color": "blue", "marker": "D"})
    
    row_ids = [chr(65+i) for i in range(num_rows)]
    
    for i, rid in enumerate(row_ids):
        base_y = start_y + (i * row_spacing)
        
        # --- 1. AP Line (Red) ---
        # Spacing: 2 Paces (~1.5m). Diagram says "2 Pace" / "1 Mtr". 
        # Converting Pace (0.75m) -> 2 Paces = 1.5m.
        ap_spacing = 1.5 
        current_x = 0
        while current_x <= frontage:
            mine_points.append({"x": current_x, "y": base_y, "type": "Anti-Personnel", "row": rid, "color": "red"})
            current_x += ap_spacing
            
        # SSM/ESM for Base Line
        markers.append({"id": f"SSM\n{rid}", "x": 0, "y": base_y, "type": "SSM", "color": "green", "marker": "^"})
        markers.append({"id": f"ESM\n{rid}", "x": current_x - ap_spacing, "y": base_y, "type": "ESM", "color": "red", "marker": "v"})

        # --- 2. AT Line (Blue) ---
        # Offset +2m Depth.
        # Spacing: 6 Paces (4.5m).
        # Start Offset: 4 Paces (3m).
        at_y = base_y + 2.0
        at_spacing = 4.5
        at_start = 3.0
        
        current_x = at_start
        while current_x <= frontage:
            mine_points.append({"x": current_x, "y": at_y, "type": "Anti-Tank", "row": rid, "color": "blue"})
            current_x += at_spacing

        # --- 3. Frag Line (Pink) ---
        # Offset +4m Depth.
        # Spacing: 12m.
        # Start Offset: 9m.
        frag_y = base_y + 4.0
        frag_spacing = 12.0
        frag_start = 9.0
        
        current_x = frag_start
        while current_x <= frontage:
            mine_points.append({"x": current_x, "y": frag_y, "type": "Fragmentation", "row": rid, "color": "magenta"})
            current_x += frag_spacing
            
        # --- Annotations (Only for first row to avoid clutter) ---
        if i == 0:
            # 1. AP Spacing
            annotations.append({"x1": 0, "y1": base_y-1, "x2": ap_spacing, "y2": base_y-1, "text": "2 Pace", "color": "red"})
            # 2. AT Spacing (Between 1st and 2nd AT)
            # Annotate from X=3 to X=7.5
            annotations.append({"x1": at_start, "y1": at_y+1, "x2": at_start+at_spacing, "y2": at_y+1, "text": "6 Pace (4.5m)", "color": "blue"})
            # 3. AT Start Offset
            annotations.append({"x1": 0, "y1": at_y+1, "x2": at_start, "y2": at_y+1, "text": "4 Pace (3m)", "color": "blue"})
            # 4. Frag Start Offset
            annotations.append({"x1": 0, "y1": frag_y+1, "x2": frag_start, "y2": frag_y+1, "text": "12 Pace (9m)", "color": "magenta"})

    return {
        "mines": pd.DataFrame(mine_points),
        "centerline": [(0, center_y), (frontage, center_y)],
        "markers": pd.DataFrame(markers),
        "annotations": pd.DataFrame(annotations),
        "meta": {"row_spacing": row_spacing, "mixed": True, "frontage": frontage, "depth": depth}
    }


# --- Persistence Logic ---

RECORDS_DIR = "records"

def ensure_records_dir():
    if not os.path.exists(RECORDS_DIR):
        os.makedirs(RECORDS_DIR)

def save_record(filename: str, traverse: list, strips: list, admin_data: dict, landmarks: list = [], lanes: list = []):
    """Saves the full record to JSON."""
    ensure_records_dir()
    data = {
        "admin_data": admin_data,
        "traverse": [asdict(p) for p in traverse],
        "strips": [asdict(s) for s in strips],
        "landmarks": [asdict(l) for l in landmarks],
        "lanes": [asdict(l) for l in lanes]
    }
    filepath = os.path.join(RECORDS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    return filepath

def load_record(filename: str):
    """Loads a record from JSON."""
    filepath = os.path.join(RECORDS_DIR, filename)
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Reconstruct objects
    traverse = [MinefieldPoint(**p) for p in data.get("traverse", [])]
    strips = [MineStrip(**s) for s in data.get("strips", [])]
    landmarks = [Landmark(**l) for l in data.get("landmarks", [])]
    lanes = [SafeLane(**l) for l in data.get("lanes", [])]
    admin_data = data.get("admin_data", {})
    
    return traverse, strips, landmarks, lanes, admin_data

def list_records():
    """Returns list of json files in records dir."""
    ensure_records_dir()
    return [f for f in os.listdir(RECORDS_DIR) if f.endswith('.json')]

# --- Logic Functions ---

# --- KML Logic ---

def dms_to_decimal(degrees, minutes, seconds):
    """Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees"""
    return degrees + (minutes / 60) + (seconds / 3600)

def generate_kml_from_layout(layout_data: dict, ref_lat: float, ref_lon: float, orientation: str = "N") -> str:
    """
    Generates a KML file from the cached tactical layout.
    Projects local (x, y) meters onto real-world coordinates starting at (ref_lat, ref_lon).
    """
    df_mines = layout_data.get('mines')
    if df_mines is None or df_mines.empty:
        return None
        
    kml = simplekml.Kml()
    filename = "minefield_plan.kml"
    
    # Approx conversion factors
    lat_deg_per_meter = 1 / 111111.0
    lon_deg_per_meter = 1 / (111111.0 * math.cos(math.radians(ref_lat)))
    
    # Styles (Matching the source project colors)
    style_map = {
        "Anti-Personnel": simplekml.Style(),
        "Anti-Tank": simplekml.Style(),
        "Fragmentation": simplekml.Style()
    }
    
    # Anti-Personnel: Red Circle
    style_map["Anti-Personnel"].iconstyle.color = 'ff0000ff' # Red
    style_map["Anti-Personnel"].iconstyle.scale = 0.5 # Smaller
    style_map["Anti-Personnel"].iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    
    # Anti-Tank: Blue Target/Diamond
    style_map["Anti-Tank"].iconstyle.color = 'ffff0000' # Blue
    style_map["Anti-Tank"].iconstyle.scale = 0.8 # Larger
    style_map["Anti-Tank"].iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/target.png'
    
    # Fragmentation: Green Star
    style_map["Fragmentation"].iconstyle.color = 'ff00ff00' # Green
    style_map["Fragmentation"].iconstyle.scale = 0.7
    style_map["Fragmentation"].iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/star.png'
    
    meta = layout_data.get('meta', {})
    frontage = meta.get('frontage', 500.0)
    depth = meta.get('depth', 200.0)
    
    # Define Perimeter Buffer (5m)
    buff = 5.0
    # Local Coordinates of the Box (Closed Loop)
    # Assuming (0,0) is Local Origin (TP1).
    local_corners = [
        (-buff, -buff),
        (frontage + buff, -buff),
        (frontage + buff, depth + buff),
        (-buff, depth + buff),
        (-buff, -buff)
    ]
    
    geo_coords = []
    
    for lx, ly in local_corners:
         # Apply same rotation logic
        if orientation == "N":
            d_lat = ly * lat_deg_per_meter
            d_lon = lx * lon_deg_per_meter
        elif orientation == "S":
            d_lat = -ly * lat_deg_per_meter
            d_lon = -lx * lon_deg_per_meter # Inverted grid
        elif orientation == "E":
            d_lat = -lx * lat_deg_per_meter
            d_lon = ly * lon_deg_per_meter
        elif orientation == "W":
            d_lat = lx * lat_deg_per_meter
            d_lon = -ly * lon_deg_per_meter
        else:
            d_lat = ly * lat_deg_per_meter
            d_lon = lx * lon_deg_per_meter
            
        geo_coords.append((ref_lon + d_lon, ref_lat + d_lat))
        
    # Add Perimeter Polygon (LineString)
    pol = kml.newlinestring(name="Perimeter Boundary")
    pol.coords = geo_coords
    pol.style.linestyle.color = 'ff0000ff' # Red
    pol.style.linestyle.width = 3
    
    # Add Corner Markers (optional, for visibility)
    # kml.newpoint(name="Boundary Corner", coords=[geo_coords[0]])
    
    for _, mine in df_mines.iterrows():
        mx = mine['x']
        my = mine['y']
        m_type = mine['type']
        
        # Apply Orientation / Rotation
        # Assuming (0,0) is at ref_lat/lon.
        # N: Forward (Y) is North. Right (X) is East.
        if orientation == "N":
            d_lat = my * lat_deg_per_meter
            d_lon = mx * lon_deg_per_meter
        elif orientation == "S":
            d_lat = -my * lat_deg_per_meter
            d_lon = mx * lon_deg_per_meter # X is still East? Or should X be inverted too? Usually grid is relative. 
            # If facing South, "Right" is West. Let's assume standard grid rotation.
            # If Front is South... Y goes South (-Lat). X (Right) goes West (-Lon).
            d_lon = -mx * lon_deg_per_meter
        elif orientation == "E":
            # Front is East. Y -> East (+Lon). X (Right) -> South (-Lat).
            d_lat = -mx * lat_deg_per_meter
            d_lon = my * lon_deg_per_meter
        elif orientation == "W":
            # Front is West. Y -> West (-Lon). X (Right) -> North (+Lat).
            d_lat = mx * lat_deg_per_meter
            d_lon = -my * lon_deg_per_meter
        else:
            d_lat = my * lat_deg_per_meter
            d_lon = mx * lon_deg_per_meter

        p_lat = ref_lat + d_lat
        p_lon = ref_lon + d_lon
        
        p = kml.newpoint(name=m_type, coords=[(p_lon, p_lat)])
        p.style = style_map.get(m_type, style_map["Anti-Tank"])
        
    kml.save(filename)
    return filename

def calculate_coords(start_x, start_y, bearing_deg, distance):
    """Calculates new coordinates based on bearing and distance."""
    bearing_rad = math.radians(bearing_deg)
    delta_e = distance * math.sin(bearing_rad)
    delta_n = distance * math.cos(bearing_rad)
    return start_x + delta_e, start_y + delta_n

def calculate_strip_coords(p1: MinefieldPoint, p2: MinefieldPoint, lateral_offset: float):
    """
    Calculates the start and end coordinates of a strip parallel to p1-p2.
    offset < 0 : Left
    offset > 0 : Right
    """
    dx = p2.easting - p1.easting
    dy = p2.northing - p1.northing
    length = math.sqrt(dx*dx + dy*dy)
    
    if length == 0:
        return p1.easting, p1.northing, p2.easting, p2.northing

    # Standard vector math for 2D offset
    # Normal vector (Right: +90 deg rotation -> (dy, -dx)) 
    # Check: North (0, 1) -> Right (1, 0). (1, -0) = (1, 0). Correct.
    ux = dx / length
    uy = dy / length
    
    nx = uy # dy/len
    ny = -ux # -dx/len
    
    # Wait, in the previous code I did:
    # nx = dy
    # ny = -dx
    # This was NOT normalized if I didn't divide by length. 
    # Ah, I see in previous code:
    # ux = dx / length
    # uy = dy / length
    # nx = dy 
    # ny = -dx 
    # This was WRONG in the previous file if I didn't normalize n!
    # Let's fix it here.
    
    # Normalized Normal Vector (Right)
    nx = dy / length
    ny = -dx / length - 0 # avoid negative zero annoyance if needed, but meh

    sx = p1.easting + nx * lateral_offset
    sy = p1.northing + ny * lateral_offset
    
    return sx, sy, ex, ey


def generate_user_defined_trace(start_e, start_n, first_bearing, frontage, strip_id=1):
    """
    Generates a strip trace based on 'First Leg Bearing'.
    - First Leg = first_bearing.
    - Subsequent Legs alternate: (first_bearing + 20), (first_bearing), (first_bearing + 20)...
      (Creating a Sawtooth pattern that drifts slightly or ZigZags relative to an axis B+10).
    - Leg Distance: ~140m (Constraint: < 150m).
    - strip_id: Used for labeling (SSM-{id}, TP-{id}-{i}, ESM-{id}).
    """
    # 1. Geometry Constants
    target_leg_dist = 140.0
    offset_angle = 20.0
    
    # Average Axis is roughly First_Bearing + (Offset/2) = B + 10
    # But calculating exact coverage depends on the projection along the "General Front" 
    # which we assume is determined by the trace itself.
    # We simply add legs until the total length (or projected length) exceeds frontage.
    # Let's use strict Euclidian length of the trace for 'Frontage' or Projected? 
    # Usually Frontage is the straight line distance start-to-end.
    # We'll approximate: 
    # Progress per pair (Leg 1 + Leg 2) along the mean axis (B+10):
    # Pair Distance = 2 * 140 * cos(10 deg). 
    
    # Let's just generate legs until sum of lengths >= frontage? 
    # Or sum of projected lengths?
    # Let's stick to projected length along the average axis (B + 10).
    
    avg_axis_rad = math.radians(first_bearing + (offset_angle / 2))
    
    # Progress per leg (approx)
    # Leg 1 (Bearing B) projected on B+10 -> cos(10)
    # Leg 2 (Bearing B+20) projected on B+10 -> cos(10)
    prog_per_leg = target_leg_dist * math.cos(math.radians(offset_angle / 2))
    
    num_legs = math.ceil(frontage / prog_per_leg)
    
    # Adjust leg distance to fit frontage exactly?
    # User said "Distance is... less than 150".
    # Fixed 140m is safe. Let's start with that or scale down if frontage is small.
    actual_leg_dist = target_leg_dist
    if frontage < prog_per_leg: # Very short strip
         actual_leg_dist = frontage / math.cos(math.radians(offset_angle / 2))
         num_legs = 1

    # Custom Labeling based on Strip ID
    # SSM-1, SSM-2 etc.
    label_ssm = f"SSM-{strip_id}"
    trace_points = [(label_ssm, start_e, start_n)]
    curr_e, curr_n = start_e, start_n
    
    # Pattern: B, B+20, B, B+20...
    # Or B, B-20? User said "+- 20". Let's default to Alternating between B and B+20.
    
    bearings = [first_bearing, (first_bearing + offset_angle) % 360]
    
    for i in range(1, num_legs + 1):
        if i == num_legs:
            # ESM-1
            label = f"ESM-{strip_id}"
        else:
            # TP-1-1 where Strip=1, TP=1
            label = f"TP-{strip_id}-{i}"
        
        # Select Bearing (0, 1, 0, 1...)
        b = bearings[(i-1) % 2]
        
        # Calc
        curr_e, curr_n = calculate_coords(curr_e, curr_n, b, actual_leg_dist)
        trace_points.append((label, curr_e, curr_n))
        
    return trace_points


def _generate_strip_mines(seg_len, num_rows, strip_depth, mine_inputs, row_height, row_1_base, spacings):
    """
    Helper to generate mines for a specific segment relative to its start.
    If num_rows == 1, generates a SINGLE MIXED ROW (Frag + AT + AP) 
    using the specific MANUAL DRILL (Pace-based offsets).
    """
    mines_rel = []
    counts = {"ap": 0, "at": 0, "frag": 0}
    
    # Constants
    PACE = 0.75 # meters
    
    # 1. Effective Length Logic
    # Mines stop 3m BEFORE the TP.
    # So if we are generating for a segment of length L, valid range is [0, L-3.0]
    # But wait, the segment passed here is 'seg_len'.
    # Does 'seg_len' include the 3m safety from previous? 
    # The caller manages origin. We just need to stop at L-3.
    # Note: If this is the LAST segment (ESM), maybe we go to end?
    # Generally, keep it consistent: Stop 3m short logic applies to TPs.
    # Let's apply valid_end = seg_len - 3.0
    valid_end = list_len = seg_len - 3.0
    if valid_end < 0: valid_end = 0 # Safety
    
    # If Single Row (User Requirement: "One Strip = One Row of Mixed Mines")
    if num_rows == 1:
        # 1. Frag Mines (M16) - Strip 1 Only (Caller filters if needed, but we generate)
        # Start: 9m. Spacing: 12m.
        # Offset: 6 Paces (~4.5m).
        # Side: Enemy Side Only (+Y).
        frag_start = 9.0
        frag_spacing = 12.0
        frag_offset = 6 * PACE # 4.5m
        
        cx = frag_start
        while cx <= valid_end:
            mines_rel.append({"x": cx, "y": frag_offset, "type": "frag", "row": 1})
            counts["frag"] += 1
            cx += frag_spacing
            
        # 2. AT Mines
        # Start: 6m. Spacing: 3m.
        # Offset: 4 Paces (~3.0m).
        # Pattern: Alternating (Enemy -> Own -> Enemy...)
        at_start = 6.0
        at_spacing = spacings.get('at', 3.0)
        at_offset_val = 4 * PACE # 3.0m
        
        cx = at_start
        side = 1 # 1 = Enemy (+), -1 = Own (-)
        # "First mine ... TOWARDS EN" -> Start with +1
        
        while cx <= valid_end:
            y_pos = side * at_offset_val
            mines_rel.append({"x": cx, "y": y_pos, "type": "at", "row": 1})
            counts["at"] += 1
            cx += at_spacing
            side *= -1 # Flip side
            
        # 3. AP Mines
        # Start: 3m. Spacing: 1m.
        # Offset: 2 Paces (~1.5m).
        # Pattern: Alternating (Enemy -> Own -> Enemy...)
        # "2 PACE TOWARDS EN" -> Start with +1
        ap_start = 3.0
        ap_spacing = spacings.get('ap', 1.0)
        ap_offset_val = 2 * PACE # 1.5m
        
        cx = ap_start
        side = 1 # Start Enemy
        
        while cx <= valid_end:
            y_pos = side * ap_offset_val
            mines_rel.append({"x": cx, "y": y_pos, "type": "ap", "row": 1})
            counts["ap"] += 1
            cx += ap_spacing
            side *= -1 # Flip side
            
        return mines_rel, counts
        
    # Standard Multi-Row Logic (Legacy or if user forces > 1)
    for r in range(1, num_rows + 1):
        # Calculate Y (Cross-distance)
        y_rel = (r - 1) * row_height - (strip_depth / 2) + (row_height/2)
        
        row_type_idx = (r - 1) % 4
        current_x = 0
        
        if row_type_idx == 0: # Row 1: Frag + AP
            # Frag (12m spacing)
            frag_spacing = 12.0
            cx = 0
            while cx <= seg_len:
                mines_rel.append({"x": cx, "y": y_rel, "type": "frag", "row": r})
                counts["frag"] += 1
                cx += frag_spacing
            # AP
            ap_spacing = spacings['ap']
            cx = 0
            while cx <= seg_len:
                mines_rel.append({"x": cx, "y": y_rel + 0.5, "type": "ap", "row": r})
                counts["ap"] += 1
                cx += ap_spacing
                
        elif row_type_idx == 1: # Row 2: AT
            at_spacing = spacings['at']
            cx = 0
            while cx <= seg_len:
                mines_rel.append({"x": cx, "y": y_rel, "type": "at", "row": r})
                counts["at"] += 1
                cx += at_spacing
                
        elif row_type_idx == 2: # Row 3: AP
            ap_spacing = spacings['ap']
            cx = 0
            while cx <= seg_len:
                mines_rel.append({"x": cx, "y": y_rel, "type": "ap", "row": r})
                counts["ap"] += 1
                cx += ap_spacing
                
        elif row_type_idx == 3: # Row 4: AT
             at_spacing = spacings['at']
             cx = 0
             while cx <= seg_len:
                mines_rel.append({"x": cx, "y": y_rel, "type": "at", "row": r})
                counts["at"] += 1
                cx += at_spacing
                
    return mines_rel, counts

def generate_minefield_data(strip_configs: list, mine_inputs, frontage=500.0, depth=200.0, detail_view=False):
    """
    Generates minefield plan for MULTIPLE STRIPS.
    strip_configs: List of dicts, each containing:
      - 'trace': List of (Label, E, N) tuples (SSM -> TPs -> ESM).
      - 'row_count': Number of rows for this strip (default 4).
      - 'depth': Depth of this strip (for row spacing spacing).
      - 'is_first': Boolean (True = Include M16, False = No M16).
    """
    
    all_abs_mines = []
    total_counts = {"ap": 0, "at": 0, "frag": 0, "dp": 0}
    
    # Process each strip
    for s_idx, config in enumerate(strip_configs):
        trace = config['trace']
        num_rows = config.get('row_count', 4)
        strip_depth = config.get('depth', 30.0) # Depth of the specific strip pattern
        is_first = config.get('is_first', False)
        
        # Calculate Row Height / Base
        row_height = strip_depth / num_rows if num_rows > 0 else 0
        row_1_base = row_height * 1
        
        # Determine Spacings (Standard Drill)
        spacings = {'ap': mine_inputs.get('ap', 1.0), 'at': mine_inputs.get('at', 3.0)}
        if spacings['ap'] <= 0: spacings['ap'] = 1.0
        if spacings['at'] <= 0: spacings['at'] = 3.0
        
        # --- Generate Mines for Each Segment (Leg) of the Trace ---
        # A trace has N points -> N-1 Segments.
        # Segment j connects Point j and Point j+1.
        
        # Map Points for easy lookup
        # trace example: [('SSM', 1000, 1000), ('TP1', 1100, 1100)...]
        
        for i in range(len(trace) - 1):
            p_start = trace[i]
            p_end = trace[i+1]
            
            start_label, start_e, start_n = p_start
            end_label, end_e, end_n = p_end
            
            # Calculate Segment Geometry
            dx = end_e - start_e
            dy = end_n - start_n
            seg_len = math.sqrt(dx*dx + dy*dy)
            seg_bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
            
            # Generate Local Mines (Relative to centerline)
            # Function _generate_strip_mines returns mines with x along frontage (0 to len).
            # We override M16 logic based on 'is_first'.
            
            # We modify _generate_strip_mines slightly to accept 'allow_frag' or filter after?
            # Let's filter after for simplicity or modify the helper.
            # Actually, let's just use the current helper and post-filter if !is_first.
            
            mines_rel, counts = _generate_strip_mines(seg_len, num_rows, strip_depth, mine_inputs, row_height, row_1_base, spacings)
            
            # Filter M16s if not first strip
            if not is_first:
                mines_rel = [m for m in mines_rel if m['type'] != 'frag']
                # Correct counts
                counts['frag'] = 0 # Since we removed them
            
            # Update Total Counts
            for k in total_counts:
                if k in counts:
                    total_counts[k] += counts[k]

            # --- Transform to Global Coordinates ---
            # Origin for this segment:
            # If Segment 0 (SSM->TP1): Start at SSM coords.
            # If Segment > 0 (TP->TP): Start at DP (TP + 3m).
            
            origin_e, origin_n = start_e, start_n
            
            if i > 0:
                # Add DP!
                # DP is 3m along the NEW bearing from the TP (which is p_start)
                dp_dist = 3.0
                rad_b = math.radians(seg_bearing)
                dp_e = start_e + (dp_dist * math.sin(rad_b))
                dp_n = start_n + (dp_dist * math.cos(rad_b))
                origin_e, origin_n = dp_e, dp_n
                
                # Add DP to results
                all_abs_mines.append({
                    "Grid Easting": round(dp_e, 1),
                    "Grid Northing": round(dp_n, 1),
                    "type": "DP", 
                    "label": f"DP-{s_idx+1}-{i}"
                })
                total_counts['dp'] += 1
            
            # Basis Vectors
            # X-Axis = Segment Bearing
            # Y-Axis = Segment Bearing - 90 (Left/Enemy Side)
            strip_azimuth = seg_bearing
            depth_azimuth = (strip_azimuth - 90) % 360
            
            sin_s, cos_s = math.sin(math.radians(strip_azimuth)), math.cos(math.radians(strip_azimuth))
            sin_d, cos_d = math.sin(math.radians(depth_azimuth)), math.cos(math.radians(depth_azimuth))
            
            count_in_seg = 0
            for m in mines_rel:
                rx = m['x']
                ry = m['y'] 
                
                dx_x = rx * sin_s
                dy_x = rx * cos_s
                dx_y = ry * sin_d
                dy_y = ry * cos_d
                
                final_e = origin_e + dx_x + dx_y
                final_n = origin_n + dy_x + dy_y
                
                count_in_seg += 1
                mine_id = f"M-{s_idx+1}-{i+1}-{count_in_seg:03d}"
                
                all_abs_mines.append({
                    "Grid Easting": round(final_e, 1),
                    "Grid Northing": round(final_n, 1),
                    "type": m['type'],
                    "row": m.get('row'),
                    "segment": i+1,
                    "strip_id": s_idx + 1,
                    "mine_id": mine_id,
                    "label": m['type'].upper()
                })

    # --- Generate Stores & Abstract Viz ---
    # We can perform a dummy viz generation for the Quick Calculator chart or reuse existing logic
    # But since we have multiple strips, a single abstract chart might not fit.
    # We will return the FULL global list for the main map.
    
    # Stores DataFrame
    total_mines_val = sum([total_counts['ap'], total_counts['at'], total_counts['frag']])
    
    # 1. Perimeter (Fencing)
    # Estimate: Total Frontage x 2 (Front/Rear) + Depth x 2 (Sides)
    # For a standard field.
    perimeter = (frontage * 2) + (depth * 2)
    
    # 2. Barbed Wire (3 Strand Fence)
    # Standard Coil = 200m usually.
    # Total Wire Length
    total_wire_m = perimeter * 3
    # Coils (assume 200m per coil for output, or just display meters)
    # User asked for "Barbed Wire Required", likely meters or bundles. 
    # Let's show meters.
    
    # 3. Marking Tape
    # One strand for perimeter? Or specific strip marking?
    # Usually Safe Lane marking + Perimeter.
    # Let's approx as Perimeter x 2 (Inner/Outer) + Extra.
    # Simple heuristic: Perimeter * 1.5
    marking_tape_m = perimeter * 1.5
    
    # 4. Lorries (Transport)
    # Weights (Approx):
    # AP = 0.5kg, AT = 10kg, Frag = 4kg
    # Wire = 0.1kg/m ? (Coil 25kg/200m => 0.125kg/m)
    # Pickets = 2kg each (Every 3m => Perimeter/3 pickets)
    
    w_ap = total_counts['ap'] * 0.5
    w_at = total_counts['at'] * 10.0
    w_frag = total_counts['frag'] * 4.0
    
    num_pickets = math.ceil(perimeter / 3.0)
    w_pickets = num_pickets * 2.0
    w_wire = total_wire_m * 0.15
    
    total_weight_kg = w_ap + w_at + w_frag + w_pickets + w_wire
    
    # 3 Ton Lorry (3000kg)
    # Capacity utilization factor 0.8 => 2400kg
    lorries = math.ceil(total_weight_kg / 2400.0)
    if lorries < 1: lorries = 1
    
    # 5. Time (Nights)
    # Rate: 1 Section (10 men) lays ~100 mines/night? 
    # Or 1 Platoon (30 men) lays ~300 mines?
    # Let's assume a Laying Party of 1 Platoon equivalent.
    # Rate = 400 mines / night (Standard drill approx)
    time_nights = math.ceil(total_mines_val / 400.0)
    
    stores_data = {
        "Total Mines": [total_mines_val],
        "Total AP": [total_counts['ap']],
        "Total AT": [total_counts['at']],
        "Total Frag": [total_counts['frag']],
        "Total DPs": [total_counts['dp']],
        "Perimeter Fence (m)": [perimeter],
        "Barbed Wire (m)": [total_wire_m],
        "Marking Tape (m)": [marking_tape_m],
        "Steel Pickets (Long)": [num_pickets],
        "Lorries (3-Ton)": [lorries],
        "Time Needed (Nights)": [time_nights]
    }
    
    # Dummy Fig (We'll use the Geodetic Plotly primarily now)
    fig, _ = plt.subplots() 
    fig.text(0.5, 0.5, "See Geodetic Map", ha='center')
    
    return None, None, pd.DataFrame(stores_data), all_abs_mines

def plot_survey_chain(points: list, mine_points: list = None):

    """
    Plots a sequence of geodetic points (Landmark -> SSM -> TPs).
    points: List of tuples (label, easting, northing)
    mine_points: Optional list of dicts {'Grid Easting', 'Grid Northing', 'type'}
    """
    if not points:
        return None
        
    e_vals = [p[1] for p in points]
    n_vals = [p[2] for p in points]
    
    fig, ax = plt.subplots(figsize=(12, 6)) # Landscape (Wider for full field)
    
    # 1. Plot Survey Line
    ax.plot(e_vals, n_vals, 'b--', linewidth=1, alpha=0.5, label='Survey Trace')
    
    # 2. Plot Survey Points and Safety Zones
    for l, e, n in points:
        if l == "Landmark" or l == "LM":
            ax.plot(e, n, 'ks', markersize=8, label='Landmark' if l=='LM' else "")
            offset_y = -5
        elif l == "SSM" or l == "ESM":
            ax.plot(e, n, 'rs', markersize=6, label=l if l in ['SSM','ESM'] else "")
            # Safety Zone Calculation (3m radius)
            # Visualize the "No Mine" area
            circle = plt.Circle((e, n), 3.0, color='gray', fill=False, linestyle='--', linewidth=0.8, alpha=0.5)
            ax.add_patch(circle)
            offset_y = 5
        else:
            # Turning Points
            ax.plot(e, n, 'bo', markersize=5, label='TP' if "TP1" in l else "")
            # Safety Zone (3m radius)
            circle = plt.Circle((e, n), 3.0, color='gray', fill=False, linestyle='--', linewidth=0.8, alpha=0.5)
            ax.add_patch(circle)
            offset_y = 5
            
        ax.text(e, n + offset_y, f"  {l}", fontsize=9, ha='left', va='center', fontweight='bold')
        
    # Legend Proxy for Safety Zone
    ax.plot([], [], 'o', markeredgecolor='gray', markerfacecolor='none', linestyle='--', markersize=10, label='Safety Zone (3m)')
        
    # 3. Plot Mines (Absolute Coordinates)
    # 3. Plot Mines (Absolute Coordinates)
    if mine_points:
        # Separate by type
        ap_x = [m['Grid Easting'] for m in mine_points if m['type'] == 'ap']
        ap_y = [m['Grid Northing'] for m in mine_points if m['type'] == 'ap']
        
        at_x = [m['Grid Easting'] for m in mine_points if m['type'] == 'at']
        at_y = [m['Grid Northing'] for m in mine_points if m['type'] == 'at']
        
        fr_x = [m['Grid Easting'] for m in mine_points if m['type'] == 'frag']
        fr_y = [m['Grid Northing'] for m in mine_points if m['type'] == 'frag']
        
        dp_x = [m['Grid Easting'] for m in mine_points if m.get('type') == 'DP']
        dp_y = [m['Grid Northing'] for m in mine_points if m.get('type') == 'DP']
        
        # Plot Anti-Personnel
        if ap_x:
            ax.plot(ap_x, ap_y, '.', color='red', markersize=3, alpha=0.6, label='Anti-Personnel')
            ax.plot(ap_x, ap_y, color='red', linewidth=0.3, alpha=0.3) # Zig-zag line
            
        # Plot Anti-Tank
        if at_x:
            ax.plot(at_x, at_y, 's', color='blue', markersize=4, alpha=0.7, label='Anti-Tank')
            ax.plot(at_x, at_y, color='blue', linewidth=0.3, alpha=0.3) # Zig-zag line

        # Plot Fragmentation (M16)
        if fr_x:
            ax.plot(fr_x, fr_y, 'x', color='orange', markersize=6, markeredgewidth=1.5, label='Fragmentation (M16)')
            ax.plot(fr_x, fr_y, color='orange', linestyle='-', linewidth=0.5, alpha=0.5, label='Mine Strip')
            
        # Plot Direction Picquets (DP)
        if dp_x:
            ax.plot(dp_x, dp_y, '^', color='green', markersize=8, markeredgecolor='black', label='Direction Picquet')
            for x, y in zip(dp_x, dp_y):
                ax.text(x, y+2, "DP", color='green', fontsize=8, fontweight='bold', ha='center')
        
    ax.set_title("Geodetic Survey & Integrated Mine Layout")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize='small', framealpha=0.9)
    
    return fig


def plot_survey_chain_interactive(points: list, mine_points: list = None, landmark_coord: tuple = None, setting_out_lines: list = None):
    """
    Plots Geodetic points and mines using Plotly for interactive zoom/pan.
    - landmark_coord: (easting, northing) tuple for the Landmark.
    - setting_out_lines: List of tuples [(StartE, StartN, EndE, EndN), ...] for layout lines (e.g. LM->SSM).
    returns: plotly.graph_objects.Figure
    """
    if not points and not mine_points:
        return None
        
    fig = go.Figure()
    
    # 0. Landmark & Setting Out Lines
    if landmark_coord:
        lx, ly = landmark_coord
        # X=LX (Easting), Y=LY (Northing)
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly],
            mode='markers+text',
            marker=dict(symbol='square', color='black', size=12, line=dict(width=2, color='white')),
            text=[f"LANDMARK\nE:{lx:.0f} N:{ly:.0f}"],
            textposition="bottom center",
            name='Landmark'
        ))
        
    if setting_out_lines:
        for x1, y1, x2, y2 in setting_out_lines:
             # X=[E1, E2], Y=[N1, N2]
             fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2],
                mode='lines',
                line=dict(color='black', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

    # 1. Survey Trace
    if points:
        e_vals = [p[1] for p in points]
        n_vals = [p[2] for p in points]
        
        # X=E, Y=N
        fig.add_trace(go.Scatter(
            x=e_vals, y=n_vals,
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='Survey Trace'
        ))
        
        # 2. Survey Points & Safety Zones
        for l, e, n in points:
            # Determine Symbol/Color
            if "Landmark" in l or "LM" in l:
                continue # Handled above explicitly if in list
            elif "SSM" in l or "ESM" in l:
                symbol = 'square'
                color = 'red'
                size = 8
                # Safety Zone (Plotly Shape) - Standard
                # X = E, Y = N
                fig.add_shape(type="circle",
                    xref="x", yref="y",
                    x0=e-3, y0=n-3, x1=e+3, y1=n+3,
                    line_color="gray", line_dash="dash", opacity=0.5
                )
            else:
                # TP
                symbol = 'circle'
                color = 'blue'
                size = 6
                # Safety Zone
                fig.add_shape(type="circle",
                    xref="x", yref="y",
                    x0=e-3, y0=n-3, x1=e+3, y1=n+3,
                    line_color="gray", line_dash="dash", opacity=0.5
                )
                
            fig.add_trace(go.Scatter(
                x=[e], y=[n],
                mode='markers+text',
                marker=dict(symbol=symbol, color=color, size=size),
                text=[f"  {l}"],
                textposition="top right",
                showlegend=False,
                hoverinfo='text'
            ))

    # 3. Mines
    if mine_points:
        # Helper to filter using list comprehension for robustness
        aps = [m for m in mine_points if m.get('type') == 'ap']
        ats = [m for m in mine_points if m.get('type') == 'at']
        frags = [m for m in mine_points if m.get('type') == 'frag']
        dps = [m for m in mine_points if m.get('type') == 'DP']
        
        # Plot AP
        if aps:
            x = [m['Grid Easting'] for m in aps]
            y = [m['Grid Northing'] for m in aps]
            # Plot X=E, Y=N
            fig.add_trace(go.Scatter(
                x=x, y=y, 
                mode='markers', # Just markers for mixed
                marker=dict(color='red', size=4, symbol='circle'),
                name='Anti-Personnel',
                hoverinfo='text',
                text=[f"AP-{m.get('row')}" for m in aps]
            ))
            
        # Plot AT
        if ats:
            x = [m['Grid Easting'] for m in ats]
            y = [m['Grid Northing'] for m in ats]
            fig.add_trace(go.Scatter(
                x=x, y=y, 
                mode='markers',
                marker=dict(color='blue', size=6, symbol='square'),
                name='Anti-Tank',
                hoverinfo='text',
                text=["AT" for m in ats]
            ))
            
        # Plot M16
        if frags:
            x = [m['Grid Easting'] for m in frags]
            y = [m['Grid Northing'] for m in frags]
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color='orange', size=8, symbol='x-thin', line_width=2),
                name='Fragmentation (M16)',
                hoverinfo='text',
                text=["M16" for m in frags]
            ))
            
        # Plot DPs
        if dps:
            x = [m['Grid Easting'] for m in dps]
            y = [m['Grid Northing'] for m in dps]
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text=["DP" for _ in dps],
                textposition="top center",
                name='Direction Picquet'
            ))
            
    # North Arrow Annotation
    # Standard North Up = Up Arrow
    fig.add_annotation(
        text="NORTH ⬆", 
        xref="paper", yref="paper",
        x=0.05, y=0.95, # Top Left
        showarrow=False,
        font=dict(size=14, color="black"),
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        opacity=0.8
    )
            
    # Layout configuration for Zoom/Pan
    fig.update_layout(
        title=dict(text="Interactive Geodetic Survey & Mine Layout (1 Grid Sq = 1m)", font=dict(size=20)),
        width=1200, height=800, # Landscape Ratio
        dragmode='pan', # Default tool
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='white', # Clean background for grid visibility
        xaxis=dict(
            title='Grid Easting (meters)',
            showgrid=True,
            gridcolor='#e0e0e0',
            zeroline=False,
            # 1:1 Aspect Ratio locked to Y
            scaleanchor='y', 
            scaleratio=1,
            # Major Grid every 10m
            dtick=10, 
            # Minor Grid every 1m (The "1 Meter Square")
            minor=dict(showgrid=True, gridcolor='#f5f5f5', dtick=1, gridwidth=0.5),
            tickformat=".0f" # Show integers for GR
        ),
        yaxis=dict(
            title='Grid Northing (meters)',
            showgrid=True,
            gridcolor='#e0e0e0',
            zeroline=False,
            # Major Grid every 10m
            dtick=10,
            # Minor Grid every 1m
            minor=dict(showgrid=True, gridcolor='#f5f5f5', dtick=1, gridwidth=0.5),
            tickformat=".0f"
        )
    )
    
    return fig


    # 3. Mines
    if mine_points:
        # Helper to filter using list comprehension for robustness
        aps = [m for m in mine_points if m.get('type') == 'ap']
        ats = [m for m in mine_points if m.get('type') == 'at']
        frags = [m for m in mine_points if m.get('type') == 'frag']
        dps = [m for m in mine_points if m.get('type') == 'DP']
        
        # Plot AP
        if aps:
            x = [m['Grid Easting'] for m in aps]
            y = [m['Grid Northing'] for m in aps]
            fig.add_trace(go.Scatter(
                x=x, y=y, 
                mode='markers+lines', # Zig-zag line
                marker=dict(color='red', size=4, symbol='circle'),
                line=dict(color='red', width=0.5),
                name='Anti-Personnel',
                hoverinfo='text',
                text=[f"AP-{m.get('row')}" for m in aps]
            ))
            
        # Plot AT
        if ats:
            x = [m['Grid Easting'] for m in ats]
            y = [m['Grid Northing'] for m in ats]
            fig.add_trace(go.Scatter(
                x=x, y=y, 
                mode='markers+lines',
                marker=dict(color='blue', size=6, symbol='square'),
                line=dict(color='blue', width=0.5),
                name='Anti-Tank',
                hoverinfo='text',
                text=[f"AT-{m.get('row')}" for m in ats]
            ))
            
        # Plot M16
        if frags:
            x = [m['Grid Easting'] for m in frags]
            y = [m['Grid Northing'] for m in frags]
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers+lines',
                marker=dict(color='orange', size=8, symbol='x-thin', line_width=2),
                line=dict(color='orange', width=0.5),
                name='Fragmentation (M16)',
                hoverinfo='text',
                text=["M16" for m in frags]
            ))
            
        # Plot DPs
        if dps:
            x = [m['Grid Easting'] for m in dps]
            y = [m['Grid Northing'] for m in dps]
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text=["DP" for _ in dps],
                textposition="top center",
                name='Direction Picquet'
            ))
            
    # Layout configuration for Zoom/Pan
    fig.update_layout(
        title="Interactive Geodetic Survey & Mine Layout (Meter Scale)",
        width=1000, height=700,
        dragmode='pan', # Default tool
        showlegend=True,
        hovermode='closest',
        xaxis=dict(
            title='Grid Easting (meters)',
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            # 1:1 Aspect Ratio locked to Y
            scaleanchor='y', 
            scaleratio=1,
            # Minor Grid (1m precision when zoomed)
            minor=dict(showgrid=True, gridcolor='#f0f0f0', dtick=1)
        ),
        yaxis=dict(
            title='Grid Northing (meters)',
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            # Minor Grid
            minor=dict(showgrid=True, gridcolor='#f0f0f0', dtick=1)
        )
    )
    
    return fig

# --- KML EXPORT LOGIC ---

def local_grid_to_latlon(ref_lat, ref_lon, ref_e, ref_n, target_e, target_n):
    """
    Converts local grid (E, N) to (Lat, Lon) based on a reference point.
    Uses simple spherical approximation suitable for small areas (< 10km).
    """
    R = 6378137.0 # Earth Radius in meters
    
    dn = target_n - ref_n
    de = target_e - ref_e
    
    # Coordinate offsets in radians
    dLat = dn / R
    dLon = de / (R * math.cos(math.pi * ref_lat / 180.0))
    
    # New Lat/Lon in degrees
    lat = ref_lat + (dLat * 180.0 / math.pi)
    lon = ref_lon + (dLon * 180.0 / math.pi)
    
    return lat, lon

def generate_kml(all_mines, survey_points, ref_lat, ref_lon, ref_e, ref_n):
    """
    Generates a KML string for the minefield.
    """
    kml = ['<?xml version="1.0" encoding="UTF-8"?>']
    kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    kml.append('<Document>')
    kml.append('<name>Minefield Plan</name>')
    
    # Styles
    # Red (AP) - KML Color AABBGGRR. Red=FF0000FF
    kml.append('<Style id="style_ap"><IconStyle><color>ff0000ff</color><scale>0.8</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>')
    # Blue (AT) - Red=00, Green=00, Blue=FF -> AABBGGRR -> FFFF0000
    kml.append('<Style id="style_at"><IconStyle><color>ffff0000</color><scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_square.png</href></Icon></IconStyle></Style>')
    # Orange (Frag) - Blue=00, Green=A5, Red=FF -> AABBGGRR -> FF00A5FF
    kml.append('<Style id="style_frag"><IconStyle><color>ff00a5ff</color><scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/star.png</href></Icon></IconStyle></Style>')
    # Green (DP) - AABBGGRR -> FF008000
    kml.append('<Style id="style_dp"><IconStyle><color>ff008000</color><scale>1.2</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon></IconStyle></Style>')
    # Survey - White
    kml.append('<Style id="style_survey"><IconStyle><color>ffffffff</color><scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/cross-hairs.png</href></Icon></IconStyle></Style>')
    # Trace Line Style (White Dashed equivalent in KML usually just solid line, opacity)
    kml.append('<Style id="style_trace"><LineStyle><color>ffffffff</color><width>2</width></LineStyle></Style>')
    # Polygon Style (Green Faded) - AABBGGRR. Alpha=40 (25%), Blue=00, Green=80, Red=00 -> 40008000
    kml.append('<Style id="style_aor"><LineStyle><color>ff008000</color><width>2</width></LineStyle><PolyStyle><color>40008000</color></PolyStyle></Style>')

    # 0. AOR Polygon (Background)
    # Calculate Bounding Box of Mines + Survey Points
    all_grid_e = [m['Grid Easting'] for m in all_mines] + [p['e'] for p in survey_points]
    all_grid_n = [m['Grid Northing'] for m in all_mines] + [p['n'] for p in survey_points]
    
    if all_grid_e and all_grid_n:
        min_e, max_e = min(all_grid_e) - 20, max(all_grid_e) + 20
        min_n, max_n = min(all_grid_n) - 20, max(all_grid_n) + 20
        
        # 4 Corners
        corners = [(min_e, min_n), (max_e, min_n), (max_e, max_n), (min_e, max_n), (min_e, min_n)]
        
        kml.append('<Folder><name>Minefield Zone</name>')
        kml.append('<Placemark><name>AOR Boundary</name><styleUrl>#style_aor</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>')
        for ce, cn in corners:
            lat, lon = local_grid_to_latlon(ref_lat, ref_lon, ref_e, ref_n, ce, cn)
            kml.append(f'{lon},{lat},0')
        kml.append('</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
        kml.append('</Folder>')
    
    # 1. Plot Survey Points & Trace
    kml.append('<Folder><name>Survey Control</name>')
    
    # Trace LineString
    trace_coords_str = []
    for p in survey_points:
        lat, lon = local_grid_to_latlon(ref_lat, ref_lon, ref_e, ref_n, p['e'], p['n'])
        trace_coords_str.append(f"{lon},{lat},0")
        
        # Add GR to label
        label_with_gr = f"{p['label']} (GR: {p['e']:.0f} {p['n']:.0f})"
        kml.append(f'<Placemark><name>{label_with_gr}</name><styleUrl>#style_survey</styleUrl><Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>')
        
    if trace_coords_str:
        kml.append('<Placemark><name>Survey Trace</name><styleUrl>#style_trace</styleUrl><LineString><coordinates>' + " ".join(trace_coords_str) + '</coordinates></LineString></Placemark>')
        
    kml.append('</Folder>')
    
    # 2. Plot Mines
    kml.append('<Folder><name>Mines</name>')
    for m in all_mines:
        m_type = m.get('type', 'survey')
        if m_type == 'ap': style = '#style_ap'
        elif m_type == 'at': style = '#style_at'
        elif m_type == 'frag': style = '#style_frag'
        elif m_type == 'DP': style = '#style_dp'
        else: style = '#style_survey'
        
        lat, lon = local_grid_to_latlon(ref_lat, ref_lon, ref_e, ref_n, m['Grid Easting'], m['Grid Northing'])
        
        # Labels Logic: 
        # - Mines (AP, AT, Frag): NO LABEL (Empty Name)
        # - DPs: Show Label + GR
        
        if m_type == 'DP':
            name_str = f"{m.get('label', 'DP')} (GR: {m['Grid Easting']:.0f} {m['Grid Northing']:.0f})"
        else:
            name_str = "" # Suppress name for standard mines
        
        # Description (Visible on Click)
        desc = f"ID: {m.get('mine_id', '')}\\nType: {str(m_type).upper()}\\nGR: {m['Grid Easting']:.1f}, {m['Grid Northing']:.1f}"
        
        kml.append(f'<Placemark><name>{name_str}</name><description>{desc}</description><styleUrl>{style}</styleUrl><Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>')
        
    kml.append('</Folder>')
    
    kml.append('</Document></kml>')
    return "\n".join(kml)

def generate_folium_map(all_mines, survey_points, ref_lat, ref_lon, ref_e, ref_n):
    """
    Generates an interactive Folium map (HTML).
    """
    import folium
    
    # Base Map (Centered on Landmark)
    m = folium.Map(location=[ref_lat, ref_lon], zoom_start=18, tiles=None) # Start with no tiles to manage layers manually
    
    # 1. Base Layers
    # Street View
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        control=True
    ).add_to(m)
    
    # Satellite View (Esri)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite Imagery',
        control=True
    ).add_to(m)
    
    # 1.5 Calculate AOR (Bounding Box)
    all_grid_e = [m['Grid Easting'] for m in all_mines] + [p['e'] for p in survey_points]
    all_grid_n = [m['Grid Northing'] for m in all_mines] + [p['n'] for p in survey_points]
    
    if all_grid_e and all_grid_n:
        min_e, max_e = min(all_grid_e) - 20, max(all_grid_e) + 20
        min_n, max_n = min(all_grid_n) - 20, max(all_grid_n) + 20
        
        corners = [(min_e, min_n), (max_e, min_n), (max_e, max_n), (min_e, max_n), (min_e, min_n)]
        poly_coords = []
        for ce, cn in corners:
            lat, lon = local_grid_to_latlon(ref_lat, ref_lon, ref_e, ref_n, ce, cn)
            poly_coords.append([lat, lon])
            
        folium.Polygon(
            locations=poly_coords,
            color='green',
            weight=2,
            fill=True,
            fill_color='green',
            fill_opacity=0.2,
            popup="Minefield Responsibility Area (Green Zone)",
            name="AOR Zone"
        ).add_to(m)

    # 2. Survey Trace Line
    # Extract coords in lat/lon
    trace_coords = []
    # We need to sort survey points? survey_points arg is usually a list of dicts.
    # Assuming they are in order of the trace.
    # Need to verify if survey_points contains the full trace or just control points.
    # In app.py we will construct specific trace list.
    
    for p in survey_points:
         lat, lon = local_grid_to_latlon(ref_lat, ref_lon, ref_e, ref_n, p['e'], p['n'])
         trace_coords.append([lat, lon])
         
         # Plot Survey Markers (White Crosshairs equivalent -> Black Circle with Target)
         folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color='black',
            weight=2,
            fill=True,
            fill_color='white',
            fill_opacity=1,
            popup=f"{p['label']} (GR: {p['e']:.0f} {p['n']:.0f})",
            tooltip=p['label']
         ).add_to(m)

    # Add PolyLine for Trace
    if trace_coords:
        folium.PolyLine(
            trace_coords,
            color="white",
            weight=2,
            dash_array='5, 10',
            opacity=0.8,
            name="Survey Trace"
        ).add_to(m)

    # 3. Mines
    # Create FeatureGroups for toggling
    fg_mines = folium.FeatureGroup(name="Mines")
    
    for mine in all_mines:
        m_type = mine.get('type', 'unknown')
        lat, lon = local_grid_to_latlon(ref_lat, ref_lon, ref_e, ref_n, mine['Grid Easting'], mine['Grid Northing'])
        
        # Style
        color = 'gray'
        fill_color = 'gray'
        radius = 3
        
        if m_type == 'ap':
            color = 'red'
            fill_color = 'red'
            radius = 3
        elif m_type == 'at':
            color = 'blue'
            fill_color = 'blue'
            radius = 5
        elif m_type == 'frag':
            color = 'orange'
            fill_color = 'orange'
            radius = 4
        elif m_type == 'DP':
            color = 'green'
            fill_color = 'green'
            radius = 6
            
        popup_txt = f"<b>{m_type.upper()}</b><br>ID: {mine.get('mine_id','')}<br>GR: {mine['Grid Easting']:.1f}, {mine['Grid Northing']:.1f}"
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            weight=1,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.8,
            popup=popup_txt,
            tooltip=f"{m_type.upper()} Mine"
        ).add_to(fg_mines)
        
    fg_mines.add_to(m)
    
    # 4. Legend (Floating HTML)
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 180px; height: 160px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: 0.9;">
     &nbsp;<b>Minefield Legend</b> <br>
     &nbsp;<i class="fa fa-circle" style="color:red"></i>&nbsp; AP Mine<br>
     &nbsp;<i class="fa fa-square" style="color:blue"></i>&nbsp; AT Mine<br>
     &nbsp;<i class="fa fa-star" style="color:orange"></i>&nbsp; Frag (M16)<br>
     &nbsp;<i class="fa fa-play" style="color:green; transform: rotate(-90deg);"></i>&nbsp; DP (Picket)<br>
     &nbsp;<i class="fa fa-circle-o" style="color:black"></i>&nbsp; Survey Point<br>
     &nbsp;<span style="color:black">-- --</span>&nbsp; Trace Wire
      </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add Layer Control
    folium.LayerControl().add_to(m)
    
    # Save
    out_file = "minefield_map.html"
    m.save(out_file)
    return out_file
