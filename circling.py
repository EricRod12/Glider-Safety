import math
import pandas as pd
import logging
from geopy.distance import geodesic
import datetime
from datetime import timedelta
import numpy as np

def compute_wind_vector_bearings_from_subset(df_subset: pd.DataFrame, num_pairs: int = 5) -> list:
    """
    Selects num_pairs pairs of fixes from the DataFrame (df_subset) and computes
    the wind vector bearing for each pair using the sponsor's formula.
    
    The pairs are chosen by evenly spacing indices. For each pair, the earlier fix is
    used as (lat_earlier, lon_earlier) and the later fix as (lat_later, lon_later).
    
    Returns a list of bearings (in degrees) for the selected pairs.
    """
    n = len(df_subset)
    if n < 2:
        raise ValueError("Not enough fixes in the subset to compute wind vector bearings.")
    
    offset = max(1, (n - 1) // num_pairs)
    bearings = []
    
    for i in range(num_pairs):
        idx1 = i * offset
        idx2 = idx1 + offset
        if idx2 >= n:
            idx2 = n - 1  # Ensure the second index is within bounds.
        
        # For the sponsor's formula, use:
        #   lat_earlier, lon_earlier  <- fix at idx1
        #   lat_later, lon_later      <- fix at idx2
        lat_earlier = df_subset.iloc[idx1]['latitude']
        lon_earlier = df_subset.iloc[idx1]['longitude']
        lat_later   = df_subset.iloc[idx2]['latitude']
        lon_later   = df_subset.iloc[idx2]['longitude']
        
        bearing = compute_bearing_sponsor(lat_earlier, lon_earlier, lat_later, lon_later)
        bearings.append(bearing)
        print(f"Pair {i+1}: Fix {idx1} (earlier) and Fix {idx2} (later) -> Bearing: {bearing:.2f}°")
    
    return bearings

def compute_bearing_sponsor(lat_earlier, lon_earlier, lat_later, lon_later):
    """
    Computes the bearing (in degrees) using the sponsor's formula.
    
    The formula is:
      =DEGREES(ATAN2(
            COS(RADIANS(lat_later)) * SIN(RADIANS(lat_earlier))
          - SIN(RADIANS(lat_later)) * COS(RADIANS(lat_earlier)) * COS(RADIANS(lon_earlier - lon_later)),
            SIN(RADIANS(lon_earlier - lon_later)) * COS(RADIANS(lat_earlier))
         ))
         
    In the sponsor's spreadsheet, lat_later comes from B49 and lat_earlier from B23, 
    and similarly for the longitudes.
    """
    num = math.cos(math.radians(lat_later)) * math.sin(math.radians(lat_earlier)) - \
          math.sin(math.radians(lat_later)) * math.cos(math.radians(lat_earlier)) * \
          math.cos(math.radians(lon_earlier - lon_later))
    den = math.sin(math.radians(lon_earlier - lon_later)) * math.cos(math.radians(lat_earlier))
    bearing = math.degrees(math.atan2(num, den))
    # Normalize the bearing to 0-360 degrees.
    return (bearing + 360) % 360

def compute_wind_vector_distance(lat1, lon1, lat2, lon2):
    """
    Compute the distance (in kilometers) between two points using the sponsor's formula:
    
        distance = acos(cos(radians(90-lat1)) * cos(radians(90-lat2)) +
                        sin(radians(90-lat1)) * sin(radians(90-lat2)) *
                        cos(radians(lon1-lon2))) * 6371
                        
    Here, lat1, lat2 are in degrees (from column B) and lon1, lon2 are in degrees (from column C).
    """
    # Convert (90 - latitude) to radians.
    lat1_rad = math.radians(90 - lat1)
    lat2_rad = math.radians(90 - lat2)
    # Difference in longitudes (using the order as in the formula).
    lon_diff_rad = math.radians(lon1 - lon2)
    # Compute the central angle using the spherical law of cosines.
    central_angle = math.acos(math.cos(lat1_rad) * math.cos(lat2_rad) +
                              math.sin(lat1_rad) * math.sin(lat2_rad) * math.cos(lon_diff_rad))
    # Multiply by Earth's radius in kilometers.
    return central_angle * 6371

def compute_wind_vector_from_subset(df_subset: pd.DataFrame, num_pairs: int = 5) -> float:
    """
    Choose 'num_pairs' pairs of fixes from the provided DataFrame (df_subset) that represents a circling segment.
    For each pair, compute the drift distance using the sponsor's formula.
    
    The pairs are chosen by evenly spacing the indices in df_subset.
    Returns the average distance (in km) from these pairs.
    """
    n = len(df_subset)
    if n < 2:
        raise ValueError("Not enough fixes in the subset to compute wind vector.")
    
    # Determine spacing between pairs.
    # We want to form num_pairs pairs, so the offset between the fix indices is:
    offset = max(1, (n - 1) // num_pairs)
    
    distances = []
    for i in range(num_pairs):
        idx1 = i * offset
        idx2 = idx1 + offset
        if idx2 >= n:
            idx2 = n - 1  # Ensure we stay within the DataFrame.
        lat1 = df_subset.iloc[idx1]['latitude']
        lon1 = df_subset.iloc[idx1]['longitude']
        lat2 = df_subset.iloc[idx2]['latitude']
        lon2 = df_subset.iloc[idx2]['longitude']
        dist = compute_wind_vector_distance(lat1, lon1, lat2, lon2)
        distances.append(dist)
        print(f"Pair {i+1}: Index {idx1} -> {idx2} | Distance: {dist:.3f} km")
    
    # Average the computed distances to get a representative wind vector magnitude.
    avg_distance = sum(distances) / len(distances)
    return avg_distance
def compute_distance_km(lat1, lon1, lat2, lon2):
    """
    Compute the distance (in kilometers) between two fixes using the spherical law of cosines,
    following the sponsor's formula:
    
        distance = acos( cos(radians(90-lat1)) * cos(radians(90-lat2))
                         + sin(radians(90-lat1)) * sin(radians(90-lat2))
                         * cos(radians(lon1-lon2)) ) * 6371
    """
    # Convert (90 - latitude) to radians for each point.
    lat1_rad = math.radians(90 - lat1)
    lat2_rad = math.radians(90 - lat2)
    
    # Compute the difference in longitudes (order as in the formula: (lon1 - lon2))
    delta_lon_rad = math.radians(lon1 - lon2)
    
    # Compute the cosine of the central angle between the two points.
    central_angle_cos = (math.cos(lat1_rad) * math.cos(lat2_rad) +
                         math.sin(lat1_rad) * math.sin(lat2_rad) * math.cos(delta_lon_rad))
    
    # To avoid math domain errors due to floating-point issues, clamp the value between -1 and 1.
    central_angle_cos = max(min(central_angle_cos, 1), -1)
    
    central_angle = math.acos(central_angle_cos)
    
    # Multiply by the Earth's radius in kilometers (6371 km)
    distance_km = central_angle * 6371
    return distance_km

def angle_difference(b1, b2):
    # Adjust difference to be in range [-180, 180]
    diff = (b2 - b1 + 180) % 360 - 180
    return diff


def wrap_angle(angle):
    """Wrap an angle (in degrees) to the range [-180, 180)."""
    return ((angle + 180) % 360) - 180

def choose_rotation(cal_bearing, reference_bearing, threshold=10):
    """
    Determine rotation (+90 / -90) based on the signed difference
    between cal_bearing and reference_bearing.
    """
    diff = wrap_angle(cal_bearing - reference_bearing)
    if diff == 0:
        return +90
    elif abs(diff) < threshold:
        return -90 if diff < 0 else +90
    else:
        return +90 if diff < 0 else -90


# Compute the normalized differences between consecutive headings.
def compute_heading_differences(headings):
    # Compute differences.
    diff = headings.diff()
    # Normalize differences to be in the range [-180, 180]
    diff = diff.apply(lambda x: (x + 180) % 360 - 180 if pd.notnull(x) else np.nan)
    return diff

# Compute cumulative turning angle.

def compute_TAS(row, wind_speed, eff_wind_heading):
    """
    Compute True Airspeed (TAS) by projecting the wind vector onto the flight vector,
    with two decision points:
      1) Choose which perpendicular to the flight heading will intersect the wind vector.
      2) Use the sign of the projected wind component to determine if it is added or subtracted.
    
    Parameters:
      row: A row from df_flight_subset containing:
           - 'Cal Bearing': calculated flight vector heading (in degrees)
           - 'Cal GS Km/time': calculated ground speed (in knots)
      wind_speed: the wind speed (in knots)
      eff_wind_heading: effective wind heading (in degrees) – the direction the wind is blowing TO.
    
    Returns:
      TAS: The computed true airspeed.
    """
    # Flight heading (direction the glider is pointed) in degrees.
    flight_heading = row['Cal Bearing']
    # Ground speed in knots.
    gs = row['Cal GS Km/time']
    
    # Compute the difference between effective wind direction and flight heading.
    diff = wrap_angle(eff_wind_heading - flight_heading)
    
    # --- Decision 1: Which perpendicular line to use ---
    # If the effective wind is counterclockwise (diff >= 0), add 90°.
    # If clockwise (diff < 0), subtract 90°.
    if diff >= 0:
        perp_line = (flight_heading + 90) % 360
    else:
        perp_line = (flight_heading - 90) % 360

    # --- Decision 2: Compute angle between wind direction and chosen perpendicular ---
    # This angle determines the magnitude of the wind component along the flight vector.
    angle_perp = wrap_angle(eff_wind_heading - perp_line)
    
    # The wind component along the flight vector is:
    wind_component = wind_speed * math.sin(math.radians(angle_perp))
    
    # Adjust ground speed to compute TAS:
    # If wind_component is positive, it indicates a tailwind that boosted ground speed,
    # so subtract it to get TAS. If negative (headwind), subtracting a negative adds its magnitude.
    TAS = gs - wind_component
    return TAS

def _wrap_angle_diff(a, b):
    """
    Return the signed difference (a - b) wrapped to [-180, 180).
    For heading changes, we often want the small difference.
    """
    diff = a - b
    # Wrap to [-180, 180)
    diff = (diff + 180) % 360 - 180
    return diff
def _filter_short_sequence(sr: pd.Series, min_length: int) -> pd.Series:

    groups = sr.diff().ne(0).cumsum()
    group_lengths = sr.groupby(groups).transform("size")
    return sr.where((sr == 0) | (group_lengths >= min_length), other=0)
def detect_circling_loops(
    df,
    min_angle=5,
    min_duration=50,
    min_full_turn=300,
    crossing_tol_m=50
):
    """
    1. Mark turning vs. not turning with heading diff >= min_angle.
    2. Filter out short runs < min_duration.
    3. For each run of turning, accumulate heading diffs and check loop closure.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 1) Compute heading
    df["heading"] = compute_heading_transition(df["latitude"], df["longitude"])
    
    # 2) Mark turning
    headings_diff = df["heading"].diff().abs().apply(lambda x: min(x, 360 - x))
    turning_flags = headings_diff.ge(min_angle).astype(int)
    
    # 3) Filter out short runs
    turning_flags = _filter_short_sequence(turning_flags, min_duration)
    
    # 4) Identify intervals of "turning"
    #    We'll group them by the cumsum trick
    intervals = []
    grp_id = (turning_flags.diff().ne(0)).cumsum()  # Each run gets a group ID
    run_groups = turning_flags.groupby(grp_id)
    
    for g_id, group_indices in run_groups.groups.items():
        # If the run is "1" (turning), then check net heading & loop closure
        if turning_flags.iloc[group_indices].iloc[0] == 1:
            start_idx = group_indices.min()
            end_idx   = group_indices.max()
            # Accumulate net heading
            run_diffs = headings_diff.iloc[start_idx:end_idx+1]
            net_heading_change = run_diffs.sum()
            
            # Check net heading
            if net_heading_change < min_full_turn:
                continue
            
            # Check loop closure (start vs end)
            lat1, lon1 = df.loc[start_idx, ["latitude", "longitude"]]
            lat2, lon2 = df.loc[end_idx, ["latitude", "longitude"]]
            distance = geodesic((lat1, lon1), (lat2, lon2)).meters
            if distance <= crossing_tol_m:
                intervals.append((start_idx, end_idx))
    
    # 5) Return intervals as needed
    #    e.g. turn them into {start_time, end_time, ...}
    segments = []
    for (start_idx, end_idx) in intervals:
        segments.append({
            "start_time": df.loc[start_idx, "timestamp"],
            "end_time":   df.loc[end_idx,   "timestamp"],
            "start_lat":  df.loc[start_idx, "latitude"],
            "start_lon":  df.loc[start_idx, "longitude"],
            "end_lat":    df.loc[end_idx,   "latitude"],
            "end_lon":    df.loc[end_idx,   "longitude"],
        })

    return segments

def extract_circling_segments_with_crossing(df, window_start, window_end,
                                            min_angle=5, min_duration=10,
                                            merge_gap=10, crossing_tol_m=50,
                                            min_full_turn=300):
    """
    Enhanced approach: still uses basic threshold detection for consecutive 
    heading changes, but also tracks net heading change and does crossing checks.
    """
    df_window = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)].copy()
    if df_window.empty or len(df_window) < 2:
        return []
    
    df_window = df_window.sort_values('timestamp').reset_index(drop=True)

    # Compute headings row-wise
    headings = []
    for i in range(len(df_window) - 1):
        hdg = _calc_bearing(df_window.loc[i, 'latitude'], df_window.loc[i, 'longitude'],
                            df_window.loc[i+1, 'latitude'], df_window.loc[i+1, 'longitude'])
        headings.append(hdg)
    headings.append(None)
    df_window['heading'] = headings

    # Evaluate differences for the threshold method
    headings_diff = pd.Series(headings).diff().abs().apply(lambda x: min(x, 360-x))
    circling_flags = headings_diff.ge(min_angle).astype(int)

    segments = []
    current_segment = None

    def finalize_segment(seg):
        """Perform crossing check and possibly store the segment."""
        seg_length = seg["end_index"] - seg["start_index"] + 1
        if seg_length < min_duration:
            return

        # Check net heading change
        if seg["accum_degs"] < min_full_turn:
            return

        # crossing check
        start_idx = seg["start_index"]
        end_idx   = seg["end_index"]
        start_point = (
            df_window.loc[start_idx, 'latitude'],
            df_window.loc[start_idx, 'longitude']
        )
        end_point = (
            df_window.loc[end_idx, 'latitude'],
            df_window.loc[end_idx, 'longitude']
        )
        crossing_distance = geodesic(start_point, end_point).meters
        if crossing_distance <= crossing_tol_m:
            segments.append(seg)

    # Track net heading change
    prev_hdg = headings[0]
    for idx, flag in circling_flags.items():
        hdg = df_window.loc[idx, 'heading'] if idx < len(df_window) else None

        if flag == 1:
            # We're turning now
            if current_segment is None:
                # start a new segment
                current_segment = {
                    "start_index": idx,
                    "end_index": idx,
                    "accum_degs": 0.0,
                    "prev_hdg": prev_hdg
                }
            else:
                # extend the existing segment
                current_segment["end_index"] = idx
                # accumulate heading change from prev_hdg -> this hdg
                if hdg is not None and current_segment["prev_hdg"] is not None:
                    d = headings_diff[idx]  # or compute small difference
                    current_segment["accum_degs"] += d
                current_segment["prev_hdg"] = hdg
        else:
            # Not turning (less than min_angle)
            # If we are in a segment, finalize it
            if current_segment is not None:
                finalize_segment(current_segment)
                current_segment = None
        prev_hdg = hdg

    # If segment is open at the end
    if current_segment is not None:
        finalize_segment(current_segment)

    # Merge short-gap segments
    merged_segments = []
    if segments:
        current = segments[0]
        for seg in segments[1:]:
            end_time_current = df_window.loc[current["end_index"], 'timestamp']
            start_time_next = df_window.loc[seg["start_index"], 'timestamp']
            gap_sec = (start_time_next - end_time_current).total_seconds()
            if gap_sec <= merge_gap:
                current["end_index"] = seg["end_index"]
            else:
                merged_segments.append(current)
                current = seg
        merged_segments.append(current)
    else:
        merged_segments = segments

    # Convert index-based to final segments
    final_segments = []
    for seg in merged_segments:
        s_idx = seg["start_index"]
        e_idx = seg["end_index"]
        final_segments.append({
            "start_time": df_window.loc[s_idx, 'timestamp'],
            "end_time": df_window.loc[e_idx, 'timestamp'],
            "start_lat": df_window.loc[s_idx, 'latitude'],
            "start_lon": df_window.loc[s_idx, 'longitude'],
            "end_lat": df_window.loc[e_idx, 'latitude'],
            "end_lon": df_window.loc[e_idx, 'longitude']
        })
    return final_segments



def compute_tas_from_phase_matched_method(df_flight: pd.DataFrame,
                                          engine_run_start: datetime.datetime,
                                          duration_sec: float,
                                          flight_heading_deg: float,
                                          phase_matched_wind_vector: tuple,
                                          subwindow_sec: float = 10,
                                          debug: bool = False) -> tuple:
    """
    Partition the engine run (or circling) window from engine_run_start for duration_sec into sub-windows.
    In each sub-window compute the average ground speed (GS) in knots and then apply an additive wind
    correction (using the provided phase_matched_wind_vector) to compute TAS.
    
    Returns (TAS_avg, TAS_min, TAS_max) in knots.
    """
    from datetime import timedelta

    # Unpack wind vector (in m/s, bearing in degrees) and convert wind speed to knots.
    wind_speed_mps, wind_dir_deg = phase_matched_wind_vector
    wind_speed_knots = wind_speed_mps * 1.94384

    window_end = engine_run_start + timedelta(seconds=duration_sec)
    df_window = df_flight[(df_flight['timestamp'] >= engine_run_start) & (df_flight['timestamp'] <= window_end)].copy()
    if df_window.shape[0] < 2:
        return (float('nan'), float('nan'), float('nan'))
    df_window = df_window.sort_values('timestamp').reset_index(drop=True)

    sub_tas = []
    current_start = engine_run_start
    while current_start < window_end:
        current_end = current_start + timedelta(seconds=subwindow_sec)
        df_sub = df_window[(df_window['timestamp'] >= current_start) & (df_window['timestamp'] < current_end)].copy()
        df_sub = df_sub.sort_values('timestamp').reset_index(drop=True)
        if df_sub.shape[0] < 2:
            current_start = current_end
            continue
        total_time = 0.0
        weighted_sum = 0.0
        for i in range(df_sub.shape[0]-1):
            dt_seg = (df_sub.loc[i+1, 'timestamp'] - df_sub.loc[i, 'timestamp']).total_seconds()
            if dt_seg <= 0:
                continue
            start_point = (df_sub.loc[i, 'latitude'], df_sub.loc[i, 'longitude'])
            end_point = (df_sub.loc[i+1, 'latitude'], df_sub.loc[i+1, 'longitude'])
            distance_m = geodesic(start_point, end_point).meters
            gs_knots = (distance_m / dt_seg) * 1.94384
            total_time += dt_seg
            weighted_sum += gs_knots * dt_seg
        if total_time <= 0:
            current_start = current_end
            continue
        gs_avg = weighted_sum / total_time
        
        # Apply additive correction using the wind vector.
        wind_blowing = (wind_dir_deg + 180) % 360
        theta_rad = math.radians(flight_heading_deg - wind_blowing)
        tailwind_component = wind_speed_knots * math.cos(theta_rad)
        tas_corrected = gs_avg + tailwind_component
        sub_tas.append(tas_corrected)
        if debug:
            print(f"DEBUG: Sub-window {current_start.time()} -> {current_end.time()}:")
            print(f"  GS_avg: {gs_avg:.2f} knots, tailwind: {tailwind_component:.2f} knots, TAS: {tas_corrected:.2f} knots")
        current_start = current_end

    if not sub_tas:
        return (float('nan'), float('nan'), float('nan'))
    tas_avg = sum(sub_tas) / len(sub_tas)
    tas_min = min(sub_tas)
    tas_max = max(sub_tas)
    return (tas_avg, tas_min, tas_max)
def get_phase_matched_drift_stats(df: pd.DataFrame, tolerance_deg: float = 5,
                                  min_time_gap: float = 10, max_time_gap: float = 120) -> tuple:
    """
    For a given DataFrame (representing one circling segment), find all phase-matched pairs,
    compute the drift vector for each pair, and return the minimum and maximum drift speeds (in m/s)
    as well as the average drift vector (as (avg_speed, avg_bearing) in m/s).
    """
    pairs = find_phase_matched_pairs(df, tolerance_deg, min_time_gap, max_time_gap)
    if not pairs:
        return (None, None, None)
    drift_vectors = []
    for i, j in pairs:
        try:
            drift_vectors.append(compute_drift_from_pair(df, i, j))
        except Exception as e:
            print(f"Error with pair ({i}, {j}): {e}")
    if not drift_vectors:
        return (None, None, None)
    
    speeds = [vec[0] for vec in drift_vectors]
    min_drift = min(speeds)
    max_drift = max(speeds)
    
    # Average drift vector (convert each to u and v, then average)
    u_components = []
    v_components = []
    for speed, bearing in drift_vectors:
        rad = math.radians(bearing)
        u_components.append(speed * math.sin(rad))
        v_components.append(speed * math.cos(rad))
    avg_u = sum(u_components) / len(u_components)
    avg_v = sum(v_components) / len(v_components)
    avg_speed = math.sqrt(avg_u**2 + avg_v**2)
    avg_bearing = (math.degrees(math.atan2(avg_u, avg_v)) + 360) % 360
    
    return (min_drift, max_drift, (avg_speed, avg_bearing))

def compute_true_airspeed_vector(gsp_knots: float, flight_heading_deg: float,
                                  wind_speed_knots: float, wind_direction_deg: float) -> float:
    """
    Compute the True Airspeed (TAS) using a full vector correction.
    
    Parameters:
      - gsp_knots: Ground Speed in knots.
      - flight_heading_deg: Flight heading (degrees true North).
      - wind_speed_knots: Wind speed in knots.
      - wind_direction_deg: Wind direction (the direction from which the wind is coming, in degrees).
      
    Returns:
      - TAS in knots.
    
    Note:
      Wind direction in meteorological convention is the direction from which the wind blows.
      To compute the effective tailwind component, we first convert it to the "blowing toward"
      direction by adding 180°.
    """
    # Compute wind blowing direction.
    wind_blowing = (wind_direction_deg + 180) % 360
    # Compute the difference (theta) between the flight heading and the wind blowing direction.
    theta_rad = math.radians(flight_heading_deg - wind_blowing)
    
    # Apply the law of cosines:
    tas = math.sqrt(gsp_knots**2 + wind_speed_knots**2 + 2 * gsp_knots * wind_speed_knots * math.cos(theta_rad))
    return tas

def compute_true_airspeed_additive(gsp_knots: float, flight_heading_deg: float,
                                   wind_speed_knots: float, wind_direction_deg: float) -> float:
    """
    Compute the True Airspeed (TAS) using an additive method:
    
      tailwind_component = wind_speed * cos(flight_heading - wind_blowing)
      TAS = GSP + tailwind_component
      
    Parameters:
      - gsp_knots: Ground Speed in knots.
      - flight_heading_deg: Flight heading (degrees true North).
      - wind_speed_knots: Wind speed in knots.
      - wind_direction_deg: Wind direction (the direction from which the wind is coming, in degrees).
      
    Returns:
      - TAS in knots.
    """
    wind_blowing = (wind_direction_deg + 180) % 360
    theta_rad = math.radians(flight_heading_deg - wind_blowing)
    tailwind_component = wind_speed_knots * math.cos(theta_rad)
    tas = gsp_knots + tailwind_component
    return tas
def find_phase_matched_pairs(df: pd.DataFrame, tolerance_deg: float = 5,
                             min_time_gap: float = 10, max_time_gap: float = 120) -> list:
    """
    For a DataFrame with columns 'timestamp', 'latitude', 'longitude',
    find index pairs (i, j) such that the heading difference between fix i and fix j
    is within tolerance_deg and the time gap between i and j is between min_time_gap
    and max_time_gap (in seconds). This is our “phase matching” step.
    
    Returns a list of tuples (i, j).
    """
    pairs = []
    headings = compute_heading_transition(df['latitude'], df['longitude'])
    n = len(headings)
    for i in range(n - 1):
        if pd.isna(headings.iloc[i]):
            continue
        for j in range(i + 1, n):
            if pd.isna(headings.iloc[j]):
                continue
            time_gap = (df.loc[j, 'timestamp'] - df.loc[i, 'timestamp']).total_seconds()
            if time_gap < min_time_gap or time_gap > max_time_gap:
                continue
            diff = abs(headings.iloc[i] - headings.iloc[j])
            diff = min(diff, 360 - diff)
            if diff <= tolerance_deg:
                pairs.append((i, j))
                break  # once a match is found for i, move to next i.
    return pairs

def compute_drift_from_pair(df: pd.DataFrame, i: int, j: int) -> tuple:
    """
    Given two indices i and j in the DataFrame, compute the drift vector.
    Using sponsor’s units:
      - Distance is computed in kilometers (from geodesic distance in meters),
      - Speed is computed in kph = (distance (km) / time (h)),
      - Then converted to knots by multiplying by 0.539957.
    Also compute the drift bearing (in degrees true North).
    
    Returns a tuple (drift_speed_knots, drift_bearing).
    """
    dt_sec = (df.loc[j, 'timestamp'] - df.loc[i, 'timestamp']).total_seconds()
    if dt_sec <= 0:
        raise ValueError("Non-positive time difference.")
    # Compute distance in kilometers.
    distance_km = geodesic((df.loc[i, 'latitude'], df.loc[i, 'longitude']),
                           (df.loc[j, 'latitude'], df.loc[j, 'longitude'])).meters / 1000.0
    # Convert dt from seconds to hours.
    dt_hr = dt_sec / 3600.0
    speed_kph = distance_km / dt_hr  # speed in kilometers per hour
    speed_knots = speed_kph * 0.539957  # convert to knots
    drift_bearing = _calc_bearing(df.loc[i, 'latitude'], df.loc[i, 'longitude'],
                                  df.loc[j, 'latitude'], df.loc[j, 'longitude'])
    return speed_knots, drift_bearing

def average_drift_vectors(vectors: list) -> tuple:
    """
    Given a list of drift vectors (each a tuple (speed_knots, bearing)),
    average them by converting to u (eastward) and v (northward) components.
    
    Returns a tuple (avg_speed_knots, avg_bearing).
    """
    u_components = []
    v_components = []
    for speed, bearing in vectors:
        rad = math.radians(bearing)
        u_components.append(speed * math.sin(rad))
        v_components.append(speed * math.cos(rad))
    avg_u = sum(u_components) / len(u_components)
    avg_v = sum(v_components) / len(v_components)
    avg_speed = math.sqrt(avg_u**2 + avg_v**2)
    avg_bearing = (math.degrees(math.atan2(avg_u, avg_v)) + 360) % 360
    return avg_speed, avg_bearing

def compute_phase_matched_wind_vector_for_segment(df_seg: pd.DataFrame, 
                                                  tolerance_deg: float = 5,
                                                  min_time_gap: float = 10,
                                                  max_time_gap: float = 120) -> tuple:
    """
    For a given circling segment (df_seg), find phase-matched pairs and compute drift vectors
    using the sponsor’s unit conversion (distance in km, speed in kph then to knots).
    Returns the averaged wind vector (avg_speed_knots, avg_bearing) or None if no pairs found.
    """
    df_seg = df_seg.sort_values('timestamp').reset_index(drop=True)
    pairs = find_phase_matched_pairs(df_seg, tolerance_deg, min_time_gap, max_time_gap)
    if not pairs:
        return None
    drift_vectors = []
    for i, j in pairs:
        try:
            drift_vectors.append(compute_drift_from_pair(df_seg, i, j))
        except Exception as e:
            logging.error(f"Error computing drift for pair ({i}, {j}): {e}")
    if not drift_vectors:
        return None
    return average_drift_vectors(drift_vectors)

def compute_phase_matched_wind_vectors(segments: list, df_flight: pd.DataFrame, 
                                         tolerance_deg: float = 5, 
                                         min_time_gap: float = 10, 
                                         max_time_gap: float = 120) -> list:
    """
    For each circling segment (given as a dictionary with keys for start_time and end_time),
    extract the corresponding fixes from df_flight and compute the phase-matched wind vector.
    
    Returns a list of tuples, each tuple (avg_drift_speed, avg_drift_bearing) in m/s.
    """
    results = []
    for seg in segments:
        # Extract the subset of flight data within the segment's time window.
        df_seg = df_flight[(df_flight['timestamp'] >= seg["start_time"]) & 
                           (df_flight['timestamp'] <= seg["end_time"])].copy()
        if df_seg.empty or len(df_seg) < 2:
            continue
        vector = compute_phase_matched_wind_vector_for_segment(df_seg, tolerance_deg, min_time_gap, max_time_gap)
        if vector is not None:
            results.append(vector)
    return results

def compute_TAS_stats_from_phase_matched(df: pd.DataFrame, flight_heading_deg: float,
                                         tolerance_deg: float = 5, min_time_gap: float = 10,
                                         max_time_gap: float = 120) -> tuple:
    """
    For a given circling segment DataFrame, find all phase-matched pairs and compute an estimated TAS
    for each pair. Then return the average, minimum, and maximum TAS (in knots) from these pairs.
    """
    pairs = find_phase_matched_pairs(df, tolerance_deg, min_time_gap, max_time_gap)
    if not pairs:
        return (None, None, None)
    
    tas_estimates = []
    for i, j in pairs:
        try:
            # Compute wind vector for the pair.
            wind_vector = compute_drift_from_pair(df, i, j)
            # Compute TAS for this pair.
            tas = compute_TAS_for_phase_matched_pair(df, i, j, flight_heading_deg, wind_vector)
            tas_estimates.append(tas)
        except Exception as e:
            print(f"Error processing pair ({i}, {j}): {e}")
    if not tas_estimates:
        return (None, None, None)
    avg_tas = sum(tas_estimates) / len(tas_estimates)
    return (avg_tas, min(tas_estimates), max(tas_estimates))

def _calc_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing from point A to point B.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    return (bearing + 360) % 360

def compute_heading_transition(lats: pd.Series, lons: pd.Series) -> pd.Series:
    if len(lats) < 2 or len(lons) < 2:
        raise ValueError("Insufficient data points to calculate heading transitions.")
    headings = []
    for i in range(len(lats) - 1):
        heading = _calc_bearing(lats.iloc[i], lons.iloc[i], lats.iloc[i + 1], lons.iloc[i + 1])
        headings.append(heading)
    headings.append(None)
    return pd.Series(headings, index=lats.index)

def compute_overall_heading(df: pd.DataFrame) -> float:
    if len(df) < 2:
        raise ValueError("Insufficient data points to calculate overall heading.")
    lat1, lon1 = df.iloc[0]["latitude"], df.iloc[0]["longitude"]
    lat2, lon2 = df.iloc[-1]["latitude"], df.iloc[-1]["longitude"]
    return _calc_bearing(lat1, lon1, lat2, lon2)

def detect_overall_circling(headings: pd.Series, min_angle: int = 5, min_duration: int = 50) -> bool:
    headings_diff = headings.diff().abs().apply(lambda x: min(x, 360 - x))
    circling_flags = headings_diff.ge(min_angle).astype(int)
    logging.debug(f"Headings Diff (first 5): {headings_diff.head()}")
    logging.debug(f"Circling Flags (first 5): {circling_flags.head()}")
    max_consecutive = 0
    current_consecutive = 0
    for flag in circling_flags:
        if flag == 1:
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
        else:
            current_consecutive = 0
    logging.debug(f"Max Consecutive Significant Changes: {max_consecutive}")
    return max_consecutive >= min_duration

def detect_circling_behavior(df_flight: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp, min_angle: int = 5, min_duration: int = 50) -> bool:
    df_window = df_flight[(df_flight['timestamp'] >= window_start) & (df_flight['timestamp'] <= window_end)].copy()
    if df_window.empty or not {'latitude', 'longitude'}.issubset(df_window.columns):
        logging.info(f"No valid data available for circling detection between {window_start} and {window_end}.")
        return False
    if len(df_window) < 2:
        logging.info(f"Not enough data points for circling detection between {window_start} and {window_end}.")
        return False
    headings = compute_heading_transition(df_window['latitude'], df_window['longitude'])
    headings = headings.dropna()
    if headings.empty:
        logging.info(f"No valid heading transitions for circling detection between {window_start} and {window_end}.")
        return False
    is_circling = detect_overall_circling(headings, min_angle=min_angle, min_duration=min_duration)
    return is_circling

def extract_circling_segments(df_flight: pd.DataFrame, window_start: pd.Timestamp,
                                window_end: pd.Timestamp, min_angle: int = 5,
                                min_duration: int = 10, merge_gap: float = 10) -> list:
    """
    Extract circling segments from df_flight within [window_start, window_end] using
    a heading-change threshold (min_angle) and a minimum duration (min_duration in fixes).
    Then merge segments that are separated by a gap less than merge_gap seconds.
    
    Returns a list of dictionaries, each containing:
      - start_time, end_time
      - start_lat, start_lon, end_lat, end_lon
    """
    df_window = df_flight[(df_flight['timestamp'] >= window_start) & (df_flight['timestamp'] <= window_end)].copy()
    if df_window.empty or len(df_window) < 2:
        return []
    
    # Compute heading differences.
    headings = compute_heading_transition(df_window['latitude'], df_window['longitude'])
    headings_diff = headings.diff().abs().apply(lambda x: min(x, 360 - x))
    circling_flags = headings_diff.ge(min_angle).astype(int)
    
    segments = []
    current_segment = None
    for idx, flag in circling_flags.items():
        if flag == 1:
            if current_segment is None:
                current_segment = {"start_index": idx, "end_index": idx}
            else:
                current_segment["end_index"] = idx
        else:
            if current_segment is not None:
                seg_length = current_segment["end_index"] - current_segment["start_index"] + 1
                if seg_length >= min_duration:
                    segments.append(current_segment)
                current_segment = None
    if current_segment is not None:
        seg_length = current_segment["end_index"] - current_segment["start_index"] + 1
        if seg_length >= min_duration:
            segments.append(current_segment)
    
    # Merge segments separated by a gap shorter than merge_gap seconds.
    merged_segments = []
    if segments:
        current = segments[0]
        for seg in segments[1:]:
            # Get end time of current and start time of next segment.
            end_time_current = df_window.iloc[current["end_index"]]["timestamp"]
            start_time_next = df_window.iloc[seg["start_index"]]["timestamp"]
            gap = (start_time_next - end_time_current).total_seconds()
            if gap <= merge_gap:
                # Merge by extending the current segment's end_index.
                current["end_index"] = seg["end_index"]
            else:
                merged_segments.append(current)
                current = seg
        merged_segments.append(current)
    else:
        merged_segments = segments

    # Convert indices to segment dictionaries.
    final_segments = []
    for seg in merged_segments:
        start_idx = seg["start_index"]
        end_idx = seg["end_index"]
        final_segments.append({
            "start_time": df_window.iloc[start_idx]["timestamp"],
            "end_time": df_window.iloc[end_idx]["timestamp"],
            "start_lat": df_window.iloc[start_idx]["latitude"],
            "start_lon": df_window.iloc[start_idx]["longitude"],
            "end_lat": df_window.iloc[end_idx]["latitude"],
            "end_lon": df_window.iloc[end_idx]["longitude"]
        })
    return final_segments
def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing (in degrees) from point A to point B.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

def select_wind_vector_from_segments(segments: list):
    """
    Given a list of circling segments (each a dictionary with keys as shown in your output),
    compute:
      1. The wind vector (drift speed and bearing) from the longest segment.
      2. The average wind vector computed over all segments.
    
    Returns a dictionary with:
      "longest": (speed, bearing)
      "average": (speed, bearing)
    """
    if not segments:
        raise ValueError("No segments provided.")
    
    # Compute duration (in seconds) for each segment and select the longest.
    durations = []
    drift_vectors = []
    for seg in segments:
        dt = (seg["end_time"] - seg["start_time"]).total_seconds()
        durations.append(dt)
        drift_vectors.append(compute_wind_vector_from_segment(seg))
    
    # Longest segment vector:
    longest_idx = durations.index(max(durations))
    longest_vector = drift_vectors[longest_idx]
    
    # For average, we convert each (speed, bearing) into u and v components.
    # Here we assume the drift vector's bearing is in degrees (from north, clockwise)
    u_components = []
    v_components = []
    for speed, bearing in drift_vectors:
        # Convert bearing to radians.
        rad = math.radians(bearing)
        # u: eastward component, v: northward component.
        u = speed * math.sin(rad)
        v = speed * math.cos(rad)
        u_components.append(u)
        v_components.append(v)
    avg_u = sum(u_components) / len(u_components)
    avg_v = sum(v_components) / len(v_components)
    avg_speed = math.sqrt(avg_u**2 + avg_v**2)
    # Compute average wind direction. arctan2 gives angle from the positive x-axis;
    # here we adjust so that 0° = north:
    avg_bearing = (math.degrees(math.atan2(avg_u, avg_v)) + 360) % 360

    return {
        "longest": longest_vector,
        "average": (avg_speed, avg_bearing)
    }

def compute_wind_vector_from_segment(segment: dict):
    """
    Given a circling segment dictionary with keys:
      - "start_time": datetime of segment start,
      - "end_time": datetime of segment end,
      - "start_lat", "start_lon": starting coordinates,
      - "end_lat", "end_lon": ending coordinates,
    compute the drift vector (wind vector) as:
       drift_speed (m/s) = displacement (m) / dt (s)
       drift_bearing (°) = bearing from start to end.
    """
    start_time = segment["start_time"]
    end_time = segment["end_time"]
    dt = (end_time - start_time).total_seconds()
    if dt <= 0:
        raise ValueError("Segment duration is zero or negative.")
    
    start_point = (segment["start_lat"], segment["start_lon"])
    end_point = (segment["end_lat"], segment["end_lon"])
    displacement = geodesic(start_point, end_point).meters
    drift_speed = displacement / dt  # m/s
    drift_bearing = _calc_bearing(segment["start_lat"], segment["start_lon"],
                                  segment["end_lat"], segment["end_lon"])
    return drift_speed, drift_bearing