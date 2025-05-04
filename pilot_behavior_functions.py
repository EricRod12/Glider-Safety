# Importing standard libraries
import os
import sys
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
import logging
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon
# Importing existing functions from 'circling' module
from circling import compute_heading_transition, _calc_bearing, compute_overall_heading, detect_overall_circling, detect_circling_behavior
import geopandas as gpd
# Importing custom IGC parser
from parser import igc2df  # Ensure this function correctly parses IGC files into DataFrames
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol
from pathlib import Path
import tqdm 
from typing import Optional
from joblib import Parallel, delayed
import ee
from pyproj import Transformer
import re
from typing import Dict, List, Tuple, Optional, Any
from ee_helpers import ensure_ee_initialized, LANDCOVER_DATASET
import os
import csv
from geopy.distance import distance 
#!/usr/bin/env python3


def compute_ground_speed_knots(df: pd.DataFrame) -> list:
    """
    Compute the instantaneous ground speed (in knots) between consecutive fixes in the DataFrame.
    Returns a list of speeds in knots.
    """
    speeds = []
    for i in range(len(df) - 1):
        dt = (df['timestamp'].iloc[i+1] - df['timestamp'].iloc[i]).total_seconds()
        if dt <= 0:
            continue
        # Compute the distance between consecutive fixes (in meters).
        distance = geodesic(
            (df['latitude'].iloc[i], df['longitude'].iloc[i]),
            (df['latitude'].iloc[i+1], df['longitude'].iloc[i+1])
        ).meters
        speed_mps = distance / dt
        speed_knots = speed_mps * 1.94384
        speeds.append(speed_knots)
    return speeds


def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Computes the bearing (in degrees from North) from point 1 (lat1, lon1) to point 2 (lat2, lon2).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dLon)
    initial_bearing = np.degrees(np.arctan2(x, y))
    return (initial_bearing + 360) % 360

def correct_speed(gs, wind_knots, theta_rad):
    """
    Applies the wind correction to a given ground speed (gs, in knots) using the wind vector,
    where theta_rad is the difference (in radians) between the flight course and the wind's toward direction.
    """
    return np.sqrt(gs**2 + wind_knots**2 - 2 * gs * wind_knots * np.cos(theta_rad))

def median_filter_speeds(speeds, window_size=5):
    if len(speeds) < window_size:
        return speeds
    filtered = []
    for i in range(len(speeds) - window_size + 1):
        filtered.append(np.median(speeds[i:i+window_size]))
    return filtered

def calc_corrected_speed_stats_in_window(start_dt, duration_sec, df, wind_speed_mps, wind_dir_deg, debug=False):
    from datetime import timedelta
    window_end = start_dt + timedelta(seconds=duration_sec)
    df_window = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= window_end)].copy()
    if df_window.shape[0] < 2:
        if debug:
            print(f"DEBUG: Not enough points between {start_dt} and {window_end} to compute speeds.")
        return (np.nan, np.nan, np.nan)
    
    df_window = df_window.sort_values('timestamp').reset_index(drop=True)
    
    gs_list = []
    total_time = 0.0
    weighted_sum = 0.0
    for i in range(df_window.shape[0] - 1):
        t1 = df_window.loc[i, 'timestamp']
        t2 = df_window.loc[i+1, 'timestamp']
        dt = (t2 - t1).total_seconds()
        if dt <= 0:
            continue
        total_time += dt
        lat1, lon1 = df_window.loc[i, 'latitude'], df_window.loc[i, 'longitude']
        lat2, lon2 = df_window.loc[i+1, 'latitude'], df_window.loc[i+1, 'longitude']
        d = distance((lat1, lon1), (lat2, lon2)).meters
        gs_mps = d / dt
        gs_knots = gs_mps * 1.94384
        gs_list.append(gs_knots)
        weighted_sum += gs_knots * dt
        if debug:
            print(f"DEBUG: Segment {i}: {t1.time()} -> {t2.time()}, dt={dt:.2f}s, gs={gs_knots:.2f} knots")
    
    GS_avg_raw = weighted_sum / total_time if total_time > 0 else np.nan
    if debug:
        print(f"DEBUG: Total window time: {total_time:.2f}s, Weighted GS_avg (raw): {GS_avg_raw:.2f} knots")
    
    first = df_window.iloc[0]
    last = df_window.iloc[-1]
    try:
        course_deg = compute_bearing(first['latitude'], first['longitude'],
                                     last['latitude'], last['longitude'])
    except Exception as e:
        if debug:
            print("DEBUG: Could not compute overall course; using 0°.", e)
        course_deg = 0.0
    
    wind_knots = wind_speed_mps * 1.94384
    wind_toward_deg = (wind_dir_deg + 180) % 360
    theta_deg = abs(course_deg - wind_toward_deg)
    if theta_deg > 180:
        theta_deg = 360 - theta_deg
    theta_rad = np.radians(theta_deg)
    if debug:
        print(f"DEBUG: Overall flight course = {course_deg:.2f}°, Wind toward = {wind_toward_deg:.2f}°, theta = {theta_deg:.2f}°")
    
    TAS_avg = correct_speed(GS_avg_raw, wind_knots, theta_rad)
    if debug:
        print(f"DEBUG: Corrected overall TAS_avg = {TAS_avg:.2f} knots")
    
    # Apply a median filter to smooth the raw ground speeds.
    smoothed_gs = median_filter_speeds(gs_list, window_size=5)
    if debug:
        print("DEBUG: Median-filtered raw GS values:", smoothed_gs)
    
    # Compute the robust percentiles (25th and 75th) from the smoothed speeds.
    robust_raw_min = np.percentile(smoothed_gs, 10) if smoothed_gs else np.nan
    robust_raw_max = np.percentile(smoothed_gs, 90) if smoothed_gs else np.nan
    
    # Compute a correction factor: factor = TAS_avg / GS_avg_raw.
    factor = TAS_avg / GS_avg_raw if GS_avg_raw != 0 else 1.0
    true_min = robust_raw_min * factor
    true_max = robust_raw_max * factor
    
    if debug:
        print(f"DEBUG: Median-filtered GS: 25th percentile = {robust_raw_min:.2f} knots, 75th percentile = {robust_raw_max:.2f} knots")
        print(f"DEBUG: Correction factor = {factor:.3f}")
        print(f"DEBUG: Corrected robust speeds: min = {true_min:.2f} knots, max = {true_max:.2f} knots")
    
    return (TAS_avg, true_min, true_max)


def calc_corrected_speed_stats_in_window_overall_course(start_dt, duration_sec, df, wind_speed_mps, wind_dir_deg, overall_course_deg, subwindow_sec=10, debug=False):
    """
    Divides the engine run event window (start_dt to start_dt+duration_sec) into sub-windows
    of subwindow_sec seconds, computes the average ground speed for each sub-window, and applies
    wind correction using the overall event course (overall_course_deg) for every sub-window.
    
    This avoids computing a separate course for each sub-window, which can be unstable.
    
    Returns a tuple (TAS_avg, robust_min, robust_max) in knots.
    """
    from datetime import timedelta
    wind_knots = wind_speed_mps * 1.94384
    # Compute the wind 'toward' direction.
    wind_toward_deg = (wind_dir_deg + 180) % 360
    # Use the overall event course for all sub-windows.
    theta_deg = abs(overall_course_deg - wind_toward_deg)
    if theta_deg > 180:
        theta_deg = 360 - theta_deg
    theta_rad = math.radians(theta_deg)
    
    event_end = start_dt + timedelta(seconds=duration_sec)
    subwindow_speeds = []
    current_start = start_dt
    while current_start < event_end:
        current_end = current_start + timedelta(seconds=subwindow_sec)
        df_sub = df[(df['timestamp'] >= current_start) & (df['timestamp'] < current_end)].copy()
        if df_sub.shape[0] < 2:
            current_start = current_end
            continue
        df_sub = df_sub.sort_values('timestamp').reset_index(drop=True)
        total_time = 0.0
        weighted_sum = 0.0
        for i in range(df_sub.shape[0]-1):
            t1 = df_sub.loc[i, 'timestamp']
            t2 = df_sub.loc[i+1, 'timestamp']
            dt = (t2 - t1).total_seconds()
            if dt <= 0:
                continue
            total_time += dt
            lat1, lon1 = df_sub.loc[i, 'latitude'], df_sub.loc[i, 'longitude']
            lat2, lon2 = df_sub.loc[i+1, 'latitude'], df_sub.loc[i+1, 'longitude']
            d = distance((lat1, lon1), (lat2, lon2)).meters
            gs_mps = d / dt
            gs_knots = gs_mps * 1.94384
            weighted_sum += gs_knots * dt
        if total_time > 0:
            gs_avg_sub = weighted_sum / total_time
        else:
            gs_avg_sub = np.nan
        # Apply wind correction using overall course.
        corrected_speed = math.sqrt(gs_avg_sub**2 + wind_knots**2 - 2 * gs_avg_sub * wind_knots * math.cos(theta_rad))
        subwindow_speeds.append(corrected_speed)
        if debug:
            print(f"DEBUG: Sub-window {current_start.time()} -> {current_end.time()}: GS_avg_sub = {gs_avg_sub:.2f} knots, using overall course = {overall_course_deg:.2f}°, corrected = {corrected_speed:.2f} knots")
        current_start = current_end
    
    # Compute overall weighted GS_avg (raw) for the event.
    df_window = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= event_end)].copy()
    df_window = df_window.sort_values('timestamp').reset_index(drop=True)
    total_time = 0.0
    weighted_sum = 0.0
    for i in range(df_window.shape[0]-1):
        t1 = df_window.loc[i, 'timestamp']
        t2 = df_window.loc[i+1, 'timestamp']
        dt = (t2 - t1).total_seconds()
        if dt <= 0:
            continue
        total_time += dt
        lat1, lon1 = df_window.loc[i, 'latitude'], df_window.loc[i, 'longitude']
        lat2, lon2 = df_window.loc[i+1, 'latitude'], df_window.loc[i+1, 'longitude']
        d = distance((lat1, lon1), (lat2, lon2)).meters
        gs_mps = d / dt
        gs_knots = gs_mps * 1.94384
        weighted_sum += gs_knots * dt
    GS_avg_raw = weighted_sum / total_time if total_time > 0 else np.nan
    TAS_avg = math.sqrt(GS_avg_raw**2 + wind_knots**2 - 2 * GS_avg_raw * wind_knots * math.cos(theta_rad))
    
    # Compute robust robust percentiles from the sub-window corrected speeds.
    if subwindow_speeds:
        robust_min = np.percentile(subwindow_speeds, 25)
        robust_max = np.percentile(subwindow_speeds, 75)
    else:
        robust_min = robust_max = np.nan
    
    if debug:
        print(f"DEBUG: Overall GS_avg (raw) = {GS_avg_raw:.2f} knots, Corrected TAS_avg = {TAS_avg:.2f} knots")
        print(f"DEBUG: Robust corrected speeds from sub-windows: min = {robust_min:.2f} knots, max = {robust_max:.2f} knots")
    
    return (TAS_avg, robust_min, robust_max)
def get_dynamic_subwindow_length(duration_sec, min_window=10, fraction=0.25):
    return max(min_window, duration_sec * fraction)

def calc_corrected_speed_stats_in_window_overall_course(start_dt: datetime, duration_sec: float,
                                                          df: pd.DataFrame, wind_speed_mps: float,
                                                          wind_dir_deg: float, overall_course_deg: float,
                                                          subwindow_sec: float, debug: bool=False,
                                                          wind_scale: float = 0.2):
    """
    Divide the event window (from start_dt to start_dt+duration_sec) into sub-windows,
    compute the weighted average ground speed (GS_avg_raw) for the event, then correct for wind.
    
    A scaling factor (wind_scale) is applied to the computed wind vector before applying
    the wind correction. This is useful if the circling drift overestimates the effective
    wind component affecting TAS.
    
    Returns a tuple (TAS_avg, robust_min, robust_max) in knots.
    """
    #from datetime import timedelta
    # Convert wind speed (m/s) to knots.
    wind_knots_full = wind_speed_mps * 1.94384
    # Apply scaling factor to get the effective wind speed.
    effective_wind_knots = wind_scale * wind_knots_full

    # Compute difference angle using overall course and wind 'toward' direction.
    wind_toward_deg = (wind_dir_deg + 180) % 360
    theta_deg = abs(overall_course_deg - wind_toward_deg)
    if theta_deg > 180:
        theta_deg = 360 - theta_deg
    theta_rad = math.radians(theta_deg)
    
    # Compute overall weighted ground speed over the event window.
    window_end = start_dt + timedelta(seconds=duration_sec)
    df_window = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= window_end)].copy()
    if df_window.shape[0] < 2:
        if debug:
            print(f"DEBUG: Not enough data points between {start_dt} and {window_end}.")
        return (float('nan'), float('nan'), float('nan'))
    df_window = df_window.sort_values('timestamp').reset_index(drop=True)
    
    total_time = 0.0
    weighted_sum = 0.0
    for i in range(df_window.shape[0]-1):
        dt_seg = (df_window.loc[i+1, 'timestamp'] - df_window.loc[i, 'timestamp']).total_seconds()
        if dt_seg <= 0:
            continue
        lat1, lon1 = df_window.loc[i, 'latitude'], df_window.loc[i, 'longitude']
        lat2, lon2 = df_window.loc[i+1, 'latitude'], df_window.loc[i+1, 'longitude']
        d = geodesic((lat1, lon1), (lat2, lon2)).meters
        gs_knots = (d / dt_seg) * 1.94384
        total_time += dt_seg
        weighted_sum += gs_knots * dt_seg
    GS_avg_raw = weighted_sum / total_time if total_time > 0 else float('nan')
    # Correct the overall ground speed with the effective wind:
    TAS_avg = math.sqrt(GS_avg_raw**2 + effective_wind_knots**2 - 2 * GS_avg_raw * effective_wind_knots * math.cos(theta_rad))
    
    if debug:
        print(f"DEBUG: Total window time: {total_time:.2f}s, Weighted GS_avg (raw): {GS_avg_raw:.2f} knots")
        print(f"DEBUG: Overall event course = {overall_course_deg:.2f}°, Wind toward = {wind_toward_deg:.2f}°, theta = {theta_deg:.2f}°")
        print(f"DEBUG: Wind full: {wind_knots_full:.2f} knots, effective (scale {wind_scale}): {effective_wind_knots:.2f} knots")
        print(f"DEBUG: Corrected overall TAS_avg = {TAS_avg:.2f} knots")
    
    # Now partition the event into sub-windows.
    subwindow_speeds = []
    current_start = start_dt
    while current_start < window_end:
        current_end = current_start + timedelta(seconds=subwindow_sec)
        df_sub = df[(df['timestamp'] >= current_start) & (df['timestamp'] < current_end)].copy()
        if df_sub.shape[0] < 2:
            current_start = current_end
            continue
        df_sub = df_sub.sort_values('timestamp').reset_index(drop=True)
        sub_total_time = 0.0
        sub_weighted_sum = 0.0
        for i in range(df_sub.shape[0]-1):
            dt_seg = (df_sub.loc[i+1, 'timestamp'] - df_sub.loc[i, 'timestamp']).total_seconds()
            if dt_seg <= 0:
                continue
            lat1, lon1 = df_sub.loc[i, 'latitude'], df_sub.loc[i, 'longitude']
            lat2, lon2 = df_sub.loc[i+1, 'latitude'], df_sub.loc[i+1, 'longitude']
            d = geodesic((lat1, lon1), (lat2, lon2)).meters
            gs_knots = (d / dt_seg) * 1.94384
            sub_total_time += dt_seg
            sub_weighted_sum += gs_knots * dt_seg
        gs_avg_sub = sub_weighted_sum / sub_total_time if sub_total_time > 0 else float('nan')
        # Apply wind correction to sub-window using overall course and effective wind.
        corrected_sub = math.sqrt(gs_avg_sub**2 + effective_wind_knots**2 - 2 * gs_avg_sub * effective_wind_knots * math.cos(theta_rad))
        subwindow_speeds.append(corrected_sub)
        if debug:
            print(f"DEBUG: Sub-window {current_start.time()} -> {current_end.time()}: GS_avg_sub = {gs_avg_sub:.2f} knots, corrected = {corrected_sub:.2f} knots")
        current_start = current_end
    
    # Compute robust statistics from sub-window speeds.
    if subwindow_speeds:
        robust_min = np.percentile(subwindow_speeds, 25)
        robust_max = np.percentile(subwindow_speeds, 75)
    else:
        robust_min = robust_max = float('nan')
    
    if debug:
        print(f"DEBUG: Robust corrected speeds from sub-windows: min = {robust_min:.2f} knots, max = {robust_max:.2f} knots")
    
    return (TAS_avg, robust_min, robust_max)

def calc_true_airspeed_stats(flight_date_str: str, engine_run_times_str, engine_run_durations,
                             df_flight: pd.DataFrame, wind_speed_mps: float, wind_dir_deg: float,
                             debug: bool=False, wind_scale: float = 0.2):
    """
    For each engine run event (with start times and durations), compute the overall
    ground speed (GS_avg_raw) and then use the wind vector (scaled by wind_scale) to
    correct the ground speed and obtain the true airspeed (TAS).
    
    Returns a list of tuples: (event_start_datetime, TAS_avg, robust_min, robust_max) in knots.
    """
    # Convert engine run start times to datetime.
    tokens = [t.strip() for t in engine_run_times_str.split(",") if t.strip()]
    event_datetimes = []
    for token in tokens:
        try:
            dt = datetime.strptime(f"{flight_date_str} {token}", "%m/%d/%Y %H%M%S")
            event_datetimes.append(dt)
        except Exception as e:
            if debug:
                print(f"DEBUG: Error converting token '{token}': {e}")
    
    # Ensure durations is iterable.
    if not isinstance(engine_run_durations, list):
        engine_run_durations = [engine_run_durations]
    if len(engine_run_durations) != len(event_datetimes):
        if debug:
            print("DEBUG: Mismatch between number of durations and start times. Using first duration for all events.")
        engine_run_durations = [engine_run_durations[0]] * len(event_datetimes)
    
    if debug:
        print("DEBUG: Engine run event start times:", [dt.strftime("%H:%M:%S") for dt in event_datetimes])
        print("DEBUG: Engine run durations (s):", engine_run_durations)
    
    event_stats = []
    for dt, dur in zip(event_datetimes, engine_run_durations):
        from datetime import timedelta
        window_end = dt + timedelta(seconds=dur)
        df_window = df_flight[(df_flight['timestamp'] >= dt) & (df_flight['timestamp'] <= window_end)].copy()
        if df_window.shape[0] < 2:
            continue
        df_window = df_window.sort_values('timestamp').reset_index(drop=True)
        try:
            overall_course = _calc_bearing(df_window.iloc[0]['latitude'], df_window.iloc[0]['longitude'],
                                           df_window.iloc[-1]['latitude'], df_window.iloc[-1]['longitude'])
        except Exception:
            overall_course = 0.0
        if debug:
            print(f"DEBUG: Overall event course = {overall_course:.2f}°")
        
        dynamic_subwindow = max(10, dur * (0.25 if dur > 120 else 0.15))
        if debug:
            print(f"DEBUG: Using dynamic subwindow length: {dynamic_subwindow:.2f} seconds")
        
        TAS_avg, robust_min, robust_max = calc_corrected_speed_stats_in_window_overall_course(
            dt, dur, df_flight, wind_speed_mps, wind_dir_deg, overall_course,
            subwindow_sec=dynamic_subwindow, debug=debug, wind_scale=wind_scale
        )
        event_stats.append((dt, TAS_avg, robust_min, robust_max))
    
    return event_stats



def calc_engine_run_speed_stats(flight_date_str, engine_run_times_str, engine_run_durations, df_flight, debug=False):
    """
    Calculates speed statistics (average, minimum, and maximum speeds in knots) during engine run events.

    Parameters:
      flight_date_str (str): Flight date in "MM/DD/YYYY" format.
      engine_run_times_str (str): A comma-separated string of engine run start times in HHMMSS format.
      engine_run_durations (float or str): Either a single duration (in seconds) or a comma-separated string 
                                           of durations (in seconds) corresponding to each engine run.
      df_flight (pd.DataFrame): DataFrame containing flight fix data with at least these columns:
                                'timestamp' (datetime), 'latitude' (float), 'longitude' (float).
      debug (bool): If True, prints debugging information.

    Returns:
      List[Tuple[datetime, float, float, float]]:
        A list of tuples (event_start_datetime, avg_speed, min_speed, max_speed) in knots for each engine run event.
        If no B records fall in an event window, the corresponding stats are NaN.
    """

    # Convert the engine run times (HHMMSS) into a list of datetime objects using the flight date.
    engine_run_tokens = [token.strip() for token in engine_run_times_str.split(",") if token.strip()]
    event_datetimes = []
    for token in engine_run_tokens:
        try:
            event_dt = datetime.strptime(f"{flight_date_str} {token}", "%m/%d/%Y %H%M%S")
            event_datetimes.append(event_dt)
        except Exception as e:
            if debug:
                print(f"DEBUG: Error converting token '{token}' with flight date '{flight_date_str}': {e}")
    
    # Parse engine_run_durations into a list of floats.
    if isinstance(engine_run_durations, str):
        # Remove any brackets if present.
        if engine_run_durations.startswith('[') and engine_run_durations.endswith(']'):
            engine_run_durations = engine_run_durations[1:-1]
        durations = []
        for x in engine_run_durations.split(","):
            x = x.strip()
            if x:
                try:
                    durations.append(float(x))
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Error converting duration '{x}': {e}")
        engine_run_durations = durations
    elif isinstance(engine_run_durations, (list, tuple)):
        engine_run_durations = [float(x) for x in engine_run_durations]
    else:
        engine_run_durations = [float(engine_run_durations)]

    # If the number of durations does not match the number of tokens,
    # assume that a single duration applies to all events.
    if len(engine_run_durations) != len(event_datetimes):
        if debug:
            print("DEBUG: Number of durations does not match number of engine run times. Using the first duration for all events.")
        engine_run_durations = [engine_run_durations[0]] * len(event_datetimes)

    if debug:
        print("DEBUG: Engine run event start times:", [dt.strftime("%H:%M:%S") for dt in event_datetimes])
        print("DEBUG: Engine run durations (s):", engine_run_durations)

    # Function to calculate speed statistics (in knots) over a given window.
    def calc_speed_stats_in_window(start_dt, duration_sec, df):
        window_end = start_dt + timedelta(seconds=duration_sec)
        # Filter df_flight to rows with timestamp in the event window.
        df_window = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= window_end)].copy()
        if df_window.empty:
            if debug:
                print(f"DEBUG: No flight data found between {start_dt} and {window_end}")
            return (np.nan, np.nan, np.nan)
        
        # Ensure sorted by timestamp.
        df_window = df_window.sort_values('timestamp').reset_index(drop=True)
        speeds = []
        # Compute speeds between consecutive points.
        for i in range(len(df_window) - 1):
            t1 = df_window.loc[i, 'timestamp']
            t2 = df_window.loc[i+1, 'timestamp']
            dt = (t2 - t1).total_seconds()
            if dt <= 0:
                continue
            lat1 = df_window.loc[i, 'latitude']
            lon1 = df_window.loc[i, 'longitude']
            lat2 = df_window.loc[i+1, 'latitude']
            lon2 = df_window.loc[i+1, 'longitude']
            # Distance in meters.
            d = distance((lat1, lon1), (lat2, lon2)).meters
            # Speed in m/s.
            speed_mps = d / dt
            # Convert m/s to knots (1 m/s = 1.94384 knots).
            speed_knots = speed_mps * 1.94384
            speeds.append(speed_knots)
        
        if not speeds:
            return (np.nan, np.nan, np.nan)
        
        return (np.mean(speeds), np.min(speeds), np.max(speeds))
    
    # Loop through each engine run event and compute speed statistics.
    event_stats = []
    for dt, dur in zip(event_datetimes, engine_run_durations):
        avg_speed, min_speed, max_speed = calc_speed_stats_in_window(dt, dur, df_flight)
        if debug:
            print(f"DEBUG: Engine run starting at {dt.strftime('%H:%M:%S')} for {dur} sec:")
            print(f"       Avg Speed: {avg_speed:.2f} knots, Min Speed: {min_speed:.2f} knots, Max Speed: {max_speed:.2f} knots")
        event_stats.append((dt, avg_speed, min_speed, max_speed))
    
    return event_stats
# --- Function to extract TAS Field Positions from I-Record ---
def extract_tas_field_positions(igc_file_path):
    """
    Opens the IGC file and finds the TAS field definition in the I-record.
    Instead of reading digits after the literal "TAS", this function extracts the two pairs of digits immediately 
    preceding "TAS". For example, if the I-record contains "4246TAS4751", then it returns (42, 46),
    indicating that the TAS field in the B-record spans columns 42 to 46 (1-indexed).
    
    Returns:
      A tuple (start_pos, end_pos) if found, or None if not found.
    """
    with open(igc_file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("I") and "TAS" in line:
                # Look for two pairs of digits immediately preceding "TAS"
                m = re.search(r"(\d{2})(\d{2})TAS", line)
                if m:
                    start_pos = int(m.group(1))
                    end_pos = int(m.group(2))
                    print(f"DEBUG: Found TAS positions: start = {start_pos}, end = {end_pos}")
                    return (start_pos, end_pos)
    print("DEBUG: TAS field not found in the I record.")
    return None

# --- Updated parse_b_records Function ---
def parse_b_records(igc_file_path, tas_positions, flight_date_str):
    """
    Parses the B records in the IGC file to extract the fix timestamp (from columns 2-7)
    and the TAS value using the positions determined from the I-record.

    The TAS field is assumed to occupy the columns from tas_positions[0] to tas_positions[1] (inclusive).
    The field width is computed as: width = end_pos - start_pos + 1.
    Conversion is applied as follows:
      - If width == 3: The raw value is in kph (whole number). Convert to knots by dividing by 1.852.
      - If width == 4: The raw value is already in knots with 2 decimal places.
      - If width == 5: The raw value is in kph to the hundredths (divide by 100 then by 1.852).
      - Otherwise: Assume kph to the hundredths (divide by 100 then by 1.852).

    Returns:
      A list of dictionaries with keys 'timestamp' (a datetime object) and 'TAS' (float, in knots).
    """
    b_records = []
    base_date = datetime.strptime(flight_date_str, "%m/%d/%Y")
    if tas_positions is None:
        raise ValueError("TAS field positions not found in I record.")
    tas_start, tas_end = tas_positions  # These positions are 1-indexed.
    field_width = tas_end - tas_start + 1

    with open(igc_file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("B"):
                # Extract the fix time from columns 2-7 (indices 1:7)
                time_str = line[1:7]
                try:
                    fix_time = base_date.replace(
                        hour=int(time_str[0:2]),
                        minute=int(time_str[2:4]),
                        second=int(time_str[4:6])
                    )
                except Exception:
                    continue

                # Extract raw TAS string from the B record using the defined positions.
                tas_str = line[tas_start-1:tas_end]  # Convert to 0-indexed slice.
                try:
                    raw_value = float(tas_str)
                except Exception:
                    raw_value = np.nan

                # Conversion logic based on field width.
                if field_width == 4:
                    # Value is already in knots with 2 decimal places.
                    tas_val = round(raw_value, 2)
                elif field_width == 3:
                    # Value is in kph (whole number); convert to knots.
                    tas_val = raw_value / 1.852
                elif field_width == 5:
                    # Value is in kph to the hundredths.
                    tas_val = (raw_value / 100.0) / 1.852
                else:
                    # Default: assume kph to the hundredths.
                    tas_val = (raw_value / 100.0) / 1.852

                b_records.append({
                    "timestamp": fix_time,
                    "TAS": tas_val
                })
    return b_records

def extract_gsp_field_positions(igc_file_path):
    """
    Opens the IGC file and finds the GSP field definition in the I‑record.
    Instead of reading digits after the literal "GSP", this function extracts the two pairs of digits immediately 
    preceding "GSP". For example, if the I‑record contains "4751GSP", then it returns (47, 51),
    indicating that the GSP field in the B‑record spans columns 47 to 51 (1‑indexed).

    Returns:
      A tuple (start_pos, end_pos) if found, or None if not found.
    """
    with open(igc_file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("I") and "GSP" in line:
                m = re.search(r"(\d{2})(\d{2})GSP", line)
                if m:
                    start_pos = int(m.group(1))
                    end_pos = int(m.group(2))
                    print(f"DEBUG: Found GSP positions: start = {start_pos}, end = {end_pos}")
                    return (start_pos, end_pos)
    print("DEBUG: GSP field not found in the I record.")
    return None

def parse_b_records_for_gsp(igc_file_path, gsp_positions, flight_date_str):
    """
    Parses the B records in the IGC file to extract the fix timestamp (from columns 2-7)
    and the GSP value using the positions determined from the I‑record.

    The GSP field is assumed to occupy the columns from gsp_positions[0] to gsp_positions[1] (inclusive).
    The field width is computed as: width = end_pos - start_pos + 1.
    
    Conversion is applied as follows:
      - If width == 3: The raw value is in kph (whole number). Convert to knots by dividing by 1.852.
      - If width == 4: The raw value is already in knots.
      - If width == 5: The raw value is in kph to the hundredths (divide by 100 then by 1.852).
      - Otherwise: Assume kph to the hundredths (divide by 100 then by 1.852).

    Returns:
      A list of dictionaries with keys 'timestamp' (a datetime object) and 'GSP' (float, in knots).
    """
    gsp_records = []
    base_date = datetime.strptime(flight_date_str, "%m/%d/%Y")
    if gsp_positions is None:
        raise ValueError("GSP field positions not found in I record.")
    gsp_start, gsp_end = gsp_positions  # These positions are 1-indexed.
    field_width = gsp_end - gsp_start + 1

    with open(igc_file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("B"):
                # Extract the fix time from columns 2-7 (indices 1:7)
                time_str = line[1:7]
                try:
                    fix_time = base_date.replace(
                        hour=int(time_str[0:2]),
                        minute=int(time_str[2:4]),
                        second=int(time_str[4:6])
                    )
                except Exception:
                    continue

                # Extract raw GSP string from the B record using the defined positions.
                gsp_str = line[gsp_start-1:gsp_end]  # Convert to 0-indexed slice.
                try:
                    raw_value = float(gsp_str)
                except Exception:
                    raw_value = np.nan

                # Conversion logic based on field width.
                if field_width == 4:
                    # Assume the value is already in knots.
                    gsp_val = raw_value
                elif field_width == 3:
                    # Value is in kph (whole number); convert to knots.
                    gsp_val = raw_value / 1.852
                elif field_width == 5:
                    # Value is in kph to the hundredths.
                    gsp_val = (raw_value / 100.0) / 1.852
                else:
                    # Default: assume kph to hundredths.
                    gsp_val = (raw_value / 100.0) / 1.852

                gsp_records.append({
                    "timestamp": fix_time,
                    "GSP": gsp_val
                })
    return gsp_records

# --- compute_time_weighted_average Function (same as before) ---
def compute_time_weighted_average(b_records_window, debug=False):
    """
    Given a list of B-records (each a dict with 'timestamp' and 'TAS') for an engine-run window,
    compute the time-weighted average TAS (in knots) using the trapezoidal rule.
    
    If debug is True, print intermediate values.
    
    Returns:
      weighted_avg (float): The time-weighted average TAS.
    """
    if len(b_records_window) == 0:
        return np.nan
    if len(b_records_window) == 1:
        if debug:
            print("DEBUG: Only one record in window; returning its TAS value:", b_records_window[0]["TAS"])
        return b_records_window[0]["TAS"]
    
    sorted_records = sorted(b_records_window, key=lambda r: r["timestamp"])
    times = [r["timestamp"] for r in sorted_records]
    tas_values = [r["TAS"] for r in sorted_records]
    
    t0 = times[0]
    time_seconds = [(t - t0).total_seconds() for t in times]
    
    if debug:
        print("DEBUG: Time offsets (sec):", time_seconds)
        print("DEBUG: TAS values in window (knots):", tas_values)
    
    weighted_integral = 0.0
    total_time = time_seconds[-1] - time_seconds[0]
    
    if debug:
        print("DEBUG: Interval contributions:")
    for i in range(len(time_seconds)-1):
        dt = time_seconds[i+1] - time_seconds[i]
        avg_val = (tas_values[i] + tas_values[i+1]) / 2.0
        weighted_integral += avg_val * dt
        if debug:
            print(f"  {times[i].time()} to {times[i+1].time()} dt={dt:.2f} sec, avg_TAS={avg_val:.2f}, contribution={avg_val*dt:.2f}")
    
    if total_time > 0:
        weighted_avg = weighted_integral / total_time
    else:
        weighted_avg = tas_values[0]
    
    if debug:
        print(f"DEBUG: Total time: {total_time:.2f} sec, Weighted integral: {weighted_integral:.2f}, Weighted avg: {weighted_avg:.2f}")
    return weighted_avg

# --- get_true_airspeed_stats_for_engine_runs_from_row Function ---
def get_true_airspeed_stats_for_engine_runs_from_row(row, debug=False):
    """
    Processes a single flight (row) to compute true airspeed (TAS) statistics during each engine run.
    
    Expects the following columns in the row:
      - "File": the IGC filename (e.g. "16TVACN1.igc"); the file is assumed to reside in the "TAS" folder.
      - "Date (MM/DD/YYYY)": the flight date as a string.
      - "ENL_Engine_Run_Start_Times", "MOP_Engine_Run_Start_Times", "RPM_Engine_Run_Start_Times":
            comma-separated strings of engine run start times in HHMMSS format.
      - "engine_run_times (s)": a list (or comma-separated string) of engine run durations in seconds.
    
    Returns:
      A list of tuples, one per engine run event, each tuple:
         (event_start_datetime, weighted_avg_TAS, min_TAS, max_TAS)
         If no TAS values are found for an event, the TAS statistics are returned as blank.
    """
    import os
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np

    base_folder = "TAS"
    alt_folder = "TAS_wgc"
    igc_filename = row["File"]
    igc_file_path = os.path.join(base_folder, igc_filename)
    if not os.path.exists(igc_file_path):
        igc_file_path = os.path.join(alt_folder, igc_filename)
    flight_date_str = row["Date (MM/DD/YYYY)"]

    # Gather engine run start time tokens from the three possible columns.
    engine_run_time_tokens = []
    for col in ["ENL_Engine_Run_Start_Times", "MOP_Engine_Run_Start_Times", "RPM_Engine_Run_Start_Times"]:
        token_str = str(row.get(col, ""))
        if token_str.endswith('.0'):
            token_str = token_str[:-2]
        if pd.notnull(token_str) and token_str != "":
            tokens = [token.strip() for token in token_str.split(",")
                      if token.strip() and token.strip().isdigit() and len(token.strip()) == 6]
            engine_run_time_tokens.extend(tokens)
    if not engine_run_time_tokens:
        if debug:
            print("DEBUG: No engine run times found for this flight.")
        return []

    # Retrieve engine run durations.
    durations_val = row.get("engine_run_times (s)", None)
    if durations_val is None or durations_val == "":
        if debug:
            print("DEBUG: No engine run durations found for this flight.")
        return []
    else:
        if isinstance(durations_val, str):
            if durations_val.startswith('[') and durations_val.endswith(']'):
                durations_val = durations_val[1:-1]
            try:
                engine_run_durations = [float(x.strip()) for x in durations_val.split(",") if x.strip()]
            except Exception as e:
                if debug:
                    print("DEBUG: Error parsing engine run durations:", e)
                return []
        elif isinstance(durations_val, (list, tuple)):
            engine_run_durations = durations_val
        else:
            engine_run_durations = [float(durations_val)]
    
    if debug:
        print("DEBUG: Engine run start time tokens:", engine_run_time_tokens)
        print("DEBUG: Engine run durations (s):", engine_run_durations)
    
    # Retrieve TAS field positions and parse B records from the IGC file.
    tas_positions = extract_tas_field_positions(igc_file_path)
    if tas_positions is None:
        raise ValueError("TAS field not found in I record in file " + igc_file_path)
    
    b_records = parse_b_records(igc_file_path, tas_positions, flight_date_str)
    if debug:
        print(f"DEBUG: Total number of B records parsed: {len(b_records)}")
    
    stats = []
    event_idx = 0
    for token in engine_run_time_tokens:
        try:
            event_dt = datetime.strptime(f"{flight_date_str} {token}", "%m/%d/%Y %H%M%S")
        except Exception as e:
            if debug:
                print(f"DEBUG: Skipping token '{token}' due to error: {e}")
            continue
        
        if event_idx < len(engine_run_durations):
            duration_sec = engine_run_durations[event_idx]
        else:
            duration_sec = 300
        event_idx += 1
        
        window_end = event_dt + timedelta(seconds=duration_sec)
        
        # Filter the B records to those within the engine run event window.
        b_records_window = [record for record in b_records if event_dt <= record["timestamp"] <= window_end]
        if debug:
            print(f"DEBUG: Engine run starting at {event_dt.strftime('%H:%M:%S')} with duration {duration_sec} sec")
            print(f"DEBUG: Number of B records in this window: {len(b_records_window)}")
            for r in b_records_window:
                print(f"    Time: {r['timestamp'].strftime('%H:%M:%S')}, TAS: {r['TAS']:.2f} knots")
        
        if not b_records_window:
            weighted_avg = ""
            min_tas = ""
            max_tas = ""
        else:
            # Compute average as a simple arithmetic mean.
            tas_values = [record["TAS"] for record in b_records_window]
            weighted_avg = sum(tas_values) / len(tas_values)
            min_tas = np.min(tas_values)
            max_tas = np.max(tas_values)
        stats.append((event_dt, weighted_avg, min_tas, max_tas))
    return stats



def get_all_engine_start_speeds(row, df_flight, window_seconds=10):
    """
    For a given flight row and its flight data (df_flight), compute the estimated ground speed (in knots)
    at each engine run event. Engine run start times are taken from the columns:
       ENL_Engine_Run_Start_Times, MOP_Engine_Run_Start_Times, RPM_Engine_Run_Start_Times.
    Each time is assumed to be in HHMMSS format. The flight date is taken from the 'Date (MM/DD/YYYY)' column.
    
    Parameters:
        row (pd.Series): A flight row.
        df_flight (DataFrame): Flight data with columns ['timestamp', 'latitude', 'longitude'].
        window_seconds (int): The window (in seconds) around the event time used to compute speed.
    
    Returns:
        str: A comma-separated string of the estimated speeds (in knots) for each engine run event.
             If no events are available, returns an empty string.
    """
    speeds = []
    flight_date_str = row.get('Date (MM/DD/YYYY)', None)
    if not flight_date_str:
        return ""
    try:
        base_date = datetime.strptime(flight_date_str, "%m/%d/%Y")
    except Exception as e:
        return ""
    
    # Iterate over each sensor.
    for sensor in ["ENL", "MOP", "RPM"]:
        col_name = f"{sensor}_Engine_Run_Start_Times"
        times_str = row.get(col_name, "")
        if pd.notnull(times_str) and isinstance(times_str, str):
            for token in times_str.split(","):
                token = token.strip()
                if token.isdigit() and len(token) == 6:  # Expecting HHMMSS
                    try:
                        event_dt = base_date.replace(
                            hour=int(token[0:2]),
                            minute=int(token[2:4]),
                            second=int(token[4:6])
                        )
                    except Exception as e:
                        continue
                    # Compute the speed at this event using get_speed_at_event.
                    spd = get_speed_at_event(df_flight, event_dt, window_seconds=window_seconds)
                    speeds.append(spd)
    # Convert speeds to strings; if a value is NaN, represent it as "NaN".
    return ",".join([f"{s:.2f}" if not pd.isna(s) else "NaN" for s in speeds])

def get_first_engine_agl(row):
    """
    For a given row, returns the engine run altitude (AGL) for the first engine event.
    It uses the 'first_event_time' and 'event_sensor' columns (from get_first_engine_event)
    and then looks up the corresponding engine run altitude value from the sensor's altitude column.

    Parameters:
      row (pd.Series): A row from the DataFrame.
    
    Returns:
      float or pd.NA: The AGL value for the first engine event or NA if not found.
    """
    chosen_time = row.get('first_event_time')
    sensors_str = row.get('event_sensor', "")
    
    if pd.isna(chosen_time) or sensors_str == "":
        return pd.NA

    # Pick the first sensor if multiple are returned.
    sensor = sensors_str.split(",")[0].strip()
    
    # Identify the corresponding columns for engine run times and altitudes.
    time_col = f"{sensor}_Engine_Run_Start_Times"
    alt_col = f"{sensor}_Engine_Run_Altitudes_AGL"
    
    # Retrieve the raw values from the row.
    time_str = row.get(time_col, "")
    alt_str = row.get(alt_col, "")
    
    if pd.isnull(time_str) or pd.isnull(alt_str) or time_str == "" or alt_str == "":
        return pd.NA
    
    # Parse the time column: expect a comma-separated string of times (as integers)
    try:
        time_list = [int(x.strip()) for x in time_str.split(",") if x.strip().isdigit()]
    except Exception as e:
        logging.warning(f"Error parsing times in column {time_col} for row {row.get('File', 'Unknown')}: {e}")
        return pd.NA

    # Parse the altitude column: expect a comma-separated string of altitude values (as floats)
    try:
        alt_list = [float(x.strip()) for x in alt_str.split(",") if x.strip()]
    except Exception as e:
        logging.warning(f"Error parsing altitudes in column {alt_col} for row {row.get('File', 'Unknown')}: {e}")
        return pd.NA

    if not time_list or not alt_list:
        return pd.NA

    try:
        # Find the index of the chosen_time in the time list.
        idx = time_list.index(chosen_time)
    except ValueError:
        # chosen_time not found in the list
        return pd.NA

    # Return the corresponding altitude, if available.
    if idx < len(alt_list):
        return alt_list[idx]
    else:
        return pd.NA


def get_agl_all_events(row, alt_cols=['ENL_Engine_Run_Altitudes_AGL', 'MOP_Engine_Run_Altitudes_AGL', 'RPM_Engine_Run_Altitudes_AGL']):
    """
    For a given row, this function concatenates the AGL values from the specified altitude columns.
    
    Parameters:
      row (pd.Series): A row from the DataFrame.
      alt_cols (list): List of column names to look for AGL values.
      
    Returns:
      str: A comma-separated string of all AGL values found in the specified columns.
           If no valid values are found, returns an empty string.
    """
    values = []
    for col in alt_cols:
        val = row.get(col, "")
        if pd.notnull(val):
            # If the value is a string, split on commas.
            if isinstance(val, str):
                # Split the string on commas and strip any extra whitespace.
                tokens = [token.strip() for token in val.split(",") if token.strip()]
                values.extend(tokens)
            else:
                # If the value is numeric, simply convert to string.
                values.append(str(val))
    # Join all values with a comma.
    return ",".join(values)


def count_pct_engine_starts_below_1000(agl_all_events_str):
    """
    Given a comma-separated string of AGL values (from the 'agl_all_events' column),
    returns a tuple (num_engine_starts_below_1000ft, pct_engine_starts_below_1000ft).
    
    If the input string is blank (or only whitespace), returns (pd.NA, pd.NA).

    Parameters:
      agl_all_events_str (str): Comma-separated AGL values.
      
    Returns:
      tuple: (count, percentage) or (pd.NA, pd.NA) if the input is blank.
    """
    # If the input is missing or blank, return NA for both values.
    if pd.isna(agl_all_events_str) or agl_all_events_str.strip() == "":
        return (pd.NA, pd.NA)
    
    # Split the string by commas and attempt to convert tokens to floats.
    tokens = agl_all_events_str.split(",")
    values = []
    for token in tokens:
        token = token.strip()
        if token:
            try:
                value = float(token)
                values.append(value)
            except Exception:
                continue  # Skip tokens that cannot be converted.
    
    total = len(values)
    if total == 0:
        return (pd.NA, pd.NA)
    
    # Count how many of the values are below 1000 ft.
    count_below = sum(1 for val in values if val < 1000)
    pct_below = (count_below / total) * 100
    return (count_below, pct_below)



def calc_speed_mean_post_event_window(df_flight, event_dt, engine_run_duration=None, default_window_minutes=5):
    """
    Compute the mean and standard deviation of ground speed (in knots) for the time window
    starting at event_dt and lasting for the duration of the engine run (converted from seconds
    to minutes). If engine_run_duration is not provided, a default window (in minutes) is used.
    
    Parameters:
        df_flight (DataFrame): Flight data with columns ['timestamp', 'latitude', 'longitude'].
        event_dt (pd.Timestamp): The reference datetime (e.g. row['first_event_datetime']).
        engine_run_duration (float, optional): Duration in seconds of the engine run event.
        default_window_minutes (int, optional): Default time window in minutes if engine_run_duration is None.
    
    Returns:
        tuple: (mean_speed, std_speed) in knots, or (np.nan, np.nan) if not enough data.
    """
    if pd.isna(event_dt):
        return np.nan, np.nan

    # Use the engine run duration if provided; otherwise, use the default window.
    if engine_run_duration is not None:
        window_minutes = engine_run_duration / 60.0
    else:
        window_minutes = default_window_minutes

    start_time = event_dt
    end_time = event_dt + pd.Timedelta(minutes=window_minutes)

    # Filter flight data to the post-event window.
    mask = (df_flight['timestamp'] >= start_time) & (df_flight['timestamp'] <= end_time)
    df_window = df_flight.loc[mask].copy()
    if df_window.shape[0] < 2:
        logging.info("Speed calculation: Not enough points in the post-event window.")
        return np.nan, np.nan

    df_window = df_window.sort_values('timestamp')

    speeds = []
    timestamps = df_window['timestamp'].values
    lats = df_window['latitude'].values
    lons = df_window['longitude'].values

    for i in range(len(df_window) - 1):
        lat1, lon1 = lats[i], lons[i]
        lat2, lon2 = lats[i+1], lons[i+1]
        dt_sec = (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 's')
        if dt_sec <= 0:
            continue
        # Calculate distance in meters and convert speed to knots (1 m/s = 1.94384 knots)
        dist_m = geodesic((lat1, lon1), (lat2, lon2)).meters
        speed_m_s = dist_m / dt_sec
        speed_knots = speed_m_s * 1.94384
        speeds.append(speed_knots)

    if not speeds:
        logging.info("Speed calculation: No valid speed segments in post-event window.")
        return np.nan, np.nan

    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    return mean_speed, std_speed


# Example helper: calculate average & std dev of speed in [event_dt - 5min, event_dt]
def calc_speed_mean_std_pre_event_window(df_flight, event_dt, window_minutes=5):
    """
    Compute the mean and std dev of ground speed (in knots) for the time window
    [event_dt - window_minutes, event_dt], based on consecutive lat/lon/time rows.

    df_flight: DataFrame with columns ['timestamp', 'latitude', 'longitude']
    event_dt:  The reference datetime (pd.Timestamp), e.g. row['first_event_datetime']
    window_minutes: int, time window length before event_dt

    Returns: (mean_speed, std_speed) in knots, or (np.nan, np.nan) if not enough data
    """
    if pd.isna(event_dt):
        return np.nan, np.nan

    start_time = event_dt - pd.Timedelta(minutes=window_minutes)
     
    end_time = event_dt
    
    # Filter to the pre-event window
    mask = (df_flight['timestamp'] >= start_time) & (df_flight['timestamp'] <= end_time)

    df_window = df_flight.loc[mask].copy()

    # Need at least 2 points to compute speed
    if df_window.shape[0] < 2:
        logging.info("Speed calculation: Not enough points in the pre-event window.")
        return np.nan, np.nan

    # Sort by timestamp
    df_window = df_window.sort_values('timestamp')

    # Compute speeds
    speeds = []
    timestamps = df_window['timestamp'].values
    lats = df_window['latitude'].values
    lons = df_window['longitude'].values

    for i in range(len(df_window) - 1):
        lat1, lon1 = lats[i], lons[i]
        lat2, lon2 = lats[i+1], lons[i+1]
        dt_sec = (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 's')
        if dt_sec <= 0:
            continue  # skip if same timestamp or data anomaly

        dist_m = geodesic((lat1, lon1), (lat2, lon2)).meters
        speed_m_s = dist_m / dt_sec
        speed_knots = speed_m_s * 1.94384  # convert m/s to knots
        speeds.append(speed_knots)

    if not speeds:
        logging.info("Speed calculation: No valid speed segments in pre-event window.")
        return np.nan, np.nan

    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    return mean_speed, std_speed



def convert_to_datetime(row):
    """
    Converts first_event_time from integer HHMMSS to full datetime by combining with flight date.
    
    Parameters:
        row (pd.Series): A row from the DataFrame.

    Returns:
        pd.Timestamp: Combined datetime object.
    """
    if pd.isna(row['first_event_time']):
        return pd.NaT

    try:
        # Convert Date (MM/DD/YYYY) column to a proper datetime.date object
        flight_date = pd.to_datetime(row['Date (MM/DD/YYYY)'], format='%m/%d/%Y').date()
        
        # Ensure first_event_time is a proper 6-digit integer
        time_int = int(row['first_event_time'])
        time_str = f"{time_int:06d}"  # Zero-pad to ensure HHMMSS format

        # Convert to datetime.time object
        time_obj = datetime.strptime(time_str, "%H%M%S").time()

        # Combine with the flight date
        first_event_datetime = datetime.combine(flight_date, time_obj)
        
        return pd.Timestamp(first_event_datetime)
    except Exception as e:
        logging.warning(f"Error converting to datetime: {e}")
        return pd.NaT

def define_time_window(event_datetime, flight_start, window_minutes=5):
    """
    Defines a time window ending at event_datetime.
    
    If the event_datetime occurs less than 'window_minutes' after flight_start,
    the window start is set to flight_start (so the window duration is the 
    difference between event_datetime and flight_start). Otherwise, a window 
    of 'window_minutes' is used.
    
    Args:
        event_datetime (pd.Timestamp): The time when the engine started.
        flight_start (pd.Timestamp): The first timestamp of the flight.
        window_minutes (int, optional): Desired window length in minutes. Defaults to 5.
        
    Returns:
        (pd.Timestamp, pd.Timestamp): The window start and end times.
    """
    if pd.isna(event_datetime) or pd.isna(flight_start):
        return pd.NaT, pd.NaT
    
    # Calculate the actual duration from flight start to the event (in minutes)
    actual_duration = (event_datetime - flight_start).total_seconds() / 60.0
    
    if actual_duration < window_minutes:
        window_start = flight_start
        window_end = event_datetime
    else:
        window_start = event_datetime - timedelta(minutes=window_minutes)
        window_end = event_datetime

    return pd.Timestamp(window_start), pd.Timestamp(window_end)

# Example usage:
# flight_start = pd.to_datetime("2024-06-24 23:00:00")
# event_datetime = pd.to_datetime("2024-06-24 23:04:00")
# In this case, since only 4 minutes have passed, the window will be from flight_start to event_datetime.
w_start, w_end = define_time_window(pd.to_datetime("2024-06-24 23:04:00"),
                                    pd.to_datetime("2024-06-24 23:00:00"), window_minutes=5)
print("Window Start:", w_start, "Window End:", w_end)


def load_flight_data(file_id: str, flight_date_str: str, igc_directory: str = 'filtered') -> Optional[pd.DataFrame]:
    """
    Loads the IGC log file for a given File ID, reads its content as text, 
    and converts it to a DataFrame using igc2df.
    
    Parameters:
        file_id (str): The identifier for the flight file (with or without extension).
        flight_date_str (str): The flight date in 'MM/DD/YYYY' format from the CSV.
        igc_directory (str): Directory where IGC files are stored. Defaults to 'filtered'.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame representation of the IGC data or None if loading fails.
    """
    from pathlib import Path
    import os
    import logging

    # Initialize Path object for the primary directory.
    file_path = Path(igc_directory) / file_id

    # Function to try alternate directory if needed.
    def try_alternate(file_id: str, alt_dir: str) -> Optional[Path]:
        alt_path = Path(alt_dir) / file_id
        # Try with both possible extensions.
        for ext in ['.IGC', '.igc']:
            candidate = alt_path.with_suffix(ext)
            if candidate.exists():
                return candidate
        return None

    # Check for missing or incorrect extension.
    if file_path.suffix.lower() != '.igc':
        # When no extension is present.
        if not file_path.suffix:
            possible_extensions = ['.IGC', '.igc']
            for ext in possible_extensions:
                temp_path = file_path.with_suffix(ext)
                if temp_path.exists():
                    file_path = temp_path
                    break
            else:
                # Try alternate directory if using default
                if igc_directory == 'filtered':
                    alt_candidate = try_alternate(file_id, 'filtered_wgc')
                    if alt_candidate is not None:
                        file_path = alt_candidate
                    else:
                        logging.error(f"IGC file '{file_id}' not found in 'filtered' or 'filtered_wgc' directories.")
                        return None
                else:
                    logging.error(f"IGC file '{file_id}' not found in '{igc_directory}' directory.")
                    return None
        else:
            # File has an extension but it might be wrong case.
            for ext in ['.IGC', '.igc']:
                temp_path = file_path.with_suffix(ext)
                if temp_path.exists():
                    file_path = temp_path
                    break
            else:
                if igc_directory == 'filtered':
                    alt_candidate = try_alternate(file_id, 'filtered_wgc')
                    if alt_candidate is not None:
                        file_path = alt_candidate
                    else:
                        logging.error(f"IGC file '{file_id}' not found in 'filtered' or 'filtered_wgc' directories.")
                        return None
                else:
                    logging.error(f"IGC file '{file_id}' not found in '{igc_directory}' directory.")
                    return None
    else:
        # File path already ends with .igc but check if it exists.
        if not file_path.exists():
            if igc_directory == 'filtered':
                alt_candidate = try_alternate(file_id, 'filtered_wgc')
                if alt_candidate is not None:
                    file_path = alt_candidate
                else:
                    logging.error(f"IGC file '{file_id}' not found in 'filtered' or 'filtered_wgc' directories.")
                    return None
            else:
                logging.error(f"IGC file '{file_id}' not found in '{igc_directory}' directory.")
                return None

    logging.info(f"Attempting to load IGC file for Flight '{file_id}': {file_path}")

    try:
        # Read the file content as text
        text_content = file_path.read_text(encoding='utf-8')
        
        # Pass flight_date_str to igc2df()
        df_flight = igc2df(text_content, flight_date_str)
        
        logging.info(f"File '{file_id}': IGC file loaded successfully.")
        return df_flight
    except Exception as e:
        logging.error(f"Error loading IGC file '{file_id}': {e}")
        return None

def determine_steady_descent_during_pre_event_window(df_flight, window_start, window_end, std_threshold=0.1):
    """
    Determines whether the descent during the specified window is steady or changing.

    Parameters:
        df_flight (pd.DataFrame): DataFrame containing flight data with 'timestamp' and 'descent_rate' columns.
        window_start (pd.Timestamp): Start time of the window.
        window_end (pd.Timestamp): End time of the window.
        std_threshold (float): Threshold for standard deviation to classify descent behavior.

    Returns:
        str: 'Steady' if std < threshold, 'Changing' otherwise.
    """
    # Ensure 'timestamp' is a pandas.Timestamp
    if not pd.api.types.is_datetime64_any_dtype(df_flight['timestamp']):
        logging.warning("The 'timestamp' column is not of datetime type.")
        return pd.NA
    
    # Filter data within the window
    df_window = df_flight[(df_flight['timestamp'] >= window_start) & (df_flight['timestamp'] <= window_end)]
    
    if df_window.empty or df_window['altitude_press'].isnull().all():
        logging.info(f"No data available or all descent rates are NaN between {window_start} and {window_end}.")
        return pd.NA  # Or another appropriate default value
    
    # Calculate descent_rate if not already present
    if 'descent_rate' not in df_window.columns:
        df_window['descent_rate'] = df_window['altitude_press'].diff() / df_window['timestamp'].diff().dt.total_seconds()
    
    # Drop NaN descent rates
    descent_rates = df_window['descent_rate'].dropna()
    
    if descent_rates.empty:
        logging.info(f"No valid descent rates between {window_start} and {window_end}.")
        return pd.NA
    
    # Calculate standard deviation of descent rates within the window
    descent_std = descent_rates.std()
    
    # Classify based on the threshold
    if descent_std < std_threshold:
        return 'Steady'
    else:
        return 'Changing'


def calculate_descent_rate(df_flight, window_start, window_end):
    """
    Calculates the descent rate (ft/s) within the specified window.
    """
    mask = (df_flight['timestamp'] >= window_start) & (df_flight['timestamp'] <= window_end)
    df_window = df_flight.loc[mask].copy()

    if df_window.empty or df_window['altitude_press'].isnull().all():
        logging.info("Descent rate calculation: No data in window or all altitudes are NaN.")
        return np.nan

    df_window = df_window.sort_values('timestamp')

    altitude_diff = df_window['altitude_press'].iloc[-1] - df_window['altitude_press'].iloc[0]
    time_diff = (df_window['timestamp'].iloc[-1] - df_window['timestamp'].iloc[0]).total_seconds()

    if time_diff == 0:
        logging.info("Descent rate calculation: Time difference is zero.")
        return 0

    descent_rate = altitude_diff / time_diff
    logging.info(f"Descent rate calculated: {descent_rate} ft/s")
    return descent_rate
def calculate_distance_traveled(df_flight, window_start, window_end):
    """
    Calculates the total distance traveled within the specified window (in miles).
    """
    mask = (df_flight['timestamp'] >= window_start) & (df_flight['timestamp'] <= window_end)
    df_window = df_flight.loc[mask].copy()

    if df_window.shape[0] < 2:
        logging.info("Distance traveled calculation: Not enough data in window.")
        return 0  # Not enough data to calculate distance

    df_window = df_window.sort_values('timestamp')

    total_distance = 0
    prev_point = None
    for _, row in df_window.iterrows():
        current_point = (row['latitude'], row['longitude'])
        if prev_point is not None:
            segment_distance = geodesic(prev_point, current_point).miles
            total_distance += segment_distance
        prev_point = current_point

    logging.info(f"Distance traveled calculated: {total_distance} miles")
    return total_distance
def detect_climb_attempt(df_flight, window_start, window_end):
    """
    Determines if a climb was attempted or achieved within the window.
    Returns:
        1 if climb attempted,
        2 if climb achieved,
        0 otherwise.
    """
    mask = (df_flight['timestamp'] >= window_start) & (df_flight['timestamp'] <= window_end)
    df_window = df_flight.loc[mask].copy()

    if df_window.empty:
        logging.info("Climb detection: No data in window.")
        return 0

    altitude_diff = df_window['altitude'].diff().fillna(0)

    if (altitude_diff > 0).any():
        logging.info("Climb detection: Climb achieved.")
        return 2  # Climb achieved
    elif (df_window['altitude'].diff().fillna(0) < 0).any():
        logging.info("Climb detection: Descent detected, no climb attempted.")
        return 1  # Descent detected, implying no climb attempted
    else:
        logging.info("Climb detection: No significant altitude change.")
        return 0  # No significant altitude change
def detect_multiple_start_stop(row):
    """
    Checks if there are multiple Engine_Run_Start_Times within the window.
    Returns:
        1 if multiple start attempts,
        0 otherwise.
    """
    count = 0
    for sensor in ['ENL', 'MOP', 'RPM']:
        times_str = row.get(f"{sensor}_Engine_Run_Start_Times", "")
        if pd.notna(times_str) and isinstance(times_str, str) and times_str.strip():
            times = [t.strip() for t in times_str.split(',') if t.strip().isdigit()]
            count += len(times)
    logging.info(f"File {row['File']}: Multiple start/stop actions: {count}")
    return 1 if count > 1 else 0
def clean_time_entries(time_str):
    """
    Cleans and validates time entries.
    Removes non-digit characters and ensures a 6-digit HHMMSS format.
    
    Parameters:
        time_str (str or float): The time entry to clean.
    
    Returns:
        str: A cleaned 6-digit time string or an empty string if invalid.
    """
    if pd.isna(time_str):
        # If the entry is NaN, return an empty string
        return ""
    
    # Ensure the input is a string; if not, convert it to string
    if not isinstance(time_str, str):
        time_str = str(time_str)
    
    # Remove any non-digit characters
    cleaned = ''.join(filter(str.isdigit, time_str))
    
    # Ensure the cleaned string is exactly 6 digits (HHMMSS)
    if len(cleaned) == 6:
        return cleaned
    else:
        # Log a warning if the time format is invalid
        logging.warning(f"Invalid time format: '{time_str}'")
        return ""


def get_first_engine_event(row, sensor_types=["ENL", "MOP", "RPM"], glider_types_df=None):
    """
    Determines the first engine event time for a flight based on engine run start times across sensors.
    For Self-Launch gliders (as determined from glider_types_df via the 'Model' and 'Launch-Type' columns),
    the event is set to the second earliest unique engine run start time (using the columns:
    ENL_Engine_Run_Start_Times, MOP_Engine_Run_Start_Times, and RPM_Engine_Run_Start_Times). 
    If a Self-Launch glider does not have at least two unique engine run events, the event columns are returned as blank.
    For all other flights, the event is set to the earliest engine run start time.

    Parameters:
        row (pd.Series): A row from the DataFrame representing a single flight.
        sensor_types (list): List of sensor prefixes to consider (default: ["ENL", "MOP", "RPM"]).
        glider_types_df (pd.DataFrame, optional): DataFrame containing glider information with columns
            'Model' and 'Launch-Type'.

    Returns:
        pd.Series: A Series containing 'first_event_time', 'event_type', and 'event_sensor'.
    """
    engine_run_times = []
    engine_run_sensors = {}

    # Gather all engine run start times from the provided sensors.
    for sensor in sensor_types:
        col_name = f"{sensor}_Engine_Run_Start_Times"
        times_str = row.get(col_name, "")
        # Only process if times_str is not null and is a string.
        if pd.notnull(times_str) and isinstance(times_str, str):
            try:
                # Split the string by commas, strip whitespace, and convert tokens to integers.
                times_list = [int(t.strip()) for t in times_str.split(",") if t.strip().isdigit()]
                engine_run_times.extend(times_list)
                for t in times_list:
                    engine_run_sensors.setdefault(t, []).append(sensor)
            except Exception as e:
                logging.warning(f"Row {row.get('File', 'Unknown')}: Error parsing {col_name}: '{times_str}'. Error: {e}")
        else:
            continue

    if not engine_run_times:
        logging.warning(f"Row {row.get('File', 'Unknown')}: No engine run start times found.")
        return pd.Series({
            'first_event_time': pd.NA,
            'event_type': pd.NA,
            'event_sensor': pd.NA
        })

    # Use unique engine run times to determine the count.
    unique_times = sorted(set(engine_run_times))
    
    # Default: use the earliest event time.
    chosen_time = min(engine_run_times)
    chosen_sensors = engine_run_sensors.get(chosen_time, [])

    # Determine if this flight is a Self-Launch glider using glider_types_df.
    is_self_launch = False
    if glider_types_df is not None and row.get("Gtype"):
        glider_model = row["Gtype"]
        glider_info = glider_types_df[glider_types_df["Model"] == glider_model]
        if not glider_info.empty:
            launch_type = glider_info.iloc[0].get("Launch-Type")
            if launch_type == "Self-Launch":
                is_self_launch = True
        else:
            logging.warning(f"Row {row.get('File', 'Unknown')}: No glider type information found for Gtype '{glider_model}'.")
    
    if is_self_launch:
        # For Self-Launch gliders, require at least two unique engine run events.
        if len(unique_times) >= 2:
            chosen_time = unique_times[1]
            chosen_sensors = engine_run_sensors.get(chosen_time, [])
            logging.info(f"Row {row.get('File', 'Unknown')}: Self-Launch glider '{row.get('Gtype')}', "
                         f"using second unique engine event time {chosen_time} from sensor(s): {','.join(chosen_sensors)}")
        else:
            logging.warning(f"Row {row.get('File', 'Unknown')}: Self-Launch glider '{row.get('Gtype')}' "
                            f"has fewer than two unique engine run events. Leaving event columns blank.")
            return pd.Series({
                'first_event_time': pd.NA,
                'event_type': pd.NA,
                'event_sensor': pd.NA
            })
    else:
        logging.info(f"Row {row.get('File', 'Unknown')}: Not a Self-Launch glider, using earliest engine event time {chosen_time}.")

    event_sensor = ','.join(chosen_sensors)
    logging.info(f"Row {row.get('File', 'Unknown')}: first_event_time set to {chosen_time} from sensor(s): {event_sensor}")
    return pd.Series({
        'first_event_time': chosen_time,
        'event_type': 'Engine_Run_Start',
        'event_sensor': event_sensor
    })
LANDCOVER_CLASSES = {
    0: "Unknown",
    20: "Shrubs",
    30: "Herbaceous vegetation",
    40: "Agriculture",
    50: "Urban",
    60: "Bare/Sparse vegetation",
    70: "Snow/Ice",
    80: "Water Bodies",
    90: "Wetland",
    100: "Moss/Lichen",
    111: "Closed Forest, Evergreen Needleleaf",
    112: "Closed Forest, Evergreen Broadleaf",
    113: "Closed Forest, Deciduous Needleleaf",
    114: "Closed Forest, Deciduous Broadleaf",
    115: "Closed Forest, Mixed",
    116: "Closed Forest, Other",
    121: "Open Forest, Evergreen Needleleaf",
    122: "Open Forest, Evergreen Broadleaf",
    123: "Open Forest, Deciduous Needleleaf",
    124: "Open Forest, Deciduous Broadleaf",
    125: "Open Forest, Mixed",
    126: "Open Forest, Other",
    200: "Oceans/Seas"
}

def get_terrain_label_at_first_engine_event_gee(lat, lon):
    """
    Retrieves the terrain (land cover) label at the given latitude and longitude using
    the globally defined LANDCOVER_DATASET and maps the numeric value to a descriptive label.
    
    Parameters:
        lat (float): Latitude.
        lon (float): Longitude.
    
    Returns:
        str: The descriptive land cover label.
    """
    # Import the global LANDCOVER_DATASET from your helper module.
    # (This ensures we get the updated global variable.)
    from ee_helpers import LANDCOVER_DATASET
    
    if LANDCOVER_DATASET is None:
        raise ValueError("LANDCOVER_DATASET is not initialized!")
    
    # Create a point geometry for the given coordinates.
    point = ee.Geometry.Point(lon, lat)
    
    # Sample the LANDCOVER_DATASET at the given point.
    sample = LANDCOVER_DATASET.sample(region=point, scale=100, numPixels=1).first()
    if sample is None:
        logging.info(f"No sample returned for point ({lat}, {lon}). Returning 'Unknown'.")
        return "Unknown"
    
    try:
        # Retrieve the numeric value from the 'discrete_classification' band.
        value = sample.get("discrete_classification").getInfo()
    except Exception as e:
        logging.warning(f"Error getting classification value at ({lat}, {lon}): {e}")
        return "Unknown"
    
    # Use the mapping to convert the numeric value to a descriptive label.
    label = LANDCOVER_CLASSES.get(value, "Unknown")
    return label


def parse_int_list(s):
    """
    Convert a comma-separated string of ints into a list of ints.
    Return an empty list if s is None or empty.
    """
    if not s or pd.isna(s):
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip()]

def parse_float_list(s):
    """
    Convert a comma-separated string of floats into a list of floats.
    Return an empty list if s is None or empty.
    """
    if not s or pd.isna(s):
        return []
    return [float(x.strip()) for x in s.split(',') if x.strip()]

def parse_engine_run_info(sensor_info, sensor="MOP"):
    """
    Parses all engine run events from the 'Sensor Info' string.

    Parameters:
        sensor_info (str): The text containing engine run information.
        sensor (str): The sensor type (e.g., "ENL", "MOP").

    Returns:
        Tuple[List[int], List[int]]: Lists of engine run times (seconds) and height gains/losses (ft).
    """
    if not isinstance(sensor_info, str) or pd.isna(sensor_info):
        return [], []

    # Regular expression to capture all engine run events
    pattern = rf"{sensor}\s+monitor\s+reports\s+Engine\s+Run\s+(\d+)\s+minutes.*?Height\s+gain/loss\s+is:\s+(-?\d+)"
    regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)

    matches = regex.findall(sensor_info)
    
    if not matches:
        return [], []

    # Convert extracted values to lists of integers
    engine_run_times = [int(min_str) * 60 for min_str, _ in matches]  # Convert minutes to seconds
    height_gains = [int(gain_str) for _, gain_str in matches]

    return engine_run_times, height_gains
def get_dem_elevation_gee(lat, lon):
    """
    Retrieves terrain elevation in meters from Copernicus GLO-30 DEM via Google Earth Engine.

    Parameters:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.

    Returns:
        float: Elevation in meters.
    """
    try:
        dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select("DEM")  # ✅ Fix
        point = ee.Geometry.Point([lon, lat])
        elevation = dem.sample(region=point, scale=30).first().get('DEM').getInfo()
        return float(elevation)  # ✅ Convert to float
    except Exception as e:
        logging.warning(f"GEE elevation retrieval failed for ({lat}, {lon}): {e}")
        return np.nan


