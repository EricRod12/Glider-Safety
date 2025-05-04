from parser import parse_i_record, _parse_b_record, igc2df
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from pathlib import Path
from geopy.distance import geodesic

# --- Helper Functions for IGC Parsing (as used previously) ---

def parse_indices(line, verbose=False):
    """
    Parse an 'I' record line to extract sensor indices.
    The two-digit numbers preceding the sensor tags (ENL, MOP, RPM) indicate the positions.
    """
    indices = {"ENL": 0, "MOP": 0, "RPM": 0}
    try:
        count = int(line[1:3].strip())
        j = 7
        for _ in range(count):
            tag = line[j:j+3]
            if tag in indices:
                indices[tag] = int(line[j-4:j-2])
            j += 7
    except ValueError:
        if verbose:
            logging.info(f"Error parsing indices in line: {line}")
    return indices

def parse_engval(line, indices, verbose=False):
    """
    Extract the noise or engine sensor values from a 'B' record using sensor indices.
    """
    values = {"ENL": 0, "MOP": 0, "RPM": 0}
    for sensor, idx in indices.items():
        try:
            if idx > 0:
                values[sensor] = int(line[idx-1:idx+2].strip())
        except ValueError:
            if verbose:
                logging.info(f"Error parsing engine value for sensor {sensor} in line: {line}")
            values[sensor] = 0
    return values

def hhmmss_to_seconds(time_str):
    """
    Converts a HHMMSS string into seconds since midnight.
    """
    try:
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        ss = int(time_str[4:6])
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return None

# --- New Function: Calculate Speed Rate Change in a Given Time Window ---

def calc_speed_rate_change_interval(df_flight, start_dt, end_dt, verbose=False):
    """
    Calculate the mean and standard deviation of the rate of change (i.e. acceleration in knots/second)
    of ground speed over the time interval from start_dt to end_dt.
    
    The function computes speeds from successive points (using lat/lon and timestamp differences)
    and then computes the acceleration as differences between adjacent speed segments divided by the
    time interval between their midpoints.
    
    Parameters:
      df_flight (DataFrame): Flight data with columns ['timestamp', 'latitude', 'longitude'].
      start_dt (pd.Timestamp): Start of the interval.
      end_dt (pd.Timestamp): End of the interval.
      verbose (bool): Whether to log details.
    
    Returns:
      tuple: (mean_acceleration, std_acceleration) in knots/second, or (np.nan, np.nan) if insufficient data.
    """
    # Filter flight data to the specified window.
    mask = (df_flight['timestamp'] >= start_dt) & (df_flight['timestamp'] <= end_dt)
    df_window = df_flight.loc[mask].copy()
    if df_window.shape[0] < 2:
        if verbose:
            logging.info("Acceleration calculation: Not enough points in the window.")
        return np.nan, np.nan
    
    df_window = df_window.sort_values('timestamp')
    
    # Build list of (midpoint_time, speed in knots) for each segment.
    speed_points = []
    timestamps = df_window['timestamp'].values
    lats = df_window['latitude'].values
    lons = df_window['longitude'].values
    for i in range(len(df_window) - 1):
        t1 = df_window.iloc[i]['timestamp']
        t2 = df_window.iloc[i+1]['timestamp']
        dt_sec = (t2 - t1).total_seconds()
        if dt_sec <= 0:
            continue
        lat1, lon1 = lats[i], lons[i]
        lat2, lon2 = lats[i+1], lons[i+1]
        # Calculate distance in meters and convert to speed in knots (1 m/s = 1.94384 knots)
        dist_m = geodesic((lat1, lon1), (lat2, lon2)).meters
        speed_m_s = dist_m / dt_sec
        speed_knots = speed_m_s * 1.94384
        mid_time = t1 + (t2 - t1) / 2
        speed_points.append((mid_time, speed_knots))
    
    if len(speed_points) < 2:
        if verbose:
            logging.info("Acceleration calculation: Not enough speed segments.")
        return np.nan, np.nan
    
    # Compute acceleration between successive speed points.
    accelerations = []
    for i in range(len(speed_points) - 1):
        t1, s1 = speed_points[i]
        t2, s2 = speed_points[i+1]
        dt = (t2 - t1).total_seconds()
        if dt <= 0:
            continue
        acceleration = (s2 - s1) / dt  # in knots/second
        accelerations.append(acceleration)
    
    if not accelerations:
        return np.nan, np.nan
    mean_acc = np.mean(accelerations)
    std_acc = np.std(accelerations)
    return mean_acc, std_acc

# --- Main CSV Processing Function for Acceleration ---

def process_csv_for_acceleration(csv_file, base_folder, alt_folder, output_csv, verbose=False):
    """
    Reads flights_final_updated.csv (tab-separated) and, for each flight row, reads the corresponding IGC file
    (using the 'File' column). It then computes the ground speed rate-of-change (acceleration) during two periods:
    
      1. During the engine run, defined by the active sensor’s engine run start time and run duration.
      2. During the five minute period immediately after the engine run.
    
    The engine run start times are provided in one of the columns
      ENL_Engine_Run_Start_Times, MOP_Engine_Run_Start_Times, or RPM_Engine_Run_Start_Times,
    and the durations (in seconds) are in the 'engine_run_times (s)' column.
    
    For each engine run, the computed mean and std acceleration (in knots/second) are stored as comma-separated
    lists in new columns added to the CSV:
      - acceleration_engine_run_mean
      - acceleration_engine_run_std
      - acceleration_post_run_mean
      - acceleration_post_run_std
    """
    df = pd.read_csv(csv_file, sep="\t")
    
    # Add new columns for acceleration metrics.
    df['acceleration_engine_run_mean'] = ""
    df['acceleration_engine_run_std'] = ""
    df['acceleration_post_run_mean'] = ""
    df['acceleration_post_run_std'] = ""
    
    # Check required column for IGC file name.
    if 'File' not in df.columns:
        print("CSV file must contain a 'File' column with IGC file names.")
        sys.exit(1)
    
    # Process each flight.
    for index, row in df.iterrows():
        igc_filename = row['File']
        
        # Determine active sensor (only one should be used for engine run detection).
        active_sensor = None
        for sensor in ['ENL', 'MOP', 'RPM']:
            col_name = f"{sensor}_Engine_Run_Start_Times"
            if pd.notna(row.get(col_name, "")) and str(row.get(col_name, "")).strip() != "":
                active_sensor = sensor
                break
        
        if active_sensor is None:
            if verbose:
                logging.info(f"Row {index}: No engine run start times found; skipping acceleration calculation.")
            continue
        
        # Get the engine run start times and durations as lists.
        start_times_str = str(row[f"{active_sensor}_Engine_Run_Start_Times"]).strip()
        durations_str = str(row['engine_run_times (s)']).strip()
        if start_times_str == "" or durations_str == "":
            if verbose:
                logging.info(f"Row {index}: Missing engine run start times or durations; skipping.")
            continue
        
        start_times = [s.strip() for s in start_times_str.split(",") if s.strip() != ""]
        durations = [d.strip() for d in durations_str.split(",") if d.strip() != ""]
        
        if len(start_times) != len(durations):
            if verbose:
                logging.info(f"Row {index}: Mismatch between number of start times and durations; skipping.")
            continue
        
        # Build engine run intervals and corresponding post-run intervals.
        # Each engine run interval is (engine_run_start, engine_run_start + duration)
        # Each post-run interval is (engine_run_end, engine_run_end + 5 minutes)
        engine_run_intervals = []
        post_run_intervals = []
        for s, d in zip(start_times, durations):
            start_sec = hhmmss_to_seconds(s)
            try:
                duration_sec = int(float(d))
            except ValueError:
                duration_sec = 0
            if start_sec is None:
                continue
            engine_run_intervals.append((start_sec, start_sec + duration_sec))
            post_run_intervals.append((start_sec + duration_sec, start_sec + duration_sec + 300))  # 300s = 5 minutes
        
        if verbose:
            logging.info(f"Row {index}: Active sensor: {active_sensor}")
            logging.info(f"Engine run intervals (sec since midnight): {engine_run_intervals}")
            logging.info(f"Post-run intervals (sec since midnight): {post_run_intervals}")
        
        # Read the IGC file from base_folder; if not found, try alt_folder.
        igc_file_path = os.path.join(base_folder, igc_filename)
        if not os.path.exists(igc_file_path):
            igc_file_path = os.path.join(alt_folder, igc_filename)
        if not os.path.exists(igc_file_path):
            if verbose:
                logging.info(f"Row {index}: IGC file '{igc_filename}' not found in either folder.")
            continue
        
        try:
            text = Path(igc_file_path).read_text()
        except Exception as e:
            if verbose:
                logging.info(f"Row {index}: Error reading IGC file '{igc_filename}': {e}")
            continue
        
        # Use flight date from the CSV column "Date (MM/DD/YYYY)".
        # Parse the date and reformat it to MM/DD/YYYY as needed.
        if "Date (MM/DD/YYYY)" in df.columns:
            flight_date_raw = row["Date (MM/DD/YYYY)"]
            try:
                # Parse the date; if it's in YYYY-MM-DD, this will convert it.
                flight_date = pd.to_datetime(flight_date_raw)
                flight_date_str = flight_date.strftime("%m/%d/%Y")
            except Exception as e:
                logging.info(f"Row {index}: Invalid date format: {flight_date_raw}. Skipping flight.")
                continue
        else:
            flight_date_str = pd.Timestamp.today().strftime("%m/%d/%Y")
        
        # Convert IGC text to a DataFrame using igc2df (assumed to be available).
        try:
            df_flight = igc2df(text, flight_date_str)
        except Exception as e:
            if verbose:
                logging.info(f"Row {index}: Error converting IGC file '{igc_filename}' to DataFrame: {e}")
            continue
        
        if df_flight is None or 'timestamp' not in df_flight.columns:
            if verbose:
                logging.info(f"Row {index}: IGC to DataFrame conversion failed; skipping flight.")
            continue
        
        # The flight DataFrame is assumed to have a 'timestamp' column (pd.Timestamp) and 'latitude', 'longitude'
        # Create lists to hold acceleration metrics for each engine run.
        engine_run_acc_mean_list = []
        engine_run_acc_std_list = []
        post_run_acc_mean_list = []
        post_run_acc_std_list = []
        
        # For each engine run, calculate acceleration during the engine run and during the 5-minute post-run period.
        for (run_start_sec, run_end_sec), (post_start_sec, post_end_sec) in zip(engine_run_intervals, post_run_intervals):
            # Create timestamps for the intervals.
            # Assuming that the flight data timestamps are for the same day,
            # we convert seconds since midnight into a Timestamp.
            day = df_flight['timestamp'].iloc[0].normalize()  # midnight of flight day
            run_start_dt = day + pd.Timedelta(seconds=run_start_sec)
            run_end_dt = day + pd.Timedelta(seconds=run_end_sec)
            post_start_dt = day + pd.Timedelta(seconds=post_start_sec)
            post_end_dt = day + pd.Timedelta(seconds=post_end_sec)
            
            run_mean_acc, run_std_acc = calc_speed_rate_change_interval(df_flight, run_start_dt, run_end_dt, verbose)
            post_mean_acc, post_std_acc = calc_speed_rate_change_interval(df_flight, post_start_dt, post_end_dt, verbose)
            engine_run_acc_mean_list.append(run_mean_acc)
            engine_run_acc_std_list.append(run_std_acc)
            post_run_acc_mean_list.append(post_mean_acc)
            post_run_acc_std_list.append(post_std_acc)
        
        # Store the results as comma-separated strings.
        df.at[index, 'acceleration_engine_run_mean'] = ",".join(f"{val:.2f}" if pd.notna(val) else "nan" for val in engine_run_acc_mean_list)
        df.at[index, 'acceleration_engine_run_std'] = ",".join(f"{val:.2f}" if pd.notna(val) else "nan" for val in engine_run_acc_std_list)

        df.at[index, 'acceleration_post_run_mean'] = ",".join(f"{val:.2f}" if pd.notna(val) else "nan" for val in post_run_acc_mean_list)

        df.at[index, 'acceleration_post_run_std'] = ",".join(f"{val:.2f}" if pd.notna(val) else "nan" for val in post_run_acc_std_list)

    
    # Write out the updated CSV (tab-separated)
    df.to_csv(output_csv, sep="\t", index=False)
    if verbose:
        logging.info(f"Updated CSV file saved to '{output_csv}'.")

# --- Main Execution ---

if __name__ == "__main__":
    # Example command-line usage:
    # python process_acceleration.py /path/to/flights_final_updated.csv base_folder alt_folder /path/to/output_csv [--verbose]
    if len(sys.argv) < 5:
        print("Usage: python process_acceleration.py /path/to/flights_final_updated.csv base_folder alt_folder /path/to/output_csv [--verbose]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    base_folder = sys.argv[2]
    alt_folder = sys.argv[3]
    output_csv = sys.argv[4]
    verbose = "--verbose" in sys.argv

    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    process_csv_for_acceleration(csv_file, base_folder, alt_folder, output_csv, verbose)

