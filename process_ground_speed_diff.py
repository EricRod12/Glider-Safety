import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from geopy.distance import geodesic
from parser import parse_i_record, _parse_b_record, igc2df

def hhmmss_to_seconds(time_str):
    try:
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        ss = int(time_str[4:6])
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return None

def calc_mean_ground_speed(df_flight, start_dt, end_dt, verbose=False):
    """
    Calculate the mean ground speed (in knots) in the specified time window using lat/lon.
    """
    mask = (df_flight['timestamp'] >= start_dt) & (df_flight['timestamp'] <= end_dt)
    df_window = df_flight.loc[mask].copy()
    if df_window.shape[0] < 2:
        if verbose:
            logging.info("Not enough points for mean ground speed calculation.")
        return np.nan
    df_window = df_window.sort_values('timestamp')
    speeds = []
    for i in range(len(df_window) - 1):
        t1 = df_window.iloc[i]['timestamp']
        t2 = df_window.iloc[i+1]['timestamp']
        dt_sec = (t2 - t1).total_seconds()
        if dt_sec <= 0:
            continue
        lat1 = df_window.iloc[i]['latitude']
        lon1 = df_window.iloc[i]['longitude']
        lat2 = df_window.iloc[i+1]['latitude']
        lon2 = df_window.iloc[i+1]['longitude']
        dist_m = geodesic((lat1, lon1), (lat2, lon2)).meters
        speed_knots = (dist_m / dt_sec) * 1.94384  # conversion from m/s to knots
        speeds.append(speed_knots)
    if not speeds:
        return np.nan
    return np.mean(speeds)

def calc_ground_speed_difference(df_flight, run_start_dt, run_end_dt, post_start_dt, post_end_dt, verbose=False):
    """
    Compute the difference: mean speed during post-run window minus mean speed during engine run.
    """
    mean_run = calc_mean_ground_speed(df_flight, run_start_dt, run_end_dt, verbose)
    mean_post = calc_mean_ground_speed(df_flight, post_start_dt, post_end_dt, verbose)
    return mean_post - mean_run

def process_csv_for_ground_speed_difference(csv_file, base_folder, alt_folder, output_csv, verbose=False):
    """
    Reads the CSV (assumed tab-separated) that contains at least:
      - File: the IGC file name,
      - A date column "Date (MM/DD/YYYY)" for flight date,
      - One of the engine run start time columns: ENL_Engine_Run_Start_Times, MOP_Engine_Run_Start_Times, or RPM_Engine_Run_Start_Times,
      - engine_run_times (s): comma-separated engine run durations in seconds.
      
    For each flight, this function:
      1. Determines the active sensor (i.e. which start time column is populated).
      2. Parses the engine run start times and durations to create intervals.
      3. Reads the corresponding IGC file from base_folder (or alt_folder if not found).
      4. Converts the IGC file text to a DataFrame (using igc2df, which must be defined elsewhere).
      5. For each engine run, calculates:
            - Mean ground speed during the engine run window.
            - Mean ground speed during the 5-minute period immediately following the engine run.
         Then computes the difference: (post-run mean speed) - (engine run mean speed).
      6. Stores the comma-separated list of differences in a new column 'ground_speed_diff'.
    """
    df = pd.read_csv(csv_file, sep="\t")
    
    # Add a new column for the ground speed difference (if multiple engine runs, store as comma-separated list)
    df['ground_speed_diff'] = ""
    
    if 'File' not in df.columns:
        print("CSV file must contain a 'File' column with IGC file names.")
        sys.exit(1)
    
    for index, row in df.iterrows():
        igc_filename = row['File']
        
        # Determine active sensor based on which engine run start times column is populated.
        active_sensor = None
        for sensor in ['ENL', 'MOP', 'RPM']:
            col_name = f"{sensor}_Engine_Run_Start_Times"
            if pd.notna(row.get(col_name, "")) and str(row.get(col_name, "")).strip() != "":
                active_sensor = sensor
                break
        if active_sensor is None:
            if verbose:
                logging.info(f"Row {index}: No engine run start times found; skipping ground speed difference calculation.")
            continue
        
        # Get engine run start times and durations (comma-separated strings).
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
                logging.info(f"Row {index}: Mismatch between start times and durations; skipping.")
            continue
        
        # Build engine run and post-run intervals.
        engine_run_intervals = []
        post_run_intervals = []
        for s, d in zip(start_times, durations):
            run_start_sec = hhmmss_to_seconds(s)
            try:
                duration_sec = int(float(d))
            except ValueError:
                duration_sec = 0
            if run_start_sec is None:
                continue
            engine_run_intervals.append((run_start_sec, run_start_sec + duration_sec))
            post_run_intervals.append((run_start_sec + duration_sec, run_start_sec + duration_sec + 300))  # 5 minutes = 300 sec
        
        # Get flight date from CSV column "Date (MM/DD/YYYY)".
        if "Date (MM/DD/YYYY)" in df.columns:
            flight_date_raw = row["Date (MM/DD/YYYY)"]
            try:
                flight_date = pd.to_datetime(flight_date_raw)
                flight_day = flight_date.normalize()  # midnight of flight day
            except Exception as e:
                if verbose:
                    logging.info(f"Row {index}: Invalid date format: {flight_date_raw}. Skipping flight.")
                continue
        else:
            flight_day = pd.Timestamp.today().normalize()
        
        # Locate the IGC file in base_folder or alt_folder.
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
        
        # Convert IGC text to a DataFrame (assuming igc2df is defined).
        try:
            df_flight = igc2df(text, flight_date.strftime("%m/%d/%Y"))
        except Exception as e:
            if verbose:
                logging.info(f"Row {index}: Error converting IGC file '{igc_filename}' to DataFrame: {e}")
            continue
        
        if df_flight is None or 'timestamp' not in df_flight.columns:
            if verbose:
                logging.info(f"Row {index}: IGC to DataFrame conversion failed; skipping flight.")
            continue
        
        # Compute ground speed difference for each engine run.
        speed_diff_list = []
        for (run_start_sec, run_end_sec), (post_start_sec, post_end_sec) in zip(engine_run_intervals, post_run_intervals):
            run_start_dt = flight_day + pd.Timedelta(seconds=run_start_sec)
            run_end_dt = flight_day + pd.Timedelta(seconds=run_end_sec)
            post_start_dt = flight_day + pd.Timedelta(seconds=post_start_sec)
            post_end_dt = flight_day + pd.Timedelta(seconds=post_end_sec)
            
            diff = calc_ground_speed_difference(df_flight, run_start_dt, run_end_dt, post_start_dt, post_end_dt, verbose)
            speed_diff_list.append(diff)
        
        # Store differences as a comma-separated string, rounding to 2 decimals.
        df.at[index, 'ground_speed_diff'] = ",".join(f"{val:.2f}" if not pd.isna(val) else "nan" for val in speed_diff_list)
    
    df.to_csv(output_csv, sep="\t", index=False)
    if verbose:
        logging.info(f"Updated CSV file saved to '{output_csv}'.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python process_ground_speed_diff.py /path/to/flights_final_updated.csv base_folder alt_folder /path/to/output_csv [--verbose]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    base_folder = sys.argv[2]
    alt_folder = sys.argv[3]
    output_csv = sys.argv[4]
    verbose = "--verbose" in sys.argv
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    process_csv_for_ground_speed_difference(csv_file, base_folder, alt_folder, output_csv, verbose)
