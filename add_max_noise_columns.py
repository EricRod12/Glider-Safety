import os
import sys
import glob
import pandas as pd

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
            print(f"Error parsing indices in line: {line}")
    return indices

def parse_engval(line, indices, verbose=False):
    """
    Extract the engine noise registration values for ENL, MOP, and RPM sensors from a 'B' record.
    """
    values = {"ENL": 0, "MOP": 0, "RPM": 0}
    for sensor, idx in indices.items():
        try:
            if idx > 0:
                values[sensor] = int(line[idx-1:idx+2].strip())
        except ValueError:
            if verbose:
                print(f"Error parsing engine value for sensor {sensor} in line: {line}")
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

def get_max_noise_per_engine_run(file_path, active_sensor, engine_run_intervals, verbose=False):
    """
    Processes a single IGC file and returns a list containing the maximum noise registration value 
    for the active sensor during each engine run interval.
    
    engine_run_intervals: list of tuples (start_sec, end_sec)
    """
    # Initialize maximum noise values for each engine run interval.
    max_noise_list = [0] * len(engine_run_intervals)
    indices = {"ENL": 0, "MOP": 0, "RPM": 0}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith("I"):
                    indices = parse_indices(line, verbose)
                elif line.startswith("B"):
                    # Get the record timestamp from the B record (characters 1-6: HHMMSS)
                    if len(line) < 7:
                        continue
                    record_time_str = line[1:7]
                    try:
                        record_time_sec = hhmmss_to_seconds(record_time_str)
                        if record_time_sec is None:
                            continue
                    except ValueError:
                        continue

                    # Check if the record time falls within any of the engine run intervals.
                    for i, (start, end) in enumerate(engine_run_intervals):
                        if start <= record_time_sec <= end:
                            values = parse_engval(line, indices, verbose)
                            if values[active_sensor] > max_noise_list[i]:
                                max_noise_list[i] = values[active_sensor]
        return max_noise_list
    except Exception as e:
        if verbose:
            print(f"Error processing file {file_path}: {e}")
        return None

def process_csv_file(csv_file, igc_folders, output_csv, verbose=False):
    """
    Reads the flights_final_with_estimates.csv file (tab-separated) and, for each row, determines which sensor's
    engine run start times to use. It computes engine run intervals from the start times and run durations,
    searches for the corresponding IGC file in the provided IGC folder list, calculates the maximum noise value
    for each engine run (for the active sensor), and then writes out an updated CSV file with three new columns.
    
    Expected columns in the CSV include:
      - File
      - ENL_Engine_Run_Start_Times
      - MOP_Engine_Run_Start_Times
      - RPM_Engine_Run_Start_Times
      - engine_run_times (s)
    
    For the active sensor, the corresponding new column (e.g., max_noise_ENL) will be filled with a comma-separated
    list of maximum noise values, one for each engine run.
    """
    # Read the CSV file (tab-separated)
    df = pd.read_csv(csv_file, sep="\t")
    
    # Add new columns for maximum noise values
    df['max_noise_ENL'] = ""
    df['max_noise_MOP'] = ""
    df['max_noise_RPM'] = ""

    # Check for required columns
    required_cols = ['File', 'ENL_Engine_Run_Start_Times', 'MOP_Engine_Run_Start_Times',
                     'RPM_Engine_Run_Start_Times', 'engine_run_times (s)']
    for col in required_cols:
        if col not in df.columns:
            print(f"Required column '{col}' not found in CSV.")
            sys.exit(1)
    
    # Process each flight
    for index, row in df.iterrows():
        igc_file_name = row['File']
        
        # Determine the active sensor based on which engine run start times column is non-empty.
        active_sensor = None
        for sensor in ['ENL', 'MOP', 'RPM']:
            col_name = f"{sensor}_Engine_Run_Start_Times"
            if pd.notna(row[col_name]) and str(row[col_name]).strip() != "":
                active_sensor = sensor
                break
        
        if active_sensor is None:
            if verbose:
                print(f"Row {index}: No engine run start times found; skipping noise calculation.")
            continue
        
        # Get the engine run start times and durations as lists.
        # Format is comma-separated values, e.g., "123456,124500"
        start_times_str = str(row[f"{active_sensor}_Engine_Run_Start_Times"]).strip()
        durations_str = str(row['engine_run_times (s)']).strip()
        if start_times_str == "" or durations_str == "":
            if verbose:
                print(f"Row {index}: Missing start times or durations; skipping.")
            continue

        start_times = [s.strip() for s in start_times_str.split(",") if s.strip() != ""]
        durations = [d.strip() for d in durations_str.split(",") if d.strip() != ""]
        
        if len(start_times) != len(durations):
            if verbose:
                print(f"Row {index}: Mismatch between number of start times and durations; skipping.")
            continue

        # Build a list of engine run intervals as tuples (start_seconds, end_seconds)
        engine_run_intervals = []
        for s, d in zip(start_times, durations):
            start_sec = hhmmss_to_seconds(s)
            try:
                duration_sec = int(float(d))
            except ValueError:
                duration_sec = 0
            if start_sec is not None:
                engine_run_intervals.append((start_sec, start_sec + duration_sec))
        
        if verbose:
            print(f"Row {index}: Active sensor: {active_sensor}, intervals: {engine_run_intervals}")
        
        # Search for the IGC file in the provided IGC folders.
        igc_file_path = None
        for folder in igc_folders:
            possible_path = os.path.join(folder, igc_file_name)
            if os.path.exists(possible_path):
                igc_file_path = possible_path
                break
        
        if igc_file_path is None:
            if verbose:
                print(f"Row {index}: IGC file '{igc_file_name}' not found in any provided folder.")
            continue
        
        # Process the IGC file to compute maximum noise values per engine run for the active sensor.
        max_noise_list = get_max_noise_per_engine_run(igc_file_path, active_sensor, engine_run_intervals, verbose)
        if max_noise_list is not None:
            # Join the per-run maximum values as a comma-separated string.
            max_noise_str = ",".join(str(val) for val in max_noise_list)
            # Set the value for the active sensor's column.
            df.at[index, f"max_noise_{active_sensor}"] = max_noise_str
            if verbose:
                print(f"Row {index}: {igc_file_name} - max noise for {active_sensor} per run: {max_noise_str}")
    
    # Write out the updated CSV (tab-separated)
    df.to_csv(output_csv, sep="\t", index=False)
    if verbose:
        print(f"Updated CSV file saved to '{output_csv}'.")

if __name__ == "__main__":
    # Command-line arguments:
    #   1. Path to flights_final_with_estimates.csv
    #   2. Comma-separated list of IGC folders (e.g., "filtered,filtered_wgc")
    #   3. Path for the output CSV file
    #   4. Optional --verbose flag
    if len(sys.argv) < 4:
        print("Usage: python add_max_noise_columns.py /path/to/flights_final_with_estimates.csv "
              "filtered,filtered_wgc /path/to/output_csv [--verbose]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    igc_folders_arg = sys.argv[2]
    output_csv = sys.argv[3]
    verbose = "--verbose" in sys.argv
    
    # Split the igc_folders_arg into a list (e.g., "filtered,filtered_wgc" -> ["filtered", "filtered_wgc"])
    igc_folders = [folder.strip() for folder in igc_folders_arg.split(",") if folder.strip() != ""]
    
    process_csv_file(csv_file, igc_folders, output_csv, verbose)
