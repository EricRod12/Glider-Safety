import os
import shutil
import argparse
import logging
import re
from datetime import datetime, timedelta
from geopy.distance import distance  # Ensure geopy is installed
import math as ma

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger()

# Constants for unit conversions
S2H = 3600          # Seconds to Hours
M2F = 3.28084       # Meters to Feet
K2M = 0.621371      # Kilometers to Miles

# Thresholds for Engine Run Detection
THRESHOLDS = {
    'ENL': {'start': 600, 'stop': 250},
    'MOP': {'start': 500, 'stop': 50},
    'RPM': {'start': 50, 'stop': 20},
}

FMT = '%H%M%S'  # Time format in IGC B-records

def safe_int_conversion(value, default=0):
    """
    Safely converts a string to an integer by extracting digits only.
    If conversion fails, returns the default value.
    """
    try:
        numeric_str = ''.join(re.findall(r'\d+', value))
        if numeric_str:
            return int(numeric_str)
        else:
            raise ValueError(f"No digits found in '{value}'")
    except ValueError as e:
        logger.warning(f"{e}. Using default={default}")
        return default

def parse_i_record(line):
    """
    Parses an I-record line to extract sensor tags and their engine value positions.
    Only considers 'MOP', 'ENL', and 'RPM' sensors.

    Parameters:
        line (str): The I-record line.

    Returns:
        dict: A dictionary mapping sensor codes to their (SS, FF) positions.
    """
    sensors = {}
    content = line[1:]  # Remove the leading 'I'

    if len(content) < 2:
        logger.warning("I-record has insufficient length for sensor count.")
        return sensors

    # Extract sensor count (cnt) from the first two characters after 'I'
    cnt_str = content[:2]

    if not cnt_str.isdigit():
        logger.warning(f"Invalid sensor count '{cnt_str}' in I-record.")
        return sensors

    cnt = int(cnt_str)
    j = 2  # Starting index for sensor definitions

    for i in range(cnt):
        if j + 7 > len(content):
            logger.warning(f"I-record line is too short to extract sensor {i+1}.")
            break

        sensor_def = content[j:j+7]
        ss_ff = sensor_def[:4]
        tag = sensor_def[4:7]

        if tag not in THRESHOLDS:
            # Skip sensors not in the desired list
            j += 7
            continue

        if not ss_ff.isdigit():
            logger.warning(f"Non-digit SSFF '{ss_ff}' for sensor '{tag}'. Skipping.")
            j += 7
            continue

        SS = int(ss_ff[:2])
        FF = int(ss_ff[2:])

        sensors[tag] = (SS, FF)

        j += 7  # Move to the next sensor definition

    return sensors

def extract_engine_value(b_record, SS, FF):
    """
    Extracts the engine value from the B-record based on SS and FF positions.

    Parameters:
        b_record (str): The B-record line.
        SS (int): Start position (1-based).
        FF (int): End position (1-based).

    Returns:
        str: The extracted engine value or 'N/A' if not available.
    """
    # Ensure SS and FF are within the B-record length
    if SS < 1 or FF > len(b_record):
        logger.warning(f"SS ({SS}) or FF ({FF}) out of bounds for B-record length {len(b_record)}.")
        return 'N/A'

    # Extract characters from SS to FF (1-based indexing)
    engine_val = b_record[SS-1:FF]
    if not engine_val:
        return 'N/A'
    else:
        return engine_val

def fix_time_str(time_str):
    """
    Adjusts a 6-digit time string (HHMMSS) that may have out-of-bound values (e.g. minutes >= 60, hours >= 24)
    and returns a valid time string in HHMMSS format. This function “carries over” extra seconds and minutes,
    and then wraps the hours modulo 24.

    Parameters:
        time_str (str): The original time string (expected to be 6 digits).

    Returns:
        str: A fixed time string in HHMMSS format.
    """
    if len(time_str) != 6 or not time_str.isdigit():
        # If not exactly 6 digits, return the original string (or raise an error if preferred)
        return time_str

    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])

    # Adjust seconds: if seconds >= 60, convert excess seconds into minutes.
    if second >= 60:
        extra_min = second // 60
        second = second % 60
        minute += extra_min

    # Adjust minutes: if minutes >= 60, convert excess minutes into hours.
    if minute >= 60:
        extra_hour = minute // 60
        minute = minute % 60
        hour += extra_hour

    # Adjust hours: wrap the hours modulo 24.
    if hour >= 24:
        hour = hour % 24

    return f"{hour:02d}{minute:02d}{second:02d}"

def has_engine_run(file_path):
    """
    Processes the entire IGC file and returns True if at least one engine run event is detected
    (i.e. for at least one sensor, the value exceeds its start threshold and later falls below its stop threshold
    for a duration of at least 60 seconds).
    """
    try:
        sensors = {}
        active_sensors = {"ENL": False, "MOP": False, "RPM": False}
        run_start_times = {"ENL": None, "MOP": None, "RPM": None}
        i_record_found = False
        engine_event_found = False

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                record_type = line[0].upper()
                if record_type == 'I':
                    sensors = parse_i_record(line)
                    if sensors:
                        i_record_found = True
                    continue
                elif record_type == 'B':
                    if not i_record_found:
                        continue
                    raw_time = line[1:7]
                    fixed_time_str = fix_time_str(raw_time)
                    try:
                        atime = datetime.strptime(fixed_time_str, FMT)
                    except ValueError:
                        logger.warning(f"Line {line_number}: Invalid time '{fixed_time_str}'. Skipping line.")
                        continue

                    for sensor, (SS, FF) in sensors.items():
                        engine_val_str = extract_engine_value(line, SS, FF)
                        if engine_val_str == 'N/A':
                            continue
                        engine_val = safe_int_conversion(engine_val_str)
                        # Check for engine run start
                        if not active_sensors[sensor] and engine_val > THRESHOLDS[sensor]['start']:
                            active_sensors[sensor] = True
                            run_start_times[sensor] = atime
                        # Check for engine run stop
                        elif active_sensors[sensor] and engine_val < THRESHOLDS[sensor]['stop']:
                            duration = (atime - run_start_times[sensor]).total_seconds()
                            if duration >= 60:
                                engine_event_found = True
                            active_sensors[sensor] = False
                            run_start_times[sensor] = None
        return engine_event_found
    except Exception as e:
        logger.error(f"Error processing file {file_path} in has_engine_run: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Filter IGC files based on Engine Run events lasting at least one minute.")
    parser.add_argument('source', type=str, help='Source directory containing IGC files.')
    parser.add_argument('filtered', type=str, help='Destination directory for filtered IGC files.')
    args = parser.parse_args()

    source_dir = args.source
    filtered_dir = args.filtered

    # Verify source directory exists
    if not os.path.isdir(source_dir):
        logger.error(f"Source directory '{source_dir}' does not exist.")
        return

    # Create filtered directory if it doesn't exist
    if not os.path.exists(filtered_dir):
        try:
            os.makedirs(filtered_dir)
            logger.info(f"Created directory '{filtered_dir}'.")
        except Exception as e:
            logger.error(f"Failed to create directory '{filtered_dir}': {e}")
            return

    # Gather all .igc files in the source directory
    igc_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.igc')]
    if not igc_files:
        logger.warning(f"No .igc files found in '{source_dir}'.")
        return

    files_copied = 0
    for igc_file in igc_files:
        file_path = os.path.join(source_dir, igc_file)
        if has_engine_run(file_path):
            # Copy the file to the filtered directory
            try:
                shutil.copy(file_path, filtered_dir)
                files_copied += 1
                # Print a progress message every 200
                if files_copied % 200 == 0:
                    print(f"Copied {files_copied} files so far...")
            except Exception as e:
                logger.error(f"Failed to copy '{igc_file}': {e}")

    print(f"Done! Total files copied to '{filtered_dir}': {files_copied}")

if __name__ == "__main__":
    main()
