import os
import sys
import glob
import re
from itertools import islice

def filter_flights(input_dir, output_dir, verbose=False):
    """
    Filters IGC flights if we find MOP/RPM engine usage in any of:
      1) Crossing MOP or RPM thresholds in B lines (like in your existing code),
      2) or sensor info lines referencing MOP/RPM engine run/noise.

    Copies the file to output_dir if either condition is met.
    Otherwise, file is skipped.
    """

    def parse_indices(line):
        """
        Parse the 'I' line to extract indices for ENL, MOP, and RPM sensors.
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

    def parse_engval(line, indices):
        """
        Extract the engine values (engval) for ENL, MOP, and RPM sensors from the 'B' line.
        """
        values = {"ENL": 0, "MOP": 0, "RPM": 0}
        for sensor, idx in indices.items():
            try:
                if idx > 0:
                    substring = line[idx-1:idx+2].strip()
                    values[sensor] = int(substring)
            except ValueError:
                values[sensor] = 0
        return values

    def detect_engine_noise(values, flags):
        """
        Detect MOP or RPM noise based on thresholds for each sensor.
        We do not consider ENL in this new logic, or you can keep it if you like.
        """
        # If you want to ignore ENL, skip it. If not, set thresholds for ENL too.
        thresholds_start = {"MOP": 500, "RPM": 50}
        thresholds_stop = {"MOP": 50, "RPM": 20}

        noise_detected = False
        # We'll only check MOP/RPM
        for sensor in ("MOP", "RPM"):
            val = values[sensor]
            if val > thresholds_start[sensor] and not flags[sensor]:
                flags[sensor] = True
                noise_detected = True
            elif val < thresholds_stop[sensor] and flags[sensor]:
                flags[sensor] = False

        return noise_detected

    # Regex to detect references in text lines
    sensor_info_pattern = re.compile(
        r"(?:"
        r"MOP\s+monitor\s+reports\s+Engine\s+Run|"
        r"RPM\s+monitor\s+reports\s+Engine\s+Run|"
        r"Motor\s+noise\s+registered\s+by\s+MOP\s+sensor|"
        r"Motor\s+noise\s+registered\s+by\s+RPM\s+sensor"
        r")",
        re.IGNORECASE
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect IGC files
    igc_files = glob.glob(os.path.join(input_dir, "*.igc")) + glob.glob(os.path.join(input_dir, "*.IGC"))
    if not igc_files:
        print("No .IGC files found in the input directory.")
        return

    copied_files = 0
    print(f"Processing {len(igc_files)} file(s)...")

    for idx, file_path in enumerate(igc_files, start=1):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_text = f.read()

            # Condition A: Does text mention MOP or RPM engine run/noise lines?
            sensor_info_found = bool(sensor_info_pattern.search(file_text))

            # Condition B: MOP/RPM threshold crossing in B lines
            lines = file_text.splitlines()
            indices = {"ENL": 0, "MOP": 0, "RPM": 0}
            flags = {"ENL": False, "MOP": False, "RPM": False}
            engine_detected = False

            # 1) parse I lines
            for line in lines:
                line = line.strip()
                if line.startswith("I"):
                    new_idx = parse_indices(line)
                    for s in ("ENL", "MOP", "RPM"):
                        if new_idx[s] > 0:
                            indices[s] = new_idx[s]

            # 2) parse B lines
            for line in lines:
                line = line.strip()
                if line.startswith("B"):
                    values = parse_engval(line, indices)
                    if detect_engine_noise(values, flags):
                        engine_detected = True
                        break

            # We only copy if either sensor_info_found OR engine_detected is True
            if sensor_info_found or engine_detected:
                output_path = os.path.join(output_dir, os.path.basename(file_path))
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    out_file.write(file_text)
                copied_files += 1
                if verbose:
                    print(f"[{idx}/{len(igc_files)}] Copied: {os.path.basename(file_path)}")
            else:
                if verbose:
                    print(f"[{idx}/{len(igc_files)}] Skipped: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"\nFiltering complete. {copied_files} out of {len(igc_files)} file(s) copied to '{output_dir}'.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python filter_flights.py /path/to/input_directory /path/to/output_directory [--verbose]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    verbose = "--verbose" in sys.argv

    filter_flights(input_dir, output_dir, verbose)
