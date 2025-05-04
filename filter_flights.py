import os
import sys
import glob
from itertools import islice

def filter_flights(input_dir, output_dir, verbose=False):
    """
    Filters flights based on engine noise thresholds.
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
                    values[sensor] = int(line[idx-1:idx+2].strip())
            except ValueError:
                values[sensor] = 0
        return values

    def detect_engine_noise(values, flags):
        """
        Detect engine noise based on thresholds for each sensor.
        """
        thresholds_start = {"ENL": 600, "MOP": 500, "RPM": 50}
        thresholds_stop = {"ENL": 250, "MOP": 50, "RPM": 20}
        noise_detected = False

        for sensor, value in values.items():
            if value > thresholds_start[sensor] and not flags[sensor]:
                flags[sensor] = True
                noise_detected = True
            elif value < thresholds_stop[sensor] and flags[sensor]:
                flags[sensor] = False

        return noise_detected

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
                indices = {"ENL": 0, "MOP": 0, "RPM": 0}
                flags = {"ENL": False, "MOP": False, "RPM": False}
                engine_detected = False

                for line in f:
                    line = line.strip()

                    # Parse the I line for sensor indices
                    if line.startswith("I"):
                        indices = parse_indices(line)

                    # Parse the B line for engine data
                    elif line.startswith("B"):
                        values = parse_engval(line, indices)
                        if detect_engine_noise(values, flags):
                            engine_detected = True
                            break

                # Copy file if engine noise detected
                if engine_detected:
                    output_path = os.path.join(output_dir, os.path.basename(file_path))
                    with open(output_path, 'w') as out_file:
                        f.seek(0)  # Reset file pointer
                        out_file.write(f.read())
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
