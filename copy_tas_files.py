#!/usr/bin/env python3
import os
import shutil
import re

def file_has_tas(file_path):
    """
    Checks whether an IGC file (file_path) has an I-record that contains a TAS field.
    Returns True if found, False otherwise.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # I-records start with "I". You might want to check only a certain number of header lines.
                if line.startswith("I"):
                    # Look for a token like "TAS" followed by digits; e.g., "TAS4751"
                    if re.search(r"TAS\d{2,}", line):
                        return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return False

def main():
    # Set the base folder paths.
    current_dir = os.getcwd()
    filtered_folder = os.path.join(current_dir, "filtered")
    tas_folder = os.path.join(current_dir, "TAS")

    # Create the TAS folder if it does not exist.
    os.makedirs(tas_folder, exist_ok=True)

    # Loop over all IGC files in the filtered folder.
    # Adjust the extension if your IGC files use a different one (e.g., .igc).
    for file_name in os.listdir(filtered_folder):
        if file_name.lower().endswith(".igc"):
            file_path = os.path.join(filtered_folder, file_name)
            if file_has_tas(file_path):
                destination = os.path.join(tas_folder, file_name)
                shutil.copy(file_path, destination)
                print(f"Copied {file_name} to TAS folder.")
            else:
                print(f"{file_name} does not have TAS in the I-record.")

if __name__ == "__main__":
    main()
