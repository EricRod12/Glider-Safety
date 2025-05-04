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
from typing import Dict, List, Tuple, Optional, Any
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
from ee_helpers import ensure_ee_initialized, LANDCOVER_DATASET
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_i_record(line: str) -> Dict[str, Tuple[int, int, str]]:
    """
    Parses the I-record to determine the positions and codes of extended fields in B-records.

    Parameters:
        line (str): A line from the IGC file starting with 'I'.

    Returns:
        Dict[str, Tuple[int, int, str]]: A dictionary where each key is the field code (e.g., 'FXA'),
                                         and the value is a tuple of (start_byte, end_byte, code).
    """
    extended_fields = {}
    
    # Remove the starting 'I' and any leading/trailing whitespace
    content = line[1:].strip()
    
    if len(content) < 2:
        logging.error(f"I-record too short to contain number of extensions: {line}")
        return extended_fields
    
    # Extract number of extensions (NN)
    num_extensions_str = content[:2]
    try:
        num_extensions = int(num_extensions_str)
    except ValueError:
        logging.error(f"Invalid number of extensions '{num_extensions_str}' in I-record: {line}")
        return extended_fields
    
    # Each extension occupies 7 characters: SS (2), FF (2), CCC (3)
    expected_length = 2 + num_extensions * 7
    if len(content) < expected_length:
        logging.error(f"I-record length mismatch. Expected at least {expected_length} characters, got {len(content)}: {line}")
        return extended_fields
    
    for i in range(num_extensions):
        start_pos = 2 + i * 7
        extension_str = content[start_pos:start_pos + 7]
        if len(extension_str) != 7:
            logging.warning(f"Incomplete extension data at extension {i+1}: '{extension_str}' in line: {line}")
            continue
        
        ss_str = extension_str[:2]
        ff_str = extension_str[2:4]
        ccc_str = extension_str[4:7]
        
        try:
            ss = int(ss_str)
            ff = int(ff_str)
        except ValueError:
            logging.warning(f"Invalid start/finish bytes '{ss_str}-{ff_str}' for extension '{ccc_str}' in line: {line}")
            continue
        
        extended_fields[ccc_str] = (ss, ff, ccc_str)
    
    return extended_fields

def _parse_b_record(line: str, extended_fields: Dict[str, Tuple[int, int, str]]) -> Optional[Dict[str, Any]]:
    """
    Parses a single 'B' record from an IGC file to extract flight data, including extended fields.
    Raises a ValueError if any required extended field has an invalid value.
    """
    if not line.startswith('B'):
        raise ValueError(f"Line does not start with 'B': {line}")
    
    if len(line) < 35:
        raise ValueError(f"Line too short for standard B-record: {line}")
    
    # Standard fields
    time_str = line[1:7]  # 'HHMMSS'
    try:
        time_obj = datetime.strptime(time_str, "%H%M%S").time()
    except ValueError:
        raise ValueError(f"Invalid time format '{time_str}' in line: {line}")
    
    lat_deg = int(line[7:9])
    lat_min = float(line[9:14])
    lat_dir = line[14]
    lon_deg = int(line[15:18])
    lon_min = float(line[18:23])
    lon_dir = line[23]
    fix = line[24]
    altitude_press_str = line[25:30]
    altitude_gnss_str = line[30:35]
    
    # Calculate latitude and longitude
    latitude = lat_deg + (lat_min / 60000)
    if lat_dir.upper() == 'S':
        latitude = -latitude
    longitude = lon_deg + (lon_min / 60000)
    if lon_dir.upper() == 'W':
        longitude = -longitude

    if fix not in ['A', 'V']:
        raise ValueError(f"Invalid fix validity '{fix}' in line: {line}")
    
    try:
        altitude_press = int(altitude_press_str)
    except ValueError:
        altitude_press = np.nan

    try:
        altitude_gnss = int(altitude_gnss_str)
    except ValueError:
        altitude_gnss = np.nan
    
    parsed_data = {
        "time": time_obj,
        "latitude": latitude,
        "longitude": longitude,
        "fix": fix,
        "altitude_press": altitude_press,
        "altitude_gnss_m": altitude_gnss
    }
    
    # Process extended fields.
    for field_code, (start_byte, end_byte, code) in extended_fields.items():
        start_idx = start_byte - 1
        end_idx = end_byte
        if len(line) < end_idx:
            raise ValueError(f"Line too short for extended field '{code}': {line}")
        field_value_str = line[start_idx:end_idx].strip()
        # If the field is empty, consider it invalid.
        if field_value_str == '':
            raise ValueError(f"Invalid value '' for field '{code}' in line: {line}")
        try:
            field_value = int(field_value_str)
        except ValueError:
            raise ValueError(f"Invalid value '{field_value_str}' for field '{code}' in line: {line}")
        parsed_data[code] = field_value
        
    return parsed_data

def igc2df(text_content, flight_date_str):
    data = {
        'timestamp': [],
        'latitude': [],
        'longitude': [],
        'altitude_press': [],
        'altitude_gnss_m': [],
        'altitude_gnss_ft': [],
    }
    
    try:
        flight_date = datetime.strptime(flight_date_str, "%m/%d/%Y").date()
    except ValueError:
        logging.error(f"Invalid date format: {flight_date_str}. Skipping flight.")
        return None

    extended_fields = {}  # To be set by parsing an I-record.
    for line in text_content.splitlines():
        if line.startswith('I'):
            extended_fields = parse_i_record(line)
            continue
        elif line.startswith('B'):
            try:
                parsed_record = _parse_b_record(line, extended_fields)
            except ValueError as e:
                logging.error(f"Skipping flight due to invalid B record: {e}")
                return None  # Skip entire flight if any B record is invalid.
            
            # Process the parsed record.
            timestamp = datetime.combine(flight_date, parsed_record["time"])
            parsed_record["timestamp"] = timestamp
            del parsed_record["time"]

            altitude_gnss_m = parsed_record.get("altitude_gnss_m", np.nan)
            parsed_record["altitude_gnss_m"] = altitude_gnss_m
            altitude_gnss_ft = altitude_gnss_m * 3.28084 if not pd.isna(altitude_gnss_m) else np.nan
            parsed_record["altitude_gnss_ft"] = altitude_gnss_ft

            data['timestamp'].append(parsed_record.get("timestamp", np.nan))
            data['latitude'].append(parsed_record.get("latitude", np.nan))
            data['longitude'].append(parsed_record.get("longitude", np.nan))
            data['altitude_press'].append(parsed_record.get("altitude_press", np.nan))
            data['altitude_gnss_m'].append(altitude_gnss_m)
            data['altitude_gnss_ft'].append(altitude_gnss_ft)
            
    df = pd.DataFrame(data)
    return df

