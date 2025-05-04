import ee
import logging

ee_initted = False
LANDCOVER_DATASET = None

def ensure_ee_initialized():
    """
    Initialize Earth Engine and global variables if not already initialized.
    """
    global ee_initted, LANDCOVER_DATASET
    if not ee_initted:
        try:
            ee.Initialize()
            ee_initted = True
            logging.info("Earth Engine initialized successfully in worker.")
            # Also initialize any globals you need:
            LANDCOVER_DATASET = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019')\
                                  .select('discrete_classification')
        except Exception as e:
            logging.warning(f"Error initializing Earth Engine in worker: {e}")