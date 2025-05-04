# Glider-Safety

This repo contains the code for the glider safety capstone project from UVA Master's in Data Science. The team consists of Eric Rodriguez, Hithesh Yedlapati, Mohammad Farooq, and Elena Tsvetkova. The sponsors are Jim Garrison and Richard Carlson. The project supervisors are professor Adam Tashman and professor William Basener.

The code that Richard provided contains the base features for our analysis. That code is found in glider-engine.py. We slightly modified that script to alter the format of the output, which is found in glider-engine_edited.py.

Other features for analysis were created in pilot_behavior_analysis.py. This is where features such as terrain altitude and terrain type were created.

The code that estimates true airspeed during engine runs is called pilot_behavior_EDA.ipynb. In that file, we used one flight as a proof of concept that our circling function, found in circling.py, works properly. Then we proceeded by separating our gliders into Self-Launch and Turbo Gliders.

The code that filters flights based on if they had an engine run during their flight is in filter_flights_engine_runs.py.
To run that code, we used the command python filter_flights_engine_runs.py capstone filtered. Capstone is the directory where all of our IGC files were found and filtered is the destination directory for the IGC files with engine runs. 

The environment we used for our libraries such as rasterio and gdal is consilidated in the yml file environment.yml.

Add max_noise_columns.py helped us to obtain the max noise registration levels for our ENL, MOP and RPM sensors during engine runs.



