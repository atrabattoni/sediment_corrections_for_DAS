# First, install dependencies (optional)
echo "Installing dependencies..."
pip install -r requirements.txt

# Run Python scripts
echo "Running Python scripts..."
echo "Running 0_ttlut.py..."
python 0_ttlut.py  # This takes a while...
echo "Running 1_map.py..."
python 1_map.py
echo "Running 2_signal_and_picks.py..."
python 2_signal_and_picks.py
echo "Running 4_Pp_to_Ps_delay.py..."
python 4_Pp_to_Ps_delay.py
echo "Running 5_station_correction.py..."
python 5_station_correction.py
echo "Running 6&7_sediment_correction.py..."
python '6&7_sediment_correction.py'
echo "Running 8_localization.py..."
python 8_localization.py
echo "Running 8_precompute_locs.py..."
python 8_precompute_locs.py
echo "Running 9_catalog.py..."
python 9_catalog.py