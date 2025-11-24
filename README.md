# VibrationDatabase

Custom machinery vibration analysis pipeline using enDAQ .IDE files, SQLite, and Python.  
Goal: make it easy to collect, store, and compare vibration spectra (FFTs) across locations and over time.

---

## Project Structure

```text
Vibration/
├── vibes.db                     # SQLite database (current working snapshot)
├── vibtool/                     # Core Python package
│   ├── __init__.py
│   ├── db.py                    # DB connection + config loader (db_path, data_root)
│   ├── schema.py                # SQL schema (pump, location, reading, fft_spectrum, band_level)
│   ├── endaq_helpers.py         # IDE → XYZ loader (endaq.ide.to_pandas)
│   ├── fft_helpers.py           # FFT computation + BLOB serialization + helpers
│
├── Scripts/                     # Utility + workflow scripts
│   ├── init_db.py               # Create empty DB from schema
│   ├── add_red_pumps.py         # Insert pump + measurement_location + orientation metadata
│   ├── ingest_red_p34a_data.py  # Ingest .IDE files → measurement_run (reading) rows
│   ├── store_fft_for_all_readings.py   # Compute FFTs for each reading (X/Y/Z)
│   ├── test_load_one_ide.py     # Smoke test: IDE → XYZ, sample rate, lengths
│   ├── test_fft_one_ide.py      # Smoke test: FFT for one reading
│   ├── inspect_one_fft.py       # Read FFT BLOBs back out, reconstruct freq axis
│   ├── plot_one_fft.py          # Plot FFT for a single reading (Plotly)
│   ├── plot_compare_locations.py# Compare multiple locations on a pump (same physical orientation)
│   ├── fft_dashboard.py         # Streamlit app for interactive FFT exploration
│   ├── start_dashboard.bat      # Convenience launcher for the Streamlit app
│   ├── migrate_runid_to_readingid.py   # DB migration (run_id → reading_id)
│   ├── migrate_band_level_only.py      # DB migration for band_level
│
└── Data/                        # Raw IDE files (ignored by git)
    └── Red/
        └── P34A/
            ├── <timestamp>_L01.IDE
            ├── <timestamp>_L02.IDE
            └── ...


Basic Workflow

Add new IDE files

Copy .IDE into Data/<Site>/<Pump>/ with names like YYYYMMDD_HHMMSS_L01.IDE.

Ingest readings

Ensure pumps/locations exist (e.g. via add_red_pumps.py).

Run:

python -m Scripts.ingest_red_p34a_data


Compute FFTs for new readings

Run:

python -m Scripts.store_fft_for_all_readings


Script finds readings without FFTs and computes spectra for X/Y/Z.

Explore spectra (local scripts)

Single reading:

python -m Scripts.plot_one_fft


Compare locations (same physical orientation):

python -m Scripts.plot_compare_locations


Interactive dashboard (Streamlit)

From repo root:

streamlit run Scripts/fft_dashboard.py


Or double-click start_dashboard.bat (on Windows).

Features:

Site / pump / location selection

Orientation-aware axes (horizontal / vertical / axial)

Min/max frequency controls (min_freq hides DC spike by default)

Overlaid FFTs for quick comparison