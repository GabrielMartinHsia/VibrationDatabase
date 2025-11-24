@echo off
REM Change to project folder
cd /d "C:\Users\Gabriel Martin-Hsia\OneDrive - Fervo Energy\Vibration"

REM Activate the venv
call ".venv\Scripts\activate.bat"

REM Run the Streamlit app, listening on all interfaces
streamlit run Scripts\fft_dashboard.py --server.address 0.0.0.0 --server.port 8501
