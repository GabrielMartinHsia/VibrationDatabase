@echo off
REM Change to project folder
cd /d "%~dp0"

REM Activate the venv
call .venv\Scripts\activate

REM Run the Streamlit app, listening on all interfaces
streamlit run Scripts\fft_dashboard.py --server.port 8501
pause
