@echo off
echo Starting IoT Anomaly Detection Application...
echo.
echo Activating virtual environment...
call v\Scripts\Activate.bat
echo.
echo Starting Streamlit app...
echo The app will open in your browser at http://localhost:8501
echo.
streamlit run app.py
pause
