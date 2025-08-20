# IoT Anomaly Detection System

A comprehensive IoT anomaly detection system with Streamlit web interface, Flask dashboard, and advanced feature engineering.

## Features

- **Anomaly Detection**: Uses Isolation Forest algorithm for unsupervised anomaly detection
- **Web Interface**: Streamlit app for easy data upload and analysis
- **Dashboard**: Flask-based dashboard for monitoring anomalies
- **Feature Engineering**: Advanced time-based and statistical features
- **Email Alerts**: Configurable email notifications for anomalies
- **Auto-retraining**: Automatic model retraining with new data
- **Visualization**: PCA plots and confusion matrix visualization

## Installation

1. **Activate virtual environment**:
   ```bash
   .\v\Scripts\Activate.ps1  # Windows PowerShell
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the Streamlit App
```bash
streamlit run app.py
```

### 2. Run the Starter Script
```bash
python iot_anomaly_detection_starter.py
```

### 3. Access Flask Dashboard
The Flask dashboard runs automatically in the background when you start the Streamlit app.

## Data Format

Your CSV file should contain these required columns:
- `timestamp`: Timestamp of the reading
- `temperature_c`: Temperature in Celsius
- `humidity_pct`: Humidity percentage
- `pressure_hpa`: Pressure in hectopascals
- `vibration_g`: Vibration in g-force

Optional column:
- `label`: Ground truth labels (0=normal, 1=anomaly)

## Configuration

### Email Alerts
To enable email alerts, set these environment variables:
```bash
set EMAIL_FROM=your_email@gmail.com
set EMAIL_PASSWORD=your_app_password
```

## Files

- `app.py`: Main Streamlit application
- `extra_features.py`: Feature engineering and utility functions
- `iot_anomaly_detection_starter.py`: Standalone anomaly detection script
- `iot_anomaly_dataset.csv`: Sample dataset
- `requirements.txt`: Python dependencies

## Model Performance

The system achieves:
- **Accuracy**: ~95.4%
- **Precision**: 97.55% (normal), 61.67% (anomaly)
- **Recall**: 97.55% (normal), 61.67% (anomaly)

## Troubleshooting

1. **Import errors**: Make sure virtual environment is activated
2. **Email not sending**: Configure environment variables for email credentials
3. **Missing data**: Ensure CSV has all required columns
4. **Plot not showing**: The starter script opens a matplotlib window - close it to continue

