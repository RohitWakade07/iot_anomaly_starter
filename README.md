# IoT Anomaly Detection System

A comprehensive anomaly detection system for IoT sensor data using machine learning and Streamlit.

## Features

- **Anomaly Detection**: Uses Isolation Forest algorithm to detect unusual patterns in sensor data
- **Interactive UI**: Streamlit-based web interface for easy data upload and analysis
- **Feature Engineering**: Advanced feature engineering including time-based and rolling features
- **Visualization**: PCA scatter plots and confusion matrix visualization
- **Email Alerts**: Automatic email notifications for detected anomalies
- **Dashboard**: Separate Flask dashboard for viewing anomaly logs and reports
- **Auto-retraining**: Automatic model retraining with new data

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Windows)

**Option 1: Using PowerShell (Recommended)**
```powershell
.\run_app.ps1
```

**Option 2: Using Batch File**
```cmd
run_app.bat
```

### Manual Start

**Running the Streamlit App**
```bash
# Activate virtual environment first
.\v\Scripts\Activate.ps1  # Windows PowerShell
# or
.\v\Scripts\Activate.bat  # Windows Command Prompt

# Then run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Running the Flask Dashboard**
```bash
# Activate virtual environment first
.\v\Scripts\Activate.ps1

# Then run the dashboard
python dashboard.py
```

The dashboard will be available at `http://localhost:5000`

## Data Format

Your CSV file should contain the following columns:

**Required:**
- `timestamp`: Timestamp of the sensor reading
- `temperature_c`: Temperature in Celsius
- `humidity_pct`: Humidity percentage
- `pressure_hpa`: Pressure in hectopascals
- `vibration_g`: Vibration in g-force

**Optional:**
- `label`: Ground truth labels (0=normal, 1=anomaly) for evaluation

## How to Use

1. **Choose Data Source**: Either use the sample dataset or upload your own CSV file
2. **Configure Settings**: Adjust the expected anomaly rate in the sidebar
3. **Analyze**: Click the "Analyze" button to run anomaly detection
4. **View Results**: 
   - See summary metrics (total readings, anomalies detected)
   - View accuracy metrics if labels are provided
   - Explore visualizations (PCA scatter plot, confusion matrix)
   - Download results as CSV

## Features Explained

### Anomaly Detection
The system uses Isolation Forest, an unsupervised learning algorithm that identifies anomalies by measuring how easily a data point can be isolated from the rest of the data.

### Feature Engineering
- **Time Features**: Hour of day, weekday, minute of day
- **Rolling Features**: Rolling mean and standard deviation of sensor values
- **Interaction Features**: Multiplicative interactions between sensors
- **Statistical Features**: Rolling statistics for trend analysis

### Email Alerts
Configure email alerts by setting environment variables:

**Windows PowerShell:**
```powershell
$env:EMAIL_FROM = "your_email@gmail.com"
$env:EMAIL_PASSWORD = "your_app_password"
```

**Windows Command Prompt:**
```cmd
set EMAIL_FROM=your_email@gmail.com
set EMAIL_PASSWORD=your_app_password
```

**Linux/Mac:**
```bash
export EMAIL_FROM="your_email@gmail.com"
export EMAIL_PASSWORD="your_app_password"
```

### Auto-retraining
Place new data in `new_data.csv` with columns `X` and `y` to automatically retrain the model.

## File Structure

```
├── app.py                 # Main Streamlit application
├── dashboard.py           # Flask dashboard for viewing logs
├── extra_features.py      # Feature engineering and utility functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── iot_anomaly_dataset.csv # Sample dataset
├── anomalies.csv         # Generated anomaly logs
├── model_report.txt      # Generated model performance report
└── model.pkl            # Saved model (generated after training)
```

## Testing

To verify that everything is working correctly, run the test script:

```bash
# Activate virtual environment first
.\v\Scripts\Activate.ps1

# Run the test
python test_app.py
```

This will test all components of the application and confirm that:
- Sample data can be loaded
- Column validation works
- Feature engineering functions properly
- Model training and prediction work
- Extra features (logging, email alerts) function correctly

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Make sure all packages are installed with `pip install -r requirements.txt`
2. **Virtual Environment**: Always activate the virtual environment before running the app
3. **Data Format**: Ensure your CSV has the required columns
4. **Email Alerts**: Configure environment variables for email functionality
5. **Port Conflicts**: If ports 8501 or 5000 are in use, the apps will automatically find available ports

### Error Messages

- **"Sample file not found"**: Upload your own CSV or ensure `iot_anomaly_dataset.csv` is present
- **"Missing required columns"**: Check that your CSV has all required sensor columns
- **"Could not read CSV"**: Verify your file is a valid CSV format

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.

