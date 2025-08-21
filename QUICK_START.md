# Quick Start Guide

## 🚀 Running the Application

### Method 1: Using the Startup Scripts (Recommended)

**PowerShell (Recommended):**
```powershell
.\run_app.ps1
```

**Command Prompt:**
```cmd
run_app.bat
```

### Method 2: Manual Commands

**Step 1: Activate Virtual Environment**
```powershell
.\v\Scripts\Activate.ps1
```

**Step 2: Run Streamlit App**
```powershell
streamlit run app.py
```

**Step 3: Run Flask Dashboard (Optional - in new terminal)**
```powershell
.\v\Scripts\Activate.ps1
python dashboard.py
```

## 🌐 Access Points

- **Streamlit App**: http://localhost:8501
- **Flask Dashboard**: http://localhost:5000

## ⚠️ Common Issues & Solutions

### Issue 1: "Warning: to view this Streamlit app on a browser..."
**Problem**: You ran `python app.py` instead of `streamlit run app.py`
**Solution**: Always use `streamlit run app.py`

### Issue 2: Environment Variables Not Set
**Problem**: Email alerts not working
**Solution**: Set environment variables correctly:
```powershell
$env:EMAIL_FROM = "your_email@gmail.com"
$env:EMAIL_PASSWORD = "your_app_password"
```

### Issue 3: Port Already in Use
**Problem**: Port 8501 or 5000 is occupied
**Solution**: The apps will automatically find available ports

### Issue 4: Virtual Environment Not Activated
**Problem**: Import errors
**Solution**: Always activate the virtual environment first:
```powershell
.\v\Scripts\Activate.ps1
```

## 🧪 Testing

Run the test script to verify everything works:
```powershell
.\v\Scripts\Activate.ps1
python test_app.py
```

## 📁 File Structure

```
├── app.py                 # Main Streamlit application
├── dashboard.py           # Flask dashboard
├── extra_features.py      # Feature engineering functions
├── test_app.py           # Test script
├── run_app.ps1           # PowerShell startup script
├── run_app.bat           # Batch startup script
├── requirements.txt      # Dependencies
└── README.md            # Full documentation
```

## 🎯 What to Do Next

1. **Start the app**: Use one of the startup methods above
2. **Upload data**: Use the sample dataset or upload your own CSV
3. **Configure settings**: Adjust anomaly rate in the sidebar
4. **Analyze**: Click "Analyze" to run anomaly detection
5. **View results**: Explore visualizations and download predictions

## 📞 Need Help?

- Check the full README.md for detailed documentation
- Run `python test_app.py` to diagnose issues
- Ensure virtual environment is activated
- Use `streamlit run app.py` (not `python app.py`)
