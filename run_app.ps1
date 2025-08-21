Write-Host "Starting IoT Anomaly Detection Application..." -ForegroundColor Green
Write-Host ""

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\v\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Checking email configuration..." -ForegroundColor Yellow
if (-not $env:EMAIL_FROM -or -not $env:EMAIL_PASSWORD) {
    Write-Host "⚠️  Email alerts not configured. Run 'python email_config.py' to set up email alerts." -ForegroundColor Yellow
} else {
    Write-Host "✅ Email alerts configured: $env:EMAIL_FROM" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting Streamlit app..." -ForegroundColor Yellow
Write-Host "The app will open in your browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

streamlit run app.py
