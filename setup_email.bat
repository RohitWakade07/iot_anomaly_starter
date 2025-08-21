@echo off
echo Email Alert Setup for IoT Anomaly Detection
echo ===========================================
echo.
echo This will help you configure email alerts.
echo.
echo Requirements:
echo 1. Gmail account
echo 2. 2-factor authentication enabled
echo 3. App password generated
echo.
echo To generate an app password:
echo 1. Go to https://myaccount.google.com/security
echo 2. Enable 2-factor authentication
echo 3. Generate an app password for "Mail"
echo.
echo.

set /p EMAIL_FROM="Enter your Gmail address: "
set /p EMAIL_PASSWORD="Enter your app password: "

echo.
echo Setting environment variables...
setx EMAIL_FROM "%EMAIL_FROM%"
setx EMAIL_PASSWORD "%EMAIL_PASSWORD%"

echo.
echo ✅ Email configuration saved!
echo Note: You may need to restart your terminal for changes to take effect.
echo.
pause
