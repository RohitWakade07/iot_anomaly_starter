import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv
import os
from flask import Flask, render_template_string
import threading
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# ---------------- Feature 1: Email Alert ----------------
def send_email_alert(subject, message, to_email):
    """
    Send email alert for anomalies.
    Note: Configure your email credentials in environment variables:
    - EMAIL_FROM: your email address
    - EMAIL_PASSWORD: your app password
    """
    try:
        import os
        from_email = os.getenv("EMAIL_FROM", "your_email@gmail.com")
        password = os.getenv("EMAIL_PASSWORD", "your_app_password")
        
        # Skip if using default credentials
        if from_email == "your_email@gmail.com" or password == "your_app_password":
            print(f"[INFO] Email alert skipped - configure EMAIL_FROM and EMAIL_PASSWORD environment variables")
            return

        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()

        print(f"[ALERT] Email sent to {to_email}")
    except Exception as e:
        print(f"[ERROR] Could not send email: {e}")

# ---------------- Feature 2: Save Anomalies to CSV ----------------
def log_anomaly_to_csv(data, anomaly_type="Unknown"):
    file_exists = os.path.isfile("anomalies.csv")
    with open("anomalies.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Sensor_Value", "Anomaly_Type"])
        writer.writerow([pd.Timestamp.now(), data, anomaly_type])
    print("[INFO] Anomaly logged in anomalies.csv")

# ---------------- Feature 3: Flask Dashboard ----------------
app = Flask(__name__)

@app.route("/")
def dashboard():
    if os.path.isfile("anomalies.csv"):
        df = pd.read_csv("anomalies.csv")
        return render_template_string("""
        <h2>IoT Anomaly Dashboard</h2>
        {{ table|safe }}
        """, table=df.to_html(classes='table table-striped', index=False))
    else:
        return "<h2>No anomalies logged yet.</h2>"

def start_dashboard():
    threading.Thread(target=lambda: app.run(debug=False, use_reloader=False)).start()

# ---------------- Feature 4: Visualization ----------------
def plot_anomalies(X, y_pred):
    plt.scatter(range(len(X)), X, c=y_pred, cmap="coolwarm")
    plt.xlabel("Sample")
    plt.ylabel("Sensor Value")
    plt.title("Anomaly Detection Plot")
    plt.show()

# ---------------- Feature 5: Export Model Report ----------------
def save_model_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    with open("model_report.txt", "w") as f:
        f.write(report)
    print("[INFO] Model performance report saved to model_report.txt")

# ---------------- Feature 6: Auto Retrain ----------------
def retrain_model_if_new_data(model_path="model.pkl", new_data_path="new_data.csv"):
    if os.path.isfile(new_data_path):
        df = pd.read_csv(new_data_path)
        if "X" in df.columns and "y" in df.columns:
            X_new, y_new = df[["X"]], df["y"]
            model = joblib.load(model_path)
            model.fit(X_new, y_new)
            joblib.dump(model, "model_updated.pkl")
            print("[INFO] Model retrained and saved as model_updated.pkl")
        else:
            print("[WARNING] new_data.csv missing required columns (X, y)")

# ================= FEATURE ENGINEERING FUNCTIONS ================

def add_time_features(df, features):
    """Add time-based features: hour of day and weekday."""
    features["hour"] = df["timestamp"].dt.hour
    features["weekday"] = df["timestamp"].dt.weekday
    return features

def add_rolling_features(df, features, window=5):
    """Add rolling mean features with NaN handling."""
    features["temperature_roll_mean"] = df["temperature_c"].rolling(window, min_periods=1).mean().bfill()
    features["humidity_roll_mean"] = df["humidity_pct"].rolling(window, min_periods=1).mean().bfill()
    features["pressure_roll_mean"] = df["pressure_hpa"].rolling(window, min_periods=1).mean().bfill()
    features["vibration_roll_mean"] = df["vibration_g"].rolling(window, min_periods=1).mean().bfill()
    return features

def add_interaction_features(features):
    """Add interaction terms between key features."""
    features["temp_humidity_interaction"] = features["temperature_c"] * features["humidity_pct"]
    features["pressure_vibration_interaction"] = features["pressure_hpa"] * features["vibration_g"]
    return features

def add_statistical_features(df, features):
    """Add rolling standard deviation with NaN handling."""
    features["pressure_roll_std"] = df["pressure_hpa"].rolling(5, min_periods=1).std().fillna(0)
    features["vibration_roll_std"] = df["vibration_g"].rolling(5, min_periods=1).std().fillna(0)
    return features
