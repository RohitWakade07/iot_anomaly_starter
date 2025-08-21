import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import base64
from flask import Flask, render_template_string, request
import threading

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def dashboard():
    # Load anomalies
    anomalies_html = "<p>No anomalies found yet.</p>"
    if os.path.exists("anomalies.csv"):
        df = pd.read_csv("anomalies.csv")
        anomalies_html = df.to_html(classes="table table-striped", index=False)

    # Load model report
    report_html = "<p>No model report available yet.</p>"
    if os.path.exists("model_report.txt"):
        with open("model_report.txt", "r") as f:
            report_html = "<pre>" + f.read() + "</pre>"

    # Visualization
    plot_html = "<p>No plot available.</p>"
    if os.path.exists("anomalies.csv"):
        df = pd.read_csv("anomalies.csv")
        if not df.empty:
            plt.figure(figsize=(6,4))
            plt.scatter(df.index, df[df.columns[1]], c="red", label="Anomalies")
            plt.title("Anomaly Visualization")
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode("utf-8")
            plot_html = f'<img src="data:image/png;base64,{plot_data}" width="500">'
            plt.close()

    # Handle new data upload
    detection_result = ""
    if request.method == "POST":
        file = request.files.get("datafile")
        if file:
            new_df = pd.read_csv(file)
            detection_result = "<h4>Uploaded Data:</h4>" + new_df.to_html(classes="table table-bordered", index=False)

    html = f"""
    <html>
    <head>
        <title>IoT Anomaly Dashboard</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="container mt-4">
        <h1 class="mb-4">🚀 IoT Anomaly Dashboard</h1>
        <h2>Anomaly Logs</h2>
        {anomalies_html}
        <h2>Model Report</h2>
        {report_html}
        <h2>Visualization</h2>
        {plot_html}
        <h2>Upload New Data for Detection</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="datafile" class="form-control mb-2" required>
            <button type="submit" class="btn btn-primary">Run Detection</button>
        </form>
        {detection_result}
    </body>
    </html>
    """
    return render_template_string(html)

def start_dashboard():
    """Start the Flask dashboard in a separate thread"""
    threading.Thread(target=lambda: app.run(debug=False, use_reloader=False, port=5000)).start()

if __name__ == "__main__":
    start_dashboard()
