import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import threading
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import extra_features as ef

# -------------------- Streamlit + Core ML Functions --------------------

def validate_columns(dataframe: pd.DataFrame) -> tuple[bool, list[str]]:
    required_columns = [
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "vibration_g",
    ]
    missing = [col for col in required_columns if col not in dataframe.columns]
    return len(missing) == 0, missing


def engineer_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
    dataframe = dataframe.copy()
    if not np.issubdtype(dataframe["timestamp"].dtype, np.datetime64):
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")

    minute_of_day = dataframe["timestamp"].dt.hour * 60 + dataframe["timestamp"].dt.minute

    features = pd.DataFrame({
        "temperature_c": dataframe["temperature_c"].values,
        "humidity_pct": dataframe["humidity_pct"].values,
        "pressure_hpa": dataframe["pressure_hpa"].values,
        "vibration_g": dataframe["vibration_g"].values,
        "minute_of_day": minute_of_day.values,
    })

    # Add extra engineered features from extra_features.py
    features = ef.add_time_features(dataframe, features)
    features = ef.add_rolling_features(dataframe, features, window=5)
    features = ef.add_interaction_features(features)
    features = ef.add_statistical_features(dataframe, features)

    labels = dataframe["label"].values if "label" in dataframe.columns else None
    return features, labels


def fit_and_predict(features: pd.DataFrame, labels: np.ndarray | None, contamination: float | None) -> tuple[np.ndarray, np.ndarray, IsolationForest, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    if contamination is None:
        if labels is not None and np.isfinite(labels).all():
            contam_value = max(1e-3, float(np.mean(labels)))
        else:
            contam_value = 0.05
    else:
        contam_value = contamination

    model = IsolationForest(
        n_estimators=250,
        max_samples="auto",
        contamination=contam_value,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    pred_raw = model.predict(X)  # 1 normal, -1 anomaly
    pred = np.where(pred_raw == -1, 1, 0)
    return pred, X, model, scaler


def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels=["Normal", "Anomaly"],
        cmap="Blues",
        colorbar=True,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_pca_scatter(X_scaled: np.ndarray, predictions: np.ndarray) -> plt.Figure:
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], s=10, c=predictions, cmap="coolwarm")
    ax.set_title("Device behavior (2D view) — 0=Normal, 1=Anomaly")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Prediction")
    fig.tight_layout()
    return fig

# -------------------- Flask Dashboard --------------------
from flask import Flask, render_template_string, request
import io
import base64

flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET", "POST"])
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
        <title>All-in-One Dashboard</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="container mt-4">
        <h1 class="mb-4">🚀 All-in-One Dashboard</h1>
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

def start_flask_dashboard():
    threading.Thread(target=lambda: flask_app.run(debug=False, use_reloader=False)).start()

# -------------------- Streamlit Main --------------------
def main() -> None:
    st.set_page_config(page_title="IoT Anomaly Detector", layout="centered")
    st.title("IoT Anomaly Detector")
    st.write("This tool helps you spot unusual device readings. Upload your CSV or use the sample data, then press Analyze.")

    with st.expander("What does this do?", expanded=True):
        st.markdown(
            "- Uses an unsupervised model (Isolation Forest) to find unusual patterns.\n"
            "- If your data has a `label` column (0=normal, 1=anomaly), you'll also see accuracy metrics.\n"
            "- Visual charts help you understand the results at a glance.\n"
            "- Extra features: Email alerts, anomaly logging, auto-retrain, live dashboard."
        )

    start_flask_dashboard()  # Launch Flask dashboard in background

    data_source = st.radio("Choose data:", ("Use sample dataset", "Upload my CSV"), index=0)

    uploaded_df: pd.DataFrame | None = None
    if data_source == "Upload my CSV":
        st.info("Required columns: `timestamp`, `temperature_c`, `humidity_pct`, `pressure_hpa`, `vibration_g`. Optional: `label`.")
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file is not None:
            try:
                uploaded_df = pd.read_csv(file)
            except Exception as exc:
                st.error(f"Could not read the CSV file: {exc}")
    else:
        try:
            uploaded_df = pd.read_csv("iot_anomaly_dataset.csv", parse_dates=["timestamp"])
        except Exception as exc:
            st.error("Sample file `iot_anomaly_dataset.csv` not found. Please upload a CSV.")
            uploaded_df = None

    if uploaded_df is None:
        st.stop()

    # Validation
    is_valid, missing = validate_columns(uploaded_df)
    if not is_valid:
        st.error("Your data is missing required columns: " + ", ".join(missing))
        st.stop()

    # Features and labels
    features, labels = engineer_features(uploaded_df)
    default_contam = float(np.mean(labels)) if labels is not None else 0.05
    default_contam = min(max(default_contam, 0.001), 0.2)

    with st.sidebar:
        st.header("Settings")
        contam_mode = st.radio(
            "How many anomalies do you expect?",
            ("Auto (estimate from labels if available)", "Manual (choose percentage)"),
            index=0,
        )
        contam = None
        if contam_mode == "Manual (choose percentage)":
            contam = st.slider(
                "Estimated anomaly rate (%)",
                min_value=0.1,
                max_value=20.0,
                value=float(round(default_contam * 100.0, 2)),
                step=0.1,
            ) / 100.0

    if st.button("Analyze"):
        pred, X_scaled, model, scaler = fit_and_predict(features, labels, contamination=contam)

        # Log anomalies & send email
        for i, val in enumerate(pred):
            if val == 1:  # anomaly detected
                ef.log_anomaly_to_csv(features.iloc[i].to_dict(), anomaly_type="IsolationForest")
                ef.send_email_alert("IoT Anomaly Detected", f"Anomaly detected:\n{features.iloc[i].to_dict()}", "admin@yourcompany.com")

        # Auto retrain if new_data.csv exists
        ef.retrain_model_if_new_data(model_path="model.pkl", new_data_path="new_data.csv")

        # Headline numbers
        num_points = len(pred)
        num_anomalies = int(pred.sum())
        st.subheader("Results")
        st.metric("Total readings analyzed", f"{num_points:,}")
        st.metric("Anomalies detected", f"{num_anomalies:,}")

        # Metrics when labels exist
        if labels is not None:
            st.markdown("### Accuracy (if labels provided)")
            report = classification_report(labels, pred, digits=3, output_dict=False)
            st.text(report)
            with open("model_report.txt", "w") as f:
                f.write(report)
            cm_fig = plot_confusion_matrix(labels, pred)
            st.pyplot(cm_fig)
            st.caption("Confusion matrix: Top-left = correct normal; bottom-right = correct anomalies.")

        # PCA visualization
        st.markdown("### Visualize device behavior")
        pca_fig = plot_pca_scatter(X_scaled, pred)
        st.pyplot(pca_fig)
        st.caption("Each dot is a reading. Color shows the model's prediction (blue=normal, red=anomaly).")

        # Preview table
        st.markdown("### Preview results table")
        preview = uploaded_df.copy()
        preview["prediction"] = pred
        st.dataframe(preview.head(50), use_container_width=True)

        # Download predictions
        with st.expander("Download predictions"):
            csv_bytes = preview.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="anomaly_predictions.csv", mime="text/csv")

if __name__ == "__main__":
    main()
=======
	classification_report,
	confusion_matrix,
	ConfusionMatrixDisplay,
)


def validate_columns(dataframe: pd.DataFrame) -> tuple[bool, list[str]]:
	required_columns = [
		"timestamp",
		"temperature_c",
		"humidity_pct",
		"pressure_hpa",
		"vibration_g",
	]
	missing = [col for col in required_columns if col not in dataframe.columns]
	return len(missing) == 0, missing


def engineer_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
	dataframe = dataframe.copy()
	if not np.issubdtype(dataframe["timestamp"].dtype, np.datetime64):
		dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")

	minute_of_day = dataframe["timestamp"].dt.hour * 60 + dataframe["timestamp"].dt.minute

	features = pd.DataFrame(
		{
			"temperature_c": dataframe["temperature_c"].values,
			"humidity_pct": dataframe["humidity_pct"].values,
			"pressure_hpa": dataframe["pressure_hpa"].values,
			"vibration_g": dataframe["vibration_g"].values,
			"minute_of_day": minute_of_day.values,
		}
	)

	labels = dataframe["label"].values if "label" in dataframe.columns else None
	return features, labels


def fit_and_predict(features: pd.DataFrame, labels: np.ndarray | None, contamination: float | None) -> tuple[np.ndarray, np.ndarray, IsolationForest, StandardScaler]:
	scaler = StandardScaler()
	X = scaler.fit_transform(features)

	if contamination is None:
		if labels is not None and np.isfinite(labels).all():
			contam_value = max(1e-3, float(np.mean(labels)))
		else:
			contam_value = 0.05
	else:
		contam_value = contamination

	model = IsolationForest(
		n_estimators=250,
		max_samples="auto",
		contamination=contam_value,
		random_state=42,
		n_jobs=-1,
	)
	model.fit(X)
	pred_raw = model.predict(X)  # 1 normal, -1 anomaly
	pred = np.where(pred_raw == -1, 1, 0)
	return pred, X, model, scaler


def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(5, 4))
	ConfusionMatrixDisplay.from_predictions(
		labels,
		predictions,
		display_labels=["Normal", "Anomaly"],
		cmap="Blues",
		colorbar=True,
		ax=ax,
	)
	ax.set_title("Confusion Matrix")
	fig.tight_layout()
	return fig


def plot_pca_scatter(X_scaled: np.ndarray, predictions: np.ndarray) -> plt.Figure:
	pca = PCA(n_components=2, random_state=42)
	X_2d = pca.fit_transform(X_scaled)
	fig, ax = plt.subplots(figsize=(6, 5))
	scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], s=10, c=predictions, cmap="coolwarm")
	ax.set_title("Device behavior (2D view) — 0=Normal, 1=Anomaly")
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	fig.colorbar(scatter, ax=ax, label="Prediction")
	fig.tight_layout()
	return fig


def main() -> None:
	st.set_page_config(page_title="IoT Anomaly Detector", layout="centered")
	st.title("IoT Anomaly Detector")
	st.write(
		"This tool helps you spot unusual device readings. Upload your CSV or use the sample data, then press Analyze."
	)

	with st.expander("What does this do?", expanded=True):
		st.markdown(
			"- Uses an unsupervised model (Isolation Forest) to find unusual patterns.\n"
			"- If your data has a `label` column (0=normal, 1=anomaly), you'll also see accuracy metrics.\n"
			"- Visual charts help you understand the results at a glance."
		)

	data_source = st.radio(
		"Choose data:",
		("Use sample dataset", "Upload my CSV"),
		index=0,
	)

	uploaded_df: pd.DataFrame | None = None
	if data_source == "Upload my CSV":
		st.info(
			"Required columns: `timestamp`, `temperature_c`, `humidity_pct`, `pressure_hpa`, `vibration_g`. Optional: `label`."
		)
		file = st.file_uploader("Upload CSV", type=["csv"])
		if file is not None:
			try:
				uploaded_df = pd.read_csv(file)
			except Exception as exc:
				st.error(f"Could not read the CSV file: {exc}")
	else:
		try:
			uploaded_df = pd.read_csv("iot_anomaly_dataset.csv", parse_dates=["timestamp"])  # from repo
		except Exception as exc:
			st.error(
				"Sample file `iot_anomaly_dataset.csv` not found or unreadable. Please upload a CSV."
			)
			uploaded_df = None

	if uploaded_df is None:
		st.stop()

	# Basic validation
	is_valid, missing = validate_columns(uploaded_df)
	if not is_valid:
		st.error("Your data is missing required columns: " + ", ".join(missing))
		st.stop()

	# Prepare features and optional labels
	features, labels = engineer_features(uploaded_df)
	default_contam = float(np.mean(labels)) if labels is not None else 0.05
	default_contam = min(max(default_contam, 0.001), 0.2)

	with st.sidebar:
		st.header("Settings")
		contam_mode = st.radio(
			"How many anomalies do you expect?",
			(
				"Auto (estimate from labels if available)",
				"Manual (choose percentage)",
			),
			index=0,
		)
		contam = None
		if contam_mode == "Manual (choose percentage)":
			contam = st.slider(
				"Estimated anomaly rate (percentage)",
				min_value=0.1,
				max_value=20.0,
				value=float(round(default_contam * 100.0, 2)),
				step=0.1,
			) / 100.0

	if st.button("Analyze"):
		pred, X_scaled, model, scaler = fit_and_predict(features, labels, contamination=contam)

		# Headline numbers
		num_points = len(pred)
		num_anomalies = int(pred.sum())
		st.subheader("Results")
		st.metric("Total readings analyzed", f"{num_points:,}")
		st.metric("Anomalies detected", f"{num_anomalies:,}")

		# Simple explanation for non-experts
		st.info(
			"An anomaly is a reading that looks very different from the usual behavior. "
			"This helps you spot potential issues quickly."
		)

		# Metrics when labels are available
		if labels is not None:
			st.markdown("### Accuracy (if labels provided)")
			report = classification_report(labels, pred, digits=3, output_dict=False)
			st.text(report)
			cm_fig = plot_confusion_matrix(labels, pred)
			st.pyplot(cm_fig)
			st.caption(
				"Confusion matrix: Top-left = correctly predicted normal; bottom-right = correctly predicted anomalies."
			)

		# PCA visualization
		st.markdown("### Visualize device behavior")
		pca_fig = plot_pca_scatter(X_scaled, pred)
		st.pyplot(pca_fig)
		st.caption(
			"Each dot is a reading. Color shows the model's prediction (blue=normal, red=anomaly)."
		)

		# Show a preview of predictions alongside original rows
		st.markdown("### Preview results table")
		preview = uploaded_df.copy()
		preview["prediction"] = pred
		st.dataframe(preview.head(50), use_container_width=True)

		with st.expander("Download predictions"):
			csv_bytes = preview.to_csv(index=False).encode("utf-8")
			st.download_button(
				"Download CSV",
				data=csv_bytes,
				file_name="anomaly_predictions.csv",
				mime="text/csv",
			)


if __name__ == "__main__":
	main()


>>>>>>> 7617e482fb9a5dd350981748818cca35bb8d08ac
