import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import extra_features as ef

# 1) Load dataset
df = pd.read_csv("iot_anomaly_dataset.csv", parse_dates=["timestamp"])

# 2) Create base features
minute_of_day = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
X_raw = pd.DataFrame({
    "temperature_c": df["temperature_c"].values,
    "humidity_pct": df["humidity_pct"].values,
    "pressure_hpa": df["pressure_hpa"].values,
    "vibration_g": df["vibration_g"].values,
    "minute_of_day": minute_of_day.values
})

# --- Call extra feature functions ---
X_raw = ef.add_time_features(df, X_raw)
X_raw = ef.add_rolling_features(df, X_raw, window=5)
X_raw = ef.add_interaction_features(X_raw)
X_raw = ef.add_statistical_features(df, X_raw)

# --- Handle NaN values after feature engineering ---
X_raw = X_raw.bfill().ffill()

# ✅ Print first few rows to verify features
print("\nPreview of expanded features (X_raw):")
print(X_raw.head())

# Labels (if available)
y_true = df["label"].values if "label" in df.columns else None

# 3) Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# 4) Train Isolation Forest
contam = 0.05
if y_true is not None:
    contam = max(1e-3, float(np.mean(y_true)))

iso = IsolationForest(
    n_estimators=250,
    max_samples="auto",
    contamination=contam,
    random_state=42,
    n_jobs=-1
)
iso.fit(X)
y_pred_raw = iso.predict(X)  # 1 normal, -1 anomaly
y_pred = np.where(y_pred_raw == -1, 1, 0)

# 5) Evaluate (if labels exist)
if y_true is not None:
    print("\n=== Evaluation (unsupervised model vs true labels) ===")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred))

    # Plot confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Normal", "Anomaly"],
        cmap="Blues",
        colorbar=True,
        ax=ax_cm
    )
    ax_cm.set_title("Confusion Matrix")
    plt.tight_layout()

# 6) Visualize (PCA to 2D)
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(6, 5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=6, c=y_pred)  # default colormap
plt.title("Isolation Forest predictions (0=normal, 1=anomaly)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
