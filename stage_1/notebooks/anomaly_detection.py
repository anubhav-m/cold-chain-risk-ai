import numpy as np
import pandas as pd
import torch
from anomalybert.inference.detector import AnomalyDetector

# =========================
# CONFIG
# =========================
MODEL_PATH = "../models/model.pt"
DATA_PATH = "processed_data.csv"   # your dataset
TOP_N = 10

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Ensure correct column names
# Expected: timestamp, temperature
df = df.rename(columns={
    "measurement_time": "timestamp",
    "temperature": "value"
})

# Convert timestamp if needed (optional)
# df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**9

timestamps = df["timestamp"].values
values = df["value"].values.astype(np.float64)

print(f"Loaded {len(df)} samples")

# =========================
# LOAD MODEL
# =========================
detector = AnomalyDetector(MODEL_PATH)

# =========================
# RUN DETECTION (TOP-N)
# =========================
results = detector.detect(
    timestamps=timestamps,
    values=values,
    top_n=TOP_N
)

print("\nTop anomalies:")
for i, r in enumerate(results, 1):
    print(f"{i}. time={r['timestamp']} value={r['value']:.2f} score={r['score']:.4f}")

# =========================
# GET FULL ANOMALY SCORES
# =========================
normalizer = detector.normalizer
tokenizer = detector.tokenizer

# Normalize
norm_values = (
    normalizer.transform(values)
    if normalizer else values.astype(np.float32)
)

# Create sequences (sliding window)
idx = np.arange(len(values), dtype=np.int64)
windows = tokenizer.tokenize(idx, norm_values)

# Run model on each window
window_scores = []
with torch.no_grad():
    for w in windows:
        input_tensor = torch.tensor(w["values"], dtype=torch.float32).unsqueeze(0)
        scores = detector.model(input_tensor)
        window_scores.append(scores.squeeze(0).numpy())

# Aggregate scores back to timeline
all_scores = tokenizer.aggregate_scores_simple(window_scores, total_len=len(values))

df["anomaly_score"] = all_scores

# =========================
# SAVE OUTPUT
# =========================
df.to_csv("output_with_scores.csv", index=False)

print("\nSaved results to output_with_scores.csv")
