import numpy as np
import pandas as pd
import torch
import joblib
from anomalybert.inference.detector import AnomalyDetector

# =========================
# CONFIG
# =========================
MODEL_PATH = "../models/transformer_model.pt"
RF_MODEL_PATH = "../models/rf_model_stratified.pkl"
DATA_PATH = "test_data_3.csv"

THRESHOLD = 0.6
MEMORY_LIMIT = 3   # sustained anomaly threshold

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

df = df.rename(columns={
    "measurement_time": "timestamp",
    "temperature": "value"
})

timestamps = df["timestamp"].values
values = df["value"].values.astype(np.float64)

print(f"Loaded {len(df)} samples")

# =========================
# LOAD MODELS
# =========================
detector = AnomalyDetector(MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)

# =========================
# STAGE 1: ANOMALY SCORES
# =========================
normalizer = detector.normalizer
tokenizer = detector.tokenizer

norm_values = (
    normalizer.transform(values)
    if normalizer else values.astype(np.float32)
)

idx = np.arange(len(values), dtype=np.int64)
windows = tokenizer.tokenize(idx, norm_values)

window_scores = []
with torch.no_grad():
    for w in windows:
        input_tensor = torch.tensor(w["values"], dtype=torch.float32).unsqueeze(0)
        scores = detector.model(input_tensor)
        window_scores.append(scores.squeeze(0).numpy())

all_scores = tokenizer.aggregate_scores_simple(window_scores, total_len=len(values))

df["anomaly_score"] = all_scores

# =========================
# STAGE 2: MEMORY (TIME-AWARE)
# =========================
count = 0
memory_flags = []

for score in df["anomaly_score"]:
    if score > THRESHOLD:
        count += 1
    else:
        count = 0
    
    memory_flags.append(1 if count >= MEMORY_LIMIT else 0)

df["sustained_anomaly"] = memory_flags

# =========================
# STAGE 3: FEATURE ENGINEERING FOR RF
# =========================

# rolling window features
window_size = 10

df["temp_mean_c"] = df["value"].rolling(window_size).mean()
df["temp_max_c"] = df["value"].rolling(window_size).max()
df["temp_min_c"] = df["value"].rolling(window_size).min()
df["temp_std_c"] = df["value"].rolling(window_size).std()

# threshold-based features
df["temp_excess"] = np.maximum(df["value"] - 8, 0)
df["temp_deficit"] = np.maximum(2 - df["value"], 0)

df["violation"] = ((df["value"] < 2) | (df["value"] > 8)).astype(int)
df["violation_count"] = df["violation"].rolling(window_size).sum()

# time above threshold
df["time_above_threshold"] = df["violation"].rolling(window_size).sum()

# recovery rate (simplified)
df["temp_recovery_rate"] = df["value"].diff().fillna(0)

# static dummy features (since no sensor input)
df["humidity_mean"] = 85
df["door_open_count"] = 0
df["fill_ratio"] = 0.6
df["vibration_index"] = 0

# fill NaNs
df.fillna(0, inplace=True)

# =========================
# STAGE 4: RF PREDICTION
# =========================
features = [
    "time_above_threshold",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "temp_std_c",
    "temp_recovery_rate",
    "temp_excess",
    "temp_deficit",
    "violation_count",
    "humidity_mean",
    "door_open_count",
    "fill_ratio",
    "vibration_index"
]

rf_preds = rf_model.predict(df[features])
rf_probs = rf_model.predict_proba(df[features])[:, 1]

df["rf_risk"] = rf_preds
df["rf_probability"] = rf_probs

# =========================
# STAGE 5: FINAL RISK ENGINE
# =========================

def get_final_risk(row):
    score = row["anomaly_score"]
    sustained = row["sustained_anomaly"]
    rule_violation = int(row["value"] < 2 or row["value"] > 8)
    rf_prob = row["rf_probability"]

    final_score = score + sustained + rule_violation + rf_prob

    if final_score < 0.5:
        return "NO"
    elif final_score < 1.5:
        return "LOW"
    elif final_score < 2.5:
        return "MEDIUM"
    else:
        return "CRITICAL"

df["final_risk"] = df.apply(get_final_risk, axis=1)

# =========================
# SAVE OUTPUT
# =========================
df.to_csv("final_output.csv", index=False)

print("\nPipeline completed. Output saved to final_output.csv")

# =========================
# SUMMARY
# =========================
print("\nRisk Distribution:")
print(df["final_risk"].value_counts())