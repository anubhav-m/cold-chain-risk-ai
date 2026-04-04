import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_FILE = "data.csv"   # change this
OUTPUT_FILE = "processed_data.csv"

# If your dataset has target_temperature column → use it
USE_TARGET_COLUMN = True

# If NOT, use fixed target
FIXED_TARGET = -50


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

print("Columns:", df.columns.tolist())

# =========================
# RENAME (if needed)
# =========================
df = df.rename(columns={
    "measurement_time": "timestamp",
    "temperature": "temperature"
})

# =========================
# CREATE DEVIATION
# =========================
if USE_TARGET_COLUMN and "target_temperature" in df.columns:
    df["value"] = df["temperature"] - df["target_temperature"]
else:
    df["value"] = df["temperature"] - FIXED_TARGET

# =========================
# SAVE NEW FILE
# =========================
df_out = df[["timestamp", "value"]]

df_out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved processed data to {OUTPUT_FILE}")