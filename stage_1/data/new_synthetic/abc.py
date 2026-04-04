import pandas as pd

df = pd.read_csv("train_data_2000.csv")

# Add probability (same as anomaly_tag for now)
df["probability"] = df["anomaly_tag"]

df.to_csv("train_data_fixed.csv", index=False)