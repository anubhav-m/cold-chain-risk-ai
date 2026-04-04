import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())

# =========================
# 2. FEATURES & TARGET
# =========================
X = df.drop(columns=["risk"])
y = df["risk"]

# =========================
# 3. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4. TRAIN MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=600,        # more trees → better learning
    max_depth=None,          # let trees grow fully
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",     # important!
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# 5. EVALUATE MODEL
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =========================
# 6. FEATURE IMPORTANCE
# =========================
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()

# =========================
# 7. TEST WITH CUSTOM INPUT
# =========================

# Example realistic input (moderate risk case)
sample = pd.DataFrame([{
    "time_above_threshold": 0,
    "temp_mean_c": 5,
    "temp_max_c": 6,
    "temp_min_c": 4,
    "temp_std_c": 1,
    "temp_recovery_rate": 0.7,
    "temp_excess": 0,
    "temp_deficit": 0,
    "violation_count": 0,
    "humidity_mean": 85,
    "door_open_count": 0,
    "fill_ratio": 0.6,
    "vibration_index": 0
}])

prediction = model.predict(sample)
probability = model.predict_proba(sample)

risk_percentage = probability[0][1] * 100

print("\nPrediction (0=Safe,1=Risk):", prediction[0])
print("Risk Percentage:", round(risk_percentage, 2), "%")

# =========================
# 8. SAVE MODEL
# =========================
joblib.dump(model, "rf_model_stratified.pkl")
print("\nModel saved as rf_model_stratified.pkl")

# =========================
# 9. LOAD MODEL (optional)
# =========================
# sample = pd.DataFrame([{
#     "time_above_threshold": 30,
#     "temp_mean_c": 12,
#     "temp_max_c": 9.2,
#     "temp_min_c": 2,
#     "temp_std_c": 2.1,
#     "temp_recovery_rate": 0.3,
#     "temp_excess": 1.2,
#     "temp_deficit": 0,
#     "violation_count": 4,
#     "humidity_mean": 85,
#     "door_open_count": 10,
#     "fill_ratio": 0.6,
#     "vibration_index": 3.2
# }])
# loaded_model = joblib.load("rf_model_stratified.pkl")
# prediction = loaded_model.predict(sample)
# probability = loaded_model.predict_proba(sample)

# risk_percentage = probability[0][1] * 100

# print("\nPrediction (0=Safe,1=Risk):", prediction[0])
# print("Risk Percentage:", round(risk_percentage, 2), "%")