import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("stage2_percentage_2000.csv")

X = df.drop(columns=["timestamp", "spoilage_percent"])
y = df["spoilage_percent"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

print("Model trained ✅")


sample = [[
    5,   # temp_dev_max
    30,   # temp_dev_avg
    2,   # temp_dev_last
    60,   # humidity_avg
    2,    # door_open_count
    30,   # time_above
    3.5,  # power_avg
    20    # cumulative_damage
]]

print(sample)
print(model.predict(sample))