import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv("sensor_multilabel_dataset.csv")
print("Dataset loaded.")

X = df[[
    "acc_x", "acc_y", "acc_z",
    "tilt_angle",
    "temperature", "humidity",
    "fire", "vibration",
    "water_float", "ultrasonic_distance"
]]


y = df[[
    "landslide",
    "high_temp",
    "vibration_alert",
    "fire_alert",
    "flood_alert"
]]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y["landslide"]  # one label for balancing
)


base_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

print("\nTraining Completed.\n")



y_pred = model.predict(X_test)

print("\nCLASSIFICATION REPORT \n")
print(classification_report(y_test, y_pred, target_names=y.columns))

joblib.dump(model, "disaster_multilabel_model.pkl")
print("\nModel saved as disaster_multilabel_model.pkl")


