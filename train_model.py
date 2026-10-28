"""Train regression and classification models for EV charging demand.
Run: python train_model.py
The included CSV is simulated for educational demonstration, not real station forecasting.
"""
from pathlib import Path
import json, math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "ev_charging_sessions.csv", parse_dates=["timestamp"])
features = ["hour", "day_of_week", "month", "is_weekend", "session_duration_hours", "energy_delivered_kwh"]
split = int(len(df) * 0.8)  # time-ordered split to reduce future-to-past leakage
X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
y_reg = df["charging_sessions"]
y_cls = df["demand_level"]

reg = RandomForestRegressor(n_estimators=180, max_depth=12, random_state=42, n_jobs=-1)
reg.fit(X_train, y_reg.iloc[:split])
reg_pred = reg.predict(X_test)
clf = RandomForestClassifier(n_estimators=180, max_depth=12, random_state=42,
                             class_weight="balanced", n_jobs=-1)
clf.fit(X_train, y_cls.iloc[:split])
cls_pred = clf.predict(X_test)

metrics = {
    "rows": int(len(df)), "train_rows": int(split), "test_rows": int(len(df)-split),
    "regression": {"MAE": round(float(mean_absolute_error(y_reg.iloc[split:], reg_pred)), 3),
                   "RMSE": round(float(math.sqrt(mean_squared_error(y_reg.iloc[split:], reg_pred))), 3),
                   "R2": round(float(r2_score(y_reg.iloc[split:], reg_pred)), 3)},
    "classification": {"accuracy": round(float(accuracy_score(y_cls.iloc[split:], cls_pred)), 3),
                       "weighted_F1": round(float(f1_score(y_cls.iloc[split:], cls_pred, average="weighted")), 3)},
    "features": features,
    "dataset_type": "simulated demonstration data"
}
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs" / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
out = df.iloc[split:][["timestamp", "hour", "day_of_week", "charging_sessions", "demand_level"]].copy()
out["predicted_sessions"] = reg_pred.round().clip(min=0).astype(int)
out["predicted_demand_level"] = cls_pred
out.to_csv(ROOT / "outputs" / "test_predictions.csv", index=False)
print(json.dumps(metrics, indent=2))
