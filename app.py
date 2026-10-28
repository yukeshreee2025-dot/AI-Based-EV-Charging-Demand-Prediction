"""Streamlit dashboard for EV charging demand prediction.
Run: streamlit run app.py
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import json
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="EV Charging Demand Predictor", layout="wide")
st.title("AI-Based EV Charging Demand Prediction")
st.caption("Educational capstone demo using simulated charging-session data. Forecasts are not live station availability.")

df = pd.read_csv(ROOT / "data" / "ev_charging_sessions.csv", parse_dates=["timestamp"])
features = ["hour", "day_of_week", "month", "is_weekend", "session_duration_hours", "energy_delivered_kwh"]
split = int(len(df)*0.8)
X = df[features]
reg = RandomForestRegressor(n_estimators=180, max_depth=12, random_state=42, n_jobs=-1)
reg.fit(X.iloc[:split], df["charging_sessions"].iloc[:split])
clf = RandomForestClassifier(n_estimators=180, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1)
clf.fit(X.iloc[:split], df["demand_level"].iloc[:split])

c1,c2,c3 = st.columns(3)
c1.metric("Total records", f"{len(df):,}")
c2.metric("Mean sessions per record", f"{df['charging_sessions'].mean():.1f}")
c3.metric("High-demand records", f"{(df['demand_level']=='High').mean()*100:.1f}%")

st.subheader("Historical demand trends")
hourly = df.groupby("hour", as_index=False)["charging_sessions"].mean().rename(columns={"charging_sessions":"Average sessions"})
st.line_chart(hourly.set_index("hour"))
st.bar_chart(df["demand_level"].value_counts().reindex(["Low","Medium","High"]).fillna(0))

st.subheader("Predict demand for a scenario")
left,right = st.columns(2)
with left:
    hour = st.slider("Hour of day", 0, 23, 18)
    dow = st.selectbox("Day of week", options=list(range(7)), format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])
    month = st.slider("Month", 1, 12, 6)
    duration = st.slider("Session duration (hours)", 0.25, 8.0, 1.5, 0.25)
    energy = st.slider("Energy delivered (kWh)", 0.5, 80.0, 12.0, 0.5)
with right:
    row = pd.DataFrame([{"hour":hour,"day_of_week":dow,"month":month,"is_weekend":int(dow>=5),
                         "session_duration_hours":duration,"energy_delivered_kwh":energy}])[features]
    predicted = max(0, int(round(reg.predict(row)[0])))
    level = clf.predict(row)[0]
    st.metric("Predicted charging sessions", predicted)
    st.metric("Predicted demand level", level)
    st.info("This is a model estimate from synthetic data. It does not indicate real-time charger availability.")

st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)
