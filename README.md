# AI-Based EV Charging Demand Prediction

## Project overview
This capstone demonstrates machine-learning regression to estimate charging-session counts and classification to label demand as Low, Medium, or High.

## Important data note
`data/ev_charging_sessions.csv` is a **simulated educational dataset** created for this project because no real public dataset was supplied. It contains 3200 records. Do not describe the results as validated real-world forecasts or live charger availability.

## Features
- Timestamp-derived hour, day of week, month, and weekend indicator
- Session duration and energy delivered
- Random Forest regression and classification
- Time-ordered 80/20 train-test split
- MAE, RMSE, R², accuracy, and weighted F1 evaluation
- Streamlit dashboard with historical trends and scenario predictions

## Setup
1. Install Python 3.10 or newer.
2. In this folder, run `pip install -r requirements.txt`.
3. Run `python train_model.py` to train and evaluate models.
4. Run `streamlit run app.py` to open the dashboard.

## Project structure
- `data/ev_charging_sessions.csv` — simulated dataset
- `train_model.py` — training and evaluation
- `app.py` — interactive dashboard
- `outputs/model_metrics.json` — example evaluation metrics
- `outputs/test_predictions.csv` — held-out test predictions
- `outputs/*.png` — charts used in report/presentation
- `Project_Report.docx` — editable report
- `Final_Presentation.pptx` — presentation deck

## Limitations
The simulated data follows hand-designed daily/seasonal patterns. It cannot capture actual station location, weather, holidays, price, grid constraints, charger outages, or local EV adoption. Real-world use requires a public or station-owner dataset, data-quality checks, and validation on later unseen dates.
