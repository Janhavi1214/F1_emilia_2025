🏎️ F1 2025 Emilia Romagna GP Prediction 


This repository contains a machine learning project that predicts the 2025 Emilia Romagna Grand Prix race times for Formula 1 drivers based on historical qualifying data, sector times, clean air race pace, team performance, and weather conditions.

The project uses FastF1 to fetch historical session data and Gradient Boosting Regressor to predict race outcomes.

📂 Repository Contents

emilia_25.py — Main Python script to predict the 2025 Emilia Romagna GP results.

f1_cache/ — Local cache for FastF1 session data.

fastf1_eg.ipynb — Optional exploratory notebook for FastF1 data exploration.

monaco_2025.py — Optional placeholder for future GP predictions.

🛠️ Technologies & Libraries

Python 3.x

FastF1
 — F1 telemetry and session data

pandas, numpy — Data manipulation

scikit-learn — Machine learning (Gradient Boosting Regressor)

matplotlib — Visualization

requests — Fetching weather data from OpenWeatherMap API

🔧 Features

Fetch Historical Session Data: Uses FastF1 to load the 2024 Emilia Romagna qualifying session.

Lap and Sector Analysis: Converts lap and sector times to seconds and aggregates average sector times per driver.

Clean Air Race Pace: Integrates clean air pace estimates for each driver.

Weather Adjustment: Fetches weather forecast data (rain probability and temperature) to adjust qualifying times.

Team Performance: Considers historical team performance as a feature.

Machine Learning Model: Trains a Gradient Boosting Regressor to predict race times.

Predicted Podium & Visualization: Outputs predicted finishing times and podium and visualizes feature importance and clean air effects.

⚡ How to Run

Clone the repository:

git clone https://github.com/Janhavi1214/F1_monaco_2025.git
cd F1_monaco_2025


Install dependencies:

pip install fastf1 pandas numpy scikit-learn matplotlib requests


Run the prediction script:

python emilia_25.py


Note: Replace YOURAPIKEY in the script with a valid OpenWeatherMap API key if you want live weather adjustments.

📊 Output

Predicted race times for each driver in seconds

Predicted podium finishers (P1, P2, P3)

Scatter plot showing effect of clean air race pace on predicted race times

Bar chart of feature importance in the model

📝 Notes

This project uses hypothetical 2025 qualifying times for demonstration.

FastF1 caches session data locally in f1_cache/ to speed up repeated runs.

Model performance is evaluated using Mean Absolute Error (MAE).

🔗 References

FastF1 Documentation

OpenWeatherMap API

Scikit-learn Gradient Boosting Regressor
