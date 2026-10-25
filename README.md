# SafeSewer dashboard with ML anomaly detection

## What's included
- `app.py`: Streamlit dashboard and presentation only.
- `ml/anomaly_detector.py`: isolated Isolation Forest implementation and sensor feature engineering.
- `requirements.txt`: application dependencies.

## Setup (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DATABASE_URL = "YOUR_POSTGRESQL_CONNECTION_STRING"
streamlit run app.py
```

You can also put the connection string in `.streamlit/secrets.toml`:
```toml
DATABASE_URL = "YOUR_POSTGRESQL_CONNECTION_STRING"
```

Do not commit real credentials or secrets to GitHub. If a database password was previously pasted into source code or a public repository, rotate it.

## How the ML detector works
The module trains an Isolation Forest on the latest valid telemetry rows available to the dashboard. It uses the three sensor values plus short-term deltas and rolling statistics. The model flags approximately 5% of the training window as outliers by design (`contamination=0.05`).

At least 30 valid rows are required. With fewer rows, the dashboard reports that there is not enough data. The model is retrained on refresh, so its results can change as the recent window changes.

## Important limitations
- This is unsupervised anomaly detection, not a supervised flood prediction model.
- An anomaly is not proof of a flood, toxic gas event, blockage, or sensor failure.
- Keep the existing engineering threshold alerts enabled.
- Validate flags against real events and sensor calibration before operational use.
- The latest 500 readings are used by the existing dashboard query; more history and labeled incidents would be useful for a stronger predictive model.
