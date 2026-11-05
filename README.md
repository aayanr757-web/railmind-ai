# 🚆 RAILMIND AI — Railway Monitoring Digital Twin (Prototype)

An interactive Python/Streamlit prototype that simulates locomotive sensor readings, injects faults, and compares an **Isolation Forest anomaly detector** with simple rule-based alerts.

> **Important:** All telemetry is synthetic. This is an educational proof of concept, **not** a certified railway safety system, validated maintenance predictor, or actual Railway Challenge locomotive software. AI predictions do not control brakes or motors.

## Features

- Configurable synthetic speed, motor temperature, vibration, and motor current telemetry
- Six selectable scenarios: normal, overheating, excess vibration, overcurrent, overspeed, combined fault
- Unsupervised anomaly detection trained on *separate normal synthetic samples*
- Rule-based monitoring for comparison
- Interactive time-series charts and alert flags
- Precision, recall, F1, and confusion matrix against injected synthetic fault labels
- CSV export and four automated smoke tests

## Architecture

```text
Synthetic train telemetry → Isolation Forest → AI anomaly flags ─┐
                         → Threshold rules → Rule-based flags ────┼→ Streamlit dashboard
Injected fault labels ────────────────────────────────────────────┘
```

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, typically `http://localhost:8501`.

## Tests

```bash
python -m unittest -v
```

## How to demo

1. Run the dashboard with the **Normal** scenario.
2. Switch to **Motor overheating** and observe temperature and anomaly flags.
3. Try **Excess vibration** or **Combined fault**.
4. Compare AI alerts against threshold alerts and review the synthetic-label evaluation.
5. Download the generated CSV.

## Technical notes and limitations

- The simulator creates correlated sensor readings, but is not a physics-validated train model.
- The model uses `IsolationForest` with a fixed 2% contamination setting; its outputs are anomaly flags, **not calibrated failure probabilities**.
- Fault injection and training distributions are synthetic; apparent accuracy may not transfer to real data.
- This is an **offline batch simulation displayed interactively**, not live sensor ingestion.
- The model does not estimate remaining useful life or predict when a real component will fail.
- Thresholds are illustrative, not derived from locomotive specifications.
- For real deployment: obtain consented real telemetry, define operating envelopes with railway engineers, add time-series validation, measure false alarms, and keep safety-critical control independent of AI.

## Repository files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `simulator.py` | Synthetic locomotive telemetry and fault injection |
| `model.py` | Anomaly detection and rule-based alerts |
| `test_project.py` | Unit smoke tests |
| `requirements.txt` | Python dependencies |

## Author

Aayan Raza — BEng (Hons) Robotics, Autonomous & Interactive Systems, Heriot-Watt University.

Independent educational portfolio prototype, inspired by an interest in railway engineering.
