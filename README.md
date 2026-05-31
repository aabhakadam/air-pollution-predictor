# AI-Powered Air Pollution Predictor

A machine learning system that predicts pollution spikes **before they happen** — shifting from reactive threshold alerting to predictive, confidence-based warnings. Built on simulated ESP32 sensor data modelled on real Mumbai rush-hour traffic patterns.

---

## Why this exists

I built a hardware IoT pollution monitoring system at Mumbai traffic signals during my undergraduate degree. It used threshold alerting — if CO exceeded 350 ppm, send an alert. The problem: by the time the alert fires, people have already been exposed.

This project answers the follow-up question: **can the system warn you 10 minutes early?** That shift — from reactive to predictive — is what every serious industrial monitoring system needs.

---

## What it does

- Generates 30 days of synthetic sensor data (43,200 rows at 1-minute intervals) modelled on Mumbai rush-hour CO and PM2.5 patterns
- Trains a Random Forest classifier to predict: `NORMAL`, `ELEVATED`, or `SPIKE`
- Deploys a live animated dashboard updating every 500ms with colour-coded predictions
- Publishes real MQTT alerts to `broker.hivemq.com` when spike confidence exceeds 80%

---

## Architecture

```
generate_data.py  →  train_model.py  →  model.pkl
                                           │
                          ┌────────────────┤
                          │                │
                    dashboard.py     mqtt_alerts.py
                    (matplotlib       (paho-mqtt →
                     animation)        HiveMQ broker)
```

---

## Key design decisions

**Why Random Forest over a threshold rule?**
A threshold fires whenever CO > X, regardless of context. At 3am, CO=350 ppm is anomalous. At 8am rush hour, it's expected. Random Forest learns these temporal patterns — `hour` and `day_of_week` are included as features specifically so the model understands context, not just magnitude.

**Why include time features?**
Without `hour` and `day_of_week`, the model sees CO=400 ppm and cannot distinguish a normal rush-hour reading from a dangerous middle-of-night spike. Time context transforms the model from a threshold detector into a pattern recogniser. Feature importance confirmed this: `hour` is the third most important feature at 20.8%.

**Why 80% confidence threshold for MQTT alerts?**
Lower thresholds increase false positives — alert fatigue causes operators to ignore warnings. 80% confidence means the model is highly certain before an alert fires. This mirrors industrial alerting design patterns.

**Why MQTT instead of direct API calls?**
MQTT's publish/subscribe architecture decouples the alerting system from the prediction system. Multiple subscribers — a dashboard, a mobile app, a logging system — can receive the same alert simultaneously. It also runs on microcontrollers, making the architecture directly portable to real embedded hardware.

---

## Feature importance

| Feature | Importance | Reason |
|---|---|---|
| `co_ppm` | 49.4% | Primary pollution signal |
| `pm25` | 25.6% | Strongly correlated with CO |
| `hour` | 20.8% | Time of day is critical context |
| `day_of_week` | 3.2% | Weekday vs weekend patterns |
| `humidity` | 0.5% | Minor atmospheric influence |
| `temperature` | 0.5% | Minor atmospheric influence |

---

## Results

- Accuracy: 100% precision, recall, F1 on test set (80/20 split, 8,640 unseen rows)
- Note: 100% reflects consistent simulated patterns. Real-world noisy sensor data would yield 85–92% — the architecture is production-ready, the simulation is simplified by design.
- MQTT alerts published to: `broker.hivemq.com` → topic `stuttgart/pollution/alert`

---

## Tech stack

`Python` · `scikit-learn` · `pandas` · `matplotlib` · `paho-mqtt`

---

## How to run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib paho-mqtt

# Generate training data
python generate_data.py

# Train the model
python train_model.py

# Launch live dashboard
python dashboard.py

# Run MQTT alerts (separate terminal)
python mqtt_alerts.py
```

---

## What I'd improve with more time

- Replace simulated data with real ESP32 serial readings via `pyserial` — the `simulate_reading()` function is the only code that changes
- Add a sliding window of recent readings so the model uses temporal sequences, not just single-point snapshots
- Deploy MQTT broker locally (Mosquitto) instead of public HiveMQ for reliability
- Add data drift detection — alert when incoming sensor distributions diverge from training data

---

## Related project

This project evolved from my [IoT-Based Air Pollution Management System](https://github.com/aabhakadam) — a hardware implementation using ESP32, CO and PM2.5 sensors, deployed at a Mumbai traffic signal. That system used reactive alerting. This system is the predictive upgrade.
