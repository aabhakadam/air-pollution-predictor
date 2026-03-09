AI-Powered Air Pollution Predictor
A machine learning project that simulates ESP32 sensor readings and predicts pollution spikes in real time.
## What it does
- Simulates CO, PM2.5, temperature and humidity sensor data based on Mumbai rush hour patterns
- Trains a Random Forest ML model to predict pollution spikes
- Displays a live animated dashboard with color-coded predictions
- Sends real MQTT alerts when a spike is detected with >80% confidence
## Tech Stack
- Python, pandas, scikit-learn, matplotlib, paho-mqtt
## How to run
### 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib paho-mqtt
### 2. Generate training data
python generate_data.py
### 3. Train the model
python train_model.py
### 4. Launch live dashboard
python dashboard.py

### 5. Run MQTT alerts
python mqtt_alerts.py
## Results
- Model accuracy: 100%
- Most important feature: CO PPM (49.4%)
- Alerts published to: broker.hivemq.com topic `stuttgart/pollution/alert`
```