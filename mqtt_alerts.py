import paho.mqtt.client as mqtt
import numpy as np
import pickle
import time
from datetime import datetime

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Connect to free public MQTT broker
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect("broker.hivemq.com", 1883, 60)
client.loop_start()
print("✅ Connected to MQTT broker!")
print("📡 Monitoring for pollution spikes...\n")

def simulate_reading():
    hour = datetime.now().hour
    day_of_week = datetime.now().weekday()
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        co = np.random.normal(400, 60)
    elif 6 <= hour <= 8 or 19 <= hour <= 21:
        co = np.random.normal(250, 40)
    else:
        co = np.random.normal(120, 30)
    return {
        'co_ppm': max(0, co),
        'pm25': max(0, co * 0.3 + np.random.normal(0, 10)),
        'temperature': np.random.normal(28, 3),
        'humidity': np.random.normal(70, 10),
        'hour': hour,
        'day_of_week': day_of_week
    }

alert_count = 0

while True:
    r = simulate_reading()
    features = [[r['co_ppm'], r['pm25'], r['temperature'],
                  r['humidity'], r['hour'], r['day_of_week']]]

    pred = model.predict(features)[0]
    conf = max(model.predict_proba(features)[0])

    status = f"[{datetime.now().strftime('%H:%M:%S')}] CO: {r['co_ppm']:.0f} ppm | PM2.5: {r['pm25']:.0f} | Prediction: {pred.upper()} ({conf:.0%})"
    print(status)

    if pred == 'spike' and conf > 0.8:
        alert_count += 1
        message = f"🚨 ALERT #{alert_count} | CO: {r['co_ppm']:.0f} ppm | PM2.5: {r['pm25']:.0f} | Confidence: {conf:.0%} | Time: {datetime.now().strftime('%H:%M:%S')}"
        client.publish("stuttgart/pollution/alert", message)
        print(f"  ⚡ ALERT SENT TO MQTT: {message}\n")

    time.sleep(2)