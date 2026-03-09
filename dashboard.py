import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pickle
from datetime import datetime

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded! Opening dashboard...")

# Color map
COLOR_MAP = {'normal': 'green', 'elevated': 'orange', 'spike': 'red'}

# Data buffers (stores last 60 readings)
co_values, pm25_values, predictions, confidences = [], [], [], []

def simulate_reading():
    """Simulates a live sensor reading based on current hour"""
    hour = datetime.now().hour
    day_of_week = datetime.now().weekday()
    is_weekend = day_of_week >= 5
    weekend_factor = 0.6 if is_weekend else 1.0

    if 8 <= hour <= 10 or 17 <= hour <= 19:
        co = np.random.normal(400, 60) * weekend_factor
    elif 6 <= hour <= 8 or 19 <= hour <= 21:
        co = np.random.normal(250, 40) * weekend_factor
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

# Set up the figure with 3 panels
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10))
fig.suptitle('🌿 Air Pollution Predictor — Live Dashboard', fontsize=14, fontweight='bold')

def update(frame):
    r = simulate_reading()
    features = [[r['co_ppm'], r['pm25'], r['temperature'],
                  r['humidity'], r['hour'], r['day_of_week']]]

    pred = model.predict(features)[0]
    conf = max(model.predict_proba(features)[0])

    co_values.append(r['co_ppm'])
    pm25_values.append(r['pm25'])
    predictions.append(pred)
    confidences.append(conf)

    # Keep last 60 readings only
    if len(co_values) > 60:
        co_values.pop(0)
        pm25_values.pop(0)
        predictions.pop(0)
        confidences.pop(0)

    colors = [COLOR_MAP[p] for p in predictions]
    x = range(len(co_values))

    # --- Panel 1: CO readings as bar chart ---
    ax1.clear()
    ax1.bar(x, co_values, color=colors, alpha=0.8)
    ax1.axhline(y=350, color='red', linestyle='--', linewidth=1, label='Spike threshold')
    ax1.axhline(y=200, color='orange', linestyle='--', linewidth=1, label='Elevated threshold')
    ax1.set_ylabel('CO (ppm)')
    ax1.set_title(f'CO Readings  |  Now: {r["co_ppm"]:.0f} ppm  |  '
                  f'Prediction: {pred.upper()}  |  Confidence: {conf:.0%}')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(0, 550)

    # --- Panel 2: PM2.5 line chart ---
    ax2.clear()
    ax2.plot(x, pm25_values, color='purple', linewidth=2)
    ax2.fill_between(x, pm25_values, alpha=0.3, color='purple')
    ax2.set_ylabel('PM2.5')
    ax2.set_title(f'PM2.5 Readings  |  Now: {r["pm25"]:.1f}')
    ax2.set_ylim(0, 200)

    # --- Panel 3: Confidence over time ---
    ax3.clear()
    ax3.plot(x, confidences, color='blue', linewidth=2)
    ax3.fill_between(x, confidences, alpha=0.3, color='blue')
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel('Confidence')
    ax3.set_xlabel('Last 60 readings (0.5s each)')
    ax3.set_title('Model Prediction Confidence')
    ax3.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, label='80% threshold')
    ax3.legend(loc='lower left', fontsize=8)

    plt.tight_layout()

ani = animation.FuncAnimation(fig, update, interval=500, cache_frame_data=False)
plt.show()