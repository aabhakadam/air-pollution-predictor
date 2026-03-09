import pandas as pd
import numpy as np

def generate_pollution_data(days=30):
    timestamps = pd.date_range('2024-01-01', periods=days*24*60, freq='1min')
    data = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        is_weekend = day_of_week >= 5
        weekend_factor = 0.6 if is_weekend else 1.0

        if 8 <= hour <= 10 or 17 <= hour <= 19:
            base_co = np.random.normal(400, 50) * weekend_factor
            label = 'spike' if base_co > 350 else 'elevated'
        elif 6 <= hour <= 8 or 19 <= hour <= 21:
            base_co = np.random.normal(250, 40) * weekend_factor
            label = 'elevated'
        else:
            base_co = np.random.normal(120, 30)
            label = 'normal'

        data.append({
            'timestamp': ts,
            'hour': hour,
            'day_of_week': day_of_week,
            'co_ppm': round(max(0, base_co), 2),
            'pm25': round(max(0, base_co * 0.3 + np.random.normal(0, 10)), 2),
            'temperature': round(np.random.normal(28, 3), 1),
            'humidity': round(np.random.normal(70, 10), 1),
            'label': label
        })
    
    return pd.DataFrame(data)

df = generate_pollution_data(days=30)
df.to_csv('pollution_data.csv', index=False)

print("✅ Data generated!")
print(f"Total rows: {len(df)}")
print("\nLabel distribution:")
print(df['label'].value_counts())
print("\nSample data:")
print(df.head(10))