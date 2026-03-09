from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import pickle

df = pd.read_csv('pollution_data.csv')
print(f"✅ Loaded {len(df)} rows")

features = ['co_ppm', 'pm25', 'temperature', 'humidity', 'hour', 'day_of_week']
X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

print("\n⏳ Training model... (takes ~30 seconds)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\n✅ Model trained!\n")
print(classification_report(y_test, predictions))

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("--- Feature Importance ---")
print(feature_importance.to_string(index=False))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n✅ Model saved to model.pkl")