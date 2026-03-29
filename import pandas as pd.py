import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("housing.csv")

# Preprocessing
df = df.drop(columns=['Address'])
X = df.drop("Price", axis=1)
y = df["Price"]

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "house_price_model.pkl")

print("✅ File created successfully")