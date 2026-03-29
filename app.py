from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")

# UI route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = np.array([[
        float(data['income']),
        float(data['house_age']),
        float(data['rooms']),
        float(data['bedrooms']),
        float(data['population'])
    ]])

    prediction = model.predict(features)

    return jsonify({
        "predicted_price": round(float(prediction[0]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)