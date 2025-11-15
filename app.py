from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return "✅ Fake News Detector API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    # Vectorize the input
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    label = "Real" if prediction == 1 else "Fake"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
