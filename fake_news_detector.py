import joblib

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Prediction function
def predict_news(text):
    """
    Input: a string (news text)
    Output: 'Fake News' or 'Real News'
    """
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "Fake News" if prediction == 0 else "Real News"
