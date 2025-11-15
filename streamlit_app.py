# 📰 Fake News Detection Web App — Premium Interactive UI
# Designed by Vidyaa Tappu

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# --- Custom CSS for Modern Dark Theme ---
st.markdown("""
<style>
/* Animated Gradient Background */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

body {
    background: linear-gradient(-45deg, #0d0d0d, #121212, #1c1c1c, #0a0a0a);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
    color: white;
}

/* Main Card Container */
.main {
    background: rgba(20, 20, 20, 0.9);
    padding: 3rem;
    border-radius: 25px;
    max-width: 900px;
    margin: auto;
    box-shadow: 0 0 40px rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
}

/* Title & Subtitle */
.title {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    text-align: left;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.2rem;
    color: #bbb;
    text-align: left;
    margin-bottom: 2rem;
}

/* Text Area */
textarea {
    background: rgba(255,255,255,0.05) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #b00020, #ff4081);
    color: white;
    border: none;
    padding: 0.9rem 1.7rem;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #ff4081;
}

/* Result Box */
.result {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 1.5rem;
    margin-top: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 0 20px rgba(255,255,255,0.05);
}

/* Footer */
.footer {
    text-align: center;
    color: #999;
    margin-top: 3rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# --- App UI ---
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown("""
<div class="title">📰 Fake News Detection Web App</div>
<div class="subtitle">Check any news headline or paragraph using AI — find out if it's Real or Fake instantly.</div>
""", unsafe_allow_html=True)

# --- Load & Train Model ---
@st.cache_data
def train_model():
    df = pd.read_csv("news.csv")
    x = df['text']
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(tfidf_train, y_train)
    y_pred = model.predict(tfidf_test)
    acc = accuracy_score(y_test, y_pred)
    return model, vectorizer, acc

model, vectorizer, acc = train_model()

# --- Input Box ---
news_input = st.text_area("🗞️ Enter News Headline or Paragraph", height=150, placeholder="Type or paste news content here...")

# --- Analyze Button ---
if st.button("Analyze News 🔎"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        input_data = [news_input]
        vectorized_input = vectorizer.transform(input_data)
        prediction = model.predict(vectorized_input)[0]

        if prediction == "FAKE":
            st.markdown('<div class="result"><h3 style="color:#ff3b3b;">🚫 Fake News Detected!</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result"><h3 style="color:#00ff88;">✅ This news appears to be Real.</h3></div>', unsafe_allow_html=True)

        st.markdown(f"<div class='result'>🎯 News Accuracy: <b>{round(acc*100, 2)}%</b></div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">Developed by <b>Vidyaa Tappu</b> • AI & Robotics Enthusiast 🤖</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
