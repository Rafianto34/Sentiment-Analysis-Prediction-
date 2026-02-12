import streamlit as st
import joblib
import re

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Spotify Sentiment Analysis",
    page_icon="🎧",
    layout="centered"
)

st.title("🎧 Spotify Sentiment Analysis")
st.write(
    "Website untuk analisis dan prediksi sentimen review Spotify "
    "menggunakan Machine Learning (Naive Bayes)."
)

# =========================
# LOAD MODEL & VECTORIZER
# =========================
@st.cache_resource
def load_model():
    nb_model = joblib.load("naive_bayes_model.pkl")
    vectorizer = joblib.load("count_vectorizer.pkl")
    return nb_model, vectorizer

nb_model, vectorizer = load_model()

# =========================
# LABEL MAP
# =========================
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# =========================
# TEXT PREPROCESSING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# USER INPUT
# =========================
st.subheader("✍️ Masukkan Review Spotify")

user_text = st.text_area(
    "Contoh:",
    "I hate this app, it crashes every time",
    height=120
)

# =========================
# PREDICTION
# =========================
if st.button("🚀 Prediksi Sentimen"):
    if user_text.strip() == "":
        st.warning("⚠️ Masukkan review terlebih dahulu.")
    else:
        cleaned_text = clean_text(user_text)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = nb_model.predict(vectorized_text)[0]
        sentiment = label_map.get(int(prediction), "neutral")

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        if sentiment == "positive":
            st.success("😊 Sentimen POSITIF")
        elif sentiment == "negative":
            st.error("😡 Sentimen NEGATIF")
        else:
            st.info("😐 Sentimen NETRAL")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("📊 Sentiment Analysis Spotify | Streamlit & Naive Bayes")
