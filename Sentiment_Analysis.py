import streamlit as st
import pandas as pd
import joblib
import pyaudio
import matplotlib.pyplot as plt
import speech_recognition as sr

# Load model and vectorizer
model = joblib.load("modelfest.pkl")
vectorizer = joblib.load("vectorizertest.pkl")

# Label decoding
label_decode = {0: "Negative", 1: "Positive", 2: "Neutral"}
color_map = {
    "Positive": "#27ae74",
    "Negative": "#dc3545",
    "Neutral": "#6c757d"
}

# Initialize history in session
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar UI
st.sidebar.title("🎙 Sentiment Analyzer")
st.sidebar.subheader("🎤 Voice Input:")
if st.sidebar.button("🔊 Speak Now"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        with mic as source:
            st.sidebar.info("Listening... Please speak clearly.")
            audio = recognizer.listen(source, timeout=5)
            st.sidebar.info("Processing your voice...")
            text_from_voice = recognizer.recognize_google(audio)
            st.session_state["voice_input"] = text_from_voice
            st.sidebar.success(f"Detected Text: {text_from_voice}")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Show History
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Prediction History")
if st.session_state.history:
    st.sidebar.dataframe(pd.DataFrame(st.session_state.history, columns=["Text", "Sentiment"]), use_container_width=True)
else:
    st.sidebar.write("No predictions made yet.")

# Main Area
st.title("💬 Sentiment Analysis Web")
st.markdown("Analyze the sentiment of your sentence using a trained machine learning model.")

# Text Input (from voice or typing)
default_text = st.session_state.get("voice_input", "")
user_input = st.text_area("✍ Enter text or use the sidebar voice button:", value=default_text, height=100)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter or speak some text.")
    else:
        # Predict
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        sentiment = label_decode[prediction]
        color = color_map[sentiment]

        # Result
        st.markdown(f"### 🧠 Sentiment: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)

        # Chart
        st.markdown("📊 *Model Confidence:*")
        prob_df = pd.DataFrame({
            "Sentiment": ["Negative", "Positive", "Neutral"],
            "Probability": proba
        })

        fig, ax = plt.subplots()
        ax.bar(prob_df["Sentiment"], prob_df["Probability"], color=[color_map[s] for s in prob_df["Sentiment"]])
        ax.set_ylabel("Probability")
        ax.set_xlabel("Sentiment")
        ax.set_title("Model Confidence")
        st.pyplot(fig)

        # History
        st.session_state.history.insert(0, (user_input, sentiment))