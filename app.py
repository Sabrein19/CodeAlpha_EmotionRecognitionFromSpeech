"""Streamlit inference UI. Run with: streamlit run app.py"""
from pathlib import Path
import tempfile, joblib, numpy as np, streamlit as st
from tensorflow import keras
from src.features import extract_features

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.caption("Upload a WAV file to classify the dominant emotion using the trained RAVDESS model.")

@st.cache_resource
def load_artifacts():
    folder = Path("models")
    return keras.models.load_model(folder / "best_model.keras"), joblib.load(folder / "feature_scaler.joblib"), joblib.load(folder / "label_encoder.joblib")

uploaded = st.file_uploader("WAV audio", type=["wav"])
if uploaded:
    st.audio(uploaded)
    try:
        model, scaler, encoder = load_artifacts()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp.write(uploaded.getbuffer()); temp_path = temp.name
        features = extract_features(temp_path)
        scaled = scaler.transform(features).reshape(1, *features.shape)
        probabilities = model.predict(scaled, verbose=0)[0]
        idx = int(np.argmax(probabilities))
        st.success(f"Prediction: **{encoder.inverse_transform([idx])[0].title()}**")
        st.metric("Confidence", f"{probabilities[idx]:.1%}")
        st.bar_chart({name: float(score) for name, score in zip(encoder.classes_, probabilities)})
    except FileNotFoundError:
        st.warning("Train the model first: `python -m src.train --dataset_dir data/raw/RAVDESS`.")
    except Exception as exc:
        st.error(f"Could not process this file: {exc}")
