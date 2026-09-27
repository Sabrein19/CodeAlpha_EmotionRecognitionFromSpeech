# Emotion Recognition from Speech

A RAVDESS-based deep-learning project that compares a 1D CNN and bidirectional LSTM, then serves the selected model with Streamlit. It extracts 40 MFCCs, 12 chroma values, and 64 mel-spectrogram values per audio frame.

## Project structure

```
emotion-recognition-speech/
├── app.py                 # Streamlit inference app
├── data/raw/              # Downloaded RAVDESS files (ignored by Git)
├── models/                # Generated Keras model and preprocessing artifacts
├── notebooks/             # Optional EDA notebooks
├── outputs/               # Generated curves, reports, and confusion matrix
└── src/                   # Reusable feature, data, model, and training code
```

## Setup and dataset

1. Create and activate a virtual environment, then run `pip install -r requirements.txt`.
2. Download the **Audio_Speech_Actors_01-24** archive from the [RAVDESS Zenodo record](https://zenodo.org/records/1188976), extract it, and place the actor folders under `data/raw/RAVDESS/`.
3. Train both models: `python -m src.train --dataset_dir data/raw/RAVDESS --epochs 40`.
4. Launch the interface: `streamlit run app.py`.

The training script uses a stratified 70/15/15 train/validation/test split, fits a `StandardScaler` only on training frames, uses balanced class weights, selects the model with the highest validation accuracy, and saves `models/best_model.keras`, the scaler, and the label encoder.

## Evaluation and results

After training, `outputs/` contains CNN/LSTM training curves, a test-set classification report, `metrics.json` (accuracy, weighted precision, recall, F1), and a confusion matrix. **No performance numbers are claimed here; run training to generate results for your environment.**

## Screenshot

_Add a screenshot of the Streamlit upload and prediction screen here after running the app._

## Future improvements

- Add augmentation such as noise, pitch shifting, and time stretching.
- Use speaker-independent cross-validation.
- Evaluate pretrained audio embeddings (for example, YAMNet) alongside these models.
