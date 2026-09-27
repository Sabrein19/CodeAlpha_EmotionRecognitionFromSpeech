"""Train, compare, evaluate, and save speech emotion models.

Run: python -m src.train --dataset_dir data/raw/RAVDESS
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from src.data import discover_ravdess, split_indices
from src.features import extract_features
from src.models import build_cnn, build_lstm


def plot_history(history, output: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in [("loss", "Loss"), ("accuracy", "Accuracy")]:
        axes[0 if key == "loss" else 1].plot(history.history[key], label="train")
        axes[0 if key == "loss" else 1].plot(history.history[f"val_{key}"], label="validation")
        axes[0 if key == "loss" else 1].set_title(f"{title} {label}")
        axes[0 if key == "loss" else 1].legend()
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def evaluate(model, x_test, y_test, classes, output: Path):
    predicted = model.predict(x_test, verbose=0).argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average="weighted", zero_division=0)
    report = classification_report(y_test, predicted, target_names=classes, zero_division=0)
    metrics = {"accuracy": float(np.mean(predicted == y_test)), "precision_weighted": float(precision), "recall_weighted": float(recall), "f1_weighted": float(f1)}
    (output / "classification_report.txt").write_text(report, encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_test, predicted)
    plt.figure(figsize=(9, 7)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(output / "confusion_matrix.png", dpi=160); plt.close()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    root, models_dir, output = Path.cwd(), Path("models"), Path("outputs")
    models_dir.mkdir(exist_ok=True); output.mkdir(exist_ok=True)
    paths, labels = discover_ravdess(args.dataset_dir)
    print(f"Extracting features from {len(paths)} speech files...")
    x = np.stack([extract_features(p) for p in paths])
    encoder = LabelEncoder(); y = encoder.fit_transform(labels)
    train_i, val_i, test_i = split_indices(labels)
    # Fit scaling only on train data to prevent validation/test leakage.
    scaler = StandardScaler().fit(x[train_i].reshape(-1, x.shape[-1]))
    x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    x_train, x_val, x_test = x[train_i], x[val_i], x[test_i]
    y_train, y_val, y_test = y[train_i], y[val_i], y[test_i]
    class_values = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=class_values, y=y_train)
    class_weight = dict(zip(class_values, weights))
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)]
    candidates = {"cnn": build_cnn(x.shape[1:], len(encoder.classes_)), "lstm": build_lstm(x.shape[1:], len(encoder.classes_))}
    scores, trained = {}, {}
    for name, model in candidates.items():
        print(f"Training {name}...")
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs, batch_size=32, class_weight=class_weight, callbacks=callbacks, verbose=2)
        plot_history(history, output / f"{name}_training_curves.png", name.upper())
        scores[name] = float(model.evaluate(x_val, y_val, verbose=0)[1]); trained[name] = model
    best_name = max(scores, key=scores.get); best_model = trained[best_name]
    best_model.save(models_dir / "best_model.keras")
    joblib.dump(scaler, models_dir / "feature_scaler.joblib")
    joblib.dump(encoder, models_dir / "label_encoder.joblib")
    (models_dir / "model_config.json").write_text(json.dumps({"best_model": best_name, "validation_accuracy": scores[best_name], "input_shape": list(x.shape[1:])}, indent=2), encoding="utf-8")
    metrics = evaluate(best_model, x_test, y_test, encoder.classes_, output)
    print(f"Best model: {best_name}; test metrics written to {output}. Metrics: {metrics}")


if __name__ == "__main__": main()
