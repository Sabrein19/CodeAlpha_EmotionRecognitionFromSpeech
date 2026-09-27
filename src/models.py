"""Keras model definitions for the CNN and LSTM comparison."""
from __future__ import annotations
from tensorflow import keras


def build_cnn(input_shape: tuple[int, int], num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(64, 5, activation="relu", padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool1D(2)(x)
    x = keras.layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="speech_cnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(input_shape: tuple[int, int], num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Bidirectional(keras.layers.LSTM(96, return_sequences=True))(inputs)
    x = keras.layers.Dropout(0.35)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(48))(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="speech_lstm")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
