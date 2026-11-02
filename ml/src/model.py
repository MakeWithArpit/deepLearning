"""
DL-024: Sequence Classifier Model (LSTM)
Takes temporal feature sequences (T frames, 120 features) and classifies
dynamic sign language gestures (e.g. Hello, Thank You, Yes, No, Help).
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input

def build_sequence_model(input_shape=(20, 120), num_classes=5):
    """
    Builds a temporal LSTM classifier for sign language sequence recognition.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True, dropout=0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False, dropout=0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    m = build_sequence_model()
    m.summary()

