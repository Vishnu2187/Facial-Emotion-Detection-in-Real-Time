# train_and_save_model.py

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
def load_dataset():
    data = pd.read_csv('fer2013.csv')
    X = []
    y = []
    for i in range(len(data)):
        pixels = np.array(data['pixels'][i].split(), dtype='float32').reshape(48, 48)
        X.append(pixels)
        y.append(data['emotion'][i])
    X = np.array(X) / 255.0
    y = to_categorical(np.array(y))
    return X.reshape(-1, 48, 48, 1), y

# Define CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model and save
if __name__ == "__main__":
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model((48, 48, 1))
    model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))
    model.save("emotion_model.h5")
    print("✅ Model trained and saved as 'emotion_model.h5'")
