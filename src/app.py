import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os
from datetime import datetime

def load_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    df['class'] = df['class'].map({b'tested_negative': 0, b'tested_positive': 1})
    return df

df = load_arff('diabetes.arff')

X = df.iloc[:, :-1].values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", "fit", timestamp)
os.makedirs(log_dir, exist_ok=True)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[tensorboard_callback])

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

mock_data = np.array([
    [5, 117, 92, 0, 0, 34.1, 0.337, 38], # negative
    [10, 168, 74, 0, 0, 38, 0.537, 34] # positve
])

mock_data_normalized = scaler.transform(mock_data)

predictions = model.predict(mock_data_normalized)

print("Predictions for mock data:")
for i, prediction in enumerate(predictions):
    print(f"Sample {i}: {'Positive' if prediction > 0.5 else 'Negative'}")
