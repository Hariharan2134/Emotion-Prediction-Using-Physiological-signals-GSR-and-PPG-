import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

data = pd.read_csv('merged_data.csv')

bvp = data['bvp'].values
gsr = data['gsr'].values
valence = data['valence'].values
arousal = data['arousal'].values

scaler_bvp = MinMaxScaler()
scaler_gsr = MinMaxScaler()

bvp_scaled = scaler_bvp.fit_transform(bvp.reshape(-1, 1)).flatten()
gsr_scaled = scaler_gsr.fit_transform(gsr.reshape(-1, 1)).flatten()

X = np.column_stack((bvp_scaled, gsr_scaled))

valence_scaled = valence / 10.0
arousal_scaled = arousal / 10.0

X_train, X_test, valence_train, valence_test, arousal_train, arousal_test = train_test_split(
    X, valence_scaled, arousal_scaled, test_size=0.2, random_state=42
)

def create_time_windows(data, labels, window_size=50):
    X_windows, y_windows = [], []
    for i in range(len(data) - window_size):
        X_windows.append(data[i:i + window_size])
        y_windows.append(labels[i + window_size])
    return np.array(X_windows), np.array(y_windows)

window_size = 10

X_train_windows, valence_train_windows = create_time_windows(X_train, valence_train, window_size)
X_train_windows, arousal_train_windows = create_time_windows(X_train, arousal_train, window_size)
X_test_windows, valence_test_windows = create_time_windows(X_test, valence_test, window_size)
X_test_windows, arousal_test_windows = create_time_windows(X_test, arousal_test, window_size)

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Bidirectional(layers.LSTM(50, return_sequences=False)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation='sigmoid'))
    return model

input_shape = (X_train_windows.shape[1], X_train_windows.shape[2])
model = build_model(input_shape)
model.compile(optimizer='adam', loss='mse')

steps_per_epoch = X_train_windows.shape[0] // 8000

y_train = np.column_stack((valence_train_windows, arousal_train_windows))
y_test = np.column_stack((valence_test_windows, arousal_test_windows))

history = model.fit(X_train_windows, y_train, epochs=10, batch_size=32,
                    steps_per_epoch=steps_per_epoch, validation_data=(X_test_windows, y_test))

def get_user_input():
    user_bvp = float(input("Enter BVP value: "))
    user_gsr = float(input("Enter GSR value: "))

    user_bvp_scaled = scaler_bvp.transform([[user_bvp]])[0][0]
    user_gsr_scaled = scaler_gsr.transform([[user_gsr]])[0][0]

    user_input = np.array([[user_bvp_scaled, user_gsr_scaled]] * window_size).reshape(1, window_size, 2)
    return user_input

user_input = get_user_input()
valence_arousal_pred = model.predict(user_input)

predicted_valence = valence_arousal_pred[0][0] * 10
predicted_arousal = valence_arousal_pred[0][1] * 10

print(f"Predicted Valence (0-10): {predicted_valence}")
print(f"Predicted Arousal (0-10): {predicted_arousal}")
