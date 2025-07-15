import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings

warnings.filterwarnings(action='ignore')

data = pd.read_csv('merged_data.csv')
sampled_data = data.sample(n=450000, random_state=42)

X = sampled_data[['bvp', 'gsr']]
y_valence = sampled_data['valence'].astype(int)
y_arousal = sampled_data['arousal'].astype(int)

X_train, X_test, y_train_val, y_test_val = train_test_split(X, y_valence, test_size=0.3, random_state=42)
X_train_ar, X_test_ar, y_train_ar, y_test_ar = train_test_split(X, y_arousal, test_size=0.3, random_state=42)

scaler_val = MinMaxScaler()
scaler_ar = MinMaxScaler()

X_train_scaled_val = scaler_val.fit_transform(X_train)
X_test_scaled_val = scaler_val.transform(X_test)

X_train_scaled_ar = scaler_ar.fit_transform(X_train_ar)
X_test_scaled_ar = scaler_ar.transform(X_test_ar)

smote = SMOTE(random_state=42)
X_train_resampled_val, y_train_resampled_val = smote.fit_resample(X_train_scaled_val, y_train_val)
X_train_resampled_ar, y_train_resampled_ar = smote.fit_resample(X_train_scaled_ar, y_train_ar)

model_valence = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model_arousal = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

model_valence.fit(X_train_resampled_val, y_train_resampled_val)
model_arousal.fit(X_train_resampled_ar, y_train_resampled_ar)

def emotion_mapping_rule(bvp, gsr, valence, arousal):
    if bvp > 60 and gsr > 10000 and valence > 6.5 and arousal > 6.5:
        return 'Excitement'
    elif bvp > 300 and gsr < 10000 and valence > 6 and arousal < 6:
        return 'Happy'
    elif bvp < 400 and gsr < 10000 and valence < 4 and arousal < 4:
        return 'Sad'
    elif bvp > 300 and gsr > 10000 and valence < 4 and arousal > 6:
        return 'Anger'
    elif bvp < 300 and gsr > 10000 and valence < 4 and arousal > 6:
        return 'Fear'
    elif 200 <= bvp <= 300 and 10000 <= gsr <= 15000 and 4 <= valence <= 5 and 4 <= arousal <= 5:
        return 'Neutral'
    return 'Undefined'

def test_multiple_values(test_cases):
    for i, (bvp_input, gsr_input) in enumerate(test_cases):
        print(f"Test {i + 1}:")
        print(f"PPG (BVP): {bvp_input}, GSR: {gsr_input}")

        user_input = pd.DataFrame([[bvp_input, gsr_input]], columns=['bvp', 'gsr'])
        user_input_normalized_val = scaler_val.transform(user_input)
        user_input_normalized_ar = scaler_ar.transform(user_input)

        predicted_valence = model_valence.predict(user_input_normalized_val)[0]
        predicted_arousal = model_arousal.predict(user_input_normalized_ar)[0]

        print(f"Predicted Valence: {predicted_valence}")
        print(f"Predicted Arousal: {predicted_arousal}")

        emotion = emotion_mapping_rule(bvp_input, gsr_input, predicted_valence, predicted_arousal)
        print(f"Predicted Emotion: {emotion}\n")

test_cases = [
    (80, 12000),
    (250, 13000),
    (90, 11500),
    (200, 10000),
    (300, 14000),
    (150, 9000)
]

test_multiple_values(test_cases)

y_pred_val = model_valence.predict(X_test_scaled_val)
y_pred_ar = model_arousal.predict(X_test_scaled_ar)

y_pred_val_binned = np.clip(np.round(y_pred_val), 0, 10).astype(int)
y_pred_ar_binned = np.clip(np.round(y_pred_ar), 0, 10).astype(int)

print("\nValence Accuracy:", accuracy_score(y_test_val, y_pred_val_binned))
print("Arousal Accuracy:", accuracy_score(y_test_ar, y_pred_ar_binned))

print("\nValence Confusion Matrix:\n", confusion_matrix(y_test_val, y_pred_val_binned))
print("Arousal Confusion Matrix:\n", confusion_matrix(y_test_ar, y_pred_ar_binned))
