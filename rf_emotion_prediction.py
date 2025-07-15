import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

data = pd.read_csv('merged_data.csv')
sampled_data = data.sample(n=1000000, random_state=42)

X = sampled_data[['bvp', 'gsr']]
y_valence = sampled_data['valence'].astype(int)
y_arousal = sampled_data['arousal'].astype(int)

X_train, X_test, y_train_val, y_test_val = train_test_split(X, y_valence, test_size=0.3, random_state=42)
X_train_ar, X_test_ar, y_train_ar, y_test_ar = train_test_split(X, y_arousal, test_size=0.3, random_state=42)

scaler_val = MinMaxScaler(feature_range=(0, 1))
X_train_scaled_val = scaler_val.fit_transform(X_train)
X_test_scaled_val = scaler_val.transform(X_test)

scaler_ar = MinMaxScaler(feature_range=(0, 1))
X_train_scaled_ar = scaler_ar.fit_transform(X_train_ar)
X_test_scaled_ar = scaler_ar.transform(X_test_ar)

print("Valence distribution:\n", y_valence.value_counts())
print("Arousal distribution:\n", y_arousal.value_counts())

smote = SMOTE(random_state=42)
X_train_resampled_val, y_train_resampled_val = smote.fit_resample(X_train_scaled_val, y_train_val)
X_train_resampled_ar, y_train_resampled_ar = smote.fit_resample(X_train_scaled_ar, y_train_ar)

rf_classifier_val = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')
rf_classifier_ar = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')

rf_classifier_val.fit(X_train_resampled_val, y_train_resampled_val)
rf_classifier_ar.fit(X_train_resampled_ar, y_train_resampled_ar)

def map_emotion(valence, arousal, ppg, gsr):
    high_ppg = ppg > 80
    high_gsr = gsr > 11000
    if valence > 7 and arousal > 7 and high_ppg and high_gsr:
        return "Happy"
    elif valence > 7 and arousal <= 5 and high_ppg and not high_gsr:
        return "Excitement"
    elif valence == 5 and arousal == 5 and not high_ppg and not high_gsr:
        return "Neutral"
    elif valence <= 4 and arousal <= 4 and not high_ppg and not high_gsr:
        return "Fear"
    elif valence <= 4 and arousal > 5 and not high_ppg and high_gsr:
        return "Sad"
    else:
        return "Anger"

def test_multiple_values(test_cases):
    for i, (bvp_input, gsr_input) in enumerate(test_cases):
        print(f"Test {i + 1}:")
        print(f"PPG (BVP): {bvp_input}, GSR: {gsr_input}")

        user_input = pd.DataFrame([[bvp_input, gsr_input]], columns=['bvp', 'gsr'])
        user_input_normalized_val = scaler_val.transform(user_input)
        user_input_normalized_ar = scaler_ar.transform(user_input)

        predicted_valence = rf_classifier_val.predict(user_input_normalized_val)
        predicted_arousal = rf_classifier_ar.predict(user_input_normalized_ar)

        print(f"Predicted Valence: {int(predicted_valence[0])}")
        print(f"Predicted Arousal: {int(predicted_arousal[0])}")

        emotion = map_emotion(int(predicted_valence[0]), int(predicted_arousal[0]), bvp_input, gsr_input)
        print(f"Predicted Emotion: {emotion}\n")

test_cases = [
    (85, 11600),
    (95, 11400),
    (60, 9500),
    (70, 10500),
    (82, 11700),
    (120, 13000)
]

test_multiple_values(test_cases)

y_pred_val = rf_classifier_val.predict(X_test_scaled_val)
y_pred_ar = rf_classifier_ar.predict(X_test_scaled_ar)

print("\nValence Accuracy:", accuracy_score(y_test_val, y_pred_val))
print("Arousal Accuracy:", accuracy_score(y_test_ar, y_pred_ar))

print("\nValence Confusion Matrix:\n", confusion_matrix(y_test_val, y_pred_val))
print("Arousal Confusion Matrix:\n", confusion_matrix(y_test_ar, y_pred_ar))

val_scores = cross_val_score(rf_classifier_val, X_train_resampled_val, y_train_resampled_val, cv=5)
arousal_scores = cross_val_score(rf_classifier_ar, X_train_resampled_ar, y_train_resampled_ar, cv=5)

print(f"\nValence Cross-Validation Accuracy: {val_scores.mean()}")
print(f"Arousal Cross-Validation Accuracy: {arousal_scores.mean()}")
