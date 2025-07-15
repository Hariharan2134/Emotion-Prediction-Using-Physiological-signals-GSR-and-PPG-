# Emotion-Prediction-Using-Physiological-signals-GSR-and-PPG-
# 🧠 Emotion Prediction from PPG and GSR Signals

This project focuses on predicting **valence**, **arousal**, and classifying **emotions** using physiological signals — **PPG (BVP)** and **GSR**. This work was completed as part of an internship project at **NIT Trichy**.

---

## 🎯 Objectives

- Predict continuous **valence** and **arousal** values.
- Classify data into one of six emotions:  
  `Happy`, `Excitement`, `Neutral`, `Fear`, `Sad`, `Anger`
- Implement multiple modeling approaches:
  - Machine Learning: `Random Forest`, `XGBoost`
  - Deep Learning: `CNN + BiLSTM`

---

## 🛠️ Tech Stack

| Component   | Tools & Libraries           |
|------------|-----------------------------|
| Language    | Python 3.x                  |
| ML Models   | scikit-learn, XGBoost       |
| DL Models   | TensorFlow, Keras           |
| Preprocessing | Pandas, NumPy, SMOTE      |
| Visualization | Matplotlib (optional)     |

---

## 📂 Project Structure

| Folder/File             | Description                                         |
|-------------------------|-----------------------------------------------------|
| `data/`                 | Sample annotation and physiological CSVs           |
| `preprocessing/`        | Merging and preprocessing scripts                  |
| `models/`               | ML & DL model scripts                              |
| `results/`              | Logs and output predictions                        |
| `README.md`             | Project overview and usage guide                   |
| `requirements.txt`      | Python dependencies                                |
| `LICENSE`               | (Optional) MIT License                             |

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/emotion-prediction-ppg-gsr-nit.git
cd emotion-prediction-ppg-gsr-nit
2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Prepare data
Make sure your physiological and annotations files are ready.
Run the merge script:

bash
Copy
Edit
python preprocessing/merge_and_prepare.py
🤖 Models & Execution
🔹 Random Forest
bash
Copy
Edit
python models/random_forest_model.py
🔹 XGBoost
bash
Copy
Edit
python models/xgboost_model.py
🔹 CNN + BiLSTM
bash
Copy
Edit
python models/cnn_bilstm_model.py
📊 Sample Output
Random Forest Valence Accuracy: 93.13%

Random Forest Arousal Accuracy: 49.69%

CNN-BiLSTM MSE Loss: ~0.0065

Emotion predictions are printed based on PPG, GSR, and model outputs.

🧭 Emotion Mapping Logic
Emotion is predicted using thresholds on predicted valence, arousal, PPG, and GSR:

Emotion	Conditions (Simplified)
Happy	High valence, high arousal, high PPG + GSR
Excitement	High valence, low arousal, high PPG
Neutral	Mid valence & arousal, normal PPG/GSR
Fear	Low valence, low arousal, low PPG + GSR
Sad	Low valence, high arousal, low PPG, high GSR
Anger	Catch-all for other intense combinations

📌 Notes
All models were trained on merged physiological and annotation data.

SMOTE is applied to handle data imbalance.

CNN-BiLSTM requires windowed input (default: 10 time steps).

yaml
Copy
Edit

---

### 📦 requirements.txt (Copy into `requirements.txt`)

```txt
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
tensorflow
