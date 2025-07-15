😊 Emotion Prediction from Physiological Signals
This project focuses on predicting human emotions based on physiological signals such as Blood Volume Pulse (BVP) and Galvanic Skin Response (GSR). By analyzing these signals, it predicts emotional states using machine learning models, helping to understand and classify emotions automatically.

🧠 Project Overview
The project processes physiological data along with annotated emotional labels to build predictive models for valence and arousal, which are core components of emotional states. Using techniques like Random Forest, XGBoost, and deep learning models (CNN-BiLSTM), it maps these predictions to specific emotions such as Happy, Sad, Fear, Anger, and more.

🛠️ Technology Stack
Language: Python 3.x
Libraries: pandas, numpy, scikit-learn, imblearn, xgboost, tensorflow/keras
Environment: Jupyter Notebook / Python scripts

📁 Dataset
Name: CASE Dataset (Continuously Annotated Signals of Emotion) 
Link: [https://springernature.figshare.com/articles/dataset/CASE_Dataset-full/8869157?file=16260497]
Format: CSV files
Contains physiological signal data (BVP, GSR) and annotations for valence and arousal levels

🔧 How It Works

Loads and merges physiological and annotation datasets into a single CSV file.

Preprocesses data by scaling and balancing classes using SMOTE.

Trains machine learning models (Random Forest, XGBoost) to predict valence and arousal.

Implements a CNN-BiLSTM deep learning model for improved prediction accuracy.

Maps predicted valence and arousal values, combined with physiological signals, to discrete emotion categories.

Evaluates model performance using accuracy, confusion matrix, and cross-validation.

▶️ How to Run

Clone or download the repository.

Install required Python libraries listed in requirements.txt or manually: pandas, numpy, scikit-learn, imblearn, xgboost, tensorflow.

Place your merged_data.csv file in the project folder.

Run the Python scripts or notebooks in order to preprocess data, train models, and test predictions.

🧩 Use Cases

Emotion recognition for healthcare and mental wellness monitoring.

Enhancing human-computer interaction by detecting user emotions.

Real-time emotion monitoring in applications such as gaming and education.

📝 Notes

Model accuracy depends on quality and size of physiological data.

Hyperparameters can be tuned for better performance.

Predicted emotions should be interpreted in context with domain knowledge.
