Emotion Prediction from Physiological Signals 😊💓

About the Project
This project predicts human emotions based on physiological signals — specifically BVP (Blood Volume Pulse) and GSR (Galvanic Skin Response). By analyzing these signals, we estimate valence (positivity) and arousal (intensity) values and map them to common emotions like Happy, Sad, Fear, Anger, and more.

What’s Inside? 📦
Data merging: Combine physiological data and emotion annotations into one file.

Machine Learning Models:

Random Forest classifier

XGBoost regressor

Deep Learning Model: CNN + BiLSTM for time-series prediction.

Emotion Mapping: Translate valence and arousal predictions into six emotions.

Sample tests: Run example inputs to see predicted emotions.

How to Use? 🚀
Put your raw data files in the right folders (see the scripts).

Run merge_files.py to create a combined dataset.

Choose a model:

Run random_forest_prediction.py for Random Forest predictions.

Run xgboost_prediction.py for XGBoost predictions.

Run cnn_bilstm_prediction.py for deep learning predictions.

Check the terminal for results and predicted emotions!

Why This Project? 🤔
Understanding emotions through wearable sensors can improve health monitoring, mental well-being apps, and human-computer interactions. This project shows how simple signals like BVP and GSR can help decode emotional states using AI.

Future Improvements 🌟
Include more physiological signals for better accuracy.

Optimize deep learning models for real-time prediction.

Deploy as a web or mobile app for easy user interaction.
