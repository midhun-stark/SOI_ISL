 🖐️ SOI_ISL – Real-Time Indian Sign Language Recognition Web App

A real-time Indian Sign Language (ISL) recognition system built using Computer Vision and Machine Learning.

This application detects hand gestures through a webcam, classifies ISL alphabets (A–Z), and converts them into text with sentence-building and speech output functionality.

---

 🚀 Features

- 🎥 Real-time webcam streaming
- ✋ Hand landmark detection using MediaPipe
- 🔤 ISL Alphabet Recognition (A–Z)
- 📊 99%+ validation accuracy
- 🧠 Prediction smoothing for stability
- 📝 Sentence builder interface
- 🔊 Text-to-speech output
- 💻 Clean and responsive web UI (TailwindCSS)

---

 🧠 Model Overview

- **Total Samples:** 24,928
- **Features:** 42 landmark features (x, y coordinates of 21 hand points)
- **Algorithm:** RandomForest Classifier
- **Validation Accuracy:** ~99.4%
- **Preprocessing:** Wrist-relative landmark normalization

Instead of using raw images, the system extracts 21 hand landmarks using MediaPipe and trains a classifier on the normalized coordinate data. This makes the model lightweight, fast, and robust.

---

 🏗️ System Architecture


Webcam → MediaPipe Hand Landmarks → Feature Extraction → RandomForest Model → Flask Backend → Web UI


---

 📂 Project Structure


SOI_ISL/
│
├── app.py # Flask backend (real-time prediction + streaming)
├── train.py # Model training script
├── extract_landmarks.py # Dataset preprocessing script
├── templates/
│ └── index.html # Frontend UI
├── .gitignore
└── README.md


> Dataset and trained model file are not included in this repository.

---

 🛠️ Installation & Setup

 1️⃣ Clone Repository


git clone https://github.com/YOUR_USERNAME/SOI_ISL.git

cd SOI_ISL


 2️⃣ Create Virtual Environment (Recommended)


python -m venv venv
venv\Scripts\activate


 3️⃣ Install Dependencies


pip install -r requirements.txt


If `requirements.txt` is not available:


pip install flask mediapipe numpy opencv-contrib-python scikit-learn joblib


---

 🧪 Model Training

If training from scratch:

 Step 1 – Extract Landmarks

python extract_landmarks.py


 Step 2 – Train Model

python train.py


This generates:

model.pkl


---

 ▶️ Run Application


python app.py


Open in browser:

http://127.0.0.1:5000


---

 📌 How It Works

1. Webcam captures real-time video.
2. MediaPipe detects 21 hand landmarks.
3. Coordinates are normalized relative to wrist.
4. Trained RandomForest model predicts ISL alphabet.
5. Prediction smoothing ensures stability.
6. User can build sentences and convert to speech.

---

 ⚠️ Limitations

- Designed for static ISL alphabet gestures (A–Z).
- Performance may vary with lighting and camera quality.
- Dynamic ISL gestures are not supported in current version.

---

 🔮 Future Improvements

- Dynamic gesture recognition (LSTM / Temporal models)
- Word-level prediction
- Mobile application version
- Cloud deployment
- Multi-language speech output
- Gesture confidence visualization

---

 📜 License

This project is developed for educational and assistive technology purposes.

---





