from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

app = Flask(__name__)

model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=15)
latest_prediction = "No Hand"
latest_confidence = 0
current_sentence = ""

def generate_frames():
    global latest_prediction, latest_confidence

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x - wrist_x)
                    landmarks.append(lm.y - wrist_y)

                prediction = model.predict([landmarks])[0]
                confidence = np.max(model.predict_proba([landmarks]))

                prediction_buffer.append(prediction)

                latest_prediction = Counter(prediction_buffer).most_common(1)[0][0]
                latest_confidence = round(float(confidence) * 100, 2)

                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction')
def prediction():
    return jsonify({
        "prediction": latest_prediction,
        "confidence": latest_confidence
    })


@app.route('/append')
def append_letter():
    global current_sentence
    if latest_prediction != "No Hand":
        current_sentence += latest_prediction
    return jsonify({"sentence": current_sentence})


@app.route('/space')
def add_space():
    global current_sentence
    current_sentence += " "
    return jsonify({"sentence": current_sentence})


@app.route('/clear')
def clear_sentence():
    global current_sentence
    current_sentence = ""
    return jsonify({"sentence": current_sentence})


@app.route('/sentence')
def get_sentence():
    return jsonify({"sentence": current_sentence})


if __name__ == "__main__":
    app.run(debug=True)