import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATASET_PATH = "dataset_raw"
OUTPUT_PATH = "dataset_landmarks"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing {label}...")

    data = []

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []

                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x - wrist_x)
                    landmarks.append(lm.y - wrist_y)

                data.append(landmarks)

    if len(data) > 0:
        np.save(os.path.join(OUTPUT_PATH, f"{label}.npy"), np.array(data))
        print(f"Saved {label} with {len(data)} samples")
    else:
        print(f"No valid hands detected for {label}")

print("Landmark extraction complete.")