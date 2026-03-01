import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATA_PATH = "dataset_landmarks"

X = []
y = []

for file in os.listdir(DATA_PATH):
    label = file.split(".")[0]
    data = np.load(os.path.join(DATA_PATH, file))

    for sample in data:
        X.append(sample)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")