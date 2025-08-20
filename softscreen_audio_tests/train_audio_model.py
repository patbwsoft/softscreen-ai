import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Step 1: Define categories
categories = ['Bark', 'Doorbell', 'Gunshot', 'Explosion', 'Baby Cry', 'Weather', 'Human']
base_path = os.getcwd()

X = []
y = []

# Step 2: Feature extraction
for label in categories:
    folder = os.path.join(base_path, label)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            filepath = os.path.join(folder, file)
            try:
                audio, sr = librosa.load(filepath, sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                X.append(mfcc_mean)
                y.append(label)
            except Exception as e:
                print(f"Failed on {file}: {e}")

# Step 3: Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Step 4: Save model
joblib.dump(clf, "audio_classifier.pkl")
print("Model retrained and saved as audio_classifier.pkl")