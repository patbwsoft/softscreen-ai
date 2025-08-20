import os
import json
import librosa
import numpy as np
from pydub import AudioSegment
import joblib

# === Load classifier ===
clf = joblib.load("audio_classifier.pkl")

# === Settings ===
audio_path = "Human/human_example1.wav"  # You can replace this later with any test file
target_folder = "Outputs"
os.makedirs(target_folder, exist_ok=True)

threshold = -30.0  # dBFS
chunk_duration = 0.2  # seconds
hold_duration = 0.8  # seconds to extend mute after spike
audio = AudioSegment.from_wav(audio_path)
sample_rate = audio.frame_rate

chunks = [audio[i:i + int(chunk_duration * 1000)] for i in range(0, len(audio), int(chunk_duration * 1000))]
softened = []
spikes = []

last_spike_time = -999

for i in range(len(chunks)):
    chunk = chunks[i]
    t = i * chunk_duration
    volume = chunk.dBFS

    if volume > threshold:
        spikes.append(t)
        print(f"[{t:.2f}s] SPIKE Detected: {volume:.2f} dBFS")
        last_spike_time = t

        # Predict label
        audio_chunk = chunk.get_array_of_samples()
        np_chunk = np.array(audio_chunk).astype(np.float32)
        try:
            mfcc = librosa.feature.mfcc(y=np_chunk, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
            label = clf.predict(mfcc_mean)[0]
            print(f"[{t:.2f}s] Predicted: {label}")
        except Exception as e:
            label = "Unknown"
            print(f"[{t:.2f}s] Prediction failed: {e}")

        softened.append(chunk - 30)
    elif t < last_spike_time + hold_duration:
        softened.append(chunk - 30)
    else:
        softened.append(chunk)

# === Export blurred version ===
output_audio = sum(softened)
output_audio.export(os.path.join(target_folder, "blurred_output.wav"), format="wav")

# === Save timestamps ===
with open(os.path.join(target_folder, "spikes.json"), "w") as f:
    json.dump(spikes, f, indent=2)

print(f"\n Blurred audio saved to: {target_folder}/blurred_output.wav")
print(f"Spike timestamps saved to: {target_folder}/spikes.json")
