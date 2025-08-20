from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio = AudioSegment.from_file("bark_example.wav")

# === Config ===
CHUNK_LENGTH_MS = 200
VOLUME_THRESHOLD = -20.0

# Break audio into chunks
chunks = make_chunks(audio, CHUNK_LENGTH_MS)

# Detect spike timestamps
spike_times = []
for i, chunk in enumerate(chunks):
    volume = chunk.dBFS
    timestamp = i * (CHUNK_LENGTH_MS / 1000)
    if volume >= VOLUME_THRESHOLD:
        spike_times.append(timestamp)
        print(f"[{timestamp:.2f}s] SPIKE: {volume:.2f} dBFS")

# === Plot the waveform ===
samples = np.array(audio.get_array_of_samples())
channels = audio.channels
if channels == 2:
    samples = samples.reshape((-1, 2))
    samples = samples.mean(axis=1)  # Convert stereo to mono

times = np.linspace(0, len(audio) / 1000, num=len(samples))

plt.figure(figsize=(12, 4))
plt.plot(times, samples, label='Audio waveform', color='blue')

# Draw red lines at spike times
for ts in spike_times:
    plt.axvline(x=ts, color='red', linestyle='--', alpha=0.6)

plt.title("Waveform with Spike Detection")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
