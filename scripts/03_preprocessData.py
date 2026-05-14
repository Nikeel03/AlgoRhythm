import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs('data/processed', exist_ok=True)
os.makedirs('plots/processed', exist_ok=True)

print("⚙️ Loading and normalizing heartbeats...")
record_path = 'data/mitdb/100'
record = wfdb.rdrecord(record_path)
ann = wfdb.rdann(record_path, 'atr')

signal = record.p_signal[:, 0]
peaks = ann.sample
window = 180 

heartbeats = []

# Slicing loop
for i in range(1, len(peaks) - 1):
    peak = peaks[i]
    if ann.symbol[i] == 'N':
        if peak > window and peak < (len(signal) - window):
            segment = signal[peak - window : peak + window]
            heartbeats.append(segment)

# Convert to a numpy array for math operations
X = np.array(heartbeats)
print(f"Initial shape: {X.shape}") # Should be (N_beats, 360)

# --- NORMALIZATION ---
# Min-Max scaling to map values between -1 and 1
X_min = np.min(X, axis=1, keepdims=True)
X_max = np.max(X, axis=1, keepdims=True)

# Avoid division by zero in flatlines
X_norm = 2 * ((X - X_min) / (X_max - X_min + 1e-8)) - 1

print(f"Normalized range: [{np.min(X_norm):.2f}, {np.max(X_norm):.2f}]")

# Save the raw numpy array so we don't have to re-slice every time
np.save('data/processed/record100_normalized.npy', X_norm)
print("✅ Saved normalized dataset to data/processed/record100_normalized.npy")

# Plot a check to ensure we didn't distort the shape
plt.figure(figsize=(6, 4))
plt.plot(X_norm[0])
plt.title("Normalized Beat (Values between -1 and 1)")
plt.savefig("plots/processed/norm_check.png")
