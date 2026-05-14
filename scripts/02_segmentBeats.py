import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure directories exist
os.makedirs('plots/segments', exist_ok=True)

print("🔪 Slicing Record 100 into individual heartbeats...")
record_path = 'data/mitdb/100'

# Load record and annotations
record = wfdb.rdrecord(record_path)
ann = wfdb.rdann(record_path, 'atr')

# Lead II is our focus
signal = record.p_signal[:, 0]
peaks = ann.sample  # These are the indices of the R-peaks

# We'll take a window of 180 samples around the peak 
# (Total 360 samples = 1 second of data at 360Hz)
window = 180 
heartbeats = []

for i in range(1, len(peaks) - 1):
    peak = peaks[i]
    # 'N' = Normal Sinus Rhythm. Let's stick to the 'healthy' ones for the GAN first.
    if ann.symbol[i] == 'N':
        # Check if we have enough room to slice
        if peak > window and peak < (len(signal) - window):
            segment = signal[peak - window : peak + window]
            heartbeats.append(segment)

print(f"✅ Extracted {len(heartbeats)} normal heartbeats.")

# Visualize the first 3 slices to check alignment
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(heartbeats[i])
    plt.title(f"Beat {i+1} (Centered)")
plt.tight_layout()
plt.savefig("plots/segments/check_alignment.png")
