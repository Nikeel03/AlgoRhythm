import torch
import matplotlib.pyplot as plt
from models import Generator
import os

# Setup
LATENT_DIM = 100
ECG_DIM = 360
os.makedirs('plots/synthetic', exist_ok=True)

# 1. Load the trained Generator
model = Generator(LATENT_DIM, ECG_DIM)
model.load_state_dict(torch.load("generator_v1.pth"))
model.eval() # Set to evaluation mode

# 2. Generate 16 random noise vectors
noise = torch.randn(16, LATENT_DIM)

# 3. Predict/Generate
with torch.no_grad():
    synthetic_beats = model(noise).numpy()

# 4. Plot the results in a 4x4 grid
plt.figure(figsize=(12, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.plot(synthetic_beats[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Synthetic #{i+1}")

plt.tight_layout()
plt.suptitle("AlgoRhythm: First Batch of Synthetic Heartbeats", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.savefig("plots/synthetic/batch_v1.png")
print("Synthetic batch saved to plots/synthetic/batch_v1.png")
