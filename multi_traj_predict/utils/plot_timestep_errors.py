import matplotlib.pyplot as plt
import numpy as np

# Example timestep data (1–5)
timesteps = np.arange(1, 6)

# Example Euclidean Distance Errors (replace with your actual results)
ede_values = {
    "Attn Bi-GRU":   [0.576837, 0.644273, 0.784669, 0.930025, 1.079544],
    "Attn Bi-LSTM":  [1.068128, 1.162858, 1.428995, 1.481310, 1.519588],
    "Attn GRU":      [2.180485, 0.615643, 0.930102, 1.308822, 1.633506],
    "Attn LSTM":     [3.554418, 2.511556, 2.255999, 2.280867, 2.337487],
    "GRU (2 Layers)": [2.929544, 3.127545, 3.807928, 4.291709, 4.630111],
    "LSTM (2 Layers)": [2.424339, 2.550485, 2.577543, 2.629100, 2.775114],
    "RNN (2 Layers)":  [5.363140, 6.798547, 6.663338, 8.616770, 7.267049],
    "GRU (1 Layer)":   [0.504184, 0.766910, 1.015590, 1.279893, 1.551766]
}

# Plot setup
plt.figure(figsize=(10, 6))
for model, errors in ede_values.items():
    plt.plot(
        timesteps, errors, marker='o', linewidth=2,
        label=model
    )

# Labels and title
plt.title("Per-Timestep Euclidean Distance Error (EDE) Across Models", fontsize=14, pad=12)
plt.xlabel("Timestep", fontsize=12)
plt.ylabel("Euclidean Distance Error (meters)", fontsize=12)
plt.xticks(timesteps)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Models", fontsize=9, title_fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

# Show or save
plt.show()
# plt.savefig("per_timestep_error_comparison.png", dpi=300, bbox_inches='tight')
