
# compares epsilon decay rate
# exponential decay

import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_epsilon = 1.0
epsilon_min = 0.1
decay_rates = [0.9, 0.95, 0.97, 0.99]
steps = np.arange(500)

# Compute epsilon decay curves
decay_curves = {}

for rate in decay_rates:
    epsilon = initial_epsilon
    values = []
    for _ in steps:
        epsilon = max(epsilon_min, epsilon * rate)
        values.append(epsilon)
    decay_curves[rate] = values

# Plotting
plt.figure(figsize=(10, 6))
for rate, values in decay_curves.items():
    plt.plot(steps, values, label=f"Decay Rate = {rate}")

plt.axhline(y=epsilon_min, color='r', linestyle='--', label='EPSILON_MIN')
plt.title("Epsilon Decay over 500 Steps")
plt.xlabel("Steps")
plt.ylabel("Epsilon")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
