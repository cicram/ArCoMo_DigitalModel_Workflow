import matplotlib.pyplot as plt
import numpy as np

# Example data and regression models
np.random.seed(0)
x = np.linspace(0, 10, 100)
models = [np.poly1d(np.polyfit(x, np.random.normal(x * i, 1, 100), 1)) for i in range(1, 16)]

# Calculate mean and standard deviation of predictions
mean_predictions = np.mean([model(x) for model in models], axis=0)
std_predictions = np.std([model(x) for model in models], axis=0)

plt.figure(figsize=(12, 8))

# Plot individual regression lines with transparency
colors = plt.cm.viridis(np.linspace(0, 1, 15))
for idx, model in enumerate(models):
    plt.plot(x, model(x), color=colors[idx], alpha=0.3, linewidth=1)

# Plot mean regression line
plt.plot(x, mean_predictions, color='blue', linewidth=2.5, label='Mean Regression Line')

# Plot confidence interval
plt.fill_between(x, mean_predictions - std_predictions, mean_predictions + std_predictions,
                 color='blue', alpha=0.2, label='Confidence Interval (±1 SD)')

# Customize plot
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.title('Regression Models with Mean and Confidence Interval', fontsize=16)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Show plot
plt.show()
