# This code simulates a SGD trajectory on a 2D loss landscape and visualizes it with a heatmap plot.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# Define a simple 2D loss function (e.g., a quadratic bowl)
def loss_function(x, y):
    return x**2 + y**2

# Gradient of the loss function
def gradient(x, y):
    return 2*x, 2*y

# Initialize SGD parameters
lr = 0.1
num_steps = 50
x, y = 2.5, 2.5
trajectory = [(x, y)]

# Simulate SGD trajectory
for _ in range(num_steps):
    dx, dy = gradient(x, y)
    x -= lr * dx + np.random.normal(scale=0.1)
    y -= lr * dy + np.random.normal(scale=0.1)
    trajectory.append((x, y))

# Convert trajectory to numpy array
trajectory = np.array(trajectory)

# Create a grid for contour plot
x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = loss_function(X, Y)

# Plot the heatmap and trajectory
plt.figure(figsize=(19.2, 10.8))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='white', label='SGD trajectory')
plt.scatter(trajectory[10, 0], trajectory[10, 1], color='red', marker='x', s=500, linewidth=3, zorder=5)
plt.title('SGD Trajectory on Loss Landscape', fontsize=34, pad=15)
plt.xlabel('x', fontsize=26)
plt.ylabel('y', fontsize=26)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
colorbar = plt.colorbar(contour)
colorbar.set_label('Loss', fontsize=26)
colorbar.ax.tick_params(labelsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.grid(True)

# Auto-save the plot
plt.savefig("plots/sgd_trajectory.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
