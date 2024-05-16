import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the tanh function
def tanh(x):
    return np.tanh(x)

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Define the softmax function for a simple two-dimensional case
def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()

# Generate a range of values from -10 to 10 for plotting
x = np.linspace(-10, 10, 400)
z = np.linspace(-10, 10, 400)  # For softmax

# Calculate the function values
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_softmax = softmax(z)

# Create a figure and axis objects
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Sigmoid Plot
axs[0, 0].plot(x, y_sigmoid, label="Sigmoid", color='blue')
axs[0, 0].set_title("(a) Sigmoid")
axs[0, 0].set_xlabel(r"$\sigma(x) = \frac{1}{1+e^{-x}}$", fontsize=15)
axs[0, 0].grid(True, color='grey', linestyle='-', linewidth=0.5)
axs[0, 0].set_xlim(-10, 10)  # Set x-axis limits


# Tanh Plot
axs[0, 1].plot(x, y_tanh, label="Tanh", color='blue')
axs[0, 1].set_title("(b) Tanh")
axs[0, 1].set_xlabel(r"$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$", fontsize=15)
axs[0, 1].grid(True, color='grey', linestyle='-', linewidth=0.5)
axs[0, 1].set_xlim(-10, 10)  # Set x-axis limits


# ReLU Plot
axs[1, 1].plot(x, y_relu, label="ReLU", color='blue')
axs[1, 1].set_title("(c) ReLU")
axs[1, 1].set_xlabel(r"$\text{ReLU}(x) = \max(0, x)$", fontsize=15)
axs[1, 1].grid(True, color='grey', linestyle='-', linewidth=0.5)
axs[1, 1].set_xlim(-10, 10)  # Set x-axis limits


# Adjust layout and display
fig.delaxes(axs[1, 0])
plt.tight_layout(pad=3.0)
plt.show()
