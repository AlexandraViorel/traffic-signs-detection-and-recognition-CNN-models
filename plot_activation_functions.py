import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()

x = np.linspace(-10, 10, 400)
z = np.linspace(-10, 10, 400)  

y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_softmax = softmax(z)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(x, y_sigmoid, label="Sigmoid", color='blue')
axs[0, 0].set_title("(a) Sigmoid")
axs[0, 0].set_xlabel(r"$\sigma(x) = \frac{1}{1+e^{-x}}$", fontsize=15)
axs[0, 0].grid(True, color='grey', linestyle='-', linewidth=0.5)
axs[0, 0].set_xlim(-10, 10)  


axs[0, 1].plot(x, y_tanh, label="Tanh", color='blue')
axs[0, 1].set_title("(b) Tanh")
axs[0, 1].set_xlabel(r"$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$", fontsize=15)
axs[0, 1].grid(True, color='grey', linestyle='-', linewidth=0.5)
axs[0, 1].set_xlim(-10, 10)  


axs[1, 1].plot(x, y_relu, label="ReLU", color='blue')
axs[1, 1].set_title("(c) ReLU")
axs[1, 1].set_xlabel(r"$\text{ReLU}(x) = \max(0, x)$", fontsize=15)
axs[1, 1].grid(True, color='grey', linestyle='-', linewidth=0.5)
axs[1, 1].set_xlim(-10, 10) 


fig.delaxes(axs[1, 0])
plt.tight_layout(pad=3.0)
plt.show()
