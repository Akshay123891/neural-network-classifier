import numpy as np
import matplotlib.pyplot as plt

# Binary patterns (5x5) for A, B, C
A = np.array([
    [0,1,1,1,0],
    [1,0,0,0,1],
    [1,1,1,1,1],
    [1,0,0,0,1],
    [1,0,0,0,1]
]).reshape(-1, 1)

B = np.array([
    [1,1,1,0,0],
    [1,0,0,1,0],
    [1,1,1,0,0],
    [1,0,0,1,0],
    [1,1,1,0,0]
]).reshape(-1, 1)

C = np.array([
    [0,1,1,1,1],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [0,1,1,1,1]
]).reshape(-1, 1)

# Labels: A=0, B=1, C=2 (one-hot encoded)
X = np.hstack([A, B, C])
Y = np.array([
    [1, 0, 0],  # A
    [0, 1, 0],  # B
    [0, 0, 1]   # C
]).T

# Sigmoid activation and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Softmax for output layer
def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / e_z.sum(axis=0)

# Cross-entropy loss
def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[1]

# Accuracy
def accuracy(y_pred, y_true):
    pred_labels = np.argmax(y_pred, axis=0)
    true_labels = np.argmax(y_true, axis=0)
    return np.mean(pred_labels == true_labels)

# Initialize weights and biases
np.random.seed(0)
input_size = 25
hidden_size = 16
output_size = 3
lr = 0.5
epochs = 1000

W1 = np.random.randn(hidden_size, input_size) * 0.1
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * 0.1
b2 = np.zeros((output_size, 1))

losses = []
accuracies = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    # Loss and accuracy
    loss = cross_entropy(A2, Y)
    acc = accuracy(A2, Y)
    losses.append(loss)
    accuracies.append(acc)

    # Backpropagation
    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T / X.shape[1]
    db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = dZ1 @ X.T / X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]

    # Update weights
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # Print occasionally
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.2f}")

# Plot loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()

# Test and visualize predictions
def predict_and_show(image, true_label):
    Z1 = W1 @ image + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    pred = np.argmax(A2)
    labels = ['A', 'B', 'C']
    
    plt.imshow(image.reshape(5, 5), cmap='gray')
    plt.title(f"Predicted: {labels[pred]}, True: {labels[true_label]}")
    plt.axis('off')
    plt.show()

# Testing
predict_and_show(A, 0)
predict_and_show(B, 1)
predict_and_show(C, 2)
