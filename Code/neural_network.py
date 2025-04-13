import numpy as np
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation  # 'relu', 'sigmoid', 'tanh', 'softmax'
        
        # Adam optimizer parameters
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.timestep = 1
        
        # Intermediate values
        self.z = None  # Pre-activation
        self.a = None  # Post-activation
        self.input = None

    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.weights) + self.bias
        self.a = self._apply_activation(self.z)
        return self.a

    def backward(self, grad_output, learning_rate):
        grad_z = grad_output * self._activation_derivative()
        grad_weights = np.dot(self.input.T, grad_z)
        grad_bias = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = np.dot(grad_z, self.weights.T)
        
        # Adam update for weights
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (grad_weights ** 2)
        m_hat = self.m_weights / (1 - self.beta1 ** self.timestep)
        v_hat = self.v_weights / (1 - self.beta2 ** self.timestep)
        self.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Adam update for biases
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * grad_bias
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (grad_bias ** 2)
        m_hat = self.m_bias / (1 - self.beta1 ** self.timestep)
        v_hat = self.v_bias / (1 - self.beta2 ** self.timestep)
        self.bias -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.timestep += 1
        return grad_input

    def _apply_activation(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'softmax':
            e_x = np.exp(z - np.max(z, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)
        else:
            return z  # Linear activation

    def _activation_derivative(self):
        if self.activation == 'relu':
            return (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            return self.a * (1 - self.a)
        elif self.activation == 'tanh':
            return 1 - self.a ** 2
        elif self.activation == 'softmax':
            return 1  # Handled via loss gradient
        else:
            return 1  # Linear activation

def categorical_cross_entropy(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

class NeuralNetwork:
    def __init__(self, layer_sizes, activations):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, X, y_true, learning_rate):
        y_pred = self.forward(X)
        if self.layers[-1].activation == 'softmax':
            grad_z = (y_pred - y_true) / X.shape[0]  # Normalize by batch size
        else:
            grad_z = (y_pred - y_true)
        for layer in reversed(self.layers):
            grad_z = layer.backward(grad_z, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        loss_history = []
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = categorical_cross_entropy(y, y_pred)
            self.backward(X, y, learning_rate)
            loss_history.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return loss_history

if __name__ == "__main__":
    # Example 1: Binary classification
    X = np.array([[0.5], [1.5], [2.0], [3.0]])
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])  # One-hot encoded

    # Network architecture: 1 input → 1 hidden (ReLU) → 2 output (Softmax)
    nn = NeuralNetwork(layer_sizes=[1, 1, 2], activations=['relu', 'softmax'])
    loss_history = nn.train(X, y, epochs=1000, learning_rate=0.01)

    # Plot training loss
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()