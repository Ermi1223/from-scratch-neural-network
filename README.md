# Neural Network from Scratch in NumPy

## Project Overview

This project implements a simple 3-layer neural network using **NumPy** and **Python**. The network includes the following features:

- **Dense layers** with weight initialization
- **Forward propagation** and **backpropagation**
- **Gradient descent optimization** using SGD and **Adam**
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax
- **Loss function**: Cross-Entropy Loss
- **Batch processing** for efficient training

The primary goal was to build a foundational neural network without high-level frameworks like TensorFlow or PyTorch.

## Key Features

- **3-Layer Neural Network**: One input layer, one hidden layer, and one output layer, each with a single neuron.
- **Batch Training**: Utilizes efficient matrix operations via NumPy to handle batch inputs.
- **Optimizers**: Implemented both **Stochastic Gradient Descent (SGD)** and **Adam** optimizers.
- **Activation Functions**: Includes ReLU, Sigmoid, Tanh, and Softmax for experimenting with various network behaviors.

## Technologies Used

- **Python** (with NumPy)
- **Matplotlib** (optional for visualizing training curves)
- **Jupyter Notebooks** (for experiments)

## Setup and Requirements

To run this project, ensure you have the following installed:

- Python 3.x
- NumPy
- (Optional) Matplotlib (for visualizations)

To install dependencies, run:

```bash
pip install numpy matplotlib
