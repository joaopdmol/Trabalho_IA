import gzip
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np

class Network(nn.Module):
    def __init__(self, sizes):
        super(Network, self).__init__()
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [torch.randn(y, 1) for y in sizes[1:]]
        self.weights = [torch.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(torch.matmul(w, x) + b)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [torch.zeros(b.shape) for b in self.biases]
        nabla_w = [torch.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [torch.zeros(b.shape) for b in self.biases]
        nabla_w = [torch.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = torch.matmul(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = torch.matmul(delta, activations[-2].transpose(0, 1))
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = torch.matmul(self.weights[-l + 1].transpose(0, 1), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = torch.matmul(delta, activations[-l - 1].transpose(0, 1))
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(torch.argmax(self.forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def load_data_shared(filename="../data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    def shared(data):
        shared_x = torch.tensor(data[0], dtype=torch.float32)
        shared_y = torch.tensor(data[1], dtype=torch.int64)
        return shared_x, shared_y
    
    return [shared(training_data), shared(validation_data), shared(test_data)]
