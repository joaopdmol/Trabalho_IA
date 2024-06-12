#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost:
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``."""
        return 0.5 * torch.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``."""
        return torch.sum(-y * torch.log(a) - (1 - y) * torch.log(1 - a))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y)


#### Main Network class
class Network(nn.Module):
    def __init__(self, sizes, cost=CrossEntropyCost):
        super(Network, self).__init__()
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """Initialize weights using a Gaussian distribution with mean 0 and std deviation 1/sqrt(n)."""
        self.biases = [torch.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [torch.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def forward(self, x):
        """Return the output of the network if ``x`` is input."""
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(torch.mm(w, x) + b)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0, evaluation_data=None,
            monitor_evaluation_cost=False, monitor_evaluation_accuracy=False, monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient descent."""
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        optimizer = optim.SGD(self.parameters(), lr=eta)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(optimizer, mini_batch, lmbda, len(training_data))
            print(f"Epoch {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            print()
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, optimizer, mini_batch, lmbda, n):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch."""
        inputs = torch.cat([x.view(-1, 1) for x, _ in mini_batch], dim=1)
        targets = torch.cat([y.view(-1, 1) for _, y in mini_batch], dim=1)
        optimizer.zero_grad()
        outputs = self.forward(inputs)
        cost = self.cost.fn(outputs, targets)
        cost.backward()
        optimizer.step()

    def accuracy(self, data):
        """Return the number of inputs in ``data`` for which the neural network outputs the correct result."""
        inputs = torch.cat([x.view(-1, 1) for x, _ in data], dim=1)
        targets = torch.tensor([y for _, y in data])
        outputs = self.forward(inputs)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        return correct

    def total_cost(self, data, lmbda):
        """Return the total cost for the data set ``data``."""
        inputs = torch.cat([x.view(-1, 1) for x, _ in data], dim=1)
        targets = torch.cat([y.view(-1, 1) for _, y in data], dim=1)
        outputs = self.forward(inputs)
        cost = self.cost.fn(outputs, targets)
        return cost.item()

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes, "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases], "cost": str(self.cost.__name__)}
        with open(filename, "w") as f:
            json.dump(data, f)


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``. Returns an instance of Network."""
    with open(filename, "r") as f:
        data = json.load(f)
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [torch.tensor(w) for w in data["weights"]]
    net.biases = [torch.tensor(b) for b in data["biases"]]
    return net


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + torch.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
