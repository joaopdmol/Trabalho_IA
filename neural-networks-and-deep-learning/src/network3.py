import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import gzip

# Funções de ativação
def linear(z): return z
def ReLU(z): return F.relu(z)
def sigmoid(z): return torch.sigmoid(z)
def tanh(z): return torch.tanh(z)

# Constantes
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

# Função para carregar os dados do MNIST
def load_data_shared(filename="../data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    def shared(data):
        shared_x = torch.tensor(data[0], dtype=torch.float32).to(DEVICE)
        shared_y = torch.tensor(data[1], dtype=torch.int64).to(DEVICE)
        return shared_x, shared_y
    
    return [shared(training_data), shared(validation_data), shared(test_data)]

class Network(nn.Module):
    def __init__(self, layers):
        super(Network, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        train_loader = DataLoader(TensorDataset(training_x, training_y), batch_size=mini_batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(validation_x, validation_y), batch_size=mini_batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=mini_batch_size, shuffle=False)

        optimizer = optim.SGD(self.parameters(), lr=eta, weight_decay=lmbda)

        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            self.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                x_batch = x_batch.view(x_batch.size(0), 1, 28, 28)  # Redimensiona os dados de entrada
                output = self.forward(x_batch)
                loss = F.cross_entropy(output, y_batch)
                loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                validation_accuracy = self.evaluate(val_loader)
                print(f"Epoch {epoch}: validation accuracy {validation_accuracy:.2%}")
                if validation_accuracy >= best_validation_accuracy:
                    print("This is the best validation accuracy to date.")
                    best_validation_accuracy = validation_accuracy
                    if test_data:
                        test_accuracy = self.evaluate(test_loader)
                        print(f'The corresponding test accuracy is {test_accuracy:.2%}')
        
        print("Finished training network.")
        print(f"Best validation accuracy of {best_validation_accuracy:.2%}")
        if test_data:
            print(f"Corresponding test accuracy of {test_accuracy:.2%}")

    def evaluate(self, data_loader):
        correct = 0
        total = 0
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.view(x_batch.size(0), 1, 28, 28)  # Redimensiona os dados de entrada
            output = self.forward(x_batch)
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        return correct / total

class ConvPoolLayer(nn.Module):
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid):
        super(ConvPoolLayer, self).__init__()
        self.activation_fn = activation_fn
        self.conv = nn.Conv2d(in_channels=filter_shape[1], out_channels=filter_shape[0], 
                              kernel_size=filter_shape[2:])
        self.pool = nn.MaxPool2d(kernel_size=poolsize)

    def forward(self, x):
        return self.activation_fn(self.pool(self.conv(x)))

class FullyConnectedLayer(nn.Module):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        super(FullyConnectedLayer, self).__init__()
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.dropout(x)
        return self.activation_fn(self.fc(x))

class SoftmaxLayer(nn.Module):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        super(SoftmaxLayer, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.dropout(x)
        return F.log_softmax(self.fc(x), dim=1)

    def cost(self, output, target):
        return F.nll_loss(output, target)

    def accuracy(self, output, target):
        pred = torch.argmax(output, dim=1)
        return (pred == target).float().mean().item()

def size(data):
    return data[0].size(0)

# Exemplo de treinamento
training_data, validation_data, test_data = load_data_shared()

net = Network([
    ConvPoolLayer(filter_shape=(20, 1, 5, 5), image_shape=(10, 1, 28, 28), poolsize=(2, 2), activation_fn=ReLU),
    FullyConnectedLayer(n_in=20*12*12, n_out=100, activation_fn=ReLU),
    SoftmaxLayer(n_in=100, n_out=10)
]).to(DEVICE)

net.SGD(training_data, epochs=60, mini_batch_size=10, eta=0.1, validation_data=validation_data, test_data=test_data)
