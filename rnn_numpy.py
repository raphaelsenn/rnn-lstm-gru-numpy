import numpy as np

np.random.seed(42)


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def one_hot(y: np.ndarray) -> np.ndarray:
    y_hot = np.zeros((len(y), 10))
    y_hot[range(y.shape[0]), y.astype(int)] = 1
    return y_hot

class RNN:
    """
    Simple L-layer recurrent neural network in numpy.
    
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            seq_len: int,
            output_size: int,
            ) -> None:

        self.layers = 2
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        
        # initialize weights
        self.weight_ih_l0 = np.random.uniform(-1/input_size**0.5, 1/input_size**0.5, size=(hidden_size, input_size))
        self.weight_hh_l0 = np.random.uniform(-1/hidden_size**0.5, 1/hidden_size**0.5, size=(hidden_size, hidden_size))
        
        self.weight_ih_l1 = np.random.uniform(-1/hidden_size**0.5, 1/hidden_size**0.5, size=(hidden_size, hidden_size))
        self.weight_hh_l1 = np.random.uniform(-1/hidden_size**0.5, 1/hidden_size**0.5, size=(hidden_size, hidden_size))

        self.weight_l3 = np.random.uniform(-1/(hidden_size*seq_len)**0.5, 1/(hidden_size*seq_len)**0.5, size=(output_size, hidden_size*seq_len))

        # initialize biases

    def forward(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0] 
        y_hats = []
        batch_size = 64
        for i in range(0, N, batch_size):
            x = X[i:i+batch_size]
            x = x.transpose(1, 0, 2)
            batch_size = x.shape[1]
            h_t = np.zeros((2, batch_size, self.hidden_size))
            h_t_minus_1 = h_t
            output = []

            for t in range(self.seq_len):
                h_t[0] = tanh(x[t] @ self.weight_ih_l0.T + h_t_minus_1[0] @ self.weight_hh_l0.T)
                h_t[1] = tanh(h_t[0] @ self.weight_ih_l1 + h_t_minus_1[1] @ self.weight_hh_l1.T)
                output.append(h_t[-1].copy())
                h_t_minus_1 = h_t 
            output = np.stack(output, axis=0)
            output = output.transpose(1, 0, 2)

            h3 = output.reshape(batch_size, -1)
            z3 = h3 @ self.weight_l3.T
            y_hat = softmax(z3)
            y_hats.append(y_hat)
        return np.concatenate(y_hats, axis=0)

    def fit(self, X_train, y_train) -> None:
        epochs = 9
        N = X_train.shape[0]
        batch_size = 128 
        learning_rate = 0.00001
        for epoch in range(epochs):
            for i in range(0, N, batch_size):
                x, y = X_train[i:i+batch_size], y_train[i:i+batch_size]

                # forward propagation through time t

                # swap batch_size and seq_len
                x = x.transpose(1, 0, 2)
                batch_size = x.shape[1]
                h_t = np.zeros((2, batch_size, self.hidden_size))
                h_t_minus_1 = h_t
                output = []
                h_t_l0 = []
                h_t_l1 = []
                for t in range(self.seq_len):
                    h_t[0] = tanh(x[t] @ self.weight_ih_l0.T + h_t_minus_1[0] @ self.weight_hh_l0.T)    # [N, hidden_size]
                    h_t[1] = tanh(h_t[0] @ self.weight_ih_l1.T + h_t_minus_1[1] @ self.weight_hh_l1.T)    # [N, hidden_size]
                    output.append(h_t[-1].copy())
                    h_t_l0.append(h_t[0].copy()) 
                    h_t_l1.append(h_t[1].copy()) 
                    h_t_minus_1 = h_t.copy() 
                output = np.stack(output, axis=0)
                output = output.transpose(1, 0, 2)  # [N, seq_len, hidden_size]
                t3 = output.reshape(batch_size, -1) # [N, hidden_size * seq_len]
                z3 = t3 @ self.weight_l3.T  # [N, 10]
                h3 = softmax(z3)    # [N, 10]
                y_hot = one_hot(y)  # [N, 10]

                # backpropagation

                # last layer
                dz3 = (h3 - y_hot) / batch_size                # [N, 10]
                dz3_dw_l3 =  dz3.T @ t3         # [output_size, hidden_size * seq_len]

                # backpropagate throw time t
                output = output.transpose(1, 0, 2) 
                V = dz3_dw_l3.reshape(self.seq_len, self.output_size, self.hidden_size)
                dh_t = np.zeros_like(output)
                # print(output.shape)
                # backpropagate throw time t
                dht_dw_hh_l1 = 0
                dht_dw_ih_l1 = 0
                dht_dw_hh_l0 = 0
                dht_dw_ih_l0 = 0
                for t in range(self.seq_len - 1, 1, -1):
                    dh_next = dh_t[t+1]  if t < self.seq_len - 1 else 0

                    dht_dz2_l0 = 1 - h_t_l0[t] ** 2     # [N, hidden_size]
                    dht_dz2_l1 = 1 - h_t_l1[t] ** 2     # [N, hidden_size]


                    dz2_dw_hh_l1 = h_t_l1[t - 1]
                    dht_dw_hh_l1 += dz2_dw_hh_l1.T @ dht_dz2_l1

                    dz2_dw_ih_l1 = h_t_l0[t]
                    dht_dw_ih_l1 += dz2_dw_ih_l1.T @ dht_dz2_l1


                    dz1_dw_hh_l0 = h_t_l0[t-1]
                    dz1_dw_ih_l0 = x[t]

                    dht_dw_hh_l0 += dz1_dw_hh_l0.T @ dht_dz2_l0
                    dht_dw_ih_l0 += dz1_dw_ih_l0.T @ dht_dz2_l0

                    
                    # print(dht_l1_dw_ih.shape)

                    # self.weight_hh_l1 = self.weight_hh_l1 - learning_rate * dht_dw_hh_l1

                    # print(dht_l0.shape, dht_l1.shape)

                # update parameters
                self.weight_l3 = self.weight_l3 - learning_rate * dz3_dw_l3
                
                self.weight_hh_l1 = self.weight_hh_l1 - learning_rate * dht_dw_hh_l1
                self.weight_ih_l1 = self.weight_ih_l1 - learning_rate * dht_dw_ih_l1


                self.weight_hh_l0 = self.weight_hh_l0 - learning_rate * dht_dw_hh_l0
                self.weight_ih_l0 = self.weight_ih_l0 - learning_rate * (dht_dw_ih_l0.T)


            verbose = True
            if verbose:
                y_pred = self.forward(X_train)
                acc = (np.argmax(y_pred, axis=1) == y_train).sum() / X_train.shape[0]
                print(f'epoch: {epoch}\tacc: {acc}')



rnn = RNN(input_size=28, hidden_size=64, seq_len=28, output_size=10)
print(rnn)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transform
from torchvision.transforms import Compose
mnist_train  = torchvision.datasets.MNIST(
    root='./MNIST/',
    train=True,
    transform=transform.ToTensor(),
    download=True
)

mnist_test  = torchvision.datasets.MNIST(
    root='./MNIST/',
    train=False,
    transform=transform.ToTensor(),
    download=True
)

dataloader_train = DataLoader(
    dataset=mnist_train,
    batch_size=64,
    shuffle=True)

dataloader_test = DataLoader(
    dataset=mnist_test,
    batch_size=64,
    shuffle=True)


x = np.random.random(size=(10000, 28, 28))
y = np.random.random(size=(10000,))

input = np.random.random(size=(64, 28, 28))
print(input.shape)
# output, hn = rnn.forward(input)
# print(output.shape)


X = mnist_train.data.data.numpy()
Y = mnist_train.targets.data.numpy()
print(X.shape)
print(y.shape)
rnn.fit(X, Y)