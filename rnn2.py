import numpy as np

np.random.seed(42)


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)


def softmax(x: np.ndarray, derv: bool=False) -> np.ndarray:
    if derv:
        return np.exp(x) * (np.sum(np.exp(x)) - np.exp(x)) / np.sum(np.exp(x))**2 
    
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def one_hot(y: np.ndarray) -> np.ndarray:
    y_hot = np.zeros((len(y), 10))
    y_hot[range(y.shape[0]), y.astype(int)] = 1
    return y_hot


class RNN:
    """
    Simple recurrent neural network in numpy.
    It consists of a 1-layer elman RNN with a fully connected layer on top (connected to all time stemps).
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            seq_len: int,
            output_size: int,
            ) -> None:

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len

        # initialize weights
        self.weight_ih = np.random.uniform(
            -1/input_size**0.5, 1/input_size**0.5,
            size=(hidden_size, input_size))
        
        self.weight_hh = np.random.uniform(
            -1/hidden_size**0.5, 1/hidden_size**0.5,
            size=(hidden_size, hidden_size))
        
        self.weight_out = np.random.uniform(
            -1/(hidden_size*seq_len)**0.5, 1/(hidden_size*seq_len)**0.5,
            size=(output_size, hidden_size*seq_len))

        # initialize biases
        self.bias_ih = np.random.uniform(-1/input_size**0.5, 1/input_size**0.5,
                                         size=hidden_size)

        self.bias_hh = np.random.uniform(-1/input_size**0.5, 1/input_size**0.5,
                                         size=hidden_size)

        self.bias_out = np.random.uniform(-1/hidden_size**0.5, 1/hidden_size**0.5,
                                          size=output_size)

    def fit(
            self,
            X_train,
            y_train,
            epochs: int=10,
            batch_size: int=64,
            learning_rate: float=0.001,
            verbose: bool=True
            ) -> None:
        """
        Simple implementation of stochastic gradient descent (SGD)
        """ 
        N = X_train.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(N)
            X_train, y_train = X_train[permutation], y_train[permutation]
            for i in range(0, N, batch_size):
                # x has shape [N, seq_len, num_features]
                x, y = X_train[i:i+batch_size], y_train[i:i+batch_size]
                x = x.transpose(1, 0, 2)                # [seq_len, N, num_features]
                batch_size = x.shape[1]
                h_t = np.zeros(
                    (batch_size, self.hidden_size)
                    )                                   # [N, hidden_size]
                h_t_minus_1 = h_t                       # [N, hidden_size]
                ht_all = []

                # #############################################################
                # forward propagation through time t

                for t in range(self.seq_len):
                    h_t = tanh(
                        x[t] @ self.weight_ih.T
                        + self.bias_ih
                        + h_t_minus_1 @ self.weight_hh.T
                        + self.bias_hh
                    )                                   # [N, hidden_size]
                    ht_all.append(h_t.copy())
                    h_t_minus_1 = h_t.copy()
                ht_all = np.stack(ht_all, axis=0)       # [seq_len, N, hidden_size]
                ht_all = ht_all.transpose(1, 0, 2)      # [N, seq_len, hidden_size]

                # propagating data through the last layer
                H = ht_all.reshape(batch_size, -1)      # [N, hidden_size * seq_len]
                z2 = H @ self.weight_out.T              # [N, hidden_size]
                h2 = softmax(z2)                        # [N, hidden_size] 
                y_hot = one_hot(y)                      # [N, hidden_size]

                # #############################################################
                # backpropagation

                # last layer
                d_out = (h2 - y_hot) / batch_size         # [N, hidden_size]
                d_out_dw =  d_out.T @ H                   # [output_size, hidden_size * seq_len]

                # [output_size, seq_len * hidden_size] -> [output_size, seq_len, hidden_size] 
                W_out = self.weight_out.reshape(
                    self.output_size,
                    self.seq_len,
                    self.hidden_size)                   # [output_size, seq_len, hidden_size]
                ht_all = ht_all.transpose(1, 0, 2)        # [N, seq_len, hidden_size]
                
                # initialize gradients
                dht_dwhh = np.zeros_like(self.weight_hh)
                # dht_dwih = np.zeros_like(self.weight_ih)
                dht_dwih = 0

                dht_dbih = 0
                dht_dbhh = 0
                dht_next = np.zeros_like(ht_all[0])

                # propagating gradients through time t
                for t in range(self.seq_len - 1, -1, -1):
                    dz2_dht = W_out[:, t, :]            # [output_size, hidden_size]
                    dht = d_out @ dz2_dht + dht_next    # [N, hidden_size] 10.19 in goodfellow

                    dht_dz1 = 1 - ht_all[t]**2          # [N, hidden_size]
                    dht_next = (dht*dht_dz1) @ self.weight_hh

                    dht_dwhh += ht_all[t-1].T @ (dht * dht_dz1)    # [N, hidden_size]
                    dht_dwih += x[t].T @ (dht * dht_dz1)           # [N, hidden_size]

                    dht_dbih += np.sum(dht * dht_dz1, axis=0)
                    dht_dbhh += np.sum(dht * dht_dz1, axis=0)
                
                # update parameters
                self.weight_out = self.weight_out - learning_rate * d_out_dw
                self.weight_hh = self.weight_hh - learning_rate * dht_dwhh
                self.weight_ih = self.weight_ih - learning_rate * dht_dwih.T
                self.bias_ih = self.bias_ih - learning_rate * dht_dbih
                self.bias_hh = self.bias_hh - learning_rate * dht_dbhh

            if verbose:
                N = X_train.shape[0]
                # calculating cross-entropy loss
                y_pred = self.forward(X_train)
                loss = -sum(np.log(y_pred[np.arange(N), y_train])) / N

                # calculating accuracy
                acc = (np.argmax(y_pred, axis=1) == y_train).sum() / N

                print(f'epoch: {epoch}\tloss: {loss:.04f}\tacc: {acc:.04f}')

    def forward(self,
                X: np.ndarray,
                batch_size: int=64) -> np.ndarray:
        N = X.shape[0]
        y_hats = []
        for i in range(0, N, batch_size):
            x = X[i:i+batch_size]
            x = x.transpose(1, 0, 2)
            batch_size = x.shape[1]
            h_t = np.zeros((batch_size, self.hidden_size))
            h_t_minus_1 = h_t
            ht_all = []

            # forward propagation through time t
            for t in range(self.seq_len):
                h_t = tanh(
                        x[t] @ self.weight_ih.T
                        + self.bias_ih
                        + h_t_minus_1 @ self.weight_hh.T
                        + self.bias_hh
                    )
                ht_all.append(h_t.copy())
                h_t_minus_1 = h_t 
            ht_all = np.stack(ht_all, axis=0)
            ht_all = ht_all.transpose(1, 0, 2)
            H = ht_all.reshape(batch_size, -1)
            z2 = H @ self.weight_out.T

            # make final predictions
            y_hat = softmax(z2)
            y_hats.append(y_hat)
        return np.concatenate(y_hats, axis=0)


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
    shuffle=False)

dataloader_test = DataLoader(
    dataset=mnist_test,
    batch_size=64,
    shuffle=False)


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


# evaluate 
X_test = mnist_test.data.data.numpy()
Y_test = mnist_test.targets.data.numpy()
N = X_test.shape[0]
# calculating cross-entropy loss
y_pred = rnn.forward(X_test)
loss = -sum(np.log(y_pred[np.arange(N), Y_test])) / N

# calculating accuracy
acc = (np.argmax(y_pred, axis=1) == Y_test).sum() / N

print(f'Accuracy on test set: {acc:.04f}')