import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int=10) -> np.ndarray:
    y_hot = np.zeros((len(y), num_classes))
    y_hot[range(y.shape[0]), y.astype(int)] = 1
    return y_hot


class RNN_1FC_ALL:
    """
    Simple recurrent neural network in numpy - designed for the MNIST dataset.
    It consists of a 1-layer elman RNN with a fully connected layer on top (NOTE: connected to all time stemps).
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
        self.bias_ih = np.random.uniform(
            -1/input_size**0.5, 1/input_size**0.5,
            size=hidden_size)

        self.bias_hh = np.random.uniform(
            -1/input_size**0.5, 1/input_size**0.5,
            size=hidden_size)

        self.bias_out = np.random.uniform(
            -1/hidden_size**0.5, 1/hidden_size**0.5,
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

        Notation of gradient - all with respect to loss L (which is negative log-likelihood):
        d_out refers to dLoss_d_out,
        d_out_dw refers to dLoss_d_out_dw,
        ...
        """ 
        N = X_train.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(N)
            X_train, y_train = X_train[permutation], y_train[permutation]
            for i in range(0, N, batch_size):
                # NOTE: x has shape                     # [N, seq_len, num_features]
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
                    )                                       # [N, hidden_size]
                    ht_all.append(h_t.copy())
                    h_t_minus_1 = h_t.copy()
                ht_all = np.stack(ht_all, axis=0)           # [seq_len, N, hidden_size]
                ht_all = ht_all.transpose(1, 0, 2)          # [N, seq_len, hidden_size]

                # propagating data through the last layer
                H = ht_all.reshape(batch_size, -1)          # [N, hidden_size * seq_len]
                z2 = H @ self.weight_out.T + self.bias_out  # [N, output_size]
                h2 = softmax(z2)                            # [N, output_size]
                y_hot = one_hot(y)                          # [N, output_size]

                # #############################################################
                # backpropagation

                # calculating gradients of the last layer
                d_out = (h2 - y_hot) / batch_size       # [N, hidden_size]
                d_out_dw =  d_out.T @ H                 # [output_size, hidden_size * seq_len]
                d_out_db = np.sum(d_out, axis=0)        # [output_size,]

                # [output_size, seq_len * hidden_size] -> [output_size, seq_len, hidden_size] 
                W_out_t = self.weight_out.reshape(
                    self.output_size,
                    self.seq_len,
                    self.hidden_size)                   # [output_size, seq_len, hidden_size]
                ht_all = ht_all.transpose(1, 0, 2)      # [N, seq_len, hidden_size]
                W_out_t = W_out_t.transpose(1, 0, 2)    # [seq_len, output_size, hidden_size]
                
                # initialize gradients
                dht_dwhh = np.zeros_like(self.weight_hh)
                dht_dwih = np.zeros_like(self.weight_ih)
                dht_dbih = np.zeros_like(self.bias_ih)
                dht_dbhh = np.zeros_like(self.bias_hh)

                dht_plus_1 = np.zeros_like(ht_all[0])
                # propagating gradients through time t (BPTT)
                # NOTE: x[t] has now shape [seq_len, N, num_features]
                for t in range(self.seq_len - 1, -1, -1):
                    dz2_dht = W_out_t[t]                           # [output_size, hidden_size]
                    dht = d_out @ dz2_dht + dht_plus_1             # [N, hidden_size]

                    dht_dz1 = 1 - ht_all[t]**2                     # [N, hidden_size]
                    dht_plus_1 = (dht*dht_dz1) @ self.weight_hh    # [N, hidden_size]

                    dht_dwhh += (dht * dht_dz1).T @ ht_all[t-1] if t > 0 else (dht * dht_dz1).T @ np.zeros_like(ht_all[0])
                    dht_dwih += (dht * dht_dz1).T @ x[t]           # [N, hidden_size]

                    dht_dbih += np.sum(dht * dht_dz1, axis=0)      # [hidden_size,]
                    dht_dbhh += np.sum(dht * dht_dz1, axis=0)      # [hidden_size,]
                
                # update parameters
                self.weight_out = self.weight_out - learning_rate * d_out_dw
                self.weight_hh = self.weight_hh - learning_rate * dht_dwhh
                self.weight_ih = self.weight_ih - learning_rate * dht_dwih
                
                self.bias_out = self.bias_out - learning_rate * d_out_db
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
            z2 = H @ self.weight_out.T + self.bias_out

            # make final predictions
            y_hat = softmax(z2)
            y_hats.append(y_hat)
        return np.concatenate(y_hats, axis=0)


class RNN_1FC:
    """
    Simple recurrent neural network in numpy - designed for the MNIST dataset.
    It consists of a 1-layer elman RNN with a fully connected layer on top.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            ) -> None:
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize weights
        self.weight_ih = np.random.uniform(
            -1/input_size**0.5, 1/input_size**0.5,
            size=(hidden_size, input_size))
        
        self.weight_hh = np.random.uniform(
            -1/hidden_size**0.5, 1/hidden_size**0.5,
            size=(hidden_size, hidden_size))
        
        self.weight_out = np.random.uniform(
            -1/(hidden_size)**0.5, 1/(hidden_size)**0.5,
            size=(output_size, hidden_size))

        # initialize biases
        self.bias_ih = np.random.uniform(
            -1/input_size**0.5, 1/input_size**0.5,
            size=hidden_size)

        self.bias_hh = np.random.uniform(
            -1/input_size**0.5, 1/input_size**0.5,
            size=hidden_size)

        self.bias_out = np.random.uniform(
            -1/hidden_size**0.5, 1/hidden_size**0.5,
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

        Notation of gradient - all with respect to loss L (which is negative log-likelihood):
        d_out refers to dLoss_d_out,
        d_out_dw refers to dLoss_d_out_dw,
        ...
        """ 
        N = X_train.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(N)
            X_train, y_train = X_train[permutation], y_train[permutation]
            for i in range(0, N, batch_size):
                # NOTE: x has shape                     # [N, seq_len, num_features]
                x, y = X_train[i:i+batch_size], y_train[i:i+batch_size]
                x = x.transpose(1, 0, 2)                # [seq_len, N, num_features]
                batch_size = x.shape[1]
                seq_len = x.shape[0]
                h_t = np.zeros(
                    (batch_size, self.hidden_size)
                    )                                   # [N, hidden_size]
                h_t_minus_1 = h_t                       # [N, hidden_size]
                ht_all = []
                # #############################################################
                # forward propagation through time t
                for t in range(seq_len):
                    h_t = tanh(
                        x[t] @ self.weight_ih.T
                        + self.bias_ih
                        + h_t_minus_1 @ self.weight_hh.T
                        + self.bias_hh
                    )                                           # [N, hidden_size]
                    ht_all.append(h_t.copy())
                    h_t_minus_1 = h_t.copy()
                ht_all = np.stack(ht_all, axis=0)               # [seq_len, N, hidden_size]

                # final output - the prediction 
                z2 = h_t @ self.weight_out.T + self.bias_out    # [N, output_siz]
                
                # #############################################################
                # backpropagation
                dL_dz2 = -2 * (y - z2) / batch_size     # [N, output_size]
                dz2_dw_out = h_t                        # [N, hidden_size]
                dL_dw_out = dL_dz2.T @ dz2_dw_out       # [output_size, hidden_size]
                dL_db_out = np.sum(dL_dz2, axis=0)      # [hidden_size,]

                dz2_dh_tau = self.weight_out            # [output_size, hidden_size]
                dL_dh_tau = dL_dz2 @ dz2_dh_tau         # [N, hidden_size]
                dL_dht_plus_1 = dL_dh_tau               # [N, hidden_size]

                # initialize gradients
                dL_dwhh = np.zeros_like(self.weight_hh)
                dL_dwih = np.zeros_like(self.weight_ih)
                dL_dbih = np.zeros_like(self.bias_ih)
                dL_dbhh = np.zeros_like(self.bias_hh)
                
                # propagating gradients through time t (BPTT)
                # NOTE: x[t] has now shape [seq_len, N, num_features]
                for t in reversed(range(seq_len)):
                    dht_dz2 = 1 - ht_all[t]**2          # tanh derivative
                    dL_dht = dht_dz2 * dL_dht_plus_1
                    
                    dL_dwhh += dL_dht.T @ ht_all[t-1] if t > 0 else dL_dht.T @ np.zeros_like(ht_all[0])
                    dL_dwih += dL_dht.T @ x[t]
                    dL_dbih += np.sum(dL_dht, axis=0)
                    dL_dbhh += np.sum(dL_dht, axis=0)
                    
                    dL_dht_plus_1 = dL_dht @ self.weight_hh

                # update parameters
                self.weight_out = self.weight_out - learning_rate * dL_dw_out
                self.weight_hh = self.weight_hh - learning_rate * dL_dwhh
                self.weight_ih = self.weight_ih - learning_rate * dL_dwih
                
                self.bias_out = self.bias_out - learning_rate * dL_db_out
                self.bias_ih = self.bias_ih - learning_rate * dL_dbih
                self.bias_hh = self.bias_hh - learning_rate * dL_dbhh
            
            if verbose:
                N = X_train.shape[0]
                # calculating cross-entropy loss
                y_pred = self.forward(X_train)
                loss = np.mean((y_train - y_pred)**2)

                # calculating accuracy
                print(f'epoch: {epoch}\tloss: {loss:.04f}')
 
    def forward(self,
                X: np.ndarray,
                batch_size: int=64) -> np.ndarray:
        N = X.shape[0]
        pred_all = [] 
        for i in range(0, N, batch_size):
            # NOTE: x has shape                     # [N, seq_len, num_features]
            x = X[i:i+batch_size]
            x = x.transpose(1, 0, 2)                # [seq_len, N, num_features]
            batch_size = x.shape[1]
            seq_len = x.shape[0]
            h_t = np.zeros(
                (batch_size, self.hidden_size)
                )                                   # [N, hidden_size]
            h_t_minus_1 = h_t                       # [N, hidden_size]
            ht_all = []
            # #############################################################
            # forward propagation through time t
            for t in range(seq_len):
                h_t = tanh(
                    x[t] @ self.weight_ih.T
                    + self.bias_ih
                    + h_t_minus_1 @ self.weight_hh.T
                    + self.bias_hh
                )                                       # [N, hidden_size]
                ht_all.append(h_t.copy())
                h_t_minus_1 = h_t.copy()
            ht_all = np.stack(ht_all, axis=0)           # [seq_len, N, hidden_size]
            z2 = h_t @ self.weight_out.T + self.bias_out
            pred_all.append(z2)
        pred_all = np.concatenate(pred_all, axis=0) 
        return pred_all