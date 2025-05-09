import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int=10) -> np.ndarray:
    y_hot = np.zeros((len(y), num_classes))
    y_hot[range(y.shape[0]), y.astype(int)] = 1
    return y_hot