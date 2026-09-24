"""Perceptron binário (rótulos 0/1) e gerador de dados do relatório."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FitResult:
    weights: np.ndarray
    bias: float
    accuracies: list[float]
    updates: list[int]
    pocket_weights: np.ndarray | None
    pocket_bias: float | None
    pocket_accuracy: float | None
    pocket_epoch: int | None
    pocket_curve: list[float]

    @property
    def epochs(self):
        return len(self.accuracies)


def make_dataset(rng, mean_0, mean_1, variance, count=1000):
    """Sorteia as duas classes, nesta ordem, usando o mesmo Generator."""
    covariance = np.eye(2) * variance
    x0 = rng.multivariate_normal(mean_0, covariance, size=count)
    x1 = rng.multivariate_normal(mean_1, covariance, size=count)
    x = np.vstack((x0, x1))
    y = np.r_[np.zeros(count, dtype=int), np.ones(count, dtype=int)]
    return x, y


def activation(x, weights, bias):
    return x @ weights + bias


def predict(x, weights, bias):
    """Degrau: 1 se a ativação é não negativa; 0 em caso contrário."""
    return (activation(x, weights, bias) >= 0).astype(int)


def accuracy(x, y, weights, bias):
    return float(np.mean(predict(x, weights, bias) == y))


def train(x, y, rng, eta=0.01, max_epochs=100, pocket=False, initial_weights=None):
    """Treina uma amostra por vez; o pocket só guarda cópias após melhorias."""
    weights = (rng.normal(0, 0.01, size=2) if initial_weights is None
               else np.array(initial_weights, dtype=float, copy=True))
    bias = 0.0
    accuracies, updates, pocket_curve = [], [], []
    best_accuracy = -1.0
    best_weights = best_bias = best_epoch = None

    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for sample, target in zip(x, y):
            prediction = int(activation(sample, weights, bias) >= 0)
            error = int(target) - prediction
            if error == 0:
                continue
            # Para y em {0,1}, o erro é +1 ou -1 em uma classificação errada.
            weights += eta * error * sample
            bias += eta * error
            mistakes += 1
            if pocket:
                current_accuracy = accuracy(x, y, weights, bias)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_weights = weights.copy()
                    best_bias = bias
                    best_epoch = epoch
        updates.append(mistakes)
        accuracies.append(accuracy(x, y, weights, bias))
        if pocket:
            pocket_curve.append(best_accuracy)
        if mistakes == 0:
            break

    return FitResult(weights.copy(), bias, accuracies, updates,
                     best_weights, best_bias,
                     best_accuracy if pocket else None,
                     best_epoch, pocket_curve)
