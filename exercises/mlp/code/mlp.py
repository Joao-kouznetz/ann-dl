"""MLP em NumPy: as mesmas rotinas atendem classificação binária e multiclasse."""

import numpy as np


class MLP:
    def __init__(self, sizes, rng):
        self.weights = [
            rng.normal(0, np.sqrt(2 / (a + b)), (a, b))
            for a, b in zip(sizes[:-1], sizes[1:])
        ]
        self.biases = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x):
        activations = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.tanh(activations[-1] @ w + b))
        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        if logits.shape[1] == 1:
            p = np.exp(-np.logaddexp(0, -logits))
        else:
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = exp / exp.sum(axis=1, keepdims=True)
        return p, activations, logits

    def loss_gradients(self, x, y):
        p, a, u = self.forward(x)
        target = y.reshape(-1, 1) if p.shape[1] == 1 else np.eye(p.shape[1])[y]
        if p.shape[1] == 1:
            loss = np.mean(np.logaddexp(0, u) - target * u)
        else:
            shift = u - u.max(axis=1, keepdims=True)
            logp = shift - np.log(np.exp(shift).sum(axis=1, keepdims=True))
            loss = -np.sum(target * logp) / len(x)
        delta = (p - target) / len(x)
        gw, gb = [None] * len(self.weights), [None] * len(self.weights)
        # Calcular todos os gradientes antes de modificar qualquer peso.
        for layer in range(len(self.weights) - 1, -1, -1):
            gw[layer] = a[layer].T @ delta
            gb[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.weights[layer].T) * (1 - a[layer] ** 2)
        return float(loss), gw, gb

    def fit(self, x, y, rng, epochs=800, lr=0.05, batch_size=64):
        history = []
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                ids = order[start : start + batch_size]
                _, gw, gb = self.loss_gradients(x[ids], y[ids])
                for k in range(len(self.weights)):
                    self.weights[k] -= lr * gw[k]
                    self.biases[k] -= lr * gb[k]
            history.append(self.loss_gradients(x, y)[0])
        return history

    def predict(self, x):
        p = self.forward(x)[0]
        return (p[:, 0] >= 0.5).astype(int) if p.shape[1] == 1 else p.argmax(axis=1)


def check_gradients():
    """Diferenças centrais conferem cada peso e viés nas duas perdas e profundidades."""
    errors = []
    for sizes in [[2, 3, 1], [2, 3, 2, 3]]:
        rng = np.random.default_rng(42)
        model = MLP(sizes, rng)
        x = rng.normal(size=(5, 2))
        y = np.array([0, 1, 0, 1, 0])
        _, gw, gb = model.loss_gradients(x, y)
        for params, grads in [(model.weights, gw), (model.biases, gb)]:
            for p, g in zip(params, grads):
                for index in np.ndindex(p.shape):
                    old = p[index]
                    h = 1e-5
                    p[index] = old + h
                    upper = model.loss_gradients(x, y)[0]
                    p[index] = old - h
                    lower = model.loss_gradients(x, y)[0]
                    p[index] = old
                    errors.append(abs((upper - lower) / (2 * h) - g[index]))
    assert max(errors) < 1e-7
    return max(errors)
