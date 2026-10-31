"""Reproduz os exercícios 2–4; nenhuma otimização usa o conjunto de teste."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mlp import MLP, check_gradients

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"


def save(number):
    plt.tight_layout()
    plt.savefig(FIG / f"figure{number}.png", dpi=150)
    plt.close()


def run():
    FIG.mkdir(exist_ok=True)
    results = {
        "gradient_check_max_error": check_gradients(),
        "epochs": 800,
        "lr": 0.05,
        "batch_size": 64,
        "runs": [],
    }
    datasets = [
        make_blobs(
            n_samples=[500, 250, 250],
            centers=np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]]),
            cluster_std=1.2,
            random_state=42,
        ),
        make_classification(
            n_samples=1500,
            n_features=4,
            n_informative=4,
            n_redundant=0,
            n_repeated=0,
            n_classes=3,
            n_clusters_per_class=2,
            class_sep=1.0,
            random_state=42,
        ),
    ]
    for task, (x, y) in enumerate(datasets, 2):
        if task == 2:
            y = (y > 0).astype(int)
        a, b, ya, yb = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        mean, std = a.mean(0), a.std(0)
        a = (a - mean) / std
        b = (b - mean) / std
        baseline = LogisticRegression(max_iter=1000).fit(a, ya)
        results[f"baseline{task}"] = float(baseline.score(b, yb))
        for seed in [42] if task == 2 else [42, 43, 44]:
            for depth in [1] if task == 2 else [1, 2]:
                rng = np.random.default_rng(seed)
                sizes = [a.shape[1]] + [32] * depth + [1 if task == 2 else 3]
                model = MLP(sizes, rng)
                # Embaralhamentos idênticos entre profundidades para cada seed.
                losses = model.fit(a, ya, np.random.default_rng(seed))
                pred = model.predict(b)
                accuracy = float(np.mean(pred == yb))
                record = {
                    "exercise": task if depth == 1 else 4,
                    "seed": seed,
                    "sizes": sizes,
                    "parameters": sum(
                        w.size + v.size for w, v in zip(model.weights, model.biases)
                    ),
                    "train_loss": losses[-1],
                    "test_accuracy": accuracy,
                    "losses": losses,
                }
                results["runs"].append(record)
                print(record["exercise"], seed, accuracy, flush=True)
                if seed != 42:
                    continue
                if task == 2:
                    plt.figure()
                    for cls in [0, 1]:
                        plt.scatter(
                            x[y == cls, 0], x[y == cls, 1], s=10, label=f"Classe {cls}"
                        )
                    plt.title("Figura 1 — Dados binários")
                    plt.xlabel("x1")
                    plt.ylabel("x2")
                    plt.legend()
                    save(1)
                    plt.figure()
                    plt.plot(np.arange(1, 801), losses, label="Treino")
                    plt.title("Figura 2 — BCE do MLP")
                    plt.xlabel("Época")
                    plt.ylabel("BCE média")
                    plt.legend()
                    save(2)
                    xx, yy = np.meshgrid(
                        np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 250),
                        np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 250),
                    )
                    zz = model.predict(
                        (np.c_[xx.ravel(), yy.ravel()] - mean) / std
                    ).reshape(xx.shape)
                    plt.figure()
                    plt.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
                    raw = b * std + mean
                    for cls in [0, 1]:
                        plt.scatter(
                            raw[yb == cls, 0],
                            raw[yb == cls, 1],
                            s=12,
                            label=f"Teste: classe {cls}",
                        )
                    plt.title("Figura 3 — Regiões do MLP e teste")
                    plt.xlabel("x1")
                    plt.ylabel("x2")
                    plt.legend()
                    save(3)
                elif depth == 1:
                    shallow = losses
                    plt.figure()
                    plt.plot(np.arange(1, 801), losses, label="Treino")
                    plt.title("Figura 4 — Entropia cruzada multiclasse")
                    plt.xlabel("Época")
                    plt.ylabel("Perda média")
                    plt.legend()
                    save(4)
                    matrix = confusion_matrix(yb, pred)
                    results["confusion"] = matrix.tolist()
                    ConfusionMatrixDisplay(matrix).plot()
                    plt.title("Figura 5 — Matriz de confusão no teste")
                    plt.xlabel("Classe predita")
                    plt.ylabel("Classe verdadeira")
                    save(5)
                else:
                    plt.figure()
                    plt.plot(np.arange(1, 801), shallow, label="1 camada oculta")
                    plt.plot(np.arange(1, 801), losses, label="2 camadas ocultas")
                    plt.title("Figura 6 — Comparação controlada (seed 42)")
                    plt.xlabel("Época")
                    plt.ylabel("Perda média")
                    plt.legend()
                    save(6)
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    run()
