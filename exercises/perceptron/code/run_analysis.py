"""Executa o relatório do perceptron e grava as Figuras 1 a 6."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from perceptron import make_dataset, predict, accuracy, train

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / 'figures'
FIGURES.mkdir(exist_ok=True)
COLORS = ('#2563a6', '#ee8b25')


def scatter_classes(ax, x, y):
    for cls in (0, 1):
        points = x[y == cls]
        ax.scatter(points[:, 0], points[:, 1], s=12, alpha=.45,
                   color=COLORS[cls], label=f'Classe {cls}', rasterized=True)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(loc='best')


def boundary(ax, weights, bias, xlim, label, style='-', color='#202020'):
    """Plota w1*x1 + w2*x2 + b = 0, inclusive quando w2 é quase zero."""
    if abs(weights[1]) > 1e-12:
        xx = np.array(xlim)
        yy = -(weights[0] * xx + bias) / weights[1]
        ax.plot(xx, yy, style, color=color, linewidth=2, label=label)
    elif abs(weights[0]) > 1e-12:
        ax.axvline(-bias / weights[0], linestyle=style, color=color,
                   linewidth=2, label=label)


def mark_errors(ax, x, y, weights, bias):
    wrong = predict(x, weights, bias) != y
    ax.scatter(x[wrong, 0], x[wrong, 1], s=44, facecolors='none',
               edgecolors='#bf2031', linewidths=1.2,
               label=f'Erros ({wrong.sum()})', rasterized=True)
    return int(wrong.sum())


def save(fig, number):
    fig.tight_layout()
    fig.savefig(FIGURES / f'figure{number}.png', dpi=160)
    plt.close(fig)


def main():
    rng = np.random.default_rng(42)
    x1, y1 = make_dataset(rng, [1.5, 1.5], [5, 5], .5)
    # Reuso do MESMO vetor inicial isola a influência da taxa de aprendizado.
    w0 = rng.normal(0, 0.01, size=2)
    fit1 = train(x1, y1, rng, eta=.01, initial_weights=w0)
    fit1_fast = train(x1, y1, rng, eta=1.0, initial_weights=w0)
    x2, y2 = make_dataset(rng, [3, 3], [4, 4], 1.5)
    fit2 = train(x2, y2, rng, eta=.01, pocket=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, x1, y1)
    ax.set_title('Figura 1 — Dados separáveis (2.000 pontos)')
    save(fig, 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, x1, y1)
    boundary(ax, fit1.weights, fit1.bias, ax.get_xlim(), 'Fronteira final')
    mark_errors(ax, x1, y1, fit1.weights, fit1.bias)
    ax.set_ylim(x1[:, 1].min()-.4, x1[:, 1].max()+.4)
    ax.set_title('Figura 2 — Fronteira do perceptron: dados separáveis')
    ax.legend(loc='best')
    save(fig, 2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, fit1.epochs+1), np.array(fit1.accuracies)*100,
            marker='o', label='Acurácia após a época')
    ax.set(xlabel='Época', ylabel='Acurácia (%)',
           title='Figura 3 — Acurácia por época: dados separáveis')
    ax.set_ylim(0, 102)
    ax.grid(alpha=.25)
    ax.legend()
    save(fig, 3)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, x2, y2)
    ax.set_title('Figura 4 — Dados sobrepostos (2.000 pontos)')
    save(fig, 4)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, weights, bias, name in (
        (axes[0], fit2.weights, fit2.bias, 'final'),
        (axes[1], fit2.pocket_weights, fit2.pocket_bias, 'pocket')):
        scatter_classes(ax, x2, y2)
        boundary(ax, weights, bias, ax.get_xlim(), f'Fronteira {name}')
        mark_errors(ax, x2, y2, weights, bias)
        ax.set_ylim(x2[:, 1].min()-.5, x2[:, 1].max()+.5)
        ax.set_title(f'{name.capitalize()}: {accuracy(x2,y2,weights,bias):.2%}')
        ax.legend(loc='best')
    fig.suptitle('Figura 5 — Fronteiras final e pocket; círculos vermelhos = erros')
    save(fig, 5)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, fit2.epochs+1)
    ax.plot(epochs, np.array(fit2.accuracies)*100, label='Pesos atuais')
    ax.plot(epochs, np.array(fit2.pocket_curve)*100, label='Melhor até a época (pocket)')
    ax.set(xlabel='Época', ylabel='Acurácia (%)',
           title='Figura 6 — Acurácia atual e pocket por época')
    ax.set_ylim(0, 100)
    ax.grid(alpha=.25)
    ax.legend()
    save(fig, 6)

    results = {
        'initial_weights': w0.tolist(),
        'exercise1': {'weights': fit1.weights.tolist(), 'bias': fit1.bias,
                      'epochs': fit1.epochs, 'accuracy': fit1.accuracies[-1],
                      'updates': fit1.updates, 'curve': fit1.accuracies,
                      'direction': (fit1.weights / np.linalg.norm(fit1.weights)).tolist()},
        'exercise1_eta1': {'weights': fit1_fast.weights.tolist(), 'bias': fit1_fast.bias,
                           'epochs': fit1_fast.epochs, 'accuracy': fit1_fast.accuracies[-1],
                           'updates': fit1_fast.updates,
                           'direction': (fit1_fast.weights / np.linalg.norm(fit1_fast.weights)).tolist()},
        'exercise2': {'weights': fit2.weights.tolist(), 'bias': fit2.bias,
                      'epochs': fit2.epochs, 'accuracy': fit2.accuracies[-1],
                      'updates': fit2.updates,
                      'pocket_weights': fit2.pocket_weights.tolist(),
                      'pocket_bias': fit2.pocket_bias,
                      'pocket_accuracy': fit2.pocket_accuracy,
                      'pocket_epoch': fit2.pocket_epoch,
                      'curve': fit2.accuracies,
                      'pocket_curve': fit2.pocket_curve},
    }
    (HERE / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({key: {k: v for k, v in value.items() if k not in ('curve','pocket_curve','updates')}
                      for key, value in results.items() if key != 'initial_weights'}, indent=2))
    return results


if __name__ == '__main__':
    main()
