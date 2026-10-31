"""Treina os dois VAEs por 20 épocas no MNIST completo e cria as seis figuras."""

import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from prepare_data import load_splits
from vae import VAE, loss_terms

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


def evaluate(model, loader, seed):
    model.eval()
    sums = np.zeros(2)
    count = 0
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for x, y in loader:
            prediction, mu, logvar = model(x, generator)
            r, k = loss_terms(prediction, x, mu, logvar)
            sums += np.array([r.item(), k.item()]) * len(x)
            count += len(x)
    return (sums / count).tolist()


def grid(images, rows, columns, number, title):
    fig, axes = plt.subplots(
        rows, columns, figsize=(columns * 0.95, rows * 1.05), squeeze=False
    )
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Figura {number} — {title}")
    fig.supxlabel("Coluna de imagens")
    fig.supylabel("Linha de imagens")
    fig.tight_layout()
    fig.savefig(FIG / f"figure{number}.png", dpi=150)
    plt.close(fig)


def run():
    FIG.mkdir(exist_ok=True)
    (HERE / "checkpoints").mkdir(exist_ok=True)
    train, val, test = load_splits()
    full = train.dataset
    # Mesmos índices do random_split existente, carregados em tensores para CPU.
    tx = full.data[train.indices].float().reshape(-1, 784) / 255
    vx = full.data[val.indices].float().reshape(-1, 784) / 255
    testx = test.data.float().reshape(-1, 784) / 255
    testy = test.targets
    trainset = TensorDataset(tx, full.targets[train.indices])
    valset = TensorDataset(vx, full.targets[val.indices])
    testloader = DataLoader(TensorDataset(testx, testy), batch_size=128)
    models = {}
    results = {
        "seed": 42,
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.001,
        "beta": 1,
        "test_mc_seed": 123,
        "validation_mc_seed": 2026,
        "torch": torch.__version__,
    }
    for dim in [16, 2]:
        torch.manual_seed(42)
        model = VAE(dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = DataLoader(
            trainset,
            batch_size=128,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(valset, batch_size=128)
        history = []
        start = time.time()
        for epoch in range(20):
            model.train()
            sums = np.zeros(2)
            for x, y in loader:
                optimizer.zero_grad()
                prediction, mu, logvar = model(x)
                r, k = loss_terms(prediction, x, mu, logvar)
                (r + k).backward()
                optimizer.step()
                sums += np.array([r.item(), k.item()]) * len(x)
            validation = evaluate(model, val_loader, 2026)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_reconstruction": sums[0] / len(tx),
                    "train_kl": sums[1] / len(tx),
                    "validation_reconstruction": validation[0],
                    "validation_kl": validation[1],
                }
            )
            print(
                f"d={dim} época={epoch + 1} treino={sums / len(tx)} val={validation}",
                flush=True,
            )
        # O teste só é percorrido após terminar o orçamento fixado do modelo.
        r, k = evaluate(model, testloader, 123)
        results[str(dim)] = {
            "test_reconstruction": r,
            "test_kl": k,
            "test_negative_elbo": r + k,
            "history": history,
            "seconds": time.time() - start,
        }
        models[dim] = model
        torch.save(model.state_dict(), HERE / "checkpoints" / f"vae{dim}.pt")
        (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, term in zip(axes, ["reconstruction", "kl"]):
        for partition, label in [("train", "Treino"), ("validation", "Validação")]:
            ax.plot(
                range(1, 21),
                [h[f"{partition}_{term}"] for h in results["16"]["history"]],
                label=label,
            )
        ax.set(
            xlabel="Época",
            ylabel="Nats por imagem",
            title="Reconstrução" if term == "reconstruction" else "KL",
        )
        ax.legend()
    fig.suptitle("Figura 1 — Termos da perda do VAE d=16")
    fig.tight_layout()
    fig.savefig(FIG / "figure1.png", dpi=150)
    plt.close(fig)
    indices = [int(torch.where(testy == digit)[0][0]) for digit in range(10)]
    results["reconstruction_indices"] = indices
    with torch.no_grad():
        mu, _ = models[16].encode(testx[indices])
        recon = models[16].decoder(mu)
        grid(
            torch.cat([testx[indices], recon]).numpy(),
            2,
            10,
            2,
            "Originais (acima) e reconstruções de μ (abaixo), dígitos 0–9",
        )
        torch.manual_seed(0)
        samples = models[16].decoder(torch.randn(64, 16))
        grid(samples.numpy(), 8, 8, 3, "Amostras do prior N(0,I), seed 0")
        mus = []
        for x, y in testloader:
            mus.append(models[2].encode(x)[0])
        mus = torch.cat(mus).numpy()
        fig, ax = plt.subplots(figsize=(9, 7))
        for digit in range(10):
            points = mus[testy.numpy() == digit]
            ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.4, label=str(digit))
        ax.set(
            xlabel="μ1",
            ylabel="μ2",
            title="Figura 4 — Médias latentes de 10.000 imagens de teste",
        )
        ax.legend(title="Dígito")
        fig.tight_layout()
        fig.savefig(FIG / "figure4.png", dpi=150)
        plt.close(fig)
        z = norm.ppf(np.linspace(0.05, 0.95, 15))
        points = np.array([[a, b] for b in z[::-1] for a in z], dtype=np.float32)
        images = (
            models[2]
            .decoder(torch.from_numpy(points))
            .numpy()
            .reshape(15, 15, 28, 28)
            .transpose(0, 2, 1, 3)
            .reshape(420, 420)
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            images, cmap="gray", vmin=0, vmax=1, extent=[z[0], z[-1], z[0], z[-1]]
        )
        ticks = np.linspace(z[0], z[-1], 15)
        centers = (np.arange(15) + 0.5) * 28
        # Ticks são posições dos centros das células, não uma escala linear de z.
        ax.set_xticks(
            z[0] + (centers / 420) * (z[-1] - z[0]),
            labels=[f"{v:.2f}" for v in ticks],
            rotation=90,
        )
        ax.set_yticks(
            z[0] + (centers / 420) * (z[-1] - z[0]), labels=[f"{v:.2f}" for v in ticks]
        )
        # Corrigir rótulos para os quantis reais (não valores igualmente espaçados).
        ax.set_xticklabels([f"{v:.2f}" for v in z], rotation=90)
        ax.set_yticklabels([f"{v:.2f}" for v in z])
        ax.set(
            xlabel="z1 (quantis normais)",
            ylabel="z2 (quantis normais)",
            title="Figura 5 — Plano decodificado por quantis do prior",
        )
        fig.tight_layout()
        fig.savefig(FIG / "figure5.png", dpi=150)
        plt.close(fig)
        ia, ib = 0, 1
        results["interpolation_indices"] = [ia, ib]
        results["interpolation_labels"] = [int(testy[ia]), int(testy[ib])]
        t = torch.linspace(0, 1, 10).reshape(-1, 1)
        pixel = (1 - t) * testx[ia] + t * testx[ib]
        mu, _ = models[16].encode(testx[[ia, ib]])
        latent = models[16].decoder((1 - t) * mu[0] + t * mu[1])
        grid(
            torch.cat([pixel, latent]).numpy(),
            2,
            10,
            6,
            "Interpolação: pixels (acima), latente d=16 (abaixo), t=0…1",
        )
    np.savez_compressed(
        HERE / "latent_summary.npz",
        centers=np.stack([mus[testy.numpy() == i].mean(0) for i in range(10)]),
    )
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    run()
