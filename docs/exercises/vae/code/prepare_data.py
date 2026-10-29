"""Exercise 1A: prepare MNIST without evaluating or training a VAE.

Run from the repository root:
    python docs/exercises/vae/code/prepare_data.py
"""

import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


HERE = Path(__file__).resolve().parent
SEED = 42
BATCH_SIZE = 128


def load_splits(data_dir=HERE / "data"):
    """Use the official test partition and the exact seeded training split."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # uint8 [0, 255] -> float32 [0, 1]
        transforms.Lambda(torch.flatten),  # (1, 28, 28) -> (784,)
    ])
    train_full = datasets.MNIST(
        str(data_dir), train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        str(data_dir), train=False, download=True, transform=transform
    )
    train, val = random_split(
        train_full, [50_000, 10_000],
        generator=torch.Generator().manual_seed(SEED),
    )
    return train, val, test


def make_loaders(train, val, test):
    """Shuffle only training data; these loaders are ready for later work."""
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                   generator=torch.Generator().manual_seed(SEED)),
        DataLoader(val, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=False),
    )


def main():
    torch.manual_seed(SEED)
    train, val, test = load_splits()
    train_loader, _, _ = make_loaders(train, val, test)

    # Check membership using indices from the original training set only.
    train_ids, val_ids = set(train.indices), set(val.indices)
    assert len(train_ids) == 50_000 and len(val_ids) == 10_000
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(range(60_000))
    assert len(test) == 10_000

    # Inspect a training batch. Do not iterate over the test set at this stage.
    images, labels = next(iter(train_loader))
    assert tuple(images.shape) == (BATCH_SIZE, 784)
    assert images.dtype == torch.float32
    assert bool(torch.isfinite(images).all())
    assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0

    # Counts describe the realized random split; it is not stratified.
    targets = train.dataset.targets
    train_counts = torch.bincount(targets[train.indices], minlength=10).tolist()
    val_counts = torch.bincount(targets[val.indices], minlength=10).tolist()
    summary = {
        "scope": "Exercise 1A only; no model trained and no test metrics computed",
        "seed": SEED,
        "train_images": len(train),
        "validation_images": len(val),
        "test_images": len(test),
        "batch_size": BATCH_SIZE,
        "training_batch_shape": list(images.shape),
        "training_batch_dtype": str(images.dtype),
        "training_batch_min": float(images.min()),
        "training_batch_max": float(images.max()),
        "train_validation_disjoint": train_ids.isdisjoint(val_ids),
        "train_validation_cover_original_training_set": len(train_ids | val_ids) == 60_000,
        "first_10_train_indices": train.indices[:10],
        "first_10_validation_indices": val.indices[:10],
        "train_class_counts": train_counts,
        "validation_class_counts": val_counts,
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__},
    }
    result_path = HERE / "data_summary.json"
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
