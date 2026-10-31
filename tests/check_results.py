"""Validação dos artefatos publicados; executar depois dos dois experimentos."""

import json, re
from pathlib import Path
import numpy as np
import nbformat

root = Path(__file__).resolve().parents[1]
for topic in ["mlp", "vae"]:
    folder = root / "docs/exercises" / topic
    report = (folder / "index.md").read_text()
    figures = re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", report)
    assert len(figures) == 6
    assert all((folder / f).is_file() for f in figures)
    assert not re.search(
        "Pendente|Não realizado nesta entrega|Não iniciado|em andamento", report
    )
    for notebook in (folder / "code").glob("*.ipynb"):
        nbformat.validate(nbformat.read(notebook, as_version=4))
mlp = json.loads((root / "docs/exercises/mlp/code/results.json").read_text())
assert mlp["gradient_check_max_error"] < 1e-7
assert len(mlp["runs"]) == 7
assert np.array(mlp["confusion"]).sum() == 300
for run in mlp["runs"]:
    assert len(run["losses"]) == 800 and np.isfinite(run["losses"]).all()
    assert 0 <= run["test_accuracy"] <= 1
    assert abs(run["train_loss"] - run["losses"][-1]) < 1e-12
vae = json.loads((root / "docs/exercises/vae/code/results.json").read_text())
for dimension in ["16", "2"]:
    r = vae[dimension]
    assert len(r["history"]) == 20
    assert abs(r["test_negative_elbo"] - r["test_reconstruction"] - r["test_kl"]) < 1e-9
    assert r["test_kl"] >= 0 and r["test_reconstruction"] > 0
    for h in r["history"]:
        assert all(np.isfinite(v) for v in h.values())
assert vae["interpolation_labels"][0] != vae["interpolation_labels"][1]
assert len(vae["reconstruction_indices"]) == 10
print("Relatórios, notebooks, 12 figuras, sementes e resultados: OK")

# Conferir os valores publicados no exercício analítico de backpropagation.
x = np.array([0.5, -0.2])
w1 = np.array([[0.3, -0.1], [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
w2 = np.array([0.5, -0.3])
b2 = 0.2
h = np.tanh(w1 @ x + b1)
prediction = np.tanh(w2 @ h + b2)
delta = 2 * (prediction - 1) * (1 - prediction**2)
delta1 = delta * w2 * (1 - h * h)
updated_w1 = w1 - 0.3 * np.outer(delta1, x)
updated_b1 = b1 - 0.3 * delta1
updated_w2 = w2 - 0.3 * delta * h
updated_b2 = b2 - 0.3 * delta
new_prediction = np.tanh(updated_w2 @ np.tanh(updated_w1 @ x + updated_b1) + updated_b2)
np.testing.assert_allclose(
    [prediction, (1 - prediction) ** 2, new_prediction, (1 - new_prediction) ** 2],
    [0.3672465626, 0.4003769, 0.7111805295, 0.0834167],
    atol=1e-6,
)
print("Cálculo analítico do MLP: OK")
