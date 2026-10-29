"""Reprodução completa do exercício Data.

Execute a partir de qualquer diretório com:
    python docs/exercises/data/code/data_analysis.py
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_rows", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# Uma única fonte de aleatoriedade para todo o relatório.
rng = np.random.default_rng(42)

# %%

means = np.array([[2, 3], [5, 6], [8, 1], [15, 4]], dtype=float)
stds = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]], dtype=float)
n_per_class = 100

def generate_clouds(scale=1.0):
    # Gera as quatro classes usando o mesmo rng global.
    blocks = [rng.normal(loc=means[k], scale=stds[k] * scale,
                         size=(n_per_class, 2)) for k in range(4)]
    X = np.vstack(blocks)
    y = np.repeat(np.arange(4), n_per_class)
    return X, y

X1, y1 = generate_clouds(scale=1.0)

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("tab10", 4)
for k, color in enumerate(colors):
    pts = X1[y1 == k]
    ax.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.68, color=color,
               label=f"Classe {k}")
    ax.scatter(*means[k], marker="X", s=190, color=color,
               edgecolor="black", linewidth=1.2)

# Esboço de fronteiras: mediatrizes entre centros vizinhos (regra do centro mais próximo).
xlim = (-4, 20)
ylim = (-7, 13)
for i, j in [(0, 1), (1, 2), (2, 3), (1, 3)]:
    delta = means[j] - means[i]
    midpoint = (means[i] + means[j]) / 2
    xs = np.linspace(*xlim, 300)
    if abs(delta[1]) > 1e-12:
        ys = midpoint[1] - delta[0] * (xs - midpoint[0]) / delta[1]
        mask = (ys >= ylim[0]) & (ys <= ylim[1])
        ax.plot(xs[mask], ys[mask], "k--", alpha=0.38, lw=1.2)

ax.set(xlim=xlim, ylim=ylim, xlabel="x1", ylabel="x2",
       title="Figura 1 — Quatro nuvens gaussianas e esboço de fronteiras")
ax.legend(title="Classe", ncol=2)
plt.show()

# %%

scales = [0.5, 1.0, 2.0, 4.0]
datasets_s = {s: generate_clouds(scale=s) for s in scales}

# Limites globais: exatamente os mesmos nos quatro painéis.
all_X = np.vstack([datasets_s[s][0] for s in scales])
pad = 0.6
shared_xlim = (all_X[:, 0].min() - pad, all_X[:, 0].max() + pad)
shared_ylim = (all_X[:, 1].min() - pad, all_X[:, 1].max() + pad)

fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
for ax, s in zip(axes.flat, scales):
    Xs, ys = datasets_s[s]
    for k, color in enumerate(colors):
        pts = Xs[ys == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.65,
                   color=color, label=f"Classe {k}")
        ax.scatter(*means[k], marker="X", s=100, color=color,
                   edgecolor="black", linewidth=0.8)
        ax.set(title=f"s = {s}", xlabel="x1", ylabel="x2",
           xlim=shared_xlim, ylim=shared_ylim)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles[:4], labels[:4], title="Classe", loc="upper center",
           ncol=4, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Figura 2 — Efeito da escala na dispersão das quatro classes", y=1.05)
fig.tight_layout()
plt.show()

# %%

# Razões de separação teóricas para s=1.
rows = []
mean_std = stds.mean(axis=1)
for i, j in combinations(range(4), 2):
    distance = np.linalg.norm(means[i] - means[j])
    ratio = distance / (mean_std[i] + mean_std[j])
    rows.append({"Par": f"({i}, {j})", "Distância entre médias": distance,
                 "r_ij (s=1)": ratio})

ratio_table = pd.DataFrame(rows).sort_values("r_ij (s=1)").reset_index(drop=True)
display(ratio_table.style.format({"Distância entre médias": "{:.4f}", "r_ij (s=1)": "{:.4f}"}))

smallest_pair = ratio_table.loc[0, "Par"]
smallest_ratio = ratio_table.loc[0, "r_ij (s=1)"]
display(Markdown(
    f"A menor razão em $s=1$ é **{smallest_ratio:.4f}**, para o par **{smallest_pair}**. "
    f"Como $r_{{ij}}$ varia com $1/s$, em $s=2$ ela se torna "
    f"**{smallest_ratio / 2:.4f}**, sem necessidade de gerar novos pontos."
))

# %%

def mixing_rate(X, y):
    # Fração cujo centro mais próximo não é o centro da própria classe.
    sq_dist = ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    nearest_center = sq_dist.argmin(axis=1)
    return np.mean(nearest_center != y)

mixing = {s: mixing_rate(*datasets_s[s]) for s in scales}
mixing_table = pd.DataFrame({
    "s": scales,
    "Taxa de mistura": [mixing[s] for s in scales],
    "Taxa de mistura (%)": [100 * mixing[s] for s in scales],
})
display(mixing_table.style.format({"Taxa de mistura": "{:.4f}", "Taxa de mistura (%)": "{:.2f}%"}))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(scales, [mixing[s] for s in scales], marker="o", ms=8, lw=2)
for s in scales:
    ax.annotate(f"{mixing[s]:.1%}", (s, mixing[s]), xytext=(0, 8),
                textcoords="offset points", ha="center")
ax.set(xlabel="Fator de escala s", ylabel="Taxa de mistura",
       title="Figura 3 — Taxa de mistura em função da dispersão",
       xticks=scales, ylim=(0, max(mixing.values()) * 1.15 + 0.01))
plt.show()

# %%

# Diagnóstico empírico: a partir do primeiro s em que existe mistura relevante.
first_mixed_scale = next(s for s in scales if mixing[s] > 0)
display(Markdown(
    f"Nos dados gerados, a taxa de mistura já é diferente de zero em "
    f"**s = {first_mixed_scale:g}** ({mixing[first_mixed_scale]:.2%}). "
    f"Porém, a perda visual clara da separação por regiões lineares ocorre a partir de "
    f"**s = 2**, quando a menor razão de separação cai para "
    f"**{smallest_ratio/2:.4f}**. Em $s=4$, a mistura se torna ainda mais intensa."
))

# %%

mu_A = np.zeros(5)
mu_B = np.full(5, 1.5)
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

XA = rng.multivariate_normal(mu_A, Sigma_A, size=500)
XB = rng.multivariate_normal(mu_B, Sigma_B, size=500)
X_gauss = np.vstack([XA, XB])
y_gauss = np.repeat(["A", "B"], 500)
print("Dataset I:", X_gauss.shape, "— 500 amostras em cada classe")

# %%

def shell_points(n, radius_mean, radius_std=0.4):
    v = rng.normal(size=(n, 5))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    rho = rng.normal(radius_mean, radius_std, size=n)
    return u * rho[:, None]

XC = shell_points(500, 2.0)
XD = shell_points(500, 5.0)
X_shell = np.vstack([XC, XD])
y_shell = np.repeat(["C", "D"], 500)
print("Dataset II:", X_shell.shape, "— 500 amostras em cada classe")

# %%

pca_gauss = PCA(n_components=2).fit(X_gauss)
pca_shell = PCA(n_components=2).fit(X_shell)
Z_gauss = pca_gauss.transform(X_gauss)
Z_shell = pca_shell.transform(X_shell)
ev_gauss = pca_gauss.explained_variance_ratio_.sum()
ev_shell = pca_shell.explained_variance_ratio_.sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for label, color in zip(["A", "B"], [colors[0], colors[1]]):
    mask = y_gauss == label
    axes[0].scatter(Z_gauss[mask, 0], Z_gauss[mask, 1], s=20,
                    alpha=0.58, color=color, label=f"Classe {label}")
for label, color in zip(["C", "D"], [colors[2], colors[3]]):
    mask = y_shell == label
    axes[1].scatter(Z_shell[mask, 0], Z_shell[mask, 1], s=20,
                    alpha=0.58, color=color, label=f"Classe {label}")

axes[0].set(title=f"Dataset I — gaussianas (PC1+PC2 = {ev_gauss:.2%})",
            xlabel="PC1", ylabel="PC2")
axes[1].set(title=f"Dataset II — cascas (PC1+PC2 = {ev_shell:.2%})",
            xlabel="PC1", ylabel="PC2")
for ax in axes:
    ax.legend(title="Classe")
fig.suptitle("Figura 4 — Projeções PCA em duas dimensões")
fig.tight_layout()
plt.show()

display(Markdown(
    f"A variância explicada por PC1+PC2 é **{ev_gauss:.2%}** no Dataset I "
    f"e **{ev_shell:.2%}** no Dataset II. No Dataset I, a projeção preserva melhor "
    f"a informação relevante para classificação, pois o deslocamento entre médias "
    f"é uma direção linear de alta variância. No Dataset II, a informação está no raio "
    f"em 5D e não em uma direção linear específica."
))

# %%

center_distance_gauss = np.linalg.norm(XA.mean(axis=0) - XB.mean(axis=0))
center_distance_shell = np.linalg.norm(XC.mean(axis=0) - XD.mean(axis=0))

radius_A, radius_B = np.linalg.norm(XA, axis=1), np.linalg.norm(XB, axis=1)
radius_C, radius_D = np.linalg.norm(XC, axis=1), np.linalg.norm(XD, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
sns.histplot(radius_A, bins=30, stat="density", alpha=0.48,
             label="Classe A", color=colors[0], ax=axes[0])
sns.histplot(radius_B, bins=30, stat="density", alpha=0.48,
             label="Classe B", color=colors[1], ax=axes[0])
sns.histplot(radius_C, bins=30, stat="density", alpha=0.48,
             label="Classe C", color=colors[2], ax=axes[1])
sns.histplot(radius_D, bins=30, stat="density", alpha=0.48,
             label="Classe D", color=colors[3], ax=axes[1])
axes[0].set(title="Dataset I — gaussianas deslocadas", xlabel="Raio ||x|| em 5D",
            ylabel="Densidade")
axes[1].set(title="Dataset II — cascas concêntricas", xlabel="Raio ||x|| em 5D",
            ylabel="Densidade")
for ax in axes:
    ax.legend(title="Classe")
fig.suptitle("Figura 5 — Distribuição do raio por classe")
fig.tight_layout()
plt.show()

display(Markdown(
    f"A distância empírica entre os centros é **{center_distance_gauss:.4f}** "
    f"no Dataset I e **{center_distance_shell:.4f}** no Dataset II."
))

# %%

def separate_shells(X, threshold_radius=3.5):
    # 0 para núcleo C e 1 para casca D.
    return (np.sum(X**2, axis=1) > threshold_radius**2).astype(int)

shell_truth = np.r_[np.zeros(500, dtype=int), np.ones(500, dtype=int)]
shell_accuracy = np.mean(separate_shells(X_shell) == shell_truth)
display(Markdown(f"Com limiar radial 3,5, essa função separa **{shell_accuracy:.2%}** das amostras geradas."))

# %%

data_path = Path(__file__).with_name("train.csv")
if not data_path.exists():
    raise FileNotFoundError(
        "Coloque o train.csv da competição Spaceship Titanic na mesma pasta do notebook."
    )

df = pd.read_csv(data_path)
display(df.head())
print("Dimensão original:", df.shape)

# %%

balance = df["Transported"].value_counts().rename_axis("Transported").to_frame("Contagem")
balance["Percentual"] = 100 * balance["Contagem"] / len(df)
display(balance.style.format({"Percentual": "{:.2f}%"}))
positive_share = df["Transported"].mean()
display(Markdown(
    f"A classe positiva (`Transported=True`) corresponde a **{positive_share:.2%}** "
    f"dos passageiros; as classes são praticamente balanceadas."
))

# %%

missing = df.isna().sum().to_frame("Ausentes")
missing["Percentual"] = 100 * missing["Ausentes"] / len(df)
display(missing.sort_values("Ausentes", ascending=False)
        .style.format({"Percentual": "{:.2f}%"}))

# %%

spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
spend_stats_full = df[spend_cols].agg(["mean", "median", "max"]).T
spend_stats_full.columns = ["Média", "Mediana", "Máximo"]
display(spend_stats_full.style.format("{:,.2f}"))

# %%

X_raw = df.drop(columns="Transported")
y = df["Transported"].astype(int)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=42
)
print("Treino:", X_train_raw.shape, "Teste:", X_test_raw.shape)
print("Positivos — treino:", f"{y_train.mean():.2%}", "teste:", f"{y_test.mean():.2%}")

# Números pedidos no resumo: FoodCourt no treino, antes de qualquer transformação.
foodcourt_train_mean = X_train_raw["FoodCourt"].mean()
foodcourt_train_median = X_train_raw["FoodCourt"].median()
display(Markdown(
    f"No treino, antes das transformações, `FoodCourt` tem média "
    f"**{foodcourt_train_mean:,.2f}** e mediana **{foodcourt_train_median:,.2f}**."
))

# %%

numeric_base = ["Age"] + spend_cols
categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]

# 1) Imputação — ajuste apenas no treino.
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

train_num = pd.DataFrame(
    num_imputer.fit_transform(X_train_raw[numeric_base]),
    columns=numeric_base, index=X_train_raw.index
)
test_num = pd.DataFrame(
    num_imputer.transform(X_test_raw[numeric_base]),
    columns=numeric_base, index=X_test_raw.index
)
train_cat = pd.DataFrame(
    cat_imputer.fit_transform(X_train_raw[categorical_cols]),
    columns=categorical_cols, index=X_train_raw.index
)
test_cat = pd.DataFrame(
    cat_imputer.transform(X_test_raw[categorical_cols]),
    columns=categorical_cols, index=X_test_raw.index
)

# 2) Engenharia: soma após imputação para não propagar NaN.
for frame in (train_num, test_num):
    frame["TotalSpend"] = frame[spend_cols].sum(axis=1)

# 3) Log1p nos gastos e no gasto total.
log_cols = spend_cols + ["TotalSpend"]
train_num[log_cols] = np.log1p(train_num[log_cols])
test_num[log_cols] = np.log1p(test_num[log_cols])

# 4) Padronização numérica — fit só no treino.
numeric_final = numeric_base + ["TotalSpend"]
scaler = StandardScaler()
train_num_scaled = scaler.fit_transform(train_num[numeric_final])
test_num_scaled = scaler.transform(test_num[numeric_final])

# 5) One-hot — compatível com versões novas e antigas do scikit-learn.
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
train_cat_encoded = encoder.fit_transform(train_cat)
test_cat_encoded = encoder.transform(test_cat)

feature_names = numeric_final + list(encoder.get_feature_names_out(categorical_cols))
X_train = pd.DataFrame(
    np.hstack([train_num_scaled, train_cat_encoded]),
    columns=feature_names, index=X_train_raw.index
)
X_test = pd.DataFrame(
    np.hstack([test_num_scaled, test_cat_encoded]),
    columns=feature_names, index=X_test_raw.index
)

print("Features finais:")
print(feature_names)

# %%

foodcourt_idx = numeric_final.index("FoodCourt")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
sns.histplot(X_train_raw["FoodCourt"].dropna(), bins=50, color=colors[0], ax=axes[0])
sns.histplot(train_num_scaled[:, foodcourt_idx], bins=50, color=colors[1], ax=axes[1])
axes[0].set(title="Antes: valores originais", xlabel="FoodCourt", ylabel="Contagem")
axes[1].set(title="Depois: log1p + padronização", xlabel="FoodCourt transformado", ylabel="Contagem")
fig.suptitle("Figura 6 — Efeito do pré-processamento sobre FoodCourt")
fig.tight_layout()
plt.show()

# %%

train_nan = int(X_train.isna().sum().sum())
test_nan = int(X_test.isna().sum().sum())
train_min, train_max = X_train.to_numpy().min(), X_train.to_numpy().max()
test_min, test_max = X_test.to_numpy().min(), X_test.to_numpy().max()

checks = pd.DataFrame({
    "Conjunto": ["Treino", "Teste"],
    "Shape": [str(X_train.shape), str(X_test.shape)],
    "NaN restantes": [train_nan, test_nan],
    "Mínimo": [train_min, test_min],
    "Máximo": [train_max, test_max],
})
display(checks.style.format({"Mínimo": "{:.4f}", "Máximo": "{:.4f}"}))

display(Markdown(
    f"As matrizes finais têm **zero NaN**. O treino tem shape **{X_train.shape}** "
    f"e varia de **{train_min:.4f}** a **{train_max:.4f}**; o teste tem shape "
    f"**{X_test.shape}** e varia de **{test_min:.4f}** a **{test_max:.4f}**. "
    f"As variáveis contínuas foram centradas e padronizadas; as one-hot ficam em 0 ou 1. "
    f"A padronização não limita rigidamente o intervalo, mas concentra a maioria dos valores "
    f"perto de zero, uma escala adequada para `tanh`."
))

# %%

summary = pd.DataFrame([
    (1, "Taxa de mistura em s = 0.5", f"{mixing[0.5]:.4f} ({mixing[0.5]:.2%})"),
    (2, "Taxa de mistura em s = 1.0", f"{mixing[1.0]:.4f} ({mixing[1.0]:.2%})"),
    (3, "Taxa de mistura em s = 2.0", f"{mixing[2.0]:.4f} ({mixing[2.0]:.2%})"),
    (4, "Taxa de mistura em s = 4.0", f"{mixing[4.0]:.4f} ({mixing[4.0]:.2%})"),
    (5, "Menor r_ij em s = 1.0 e par", f"{smallest_ratio:.4f}, par {smallest_pair}"),
    (6, "Distância entre centros — Dataset I", f"{center_distance_gauss:.4f}"),
    (7, "Distância entre centros — Dataset II", f"{center_distance_shell:.4f}"),
    (8, "Variância explicada PC1 + PC2 — Dataset I", f"{ev_gauss:.4f} ({ev_gauss:.2%})"),
    (9, "Variância explicada PC1 + PC2 — Dataset II", f"{ev_shell:.4f} ({ev_shell:.2%})"),
    (10, "Participação da classe positiva em Transported", f"{positive_share:.4f} ({positive_share:.2%})"),
    (11, "Média e mediana de FoodCourt no treino, antes", f"{foodcourt_train_mean:,.2f}; {foodcourt_train_median:,.2f}"),
    (12, "Shape final da matriz de treino", str(X_train.shape)),
    (13, "Mínimo e máximo após escala — treino; teste",
     f"treino [{train_min:.4f}, {train_max:.4f}]; teste [{test_min:.4f}, {test_max:.4f}]"),
], columns=["#", "Item", "Valor"])
display(summary.style.hide(axis="index"))
