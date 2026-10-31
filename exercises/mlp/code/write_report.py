"""Atualiza os itens numéricos do relatório a partir da execução registrada."""

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "results.json").read_text())
runs = r["runs"]
binary = runs[0]
shallow = [x for x in runs if x["exercise"] == 3]
deep = [x for x in runs if x["exercise"] == 4]
a, b = shallow[0], deep[0]
mean1 = np.mean([x["test_accuracy"] for x in shallow])
mean2 = np.mean([x["test_accuracy"] for x in deep])
c = np.array(r["confusion"])
pairs = [(int(c[i, j] + c[j, i]), i, j) for i in range(3) for j in range(i + 1, 3)]
count, i, j = max(pairs)
tied_pairs = ", ".join(f"{u}/{v}" for n, u, v in pairs if n == count)
page = HERE.parent / "index.md"
prefix = (
    page.read_text()
    .split("## Results summary")[0]
    .split("## Exercise 2")[0]
    .split("## Código reproduzível")[0]
)
prefix = prefix.replace(
    "Esta entrega abrange **somente o Exercise 1**, itens A, B e C. Os Exercises 2, 3 e 4 não foram realizados nesta entrega.",
    "Esta entrega resolve os Exercises 1 a 4. O Exercise 1 preserva a derivação analítica; os demais usam o mesmo MLP implementado em NumPy.",
)
prefix = prefix.replace(
    "O desenvolvimento abaixo é analítico",
    "O desenvolvimento do Exercise 1 é analítico",
)
prefix = prefix.replace(
    "OpenAI Codex auxiliou na elaboração e conferência dos cálculos, na redação das derivações em Markdown e na publicação da página.",
    "OpenAI Codex auxiliou no código, execução, verificação de gradientes, gráficos e redação dos quatro exercícios.",
)
# Preservar o bloco de autoria ao regenerar o relatório.
authorship = '**Autoria e uso de IA.** Este relatório foi desenvolvido com colaboração do Codex (OpenAI). Para auxiliar no desenvolvimento dos códigos.'
if authorship not in prefix:
    title = next(line for line in prefix.splitlines() if line.startswith("# "))
    prefix = prefix.replace(title + "\n", title + "\n\n" + authorship + "\n", 1)

text = f"""
## Código reproduzível

[Notebook executável](code/MLP.ipynb) · [MLP em NumPy](code/mlp.py) · [Experimentos](code/run_analysis.py) · [Resultados completos](code/results.json).

```bash
python docs/exercises/mlp/code/run_analysis.py
python docs/exercises/mlp/code/write_report.py
```

Nenhum framework de redes é usado no MLP. NumPy implementa ativações, perdas, gradientes e SGD;
scikit-learn gera/divide os dados e calcula as matrizes. LogisticRegression é usada apenas nos baselines,
como autorizado. A padronização usa média e desvio do treino (NumPy); os índices e geradores de dados
seguem o enunciado. Não foram escolhidos hiperparâmetros por acurácia no teste.

## Exercise 2

### A — Generate the data

Geramos 1.000 pontos com centros (0,0), (3,3), (-3,-3), desvios 1,2 e tamanhos 500/250/250.
Os dois centros externos formam a classe 1; o central, a classe 0. O gerador e o split estratificado
80/20 usam random_state=42, deixando 800 exemplos de treino e 200 de teste, com 100 por classe no teste.

![Figura 1 — Pontos das duas classes](figures/figure1.png)

Os grupos externos cercam o centro em direções opostas; uma reta não isola as duas extremidades do centro.

### B — Establish the baseline

LogisticRegression foi ajustada só no treino padronizado: **{r["baseline2"]:.2%} de acurácia no teste**.
Isso descreve esta amostra finita; abaixo de 50% não significa que todo classificador linear seja pior
que um sorteio em toda população. A simetria aproxima a média das duas classes e a reta não capta
que classe 1 ocupa duas regiões opostas.

### C — Implement the MLP

Arquitetura **2 → 32 (tanh) → 1 (sigmoid)**, {binary["parameters"]} parâmetros, SGD com taxa **0,05**,
mini-batches **64**, **800 épocas**. A BCE estável é logaddexp(0,u)−y·u, evitando log(0).
A acurácia de teste foi **{binary["test_accuracy"]:.2%}** e a BCE final de treino, **{binary["train_loss"]:.6f}**.
A inicialização Xavier normal e o embaralhamento usam geradores NumPy com seed 42.

Para uma camada H=tanh(AW+b), temos dL/dW=Aᵀδ, dL/db=Σδ e
δ anterior=(δWᵀ)⊙(1−A²). Na saída sigmoid+BCE, dL/du=(p−y)/B.
A divisão por B ocorre uma vez, na saída; cada gradiente propaga a perda média.
Todos os gradientes são calculados antes da atualização θ←θ−η∇L.
Diferenças finitas centrais verificaram cada peso e viés em redes binária e multiclasse com
profundidades diferentes: maior erro absoluto **{r["gradient_check_max_error"]:.3e}**.

![Figura 2 — Perda de treino](figures/figure2.png)

A curva registra a perda no treino completo após cada época, sem misturar perdas de pesos diferentes.

![Figura 3 — Regiões e pontos de teste](figures/figure3.png)

### D — Analysis

O MLP alcançou {binary["test_accuracy"]:.2%}, contra {r["baseline2"]:.2%} do baseline, porque combina
unidades ocultas em regiões não lineares. A Figura 3 atribui as duas extremidades à classe 1 e o
miolo à classe 0; uma única reta só divide o plano em dois semiplanos. Há sobreposição das Gaussianas,
então não se espera classificação perfeita. A fronteira não é inferida da acurácia: está desenhada
com predições da rede em uma grade, sobre os pontos de teste que não participaram do ajuste.

## Exercise 3

### A — Generate the data

make_classification usa 1.500 exemplos, quatro atributos informativos, nenhum redundante/repetido,
três classes, dois clusters por classe, class_sep=1 e random_state=42. Split estratificado 80/20:
1.200 exemplos de treino e 300 de teste. A padronização é ajustada no treino.

### B — Adapt the network

Reutilizamos sem alteração a classe MLP, inclusive forward, backward e atualização. Apenas o
argumento da camada de saída muda para três unidades. Para logits u, pₖ=exp(uₖ)/Σexp(uⱼ), calculados
subtraindo max(u) para estabilidade. Com alvo one-hot y e Σyₖ=1:

$$L=-\\sum_k y_k\\log p_k=-\\sum_k y_k u_k+\\log\\sum_j e^{{u_j}}.$$

Logo ∂L/∂uᵢ=−yᵢ+exp(uᵢ)/Σexp(uⱼ)=pᵢ−yᵢ; para B amostras, δ=(p−Y)/B.
A forma é a mesma da saída binária: o laço de retropropagação e o SGD não mudam.

### C — Train and evaluate

Arquitetura **4 → 32 (tanh) → 3 (softmax)**, **{a["parameters"]} parâmetros**. Mesmo SGD, taxa 0,05,
batch 64 e 800 épocas. Na seed 42: perda final de treino **{a["train_loss"]:.6f}** e acurácia de teste
**{a["test_accuracy"]:.2%}**. Baseline logístico: **{r["baseline3"]:.2%}**.

![Figura 4 — Entropia cruzada](figures/figure4.png)

![Figura 5 — Matriz no teste](figures/figure5.png)

### D — Analysis

Há empate entre os pares **{tied_pairs}**, cada um com **{count} erros nas duas direções**.
Para o par {i}/{j}, as contagens são
({c[i, j]} de {i} para {j}; {c[j, i]} de {j} para {i}). As quatro dimensões informativas descrevem
seis clusters: uma classe não corresponde necessariamente a um único grupo convexo. A camada
oculta permite compor regiões para esses grupos; o baseline tem fronteiras lineares. Essa estrutura
é uma explicação plausível da confusão, não uma identificação de clusters observáveis em uma única
projeção 2D. O teste mede a rede congelada após o orçamento fixado.

## Exercise 4

### A — Train the deeper network

Arquitetura **4 → 32 (tanh) → 32 (tanh) → 3 (softmax)**. Mantivemos o dataset, split, escala,
SGD, taxa, batch e 800 épocas. Só acrescentamos a camada de 32 unidades. Usamos seeds 42, 43 e 44
nas duas arquiteturas, com os mesmos embaralhamentos de batches para cada seed. Os formatos de
pesos são diferentes, portanto seed igual não implica pesos idênticos em todas as camadas.

### B — Compare

| Rede (seed 42) | Camadas ocultas | Parâmetros | Perda final no treino | Acurácia no teste |
| --- | ---: | ---: | ---: | ---: |
| Exercise 3 | 1 | {a["parameters"]} | {a["train_loss"]:.6f} | {a["test_accuracy"]:.2%} |
| Exercise 4 | 2 | {b["parameters"]} | {b["train_loss"]:.6f} | {b["test_accuracy"]:.2%} |

![Figura 6 — Mesmo orçamento de treinamento](figures/figure6.png)

| Seed | Uma camada | Duas camadas |
| --- | ---: | ---: |
"""
for x, y in zip(shallow, deep):
    text += f"| {x['seed']} | {x['test_accuracy']:.2%} | {y['test_accuracy']:.2%} |\n"
text += f"""
Médias: **{mean1:.2%}** (uma camada) e **{mean2:.2%}** (duas), diferença
**{(mean2 - mean1) * 100:.3f} pontos percentuais**. Desvios amostrais entre seeds:
{np.std([x["test_accuracy"] for x in shallow], ddof=1) * 100:.3f} e
{np.std([x["test_accuracy"] for x in deep], ddof=1) * 100:.3f} pontos percentuais.

### C — Analysis

Nesta execução a rede profunda teve ganho médio pequeno, não uma vantagem universal. Ela tem
mais parâmetros e pode compor características de características; também torna a otimização mais
longa e não garante generalização melhor. A diferença em três seeds é descritiva, sem teste de
significância ou busca até obter um vencedor. Os seis clusters em quatro atributos já podem ser
representados com uma camada de 32 unidades. Profundidade tende a ser mais útil em estruturas
composicionais, como partes de imagens combinadas em objetos, desde que existam dados e orçamento
adequados. A Figura 6 e a tabela distinguem ajuste de treino de generalização no teste.

## Results summary

| # | Quantity | Value |
| --- | --- | --- |
| 1 | Ex. 1 — z e h | z=[0.270000,−0.180000]; h=[0.263625,−0.178081] |
| 2 | Ex. 1 — u, predição e perda | 0.385237; 0.367247; 0.400377 |
| 3 | Ex. 1 — parâmetros atualizados | W1=[[0.376405,−0.130562],[0.152295,0.419082]]; b1=[0.252811,−0.295410]; W2=[0.586587,−0.358490]; b2=0.528448. São nove escalares, apesar do texto citar oito. |
| 4 | Ex. 2 — baseline no teste | {r["baseline2"]:.2%} |
| 5 | Ex. 2 — arquitetura e acurácia | 2–32–1; {binary["test_accuracy"]:.2%} |
| 6 | Ex. 3 — arquitetura e acurácia | 4–32–3; {a["test_accuracy"]:.2%} |
| 7 | Ex. 3 — par mais confundido | Empate: {tied_pairs}, {count} erros bidirecionais por par |
| 8 | Ex. 4 — arquitetura e acurácia | 4–32–32–3; {b["test_accuracy"]:.2%} |
| 9 | Ex. 4 − Ex. 3, média de três seeds | {(mean2 - mean1) * 100:.3f} pontos percentuais |
"""
page.write_text(prefix.rstrip() + "\n\n" + text.lstrip())
