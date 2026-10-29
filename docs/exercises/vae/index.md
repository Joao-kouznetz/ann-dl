---
exercise: vae
ai_use: "OpenAI Codex auxiliou na implementação e execução da preparação do MNIST, na documentação dos resultados e na publicação desta página."
---

# VAE — Preparação do MNIST

**Autor:** João Bresser Pereira  
**Disciplina:** Artificial Neural Networks and Deep Learning — 2026.2

[Enunciado da atividade](https://insper.github.io/ann-dl/2026.2/exercises/vae/) · [Código-fonte do relatório](https://github.com/Joao-kouznetz/ann-dl/blob/main/docs/exercises/vae/index.md)

**Status: em andamento.** Esta página apresenta o início da atividade: o **Exercise 1, item A**, implementado e executado. Os modelos ainda não foram construídos ou treinados; as métricas e as figuras das etapas seguintes permanecem pendentes.

## Exercise 1

### A — Load and split the data

O MNIST foi carregado com `torchvision.datasets.MNIST`. Cada imagem tem 28 × 28 pixels em um único canal. A transformação `ToTensor()` converte os valores inteiros dos pixels para `float32` no intervalo de 0 a 1. Em seguida, `torch.flatten` transforma cada imagem em um vetor de **784 valores**.

As 60.000 imagens do conjunto de treinamento original foram divididas usando `random_split`, com um gerador próprio inicializado pela semente **42**. O conjunto oficial de teste foi carregado separadamente.

| Conjunto | Número de imagens | Origem | Uso previsto |
| --- | ---: | --- | --- |
| Treino | **50.000** | Subconjunto do treino original | Ajustar os parâmetros do modelo |
| Validação | **10.000** | Restante do treino original | Acompanhar o treinamento |
| Teste | **10.000** | Partição oficial de teste | Calcular os resultados finais |

A divisão é aleatória, **sem estratificação**. As contagens observadas em treino e validação foram:

| Dígito | Treino | Validação |
| --- | ---: | ---: |
| 0 | 4.923 | 1.000 |
| 1 | 5.613 | 1.129 |
| 2 | 4.958 | 1.000 |
| 3 | 5.093 | 1.038 |
| 4 | 4.863 | 979 |
| 5 | 4.546 | 875 |
| 6 | 4.933 | 985 |
| 7 | 5.204 | 1.061 |
| 8 | 4.882 | 969 |
| 9 | 4.985 | 964 |
| **Total** | **50.000** | **10.000** |

Os índices de treino e validação não se sobrepõem e, juntos, cobrem todas as 60.000 imagens originais. Como referência para reproduzir a divisão, os primeiros cinco índices de treino são **37542, 43087, 2992, 25810 e 39634**; os de validação são **59753, 9414, 10988, 15501 e 49861**. Esses índices se referem ao conjunto original de treinamento.

Um lote de treino foi inspecionado com sucesso: formato **`(128, 784)`**, tipo **`torch.float32`**, mínimo **0.0** e máximo **1.0**, sem valores não finitos. Os carregadores usam lotes de **128 imagens**; somente o treino tem embaralhamento. O último lote pode ser menor.

O carregador de teste está preparado para as próximas etapas, mas não foi percorrido nesta execução. Nenhuma métrica de teste foi calculada. Os rótulos são preservados para as visualizações futuras; a reconstrução do VAE utilizará a própria imagem como alvo.

**Como reproduzir**

Na raiz do repositório, execute:

```bash
python -m pip install -r docs/exercises/vae/code/requirements.txt
python docs/exercises/vae/code/prepare_data.py
```

A primeira execução precisa de acesso à internet para baixar o MNIST. O script salva os dados em `docs/exercises/vae/code/data/`, pasta ignorada pelo Git, e grava as verificações em [data_summary.json](code/data_summary.json). O ambiente utilizado tinha **PyTorch 2.14.1** e **torchvision 0.29.1**, registrados também no arquivo de dependências.

**Código executado**

[Baixar prepare_data.py](code/prepare_data.py) · [Dependências](code/requirements.txt)

```python
--8<-- "docs/exercises/vae/code/prepare_data.py"
```

### B — Build the model

Pendente. A próxima etapa é implementar o VAE, incluindo a reparametrização e os dois termos da perda.

### C — Train

Pendente. Ainda não há treinamento, curvas de perda ou resultados finais do modelo.

### D — Reconstructions and samples

Pendente. As reconstruções e as amostras serão produzidas depois do treinamento.

### E — Analysis

Pendente. As respostas serão fundamentadas nos resultados medidos e nas figuras das etapas anteriores.

## Exercise 2

Não iniciado. A exploração do espaço latente ficará para a continuação da atividade.

## Results summary

O item 1A foi executado e os dados estão preparados. Os valores abaixo dependem de modelos treinados; por isso estão explicitamente identificados como pendentes.

| # | Quantity | Value |
| --- | --- | --- |
| 1 | Ex. 1 — test reconstruction, KL and negative ELBO ($d=16$) | Pendente — modelo ainda não treinado. |
| 2 | Ex. 1 — validation KL at the first and at the last epoch | Pendente — treinamento ainda não iniciado. |
| 3 | Ex. 2 — test reconstruction and KL ($d=2$) | Pendente — Exercise 2 não iniciado. |
| 4 | Ex. 2 — the two test indices interpolated | Pendente — imagens ainda não selecionadas. |
