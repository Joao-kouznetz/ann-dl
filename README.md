# ANN-DL

Portfólio de exercícios de **Artificial Neural Networks and Deep Learning**.

- Site publicado: <https://Joao-kouznetz.github.io/ann-dl/>
- Exercício Data: <https://Joao-kouznetz.github.io/ann-dl/exercises/data/>
- Exercício Perceptron: <https://Joao-kouznetz.github.io/ann-dl/exercises/perceptron/>
- Exercício MLP: <https://Joao-kouznetz.github.io/ann-dl/exercises/mlp/>
- Exercício VAE: <https://Joao-kouznetz.github.io/ann-dl/exercises/vae/>

## Executar localmente

```bash
python3 -m pip install -r requirements.txt
mkdocs serve
```

O notebook e o dataset do exercício Data ficam em `docs/exercises/data/code/`.
Os notebooks dos Exercícios 1 e 2 do Perceptron ficam em `docs/exercises/perceptron/code/`.
Para recriar os resultados e as seis figuras:

```bash
python docs/exercises/perceptron/code/run_analysis.py
```

Para reproduzir os exercícios MLP e VAE:

```bash
python docs/exercises/mlp/code/run_analysis.py
python docs/exercises/mlp/code/write_report.py
python -m pip install -r docs/exercises/vae/code/requirements.txt
python docs/exercises/vae/code/prepare_data.py
python docs/exercises/vae/code/train.py
python docs/exercises/vae/code/write_report.py
```

O script baixa o MNIST na primeira execução. Os dados ficam fora do controle de versão.

Notebooks: [MLP](docs/exercises/mlp/code/MLP.ipynb) e [VAE](docs/exercises/vae/code/VAE.ipynb).
O MLP usa apenas NumPy na rede; o VAE usa PyTorch CPU. O download e o treinamento
podem levar alguns minutos. As sementes, curvas e métricas medidas estão nos relatórios.

MLP e VAE foram concluídos com auxílio de inteligência artificial (OpenAI Codex),
incluindo código, execução dos experimentos, análises e redação.
