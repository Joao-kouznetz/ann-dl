# ANN-DL

Portfólio de exercícios de **Artificial Neural Networks and Deep Learning**.

- Site publicado: <https://Joao-kouznetz.github.io/ann-dl/>
- Exercício Data: <https://Joao-kouznetz.github.io/ann-dl/exercises/data/>
- Exercício Perceptron: <https://Joao-kouznetz.github.io/ann-dl/exercises/perceptron/>
- Exercício MLP (Exercise 1): <https://Joao-kouznetz.github.io/ann-dl/exercises/mlp/>
- Exercício VAE (início — Exercise 1A): <https://Joao-kouznetz.github.io/ann-dl/exercises/vae/>

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

Para reproduzir a preparação do MNIST do exercício VAE:

```bash
python -m pip install -r docs/exercises/vae/code/requirements.txt
python docs/exercises/vae/code/prepare_data.py
```

O script baixa o MNIST na primeira execução. Os dados ficam fora do controle de versão.
