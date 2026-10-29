---
exercise: mlp
ai_use: "OpenAI Codex auxiliou na elaboração e conferência dos cálculos, na redação das derivações em Markdown e na publicação da página."
---

# MLP — Backpropagation passo a passo

**Autor:** João Bresser Pereira  
**Disciplina:** Artificial Neural Networks and Deep Learning — 2026.2

[Enunciado da atividade](https://insper.github.io/ann-dl/2026.2/exercises/mlp/) · [Código-fonte deste relatório](https://github.com/Joao-kouznetz/ann-dl/blob/main/docs/exercises/mlp/index.md)

Esta entrega abrange **somente o Exercise 1**, itens A, B e C. Os Exercises 2, 3 e 4 não foram realizados nesta entrega. O desenvolvimento abaixo é analítico, com as substituições numéricas explícitas; os valores são exibidos com seis casas decimais, mantendo a precisão completa nos cálculos intermediários.

## Exercise 1

Uma etapa de backpropagation em uma rede com duas entradas, dois neurônios ocultos e uma saída. As duas camadas usam a ativação $\tanh$ e a perda, para uma única amostra, é:

$$
L=(y-\hat y)^2.
$$

Os dados e os parâmetros iniciais são:

$$
\mathbf x=\begin{bmatrix}0.5\\-0.2\end{bmatrix},\qquad y=1.0,\qquad \eta=0.3,
$$

$$
\mathbf W^{(1)}=\begin{bmatrix}0.3&-0.1\\0.2&0.4\end{bmatrix},\qquad
\mathbf b^{(1)}=\begin{bmatrix}0.1\\-0.2\end{bmatrix},
$$

$$
\mathbf W^{(2)}=\begin{bmatrix}0.5&-0.3\end{bmatrix},\qquad b^{(2)}=0.2.
$$

Cada linha de $\mathbf W^{(1)}$ corresponde a um neurônio oculto. Embora o enunciado mencione oito parâmetros, os valores fornecidos definem **nove parâmetros escalares**: quatro pesos ocultos, dois vieses ocultos, dois pesos de saída e um viés de saída. Todos são atualizados no item C.

### A — Forward pass

**1. Pré-ativações da camada oculta**

$$
\mathbf z^{(1)}=\mathbf W^{(1)}\mathbf x+\mathbf b^{(1)}.
$$

Para cada neurônio:

$$
z^{(1)}_1=0.3(0.5)+(-0.1)(-0.2)+0.1
=0.15+0.02+0.1=\boxed{0.270000},
$$

$$
z^{(1)}_2=0.2(0.5)+0.4(-0.2)-0.2
=0.10-0.08-0.2=\boxed{-0.180000}.
$$

**2. Ativações da camada oculta**

Aplicando $\tanh(z)=(e^{2z}-1)/(e^{2z}+1)$ a cada componente:

$$
h^{(1)}_1=\tanh(0.27)=\boxed{0.263625},\qquad
h^{(1)}_2=\tanh(-0.18)=\boxed{-0.178081}.
$$

**3. Pré-ativação da saída**

$$
\begin{aligned}
u^{(2)}&=\mathbf W^{(2)}\mathbf h^{(1)}+b^{(2)}\\
&=0.5(0.2636248355)+(-0.3)(-0.1780808681)+0.2\\
&=0.1318124177+0.0534242604+0.2\\
&=\boxed{0.385237}.
\end{aligned}
$$

**4. Predição e perda**

$$
\hat y=\tanh(0.3852366782)=\boxed{0.367247},
$$

$$
L=(1-0.3672465626)^2=\boxed{0.400377}.
$$

A saída está abaixo do alvo $y=1$; a atualização deve aproximar a predição desse alvo para reduzir a perda.

### B — Backward pass

Usamos a regra da cadeia, começando pela perda e propagando os gradientes até os parâmetros da primeira camada. A derivada da ativação é:

$$
\frac{d\tanh(v)}{dv}=1-\tanh^2(v).
$$

**1. Gradiente em relação à predição e à pré-ativação da saída**

$$
\frac{\partial L}{\partial\hat y}
=\frac{\partial(y-\hat y)^2}{\partial\hat y}
=2(\hat y-y)
=2(0.3672465626-1)
=\boxed{-1.265507}.
$$

$$
\frac{\partial\hat y}{\partial u^{(2)}}
=1-\hat y^2=1-(0.3672465626)^2
=\boxed{0.865130}.
$$

Definindo $\delta^{(2)}=\partial L/\partial u^{(2)}$:

$$
\delta^{(2)}
=\frac{\partial L}{\partial\hat y}\frac{\partial\hat y}{\partial u^{(2)}}
=(-1.2655068747)(0.8651299622)
=\boxed{-1.094828}.
$$

**2. Gradientes dos pesos e do viés de saída**

Como $u^{(2)}=W^{(2)}_1h^{(1)}_1+W^{(2)}_2h^{(1)}_2+b^{(2)}$, temos $\partial u^{(2)}/\partial W^{(2)}_j=h^{(1)}_j$ e $\partial u^{(2)}/\partial b^{(2)}=1$. Portanto:

$$
\frac{\partial L}{\partial W^{(2)}_1}
=\delta^{(2)}h^{(1)}_1
=(-1.0948279147)(0.2636248355)
=\boxed{-0.288624},
$$

$$
\frac{\partial L}{\partial W^{(2)}_2}
=\delta^{(2)}h^{(1)}_2
=(-1.0948279147)(-0.1780808681)
=\boxed{0.194968},
$$

$$
\frac{\partial L}{\partial b^{(2)}}
=\delta^{(2)}=\boxed{-1.094828}.
$$

**3. Propagação até as ativações e pré-ativações ocultas**

Primeiro, $\partial u^{(2)}/\partial h^{(1)}_j=W^{(2)}_j$:

$$
\frac{\partial L}{\partial h^{(1)}_1}
=\delta^{(2)}W^{(2)}_1
=(-1.0948279147)(0.5)
=\boxed{-0.547414},
$$

$$
\frac{\partial L}{\partial h^{(1)}_2}
=\delta^{(2)}W^{(2)}_2
=(-1.0948279147)(-0.3)
=\boxed{0.328448}.
$$

As derivadas locais das ativações ocultas são:

$$
1-(h^{(1)}_1)^2=\boxed{0.930502},\qquad
1-(h^{(1)}_2)^2=\boxed{0.968287}.
$$

Assim, definindo $\delta^{(1)}_j=\partial L/\partial z^{(1)}_j$:

$$
\delta^{(1)}_1
=\frac{\partial L}{\partial h^{(1)}_1}\bigl(1-(h^{(1)}_1)^2\bigr)
=(-0.5474139574)(0.9305019461)
=\boxed{-0.509370},
$$

$$
\delta^{(1)}_2
=\frac{\partial L}{\partial h^{(1)}_2}\bigl(1-(h^{(1)}_2)^2\bigr)
=(0.3284483744)(0.9682872044)
=\boxed{0.318032}.
$$

Os pesos utilizados nessa propagação são os **pesos originais** da saída. Nenhum parâmetro é alterado antes de todos os gradientes serem calculados.

**4. Gradientes dos pesos e dos vieses ocultos**

De $z^{(1)}_j=\sum_iW^{(1)}_{ji}x_i+b^{(1)}_j$, obtemos:

$$
\frac{\partial L}{\partial W^{(1)}_{ji}}=\delta^{(1)}_jx_i,\qquad
\frac{\partial L}{\partial b^{(1)}_j}=\delta^{(1)}_j.
$$

Aplicando a expressão a cada peso:

$$
\begin{aligned}
\frac{\partial L}{\partial W^{(1)}_{11}}&=(-0.5093697527)(0.5)=\boxed{-0.254685},\\
\frac{\partial L}{\partial W^{(1)}_{12}}&=(-0.5093697527)(-0.2)=\boxed{0.101874},\\
\frac{\partial L}{\partial W^{(1)}_{21}}&=(0.3180323583)(0.5)=\boxed{0.159016},\\
\frac{\partial L}{\partial W^{(1)}_{22}}&=(0.3180323583)(-0.2)=\boxed{-0.063606}.
\end{aligned}
$$

Para os vieses:

$$
\frac{\partial L}{\partial b^{(1)}_1}=\boxed{-0.509370},\qquad
\frac{\partial L}{\partial b^{(1)}_2}=\boxed{0.318032}.
$$

Em forma matricial, os gradientes da primeira camada são:

$$
\frac{\partial L}{\partial\mathbf W^{(1)}}=
\begin{bmatrix}-0.254685&0.101874\\0.159016&-0.063606\end{bmatrix},\qquad
\frac{\partial L}{\partial\mathbf b^{(1)}}=
\begin{bmatrix}-0.509370\\0.318032\end{bmatrix}.
$$

### C — Parameter update

A descida do gradiente atualiza cada parâmetro pela regra:

$$
\theta_{\mathrm{novo}}=\theta_{\mathrm{original}}-0.3\frac{\partial L}{\partial\theta}.
$$

Todas as atualizações usam os gradientes do mesmo forward pass. A tabela mostra a substituição e o resultado para cada um dos nove parâmetros; os resultados usam os gradientes sem arredondamento.

| Parâmetro | Valor original | Gradiente | Atualização com $\eta=0.3$ | Novo valor |
| --- | ---: | ---: | --- | ---: |
| $W^{(1)}_{11}$ | 0.300000 | −0.254685 | $0.3-0.3(-0.2546848763)$ | **0.376405** |
| $W^{(1)}_{12}$ | −0.100000 | 0.101874 | $-0.1-0.3(0.1018739505)$ | **−0.130562** |
| $W^{(1)}_{21}$ | 0.200000 | 0.159016 | $0.2-0.3(0.1590161791)$ | **0.152295** |
| $W^{(1)}_{22}$ | 0.400000 | −0.063606 | $0.4-0.3(-0.0636064717)$ | **0.419082** |
| $b^{(1)}_1$ | 0.100000 | −0.509370 | $0.1-0.3(-0.5093697527)$ | **0.252811** |
| $b^{(1)}_2$ | −0.200000 | 0.318032 | $-0.2-0.3(0.3180323583)$ | **−0.295410** |
| $W^{(2)}_1$ | 0.500000 | −0.288624 | $0.5-0.3(-0.2886238289)$ | **0.586587** |
| $W^{(2)}_2$ | −0.300000 | 0.194968 | $-0.3-0.3(0.1949679055)$ | **−0.358490** |
| $b^{(2)}$ | 0.200000 | −1.094828 | $0.2-0.3(-1.0948279147)$ | **0.528448** |

**A perda aumenta ou diminui?**

Ela **diminui** neste caso. Para confirmar com os parâmetros atualizados, repetimos o forward pass:

$$
\begin{aligned}
z^{(1)}_{1,\mathrm{novo}}
&=0.3764054629(0.5)+(-0.1305621852)(-0.2)+0.2528109258\\
&=\boxed{0.467126},\\
z^{(1)}_{2,\mathrm{novo}}
&=0.1522951463(0.5)+0.4190819415(-0.2)-0.2954097075\\
&=\boxed{-0.303079}.
\end{aligned}
$$

$$
\mathbf h^{(1)}_{\mathrm{novo}}=
\begin{bmatrix}\tanh(0.4671260943)\\\tanh(-0.3030785226)\end{bmatrix}
=\begin{bmatrix}0.435874\\-0.294127\end{bmatrix}.
$$

$$
\begin{aligned}
u^{(2)}_{\mathrm{novo}}
&=0.5865871487(0.4358743310)+(-0.3584903716)(-0.2941273491)+0.5284483744\\
&=\boxed{0.889568},\\
\hat y_{\mathrm{novo}}&=\tanh(0.8895684781)=\boxed{0.711181},\\
L_{\mathrm{novo}}&=(1-0.7111805295)^2=\boxed{0.083417}.
\end{aligned}
$$

A predição passa de **0.367247** para **0.711181**, aproximando-se do alvo **1.000000**, e a perda cai de **0.400377** para **0.083417**. O gradiente indica a direção local de maior aumento da perda; subtraí-lo produz uma direção de descida. Uma taxa de aprendizado arbitrariamente grande não garante redução, mas a segunda avaliação confirma que $\eta=0.3$ reduz a perda para esta amostra.

## Results summary

As linhas 1 a 3 resumem o exercício realizado. As demais linhas são mantidas na ordem do enunciado e identificam explicitamente os itens fora do escopo desta entrega.

| # | Quantity | Value |
| --- | --- | --- |
| 1 | Ex. 1 — $\mathbf z^{(1)}$ e $\mathbf h^{(1)}$ | $\mathbf z^{(1)}=[0.270000,-0.180000]^T$; $\mathbf h^{(1)}=[0.263625,-0.178081]^T$. |
| 2 | Ex. 1 — $u^{(2)}$, $\hat y$ e $L$ | $u^{(2)}=0.385237$; $\hat y=0.367247$; $L=0.400377$. |
| 3 | Ex. 1 — parâmetros atualizados | $\mathbf W^{(1)}=[[0.376405,-0.130562],[0.152295,0.419082]]$; $\mathbf b^{(1)}=[0.252811,-0.295410]^T$; $\mathbf W^{(2)}=[0.586587,-0.358490]$; $b^{(2)}=0.528448$. São nove parâmetros; a perda após a atualização é $0.083417$. |
| 4 | Ex. 2 — acurácia do baseline linear no teste | Não realizado nesta entrega. |
| 5 | Ex. 2 — arquitetura e acurácia no teste | Não realizado nesta entrega. |
| 6 | Ex. 3 — arquitetura e acurácia no teste | Não realizado nesta entrega. |
| 7 | Ex. 3 — par de classes mais confundido | Não realizado nesta entrega. |
| 8 | Ex. 4 — arquitetura e acurácia no teste | Não realizado nesta entrega. |
| 9 | Ex. 4 — diferença média de acurácia em relação ao Ex. 3, com pelo menos três sementes | Não realizado nesta entrega. |
