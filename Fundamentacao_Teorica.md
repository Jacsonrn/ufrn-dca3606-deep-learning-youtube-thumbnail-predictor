# 🧠 Fundamentação Teórica do Projeto: Deep Learning na Predição de Performance de Thumbnails

Este documento centraliza a teoria matemática, arquitetural e estratégica por trás de todas as decisões do código, **organizado estritamente pelo cronograma de dias de desenvolvimento do projeto** (Unidade 2 — DCA3606 / UFRN). Ele serve como documentação de referência para o grupo e guia de preparação para a arguição técnica.

---

# 📅 DIA 1: Pipeline de Classificação (Predição de Potencial Viral) e Fundações de Visão Computacional

Neste primeiro dia, estabelecemos a infraestrutura de dados em visão computacional, criamos a variável-alvo de classificação sob protocolo estrito anti-vazamento (*data leakage*) e implementamos as **5 abordagens obrigatórias** em dois regimes de resolução:
1. **Baseline de Baixa Resolução:** $64 \times 64 \times 3$ RGB (executado em CPU local).
2. **Alta Resolução Nativa Máxima:** $480 \times 360 \times 3$ RGB (executado com aceleração em GPU NVIDIA A100-SXM4-80GB).

---

### 📊 Tabela Empírica Consolidada: Baseline (64x64) vs. Alta Resolução (480x360)

| Modelo | Acurácia (64x64) | Acurácia (480x360) | $\Delta$ Acurácia | Macro-F1 (64x64) | Macro-F1 (480x360) | $\Delta$ Macro-F1 | Recall (64x64) | Recall (480x360) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Pixels)** | 56,08% | **61,57%** | **+5,49%** | 55,83% | **61,35%** | **+5,52%** | 58,70% | 63,77% |
| **CNN 2D** | 54,12% | **58,04%** | **+3,92%** | 52,96% | **55,03%** | **+2,07%** | 64,49% | **77,54%** |
| **ResNet 2D** | 54,90% | 53,33% | -1,57% | 53,39% | 39,48% | -13,90% | 67,39% | **93,48%** |
| **MLP / DNN** | 49,02% | 47,84% | -1,18% | 48,73% | 46,66% | -2,08% | 52,17% | 30,43% |
| **RF (Embeddings AE)** | 48,24% | 45,10% | -3,14% | 48,22% | 45,08% | -3,14% | 46,38% | 39,86% |

---

### 📚 Fundamentação Teórica e Análise Arquitetural do Dia 1

#### 1. Formulação Tensorial da Imagem e o Dilema da Resolução Espacial
Em Visão Computacional com Deep Learning, uma imagem digital colorida é representada como um tensor tridimensional de ordem 3:
$$X_i \in \mathbb{R}^{C \times H \times W} \quad \text{onde } C=3 \text{ (RGB)}$$
* **Normalização em $[0, 1]$:** Os canais de cor originais (inteiros em $[0, 255]$) foram reescalados dividindo por $255.0$. Isso centraliza e homogeneíza a variância de entrada, impedindo a saturação imediata de gradientes em funções de ativação.
* **O Impacto do Salto de Resolução ($64 \times 64 \to 480 \times 360$):**
  - Em $64 \times 64$, cada imagem possui $12.288$ escalares de entrada.
  - Em $480 \times 360$ (resolução nativa das thumbnails `hqdefault.jpg` do YouTube), cada imagem possui **$518.400$ escalares** (um aumento de **$42{,}2 \times$** no volume de dados brutos).
  - O aumento de resolução preserva frequências espaciais altas: contornos nítidos de tipografia de *clickbait*, detalhes de saturação e microexpressões em rostos humanos.

#### 2. Protocolo Anti-Vazamento (*Data Leakage*) na Binarização do Alvo
A variável contínua de visualizações ($\text{views} \in \mathbb{N}$) foi binarizada em **Alto Engajamento** ($y=1$) vs. **Baixo Engajamento** ($y=0$) utilizando a **Mediana** como limiar de corte ($\tau$):
$$y_{\text{class}} = \begin{cases} 1, & \text{se } \text{views} \ge \tau \\ 0, & \text{se } \text{views} < \tau \end{cases}$$
* **A Regra de Ouro:** O limiar $\tau = 610.055$ visualizações foi computado **estritamente sobre o conjunto de Treino (70%)**.
* **Justificativa Teórica:** O cálculo da mediana sobre o conjunto total antes do particionamento violaria a independência das distribuições amostrais, vazando parâmetros dos dados de teste para a rotulagem de treino.

#### 3. A Falha Teórica da MLP e a Maldição da Dimensionalidade
A MLP exige o achatamento (*flatten*) da matriz bidimensional em um vetor $x_{\text{flat}} \in \mathbb{R}^D$:
* **Em $64 \times 64$ ($D = 12.288$):** A primeira camada linear ($12.288 \to 512$) exigiu $6{,}29$ milhões de parâmetros.
* **Em $480 \times 360$ ($D = 518.400$):** A mesma primeira camada linear ($518.400 \to 512$) explodiu para **$265.421.312$ parâmetros treináveis** (~265 milhões de pesos).
* **Consequência Matemática:** Para apenas $1.186$ amostras de treino, a razão amostra/parâmetro caiu para $1.186 / 265.421.312 \approx 4{,}4 \times 10^{-6}$. O modelo colapsou em memorização rápida (overfitting imediato na época 7), atingindo apenas 47,84% de acurácia no teste e despencando o Recall para 30,43%. A ausência de invariância translacional tornou a MLP conceitualmente inviável em alta resolução.

#### 4. Convolução 2D e o Desacoplamento de Resolução por Pooling Adaptativo
A **CNN 2D** superou as limitações da MLP através do operador de convolução discreta bidimensional:
$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) \cdot K(m, n)$$
* **Campos Receptivos Locais e Compartilhamento de Pesos:** Os filtros $3 \times 3$ deslizam por toda a thumbnail, detectando padrões locais (bordas, texturas) independentemente de sua coordenada espacial.
* **Desacoplamento via `AdaptiveAvgPool2d((4, 4))`:** A transição dos mapas de feições convolucionais para o classificador linear ocorre através de pooling adaptativo. Isso manteve a arquitetura com exatamente **$358.306$ parâmetros**, tanto em $64 \times 64$ quanto em $480 \times 360$.
* **Ganho Empírico:** Com acesso a detalhes visuais finos em $480 \times 360$, a CNN saltou de 54,12% para **58,04% de acurácia** (+3,92%) e o Recall subiu de 64,49% para **77,54%** (+13,05%).

#### 5. ResNet 2D e a "Superestrada de Gradientes": Detecção de Virais
A **ResNet** incorpora os Blocos Residuais básicos com conexões de atalho:
$$\mathcal{H}(x) = \mathcal{F}(x, \{W_i\}) + x \implies \frac{\partial \mathcal{H}}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + I$$
* **Comportamento em Alta Resolução:** A preservação do gradiente pelo termo identidade $I$ permitiu à ResNet aprender combinações de traços primitivos e composições semânticas complexas.
* **Fenômeno do Recall de 93,48%:** A ResNet atingiu uma sensibilidade recorde na classe viral, capturando **129 das 138 thumbnails de alto engajamento** do conjunto de teste. A rede especializou-se em identificar qualquer sinal visual associado ao sucesso (faces em choque, textos coloridos), aceitando um volume maior de falsos positivos em troca de quase não perder vídeos virais reais.

#### 6. Autoencoder Convolucional e o Gargalo de Compressão Extrema
O Autoencoder Convolucional foi treinado de forma **não-supervisionada** com função de perda de reconstrução MSE:
$$\mathcal{L}_{\text{MSE}}(X, \widehat{X}) = \frac{1}{N} \sum_{i=1}^N \| X_i - \widehat{X}_i \|_2^2$$
* **Gargalo Latente Constante ($Z \in \mathbb{R}^{64}$):**
  - Em $64 \times 64$, a taxa de compressão era de $192 : 1$.
  - Em $480 \times 360$, a taxa de compressão atingiu **$8.100 : 1$** ($518.400$ números comprimidos em 64).
* **Colapso da Informação Útil:** A compressão extrema de $8.100:1$ forçou o encoder a descartar textos e expressões locais para reter apenas gradientes de fundo e iluminação média (fatores que minimizam o erro quadrático global da imagem). Por consequência, o classificador clássico treinado sobre esses embeddings caiu para 45,10% de acurácia.

#### 7. Superioridade do Random Forest nos Pixels Brutos
O Random Forest nos pixels brutos destacou-se como o modelo mais robusto e equilibrado:
* **Desempenho:** Atingiu **61,57% de Acurácia** e **61,35% de Macro-F1** em $480 \times 360$ (ganho de mais de +5,5% sobre a baixa resolução).
* **Fundamentação:** O particionamento não-paramétrico baseado no Índice de Impureza de Gini:
$$I_G(t) = 1 - \sum_{i=1}^C p(i|t)^2$$
permite ao ensemble de 100 árvores encontrar limiares de decisão independentes em regiões cruciais da imagem sem sofrer com instabilidades de gradiente ou armadilhas de mínimos locais em conjuntos de dados de escala moderada (~1.200 amostras).

---

### ❓ 15 Questões de Revisão Técnica — Dia 1 (Preparação para Apresentação e Arguição)

1. **Por que o limiar de classificação (mediana de visualizações) foi calculado exclusivamente no conjunto de treino?**
   * *Resposta esperada:* Para evitar o vazamento de dados (*data leakage*). Se a mediana fosse calculada sobre todo o conjunto, informações estatísticas do teste contaminariam o treino, invalidando o teste cego.

2. **Qual é a principal deficiência teórica de utilizar uma MLP para classificação de imagens?**
   * *Resposta esperada:* A perda da coerência espacial 2D e da invariância à translação ao achatar o tensor em um vetor 1D, além da explosão exponencial de parâmetros na primeira camada linear.

3. **Como a operação de convolução 2D supera as limitações da MLP em visão computacional?**
   * *Resposta esperada:* Através dos *campos receptivos locais* ($3 \times 3$) e do *compartilhamento de pesos* (mesmos filtros em toda a imagem), preservando as relações topológicas com menos parâmetros.

4. **O que é o problema do desvanecimento do gradiente e como os blocos residuais da ResNet o mitigam?**
   * *Resposta esperada:* Multiplicações sucessivas de derivadas menores que 1 atenuam o gradiente em redes profundas. A conexão de atalho $\mathcal{H}(x) = \mathcal{F}(x) + x$ garante uma derivada $\frac{\partial \mathcal{H}}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + I$, mantendo um fluxo de gradiente unitário mínimo constante.

5. **Por que a ResNet obteve o maior Recall de engajamento (67,39% em 64x64 e 93,48% em 480x360)?**
   * *Resposta esperada:* A facilidade de fluxo de gradiente permitiu à rede aprender padrões visuais sutis e correlacionados a thumbnails virais sem saturação, tornando-a um detector altamente sensível.

6. **Qual é a diferença fundamental no treinamento do Autoencoder Convolucional em relação aos demais modelos?**
   * *Resposta esperada:* O Autoencoder foi treinado de forma **não-supervisionada** via perda MSE de reconstrução da própria imagem, sem qualquer acesso aos dados de visualizações ou rótulos de engajamento.

7. **Por que o Random Forest treinado com embeddings do Autoencoder teve desempenho inferior ao treinado em pixels brutos?**
   * *Resposta esperada:* Porque a função de perda de reconstrução prioriza a variância visual global da imagem (iluminação e fundos), descartando detalhes locais e tipográficos que definem a decisão humana de clicar.

8. **Por que o Random Forest nos pixels brutos superou as redes neurais em acurácia geral?**
   * *Resposta esperada:* Modelos ensemble particionam o espaço por cortes ortogonais não-paramétricos sem depender de gradientes estocásticos, sendo imunes a platôs de otimização em bases de escala moderada (~1.200 amostras).

9. **Qual foi o papel do Early Stopping e da divisão com conjunto de validação?**
   * *Resposta esperada:* Monitorar a perda de generalização em dados não treinados a cada época, interrompendo o treino após 5 épocas consecutivas sem melhora e restaurando os melhores pesos salvos.

10. **Por que a métrica Macro-F1 foi priorizada em relação à Acurácia simples?**
    * *Resposta esperada:* O Macro-F1 calcula a média harmônica de precisão e sensibilidade para cada classe individualmente, impedindo que um modelo enviesado para a classe majoritária receba uma nota artificialmente alta.

11. **O que aconteceu com a MLP quando a resolução aumentou de $64 \times 64$ para $480 \times 360$?**
    * *Resposta esperada:* Ocorreu um colapso pela *Maldição da Dimensionalidade*. A entrada saltou para $518.400$ valores e a primeira camada atingiu $265$ milhões de pesos. Para apenas $1.186$ amostras de treino, a rede entrou em sobreajuste imediato.

12. **Por que o número de parâmetros da CNN não explodiu com o aumento da resolução como na MLP?**
    * *Resposta esperada:* Porque os kernels convolucionais compartilham pesos independentemente da dimensão da imagem, e a camada `AdaptiveAvgPool2d((4, 4))` converte qualquer resolução para uma dimensão fixa antes das camadas densas, mantendo fixos os ~358 mil pesos.

13. **Por que o aumento de resolução elevou a performance do Random Forest nos pixels brutos para 61,57%?**
    * *Resposta esperada:* Em $480 \times 360$, a nitidez das bordas do texto de *clickbait* e o contraste das faces tornaram-se visíveis nos pixels, permitindo que as árvores de decisão encontrassem limiares de corte muito mais informativos.

14. **Por que a taxa de compressão do Autoencoder de $8.100:1$ prejudicou o classificador posterior?**
    * *Resposta esperada:* Comprimir $518.400$ valores em um vetor latente $Z$ de apenas 64 dimensões ultrapassou a capacidade representacional do gargalo, descartando informações essenciais de engajamento para focar apenas em iluminâncias médias.

15. **Qual é o papel da aceleração por GPU no pipeline em alta resolução nativa?**
    * *Resposta esperada:* A GPU executa operações tensoriais em arquitetura massivamente paralela (SIMT), reduzindo o tempo de processamento de convoluções e retropropagação de várias horas em CPU para menos de dois minutos.
