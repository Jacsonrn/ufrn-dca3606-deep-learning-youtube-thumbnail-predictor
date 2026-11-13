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


---

# 📅 DIA 2: Pipeline de Regressão Contínua (Previsão do Número Real de Views)

No segundo dia, abordamos o desafio de prever diretamente a quantidade contínua de visualizações ($y \in \mathbb{R}^+$) a partir da thumbnail, enfrentando a forte assimetria das distribuições de cauda pesada (*Power-Law*) típicas de plataformas de compartilhamento de vídeo.

---

### 📊 Tabela Empírica Consolidada de Regressão: 64x64 (CPU) vs. 480x360 (A100)

| Modelo | R² Log (64x64) | R² Log (480x360) | Δ R² Log | MAE Log (64x64) | MAE Log (480x360) | Δ MAE Log | MAE Real (480x360) | Veredito Científico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Pixels)** | +4,06% | **+6,89%** | **+2,82%** | 0,8635 | **0,8457** | **-0,0178** | **24.127.787** | 🏆 **Campeão Geral:** Maior ganho com nitidez de alta resolução (+69% no R²). |
| **CNN 2D** | -0,27% | **+3,02%** | **+3,29%** | 0,8818 | **0,8687** | **-0,0132** | 24.193.754 | 🚀 **Melhor Rede Neural:** Virou de R² negativo para positivo com resolução nativa. |
| **MLP / DNN** | **+4,92%** | +2,24% | **-2,68%** | **0,8456** | 0,8546 | +0,0090 | 24.143.606 | 📉 **Degradação Dimensional:** Perdeu metade do R² devido aos 265M de parâmetros. |
| **ResNet 2D** | +0,07% | -1,08% | -1,15% | 0,8832 | 0,8876 | +0,0044 | 24.227.226 | ⚠️ Gradientes residuais geraram maior variância pontual em alvos contínuos. |
| **RF (Embeddings AE)** | -10,90% | -10,46% | +0,44% | 0,9354 | 0,9187 | -0,0167 | 24.265.423 | 🛑 Gargalo de compressão 8.100:1 eliminou feições contínuas de engajamento. |

---

### 📚 Fundamentação Teórica da Regressão Contínua

#### 1. A Matemática da Distribuição de Cauda Pesada (*Power-Law / Pareto*)
Em sistemas de mídia social, o consumo de conteúdo segue uma distribuição livre de escala: a grande maioria dos vídeos acumula poucos milhares de acessos, enquanto uma fração ínfima de vídeos virais ultrapassa dezenas de milhões de visualizações.  
Se o Erro Quadrático Médio ($\text{MSE}$) fosse minimizado diretamente na escala linear:
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \implies \frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -2(y_i - \hat{y}_i)$$
Para um vídeo com $y_i = 100.000.000$, um erro de $20\%$ gera um gradiente de magnitude $4 \times 10^7$, enquanto para um vídeo com $y_i = 10.000$, o mesmo erro percentual gera um gradiente de apenas $4 \times 10^3$ (uma disparidade de **$10.000 \times$**). O otimizador Adam seria monopolizado pelos outliers virais, desestabilizando os pesos da rede.

#### 2. Estabilização de Variância por Transformação Logarítmica
Para restaurar a **homocedasticidade** (variância constante dos resíduos ao longo de toda a reta preditiva), aplicamos a transformação monotônica estrita:
$$y_{\log} = \log_{10}(\text{views} + 1)$$
Em seguida, ajustamos um `StandardScaler` $(\mu_{\text{treino}}, \sigma_{\text{treino}})$ **exclusivamente sobre o conjunto de Treino**:
$$z_{\text{treino}} = \frac{y_{\log} - \mu_{\text{treino}}}{\sigma_{\text{treino}}}$$
Isso projeta a variável-alvo em uma distribuição padronizada $\sim \mathcal{N}(0, 1)$, permitindo convergência estável das redes neurais.

#### 3. Por que o $R^2$ Linear é Negativo e o $R^2_{\log}$ é Positivo?
O Coeficiente de Determinação é definido formalmente por:
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
Na escala linear bruta, o erro quadrático em vídeos com centenas de milhões de visualizações explode a soma dos quadrados dos resíduos ($\text{SS}_{\text{res}}$), tornando-a maior do que a variância amostral ($\text{SS}_{\text{tot}}$) e resultando em $R^2 \approx -0{,}008$.  
Na escala $\log_{10}$, os resíduos são homocedásticos, revelando que o **Random Forest explica 6,89% de toda a variância** e a **CNN explica 3,02%**, demonstrando capacidade preditiva genuína a partir de sinais visuais.

#### 4. Interpretação Física do $\text{MAE}_{\log} \approx 0{,}845$ dex
Em física e ciência de dados astronômicos, a unidade logarítmica base 10 é denominada **dex**. Um erro absoluto médio de $0{,}8457$ dex indica que:
$$\frac{\widehat{\text{views}}}{\text{views}} \approx 10^{\pm 0{,}8457} \approx [0{,}14; \; 7{,}0]$$
O modelo prevê o alcance do vídeo dentro de um fator médio de $\approx 7 \times$. Dado que o espaço amostral abrange 6 ordens de magnitude ($10^3$ a $10^9$), restringir a predição a menos de uma ordem de grandeza analisando apenas uma imagem estática comprova o aprendizado de padrões reais de interesse visual.

#### 5. Interpolação Contínua (*Piecewise Linear*) vs. Função Degrau (*Piecewise Constant*)
* **Random Forest:** Composto por árvores ortogonais que calculam a média das folhas, formando uma **função degrau**. Em $64 \times 64$, sua incapacidade de interpolar finamente fez a MLP superá-lo ligeiramente; em $480 \times 360$, a riqueza de limiares visuais finos permitiu ao ensemble atingir o ápice de $R^2_{\log} = 6{,}89\%$.
* **Redes Neurais (MLP e CNN):** Produzem uma **superfície contínua por partes**. Enquanto a MLP colapsou em $480 \times 360$ pela explosão de 265 milhões de parâmetros, a CNN 2D manteve estabilidade com pooling adaptativo (`AdaptiveAvgPool2d((4, 4))`), saltando de $R^2$ negativo para $+3{,}02\%$.

---

### ❓ 15 Questões de Revisão Técnica — Dia 2 (Regressão Contínua)

1. **Por que o modelo treinado diretamente na contagem bruta de visualizações desestabilizaria o gradiente?**
   * *Resposta esperada:* Porque a distribuição segue uma lei de potência com outliers de centenas de milhões de visualizações. O gradiente da perda MSE é proporcional ao erro linear $(y - \hat{y})$, fazendo com que poucos vídeos virais gerassem gradientes gigantescos que destruiriam os pesos da rede.

2. **Qual é o papel da transformação logarítmica $y_{\log} = \log_{10}(\text{views} + 1)$?**
   * *Resposta esperada:* Estabilizar a variância residual (garantir homocedasticidade) e comprimir as ordens de magnitude em uma distribuição simétrica próxima da normal, permitindo que o otimizador trate pequenos e grandes canais de forma equilibrada.

3. **Por que somamos $+1$ dentro do logaritmo ($\log_{10}(\text{views} + 1)$)?**
   * *Resposta esperada:* Para garantir estabilidade matemática quando $\text{views} = 0$, uma vez que $\log(0)$ é indefinido ($-\infty$), além de garantir que $\log_{10}(0 + 1) = 0$.

4. **Como garantimos ausência de vazamento de dados (*data leakage*) no `StandardScaler` da regressão?**
   * *Resposta esperada:* Ajustando os parâmetros de média ($\mu$) e desvio padrão ($\sigma$) estritamente sobre os dados de treino (`scaler.fit_transform`), e apenas aplicando esses parâmetros congelados nos conjuntos de validação e teste (`scaler.transform`).

5. **Por que o $R^2$ na escala linear de visualizações resultou em valores negativos (~ -0,008)?**
   * *Resposta esperada:* Porque a métrica $R^2$ quadrática é hipersensível a outliers extremos. Erros em vídeos com mais de 200 milhões de views elevam a soma dos quadrados dos resíduos além da variância total, gerando $R^2 < 0$, mesmo que o modelo acerte a grande maioria dos vídeos comuns.

6. **O que significa fisicamente obter um $\text{MAE}_{\log} = 0{,}8457$ em escala logarítmica?**
   * *Resposta esperada:* Significa que a predição média do modelo erra por um fator de $10^{0{,}8457} \approx 7 \times$ a ordem de grandeza real do vídeo, o que é uma estimativa consistente em uma plataforma onde as visualizações variam por um fator de mais de $1.000.000 \times$.

7. **Por que o Random Forest nos pixels brutos superou todas as redes neurais em $480 \times 360$ ($R^2_{\log} = 6{,}89\%$)?**
   * *Resposta esperada:* Porque o ensemble de árvores com `max_features='sqrt'` particiona os pixels nítidos de alta resolução em limiares ortogonais livres de gradiente, identificando elementos de alto contraste e textos de *clickbait* sem sofrer com a maldição da dimensionalidade.

8. **O que causou o travamento inicial do `RandomForestRegressor` em alta resolução no cluster?**
   * *Resposta esperada:* Por padrão no scikit-learn, o `RandomForestRegressor` adota `max_features=1.0` (avalia todas as $518.400$ features em cada nó), ao contrário do classificador que usa `sqrt`. Isso aumentou a carga computacional em mais de $720 \times$, travando a CPU na ordenação contínua de matrizes.

9. **Como a alteração para `max_features='sqrt'` resolveu o problema de complexidade do Random Forest?**
   * *Resposta esperada:* Reduziu o espaço de busca de $518.400$ para $\sqrt{518.400} = 720$ variáveis aleatórias por nó, acelerando o treinamento em mais de 700 vezes e concluindo em menos de 30 segundos.

10. **Por que o desempenho da MLP degradou de $R^2_{\log} = 4{,}92\%$ em $64 \times 64$ para $2{,}24\%$ em $480 \times 360$?**
    * *Resposta esperada:* Devido à *Maldição da Dimensionalidade*. A primeira camada linear inflou para mais de 265 milhões de parâmetros para apenas 1.186 exemplos de treino, forçando a rede a memorizar ruídos e perder poder de generalização.

11. **Por que a CNN 2D apresentou a maior evolução positiva ($\Delta R^2_{\log} = +3{,}29\%$) com o aumento da resolução?**
    * *Resposta esperada:* Porque os kernels convolucionais combinados com `AdaptiveAvgPool2d` mantiveram o número de parâmetros constante (~358 mil) enquanto extraíram feições ricas e nítidas dos pixels em alta definição, permitindo capturar texturas e expressões faciais informativas.

12. **Por que o Random Forest treinado nos embeddings do Autoencoder teve o pior $R^2_{\log}$ (-10,46%)?**
    * *Resposta esperada:* O gargalo de compressão latente de $8.100:1$ ($518.400 \to 64$) reteve apenas a iluminação de fundo e cores médias para reconstrução geométrica, descartando sinais locais correlacionados à variação contínua de engajamento.

13. **Qual é a diferença fundamental entre as saídas das redes de classificação e de regressão implementadas?**
    * *Resposta esperada:* Na classificação, a camada final projeta para $2$ logits com perda Cross-Entropy; na regressão, a camada final projeta para $1$ único neurônio linear contínuo com perda MSE, sem função de ativação de corte.

14. **O que o gráfico de distribuição de resíduos centrado em zero comprova?**
    * *Resposta esperada:* Comprova que as predições não possuem viés sistemático de calibração; os erros estão uniformemente distribuídos entre superestimar e subestimar, validando a eficácia da normalização logarítmica.

15. **Como as predições na escala logarítmica são convertidas para visualizações no mundo real?**
    * *Resposta esperada:* Invertendo a padronização via desvio e média do treino ($\hat{y}_{\log} = \hat{z} \cdot \sigma + \mu$) e aplicando a função exponencial inversa com corte não-negativo: $\widehat{\text{views}} = \max(0, \; 10^{\hat{y}_{\log}} - 1)$.

---

# 🟨 DIA 3: Análise dos Embeddings, Teoria da Informação de Shannon e Custo Computacional

## 1. Compressão Latente e a Teoria de Rate-Distortion de Shannon

O Autoencoder Convolucional treinado na Abordagem 4 opera sob os preceitos da **Teoria de Codificação com Perda (*Lossy Source Coding*) de Claude Shannon**. O objetivo da rede é aprender duas funções parametrizadas:
$$\text{Encoder: } f_\theta : \mathcal{X} \to \mathcal{Z}, \quad \text{com } \mathcal{X} \subset \mathbb{R}^d, \; \mathcal{Z} \subset \mathbb{R}^k \; (k \ll d)$$
$$\text{Decoder: } g_\phi : \mathcal{Z} \to \hat{\mathcal{X}}, \quad \text{onde } \hat{X} \approx X$$

A minimização da perda empírica sob a norma $\mathcal{L}_2$ (MSE):
$$\mathcal{L}_{\text{recon}}(\theta, \phi) = \frac{1}{N} \sum_{i=1}^N \| X_i - g_\phi(f_\theta(X_i)) \|_2^2$$

Força o vetor latente $Z \in \mathbb{R}^{64}$ a reter as componentes de maior energia do sinal espectral da imagem. 

### A Taxa de Compressão de Informação ($d \to k$)
* Em $64 \times 64 \times 3$: $d = 12.288 \implies \text{Compressão} = \frac{12.288}{64} = 192:1$ (eliminação de $99{,}48\%$ da dimensionalidade).
* Em $480 \times 360 \times 3$: $d = 518.400 \implies \text{Compressão} = \frac{518.400}{64} = 8.100:1$ (eliminação de $99{,}988\%$).

Pelo Teorema da Taxa de Distorção $R(D)$, ao comprimir agressivamente para 64 números reais, o encoder é matematicamente forçado a codificar **iluminância média, paleta cromática dominante e silhuetas macroscópicas**, descartando detalhes de alta frequência espacial (como tipografia de títulos embutidos e expressões de micro-olhares), os quais são cruciais para o usuário humano decidir clicar no vídeo.

---

## 2. Geometria de Variedades: PCA vs. t-SNE no Espaço Latente

Para inspecionar a topologia do espaço $\mathcal{Z} \subset \mathbb{R}^{64}$, aplicamos duas técnicas com premissas matemáticas distintas:

### 2.1 Análise de Componentes Principais (PCA — Linear Global)
O PCA computa a decomposição espectral da matriz de covariância empírica de $\mathbf{Z} \in \mathbb{R}^{N \times 64}$:
$$\Sigma = \frac{1}{N} (\mathbf{Z} - \bar{\mathbf{Z}})^T (\mathbf{Z} - \bar{\mathbf{Z}}) = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$$
Onde $\mathbf{V}$ contém os autovetores ortonormais e $\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_{64})$ contém as variâncias decrescentes. A projeção 2D projeta sobre os dois autovetores de maior autovalor, maximizando a dispersão global:
$$\mathbf{Z}_{\text{PCA}} = (\mathbf{Z} - \bar{\mathbf{Z}}) \mathbf{V}_{[:, :2]}$$
* **Limitação:** Como a transformação é uma rotação rígida ortogonal, ela falha em desenrolar variedades não-lineares intrínsecas da representação visual.

### 2.2 t-Distributed Stochastic Neighbor Embedding (t-SNE — Não-Linear Local)
O t-SNE modela as distâncias euclidianas no espaço $\mathbb{R}^{64}$ como probabilidades condicionais gaussianas:
$$p_{j|i} = \frac{\exp(-\|z_i - z_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|z_i - z_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
No espaço de baixa dimensão $\mathbb{R}^2$, utiliza a distribuição de Cauchy ($t$-Student com 1 grau de liberdade) para mitigar o problema de aglomeração (*crowding problem*):
$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_k \sum_{l \neq k} (1 + \|y_k - y_l\|^2)^{-1}}$$
A configuração dos pontos $\mathbf{Y} \in \mathbb{R}^{N \times 2}$ é otimizada minimizando a **Divergência de Kullback-Leibler**:
$$\mathcal{L}_{\text{t-SNE}} = D_{KL}(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \left( \frac{p_{ij}}{q_{ij}} \right)$$
O gradiente resultante atua como forças elásticas de atração entre pontos semanticamente próximos e repulsão para pontos distantes, revelando agrupamentos locais e a continuidade do gradiente de views.

---

## 3. Custo Computacional, Complexidade Algorítmica e Latência

### 3.1 Parâmetros e Complexidade Estrutural
* **MLP Densa:** A primeira camada linear acopla todos os pixels com 512 neurônios:
  $$\mathcal{P}_{\text{MLP}_1} = (d \times H_1) + H_1 = (12.288 \times 512) + 512 = 6.291.968 \text{ pesos}$$
  Totalizando **6,44 milhões de parâmetros** (24,57 MB). A MLP não explora invariância translacional.
* **CNN Convolucional 2D:** Utiliza compartilhamento espacial de pesos (*weight sharing*):
  $$\mathcal{P}_{\text{Conv}} = (C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w) + C_{\text{out}}$$
  Totalizando apenas **356 mil parâmetros** (1,36 MB) — **$18\times$ menos parâmetros que a MLP**.
* **ResNet 2D:** Adiciona blocos residuais sem aumentar a complexidade paramétrica excessivamente, totalizando **481 mil parâmetros** (1,84 MB).

### 3.2 O Paradoxo de Latência de Inferência vs. Quantidade de Parâmetros
O benchmark formal em CPU revelou que:
* **MLP:** Latência de $63$ a $87\text{ ms}$ para 100 imagens ($1.146$ a $1.573\text{ FPS}$).
* **CNN:** Latência de $725$ a $1.751\text{ ms}$ para 100 imagens ($57$ a $138\text{ FPS}$).
* **ResNet:** Latência de $3.064$ a $4.157\text{ ms}$ para 100 imagens ($24$ a $33\text{ FPS}$).

> [!NOTE]
> **Explicação Teórica:** Em arquiteturas de CPU modernas (x86 com instruções AVX-512), a MLP executa uma multiplicação densa contígua de matrizes (GEMM - *General Matrix Multiply*), atingindo máxima eficiência de pipeline de registradores. Já as redes convolucionais e residuais demandam expansão em memória de tensores via `im2col` e múltiplas operações de pooling e ramificação de grafo computacional, resultando em maior latência temporal apesar de utilizarem $18\times$ menos memória estática.

---

## 4. Banco de Arguição Oral — Dia 3: Embeddings e Custo Computacional (Questões 31 a 40)

31. **O que é o gargalo latente $Z$ e qual é a função da camada Sigmoid no final do Decoder?**
    * *Resposta esperada:* O gargalo latente $Z \in \mathbb{R}^{64}$ é a menor camada do autoencoder, cuja dimensão reduzida força a rede a descartar ruídos e reter apenas as características determinantes para a reconstrução. A ativação Sigmoid no final do decoder é mandatória porque os tensores de entrada foram normalizados no intervalo $[0, 1]$, garantindo que as intensidades dos canais RGB reconstruídos estejam limitadas a esse mesmo espaço físico contínuo.

32. **Por que o modelo clássico Random Forest treinado sobre os embeddings $Z$ foi de $5\times$ a $10\times$ mais rápido do que treinado sobre os pixels $X$?**
    * *Resposta esperada:* Devido à complexidade computacional da divisão de nós em árvores de decisão. Para $N$ amostras e dimensão $D$, avaliar os pontos de corte candidatos custa $\mathcal{O}(N \cdot D \log N)$. Ao reduzir a dimensão de $d = 12.288$ para $k = 64$ ($192\times$ menor), o número de atributos avaliados em cada nó decai drasticamente, reduzindo o tempo de treinamento de $8{,}14\text{ s}$ para $1{,}50\text{ s}$ na classificação e de $9{,}37\text{ s}$ para $0{,}92\text{ s}$ na regressão.

33. **Por que o Autoencoder gerou uma acurácia maior em classificação com Random Forest ($59{,}22\%$) do que o Random Forest com pixels brutos ($56{,}08\%$) em 64x64?**
    * *Resposta esperada:* Porque a compressão latente atuou como um **filtro regularizador de remoção de ruído (*denoising/manifold filtering*)**. Os pixels brutos de uma imagem redimensionada para $64 \times 64$ contêm ruídos de interpolação e redundância espacial que podem iludir as árvores. As convoluções do encoder sintetizaram feições visuais agregadas, facilitando a partição ortogonal do ensemble.

34. **Por que o PCA e o t-SNE apresentaram distribuições visuais diferentes para os mesmos embeddings $Z$?**
    * *Resposta esperada:* Porque o PCA é um método linear que busca eixos ortogonais de máxima variância global, tendendo a aglomerar pontos em uma nuvem elipsoidal contínua quando a variedade subjacente é não-linear. O t-SNE, por sua vez, é um método estocástico não-linear baseado em divergência de Kullback-Leibler que prioriza a conservação das probabilidades de vizinhança local, conseguindo separar variedades dobradas em agrupamentos semânticos mais nítidos.

35. **Qual é a função da distribuição $t$-Student no espaço projetado do t-SNE e por que não se usa uma Gaussiana em 2D?**
    * *Resposta esperada:* Para resolver o "problema de aglomeração" (*crowding problem*). Em alta dimensão ($64D$), há um volume hiperesférico imenso para acomodar pontos moderadamente distantes. Em 2D, a área disponível é muito menor. A distribuição $t$-Student possui caudas muito mais pesadas ($1 / (1 + d^2)$) que a Gaussiana ($e^{-d^2}$), o que repele pontos moderadamente distantes para a periferia, abrindo espaço visual para que agrupamentos locais não fiquem todos sobrepostos no centro da projeção.

36. **O que a análise visual de reconstrução (painel Original vs. Reconstruída) evidenciou sobre a representação aprendida?**
    * *Resposta esperada:* Evidenciou que o Autoencoder convolucional prioriza grandezas de baixa frequência espacial (luminância média, blocos de cor quente vs. fria e silhuetas de rostos), mas borra e perde completamente letras de fontes pequenas e textos estilizados. Isso ocorre porque o critério MSE calcula a média quadrática pixel a pixel, penalizando severamente o erro em grandes áreas coloridas e tratando o texto fino como ruído de alta frequência estatisticamente irrelevante.

37. **Por que a MLP possui 18 vezes mais parâmetros que a CNN, mas apresentou latência de inferência cerca de 20 vezes menor em CPU?**
    * *Resposta esperada:* Porque uma camada linear totalmente conectada é uma operação direta de multiplicação matriz-vetor (GEMM) que é altamente otimizada em hardware com paralelismo SIMD (como SSE/AVX em CPUs x86). Já as convoluções 2D exigem aninhamento de loops, transformações de janela (`im2col`), e camadas intermediárias de `BatchNorm` e `MaxPool`, demandando mais saltos de ponteiros de memória (*memory bandwidth*) e profundidade de grafo computacional, resultando em maior latência temporal.

38. **O que é o Throughput de inferência (medido em FPS) e por que ele deve ser avaliado junto com a latência?**
    * *Resposta esperada:* A latência mede o tempo de resposta pontual de uma requisição (quanto tempo 1 batch leva para sair), enquanto o Throughput (Frames Per Second) mede a capacidade total de vazão do sistema ($N / \Delta t$). Em sistemas de recomendação em tempo real do YouTube que precisam filtrar milhões de thumbnails por segundo, um modelo com throughput de $1.500\text{ FPS}$ (como a MLP) pode ser preferível em pré-filtragem, enquanto um modelo de alta precisão e menor throughput (como a ResNet a $32\text{ FPS}$) é reservado para o ranqueamento final.

39. **Por que a ResNet 2D apresentou a maior latência de inferência entre todas as arquiteturas avaliadas?**
    * *Resposta esperada:* Devido à complexidade do grafo computacional e aos blocos residuais. A cada bloco residual, o tensor precisa ser bifurcado, processado por duas convoluções consecutivas com normalização e depois somado ponto a ponto com o fluxo de identidade ($F(x) + x$). Essa adição residual e a profundidade de camadas exigem mais alocações de tensores intermediários e sincronizações de memória, elevando o tempo de execução por lote.

40. **Diante de todos os resultados (Acurácia, $R^2$, Embeddings e Latência), qual solução arquitetural você defenderia como a mais viável para a indústria?**
    * *Resposta esperada:* Defenderia uma **arquitetura híbrida em dois estágios**:
      1. Para **Classificação de Virais de Alto Risco**, a **ResNet 2D** é a escolha ideal, pois seu recall de $93{,}48\%$ garante que quase nenhum conteúdo de alto potencial seja descartado.
      2. Para **Regressão Contínua em Larga Escala**, o **Random Forest** calibrado com `max_features='sqrt'` em alta resolução (ou treinado sobre embeddings latentes $Z$) oferece a melhor relação custo-benefício, entregando o maior $R^2_{\log}$ ($+6{,}89\%$) com treinamento em frações de segundo e imunidade a problemas de convergência por gradiente.
