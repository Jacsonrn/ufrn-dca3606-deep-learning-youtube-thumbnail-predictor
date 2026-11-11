# 🧠 Fundamentação Teórica do Projeto: Deep Learning na Predição de Performance de Thumbnails

Este documento centraliza a teoria matemática, arquitetural e estratégica por trás de todas as decisões do código, **organizado estritamente pelo cronograma de dias de desenvolvimento do projeto** (Unidade 2 — DCA3606 / UFRN). Ele serve como documentação de referência para o grupo e guia de preparação para a arguição técnica.

---

# 📅 DIA 1: Pipeline de Classificação (Predição de Potencial Viral) e Fundações de Visão Computacional

Neste primeiro dia, estabelecemos a infraestrutura de dados em visão computacional, criamos a variável-alvo de classificação sob protocolo estrito anti-vazamento (*data leakage*) e implementamos as **5 abordagens obrigatórias** para prever o engajamento de thumbnails do YouTube.

---

### 📚 Fundamentação Teórica do Dia 1

#### 1. Formulação Tensorial da Imagem e Resolução $64 \times 64 \times 3$
Em Visão Computacional com Deep Learning, uma imagem digital colorida é representada como um tensor tridimensional de ordem 3:
$$X_i \in \mathbb{R}^{C \times H \times W} \quad \text{onde } C=3 \text{ (RGB)}, \; H=64, \; W=64$$
* **Normalização em $[0, 1]$:** Os canais de cor originais (inteiros de 8 bits em $[0, 255]$) foram reescalados dividindo por $255.0$. Isso centraliza e homogeneíza a variância de entrada, impedindo que gradientes explodam logo nas primeiras iterações do otimizador.
* **O Trade-off da Resolução $64 \times 64$:** Thumbnails no YouTube possuem resolução nativa de $1280 \times 720$. Alimentar redes neurais com essa resolução exigiria cerca de 2,76 milhões de variáveis por imagem, tornando o custo computacional por época inviável em processamento de CPU. A redução para $64 \times 64$ retém elementos semânticos fundamentais (luminância, contraste, posicionamento de rostos e formas geométricas de texto) ao mesmo tempo em que restringe a dimensão de entrada a $12.288$ valores numéricos, permitindo o treinamento das 5 arquiteturas em tempo hábil.

#### 2. Protocolo Anti-Vazamento (*Data Leakage*) na Binarização do Alvo
A variável bruta contínua é a contagem de visualizações ($\text{views} \in \mathbb{N}$). Para transformá-la em um problema de classificação binária (Alto Engajamento / Viral vs. Baixo Engajamento), adotamos a **Mediana** como limiar de corte ($\tau$):
$$y_{\text{class}} = \begin{cases} 1, & \text{se } \text{views} \ge \tau \\ 0, & \text{se } \text{views} < \tau \end{cases}$$
* **A Regra de Ouro:** O limiar $\tau = 610.055$ views foi calculado **exclusivamente no conjunto de Treino (70%)**.
* **Por que isso é crítico?** Se a mediana fosse calculada sobre todo o dataset antes da partição, informações de distribuição dos conjuntos de validação e teste contaminariam a definição das classes. Aplicando o limiar fixo do treino nos dados não vistos de validação e teste, preservamos a integridade de um teste cego realista.

#### 3. A Falha Conceitual da MLP / DNN Profunda em Imagens
A primeira abordagem supervisionada é uma rede densa (*Multilayer Perceptron*). Para que a MLP processe a thumbnail, o tensor $64 \times 64 \times 3$ precisa ser "achatado" (*flatten*) em um vetor unidimensional:
$$x_{\text{flat}} \in \mathbb{R}^{12.288}$$
Essa operação introduz duas falhas teóricas graves em visão computacional:
1. **Perda da Invariância à Translação:** Para a MLP, o pixel da posição $(0, 0)$ e o pixel da posição $(0, 1)$ têm a mesma relação de independência que o pixel $(0, 0)$ tem com o pixel $(63, 63)$. A matriz de pesos trata cada posição como uma variável independente, sendo incapaz de reconhecer um rosto ou logotipo se ele estiver deslocado alguns centímetros para o lado.
2. **Explosão Paramétrica:** Apenas a primeira camada densa ($12.288 \to 512$ neurônios) consome $6.291.968$ parâmetros treináveis. Com apenas $1.186$ amostras de treino, a rede entra em forte memorização, atingindo ~49% de acurácia no teste (desempenho comparável ao acaso).

#### 4. Convolução 2D, Compartilhamento de Pesos e Campos Receptivos Locais
A **CNN 2D** resolve as limitações da MLP aplicando o operador de convolução discreta bidimensional:
$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) \cdot K(m, n)$$
* **Filtros Locais (*Kernels* $3 \times 3$):** Cada neurônio convolucional conecta-se apenas a uma pequena vizinhança de pixels (*Local Receptive Field*), capturando correlações espaciais imediatas (bordas, texturas e cantos).
* **Compartilhamento de Pesos (*Weight Sharing*):** O mesmo filtro $3 \times 3$ desliza sobre toda a extensão da imagem. Se um filtro aprende a detectar alto contraste ou saturação de cores, ele detectará esse padrão em qualquer canto da thumbnail.
* **Hierarquia Visual:** Na primeira camada convolucional, a rede extrai traços primitivos de baixo nível; nas camadas subsequentes (após `MaxPool2d`), os campos receptivos aumentam, permitindo detectar estruturas semânticas compostas. A CNN alcançou 54,12% de acurácia com apenas 358 mil parâmetros (17 vezes menos parâmetros que a MLP).

#### 5. A ResNet 2D e o Teorema do Mapeamento Residual
Conforme a profundidade de uma rede neural cresce, o algoritmo de retropropagação (*backpropagation*) sofre com o **Desvanecimento do Gradiente** (*Vanishing Gradient*):
$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial a_L} \prod_{l=2}^{L} W_l \cdot \sigma'(z_{l-1})$$
Se as derivadas $\sigma'(z)$ forem menores que 1, a multiplicação sucessiva através de muitas camadas faz o gradiente convergir exponencialmente para zero, impedindo o aprendizado das camadas iniciais.

A **ResNet** (Kaiming He et al., 2015) introduz as **Conexões de Atalho (*Skip / Shortcut Connections*)**:
$$\mathcal{H}(x) = \mathcal{F}(x, \{W_i\}) + x$$
Onde $\mathcal{F}(x)$ é a transformação das camadas convolucionais e $x$ é a identidade do sinal original.
* **A "Superestrada de Gradientes":** Ao derivar em relação à entrada $x$:
$$\frac{\partial \mathcal{H}}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + I$$
Mesmo que o gradiente da convolução $\frac{\partial \mathcal{F}}{\partial x}$ se aproxime de zero, o termo identidade $I$ garante que o gradiente flua sem atenuação até as camadas mais rasas.
* **Impacto no Experimento:** A ResNet obteve o maior **Recall de Alto Engajamento (67,39%)** entre todas as redes neurais profundas testadas, mostrando alta sensibilidade para capturar thumbnails virais.

#### 6. Autoencoder Convolucional (CAE) e Representação Latente Não-Supervisionada
O **Convolutional Autoencoder** atua como um compressor não-linear composto por duas sub-redes:
1. **Encoder $f_\theta$:** Comprime $X \in \mathbb{R}^{64 \times 64 \times 3}$ ($12.288$ números) em um gargalo latente (*bottleneck*) $Z \in \mathbb{R}^{64}$.
2. **Decoder $g_\phi$:** Reconstrói a imagem aproximada $\widehat{X} = g_\phi(Z) \in \mathbb{R}^{64 \times 64 \times 3}$.

* **Função de Perda MSE Não-Supervisionada:**
$$\mathcal{L}_{\text{MSE}}(X, \widehat{X}) = \frac{1}{N} \sum_{i=1}^N \| X_i - \widehat{X}_i \|_2^2$$
O Autoencoder foi treinado **sem acesso aos rótulos de engajamento ($y$)**.
* **Hipótese de Representação:** O vetor latente $Z$ atua como uma assinatura compacta que retém apenas a variância visual máxima. Quando congelamos o encoder e treinamos um Random Forest sobre $Z$, o modelo alcançou 48,24% de acurácia. Esse resultado demonstra que os padrões visuais que minimizam o erro de reconstrução de imagem (fundo, iluminação média) não coincidem necessariamente com os gatilhos psicológicos que levam um usuário a clicar em um vídeo.

#### 7. Controle Clássico com Random Forest (Pixels Brutos vs. Embeddings)
O modelo clássico de controle consiste em um ensemble de **100 Árvores de Decisão** (*Random Forest*) com amostragem Bootstrap (*Bagging*) e seleção aleatória de features nos nós.
* **Desempenho nos Pixels Brutos:** O Random Forest diretamente em $X_{\text{flat}}$ obteve a maior Acurácia geral (56,08%) e o maior Macro-F1 (55,83%).
* **Fundamentação:** Em conjuntos de dados de escala moderada (~1.200 amostras), redes neurais profundas enfrentam superfícies de perda não-convexas repletas de pontos de sela e platôs de gradiente. O Random Forest, por operar através de divisões ortogonais baseadas em entropia e impureza de Gini ($I_G(p) = 1 - \sum p_i^2$), constrói fronteiras de decisão robustas sem depender de otimizadores baseados em gradiente, tornando-se uma linha de base sólida contra o sobreajuste.

#### 8. Métricas de Avaliação Multicritério e a Matriz de Confusão
Para avaliação rigorosa no conjunto de teste ($N = 255$), foram consolidadas 4 métricas formais:
* **Acurácia:** $\frac{TP + TN}{TP + TN + FP + FN}$ (fração global de acertos).
* **Precisão:** $\frac{TP}{TP + FP}$ (confiabilidade quando o modelo afirma que um vídeo será viral).
* **Recall (Sensibilidade):** $\frac{TP}{TP + FN}$ (capacidade de encontrar os vídeos virais reais).
* **Macro-F1 Score:** Média harmônica balanceada calculada independentemente para cada classe:
$$\text{F1}_{\text{macro}} = \frac{\text{F1}_{\text{Baixo}} + \text{F1}_{\text{Alto}}}{2}$$
O Macro-F1 é a métrica mais confiável do estudo, pois impede que um classificador tendencioso a prever apenas a classe majoritária receba uma pontuação falsamente alta.

---

### ❓ 10 Questões de Revisão Técnica — Dia 1 (Preparação para Apresentação e Arguição)

1. **Por que o limiar de classificação (mediana de visualizações) foi calculado exclusivamente no conjunto de treino?**
   * *Resposta esperada:* Para evitar o vazamento de dados (*data leakage*). Se a mediana fosse calculada sobre todo o conjunto de dados, os limites estatísticos dos conjuntos de validação e teste interfeririam no rótulo de treino, invalidando o rigor do teste cego.

2. **Qual é a principal deficiência teórica de utilizar uma MLP (rede densa) para classificação de imagens de thumbnails?**
   * *Resposta esperada:* A perda da invariância à translação e da coerência espacial 2D ao achatar a imagem em um vetor 1D, além da explosão de parâmetros treináveis na primeira camada linear ($12.288 \times 512 \approx 6,3$ milhões de pesos), gerando forte sobreajuste.

3. **Como a operação de convolução 2D supera as limitações da MLP em visão computacional?**
   * *Resposta esperada:* Através dos conceitos de *campos receptivos locais* (conexões restritas a vizinhanças $3 \times 3$) e *compartilhamento de pesos* (os mesmos filtros deslizam por toda a imagem), preservando as relações espaciais e reduzindo drasticamente a quantidade de parâmetros.

4. **O que é o problema do desvanecimento do gradiente e como os blocos residuais da ResNet o mitigam matematicamente?**
   * *Resposta esperada:* Em redes profundas, a retropropagação multiplica derivadas sucessivas menores que 1, fazendo o gradiente tender a zero nas camadas iniciais. A ResNet adiciona uma conexão de atalho identidade $\mathcal{H}(x) = \mathcal{F}(x) + x$, cuja derivada $\frac{\partial \mathcal{H}}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + I$ preserva um gradiente unitário mínimo constante que flui sem atenuação.

5. **Por que a ResNet obteve o maior Recall de engajamento (67,39%) entre os modelos neurais?**
   * *Resposta esperada:* As conexões residuais permitiram que a rede combinasse representações visuais de baixo nível (cores e contraste) com representações de alto nível sem saturação de gradientes, tornando-a especialmente sensível na identificação de thumbnails com alto potencial viral.

6. **Qual é a diferença fundamental no treinamento do Autoencoder Convolucional em comparação com os outros modelos de Deep Learning implementados?**
   * *Resposta esperada:* O Autoencoder foi treinado de forma puramente **não-supervisionada** usando como função de perda o Erro Quadrático Médio (MSE) de reconstrução da própria imagem de entrada, sem receber qualquer rótulo de visualizações ou engajamento.

7. **Por que o Random Forest treinado nos embeddings latentes ($Z \in \mathbb{R}^{64}$) obteve acurácia inferior (48,24%) ao treinado nos pixels brutos (56,08%)?**
   * *Resposta esperada:* Porque a compressão não-supervisionada de 12.288 para 64 dimensões otimizou a reconstrução média da imagem (priorizando iluminação de fundo e paleta geral), descartando sutilezas locais e tipográficas que influenciam a decisão humana de clique.

8. **Por que o modelo clássico (Random Forest nos pixels brutos) superou as redes neurais em acurácia geral neste conjunto de dados?**
   * *Resposta esperada:* Modelos baseados em ensemble de árvores de decisão particionam o espaço por meio de cortes ortogonais sem depender de gradientes estocásticos, sendo muito menos suscetíveis a platôs de otimização em bases de dados com escala moderada (~1.200 amostras de treino).

9. **Qual foi o papel do Early Stopping e da divisão com conjunto de validação durante o treinamento do PyTorch?**
   * *Resposta esperada:* O conjunto de validação serviu para monitorar a perda de generalização a cada época. O Early Stopping interrompeu o treinamento quando a perda na validação parou de melhorar por 5 épocas consecutivas (*patience=5*), restaurando os melhores pesos aprendidos e evitando memorização excessiva do treino.

10. **Por que a métrica Macro-F1 foi priorizada em vez da Acurácia simples na análise comparativa?**
    * *Resposta esperada:* Porque o Macro-F1 calcula a média harmônica entre precisão e sensibilidade para cada classe de forma isolada e ponderada igualmente, impedindo que distorções ou desbalanceamentos favoreçam um classificador trivial.
