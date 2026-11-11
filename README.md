# 🎬 Previsor de Performance de Thumbnails com Deep Learning: Classificação Viral e Regressão Contínua

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/NVIDIA%20A100-SXM4%2080GB-76B900?logo=nvidia&logoColor=white" alt="NVIDIA A100" />
  <img src="https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=white" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Ensemble-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/UFRN-DCA3606-navy" alt="UFRN" />
  <img src="https://img.shields.io/badge/Status-Dia%201%20Conclu%C3%ADdo-success" alt="Status" />
</p>

Projeto de inteligência artificial aplicada à visão computacional desenvolvido para a disciplina **DCA3606 — Inteligência Artificial** no Departamento de Engenharia de Computação e Automação (DCA) da Universidade Federal do Rio Grande do Norte (UFRN).
* **Docente:** Prof. Marcelo A. C. Fernandes
* **Discente:** Jacson Arruda Ribeiro
* **Dataset:** `advaypatil_youtube-statistics` (Hugging Face Hub) integrado à CDN oficial do YouTube (`img.youtube.com/vi/{id}/hqdefault.jpg`) — 1.867 registros com 1.695 thumbnails reais e íntegras em alta resolução.

---

## 🎯 Visão Geral do Projeto

Este repositório contém a implementação completa, o protocolo anti-vazamento estatístico e a avaliação experimental das **5 abordagens obrigatórias** da ementa para a predição de engajamento humano a partir de imagens de capa de vídeos do YouTube:
1. **Abordagem 1 — MLP / DNN Profunda:** Rede densa com múltiplas camadas lineares, Dropout e Normalização em Lote (*Batch Normalization*).
2. **Abordagem 2 — CNN Convolucional 2D:** Extração hierárquica de feições espaciais com blocos de Convolução $3 \times 3$, ReLU, MaxPool e `AdaptiveAvgPool2d`.
3. **Abordagem 3 — ResNet 2D Residual:** Blocos residuais com conexões de atalho identidade (*Skip Connections* $\mathcal{F}(x) + x$) para fluxo ininterrupto de gradientes.
4. **Abordagem 4 — Autoencoder Convolucional (CAE) + Random Forest:** Compressão não-supervisionada da imagem em espaço latente $Z \in \mathbb{R}^{64}$ seguido de classificador clássico.
5. **Abordagem 5 — Random Forest Clássico nos Pixels Brutos:** Linha de base (*baseline*) não-paramétrica baseada em ensemble de 100 árvores de decisão particionando pixels achatados (*flatten*).

### 🔬 Protocolo Científico Estrito (Prevenção de *Data Leakage*)
* **Particionamento:** 70% Treino ($N = 1.186$), 15% Validação ($N = 254$) e 15% Teste cego ($N = 255$), sob semente estocástica reproduzível (`random_state=42`).
* **Binarização Anti-Vazamento:** O limiar de corte de engajamento ($\tau = 610.055$ visualizações) foi calculado **estritamente sobre o conjunto de Treino**. Amostras de validação e teste foram binarizadas contra esse limiar fixo, eliminando contaminação amostral.
* **Avaliação em Dois Regimes de Resolução:**
  - **Baseline CPU:** Imagens reduzidas para $64 \times 64 \times 3$ RGB ($12.288$ features).
  - **Nativo GPU (NVIDIA A100-SXM4-80GB):** Imagens preservadas em alta resolução $480 \times 360 \times 3$ RGB (**$518.400$ features**).

---

## 📊 Síntese Comparativa de Desempenho (Dia 1 — Classificação)

Comparação empírica no conjunto de teste cego ($N = 255$ amostras inéditas):

| Modelo | Parâmetros Treináveis | Acurácia (64x64) | Acurácia (480x360) | $\Delta$ Acurácia | Macro-F1 (480x360) | Recall Viral (480x360) | Veredito Técnico e Arquitetural |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Pixels)** | — (Ensemble) | 56,08% | **61,57%** | **+5,49%** | **61,35%** | 63,77% | **Campeão Geral:** Particionamento não-paramétrico sobre bordas nítidas de texto e saturação. |
| **CNN 2D** | **358.306** (Fixo) | 54,12% | **58,04%** | **+3,92%** | **55,03%** | **77,54%** | **Melhor Rede Neural:** Invariância à translação e desacoplamento de escala via `AdaptiveAvgPool2d`. |
| **ResNet 2D** | 350.338 (Fixo) | 54,90% | 53,33% | -1,57% | 39,48% | **93,48%** | **Detector Viral Quase Perfeito:** Capturou 129 de 138 virais reais devido à superestrada de gradientes. |
| **MLP / DNN** | 265.421.312 | 49,02% | 47,84% | -1,18% | 46,66% | 30,43% | **Colapso Dimensional:** Maldição da dimensionalidade ($N/D \approx 0{,}0022$), memorização precoce na época 7. |
| **RF (Embeddings AE)** | — (Espaço $Z$) | 48,24% | 45,10% | -3,14% | 45,08% | 39,86% | **Gargalo de Shannon:** Compressão extrema de $8.100:1$ descartou feições discriminativas de clique. |

---

## 📈 Prova Visual dos Resultados

### 1. Confrontação Multi-Resolução: CPU (64x64) vs. A100 GPU (480x360)
<p align="center">
  <img src="reports/figures/comparison_resolution_cpu_vs_a100.png" width="95%" alt="Comparação de Resoluções CPU vs GPU" />
</p>

### 2. Matrizes de Confusão Lado a Lado (Alta Resolução Nativa — 480x360)
<p align="center">
  <img src="reports/figures/confusion_matrices_classification_480x360.png" width="98%" alt="Matrizes de Confusão 480x360" />
</p>

<details>
<summary><b>🔍 Clique para expandir os Gráficos de Análise Exploratória (EDA) e Particionamento</b></summary>
<br>

<p align="center">
  <b>Amostragem Real das Thumbnails do Dataset com Rótulos de Visualização</b><br>
  <img src="reports/figures/sample_thumbnails_grid.png" width="95%" alt="Grid de Thumbnails do YouTube" />
</p>

<p align="center">
  <b>Distribuição Heavy-Tail de Visualizações (Lei de Potência)</b><br>
  <img src="reports/figures/target_distribution_views.png" width="65%" alt="Distribuição de Views" />
</p>

<p align="center">
  <b>Equilíbrio Rigoroso de Classes nas Partições Anti-Leakage</b><br>
  <img src="reports/figures/class_balance_splits.png" width="70%" alt="Equilíbrio de Classes" />
</p>
</details>

---

## 💡 Principais Descobertas Científicas de Baixo Nível

1. **A Maldição da Dimensionalidade na MLP:** Ao saltar para $480 \times 360$, o vetor achatado atingiu $518.400$ posições, forçando a primeira camada linear a alocar $265{,}4$ milhões de pesos. Com apenas $1.186$ imagens de treino, a rede entrou em sobreajuste precoce (*early stopping* na época 7), comprovando a inviabilidade de redes densas em visão de alta resolução.
2. **Desacoplamento Paramétrico da CNN:** Graças ao compartilhamento de pesos e à camada `AdaptiveAvgPool2d((4, 4))`, o número de parâmetros da CNN 2D manteve-se rigorosamente estável em $358$ mil. A maior resolução forneceu nitidez em tipografia de *clickbait* e expressões faciais, gerando salto de **+3,92% em Acurácia** e **+13,05% em Recall**.
3. **Sensibilidade Viral da ResNet (93,48% de Recall):** A conexão residual identidade $\frac{\partial \mathcal{H}}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + I$ preservou gradientes em alta resolução, tornando a ResNet um detector hipersensível capaz de identificar praticamente qualquer thumbnail viral (129 acertos em 138).
4. **O Gargalo Informacional de Shannon no Autoencoder:** O gargalo latente $Z \in \mathbb{R}^{64}$ funcionou adequadamente em $64 \times 64$ (taxa de $192:1$). Em $480 \times 360$, a taxa saltou para **$8.100:1$**, forçando o encoder a reter apenas iluminância média e fundos, eliminando as sutilezas que determinam a decisão humana de clique.

---

## 📁 Estrutura do Repositório

```text
.
├── data/
│   ├── raw/thumbnails/             # Cache local concorrente das imagens JPEG
│   └── processed/                  # Tensores particionados e compactados (.npz)
├── models/                         # Pesos treinados das redes (.pt) e modelos sklearn
├── reports/                        # Métricas formais e figuras consolidadas
│   ├── figures/                    # Gráficos em 300 DPI (barras, matrizes, EDA)
│   ├── classification_metrics.csv  # Tabela exportada de métricas
│   └── comparison_classification_64x64_vs_480x360.csv
├── reports_a100/                   # Backup bruto das métricas geradas no cluster de GPU
├── src/                            # Pipeline de código-fonte modular
│   ├── architectures.py            # Definições em PyTorch (MLP, CNN, ResNet, CAE)
│   ├── compare_resolutions.py      # Script de consolidação e plotagem pareada
│   ├── data_loader.py              # Download concorrente e partição sem leakage
│   ├── eda_report.py               # Geração automatizada de gráficos exploratórios
│   └── train_classification.py     # Pipeline unificado com interface CLI (--res, --device)
├── job.sh                          # Script de submissão em lotes via SLURM para cluster
├── Fundamentacao_Teorica.md        # Teoria exaustiva, formulações e 15 questões de arguição
├── requirements.txt                # Dependências com travas de versão
└── README.md                       # Apresentação do projeto e instruções de reprodução
```

---

## 🚀 Como Reproduzir o Projeto

### 1. Clonar o Repositório e Configurar o Ambiente Virtual
```bash
git clone https://github.com/Jacsonrn/ufrn-dca3606-deep-learning-youtube-thumbnail-predictor.git
cd ufrn-dca3606-deep-learning-youtube-thumbnail-predictor

# Criar e ativar ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Execução Flexível do Pipeline de Classificação
O script [`src/train_classification.py`](src/train_classification.py) possui uma interface de linha de comando parametrizada para executar em qualquer ambiente:

```bash
# Execução local rápida em CPU (baixa resolução):
python src/train_classification.py --res 64x64 --device cpu

# Execução nativa completa em GPU / Cluster (alta resolução):
python src/train_classification.py --res 480x360 --device auto --batch-size 32 --epochs 25
```

### 3. Gerar a Comparação Multi-Resolução Pareada
```bash
python src/compare_resolutions.py
```

### 4. Execução em Cluster Supercomputador (SLURM)
```bash
sbatch job.sh
```

---

## 🗺️ Roadmap de Desenvolvimento

- [x] **Dia 1: Pipeline de Classificação (Concluído)**
  - Ingestão concorrente de metadados e imagens do YouTube.
  - Particionamento anti-leakage e treino das 5 abordagens.
  - Avaliação multi-resolução ($64 \times 64$ CPU vs. $480 \times 360$ NVIDIA A100).
  - Tabela comparativa consolidada e prova visual completa.
- [ ] **Dia 2: Pipeline de Regressão Contínua (Em Andamento)**
  - Transformação logarítmica do alvo contínuo ($y = \log_{10}(\text{views} + 1)$).
  - Modelos regressores (MLP, CNN, ResNet, RF nos Embeddings e RF nos Pixels).
  - Avaliação com MAE, RMSE e Coeficiente $R^2$ na escala real desnormalizada.
- [ ] **Dia 3: Estudo de Ablação, Consolidação Final e Artigo Técnico**
  - Ajuste fino de hiperparâmetros e análise crítica de erros residuais.

---

## 👨‍💻 Autor
**Jacson Arruda Ribeiro**  
Universidade Federal do Rio Grande do Norte (UFRN) — Departamento de Engenharia de Computação e Automação (DCA)  
Discente em Engenharia de Computação
