#!/bin/bash
#SBATCH --job-name=ThumbnailPredictor
#SBATCH --output=slurm_thumbnail_%j.log
#SBATCH --error=slurm_thumbnail_%j.log
#SBATCH --partition=gpu-4-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-user=jacson.arruda.123@ufrn.edu.br
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ==============================================================================
# 1. LOCALIZAÇÃO DO REPOSITÓRIO (SLURM_SUBMIT_DIR)
# ==============================================================================
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO_DIR"
echo ">> [SISTEMA] Diretório de trabalho real do projeto: $(pwd)"

# ==============================================================================
# 2. BLINDAGEM DO AMBIENTE
# ==============================================================================
export PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset PYTHONHOME
module purge

# ==============================================================================
# 3. AUTO-DESCOBERTA DO CONDA NO CLUSTER
# ==============================================================================
CONDA_DIR=""

# Tentativa 1: Via comando conda existente no sistema
if command -v conda &> /dev/null; then
    CONDA_DIR="$(conda info --base 2>/dev/null)"
fi

# Tentativa 2: Varredura dos diretórios padrão de usuários do NPAD
if [ -z "$CONDA_DIR" ] || [ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    for candidate in \
        "$HOME/miniconda3" \
        "$HOME/anaconda3" \
        "/mnt/beegfs/scratch/$USER/miniconda3" \
        "/mnt/beegfs/scratch/$USER/anaconda3" \
        "/opt/npad/miniconda3" \
        "/opt/npad/anaconda3"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            CONDA_DIR="$candidate"
            break
        fi
    done
fi

# Tentativa 3: Se não existir em nenhum lugar, instala Miniconda automaticamente
if [ -z "$CONDA_DIR" ] || [ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    echo ">> [SISTEMA] Conda não encontrado. Instalando Miniconda automaticamente em $HOME/miniconda3..."
    INSTALLER="/tmp/miniconda_installer_$USER.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$HOME/miniconda3" -u
    rm -f "$INSTALLER"
    CONDA_DIR="$HOME/miniconda3"
fi

echo ">> [SISTEMA] Localização do Conda confirmada: $CONDA_DIR"
source "$CONDA_DIR/etc/profile.d/conda.sh"

# ==============================================================================
# 4. ATIVAÇÃO DO AMBIENTE E PROTEÇÃO DE CAMINHO
# ==============================================================================
ENV_NAME="thumbnail_dl_env"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo ">> [SISTEMA] Criando novo ambiente Conda ($ENV_NAME) com Python 3.11..."
    conda create -y -n $ENV_NAME python=3.11
fi
conda activate $ENV_NAME

if [ -z "$CONDA_PREFIX" ]; then
    echo ">> [ERRO FATAL] Falha ao ativar o ambiente Conda $ENV_NAME."
    exit 1
fi

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON_BIN="$CONDA_PREFIX/bin/python"

echo ">> [VERIFICAÇÃO DE INTERPRETADOR]"
echo ">> Binário Python: $PYTHON_BIN"
$PYTHON_BIN --version

# ==============================================================================
# 5. INSTALAÇÃO DAS DEPENDÊNCIAS COM CUDA NO PYTHON 3.11 DO CONDA
# ==============================================================================
if [ ! -f "$REPO_DIR/.instalacao_ok" ]; then
    echo ">> [SISTEMA] Primeira execução. Instalando dependências com suporte a GPU na A100..."
    $PYTHON_BIN -m pip install --upgrade pip
    
    # PyTorch com suporte a CUDA 12.4
    $PYTHON_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    
    # Bibliotecas de Data Science e Visão
    $PYTHON_BIN -m pip install scikit-learn pandas matplotlib seaborn huggingface_hub tqdm requests pillow tabulate
    
    touch "$REPO_DIR/.instalacao_ok"
else
    echo ">> [SISTEMA] Dependências já instaladas anteriormente."
fi

# ==============================================================================
# 6. CARREGAMENTO DOS MÓDULOS CUDA DO NPAD
# ==============================================================================
echo ">> [SISTEMA] Carregando compiladores CUDA 12.6 do NPAD..."
module load compilers/nvidia/cuda/12.6

# ==============================================================================
# 7. DIAGNÓSTICO DE HARDWARE
# ==============================================================================
echo ">> ================= DIAGNÓSTICO DE GPU ================="
nvidia-smi
$PYTHON_BIN -c "import torch; print(f'>> PyTorch CUDA Ativo: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NENHUMA (Fallback CPU)\"}')"
echo ">> ======================================================"

# ==============================================================================
# 8. EXECUÇÃO DOS PIPELINES DO PROJETO EM ALTA RESOLUÇÃO NATIVA
# ==============================================================================
echo ">> [1/3] Gerando Análise Exploratória e Gráficos..."
$PYTHON_BIN -u src/eda_report.py

echo ">> [2/3] Executando Pipeline de Classificação em Alta Resolução Nativa (480x360)..."
$PYTHON_BIN -u src/train_classification.py

if [ -f "src/train_regression.py" ]; then
    echo ">> [3/3] Executando Pipeline de Regressão em Alta Resolução Nativa (480x360)..."
    $PYTHON_BIN -u src/train_regression.py
fi

echo ">> [SUCESSO] Pipeline concluído com sucesso no NPAD!"
