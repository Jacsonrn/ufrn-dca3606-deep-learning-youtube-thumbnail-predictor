#!/bin/bash
#SBATCH --job-name=ThumbnailPredictor
#SBATCH --output=slurm_thumbnail_%j.log
#SBATCH --error=slurm_thumbnail_%j.log  # Unifica stdout e stderr no mesmo arquivo
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
# 1. BLINDAGEM DO AMBIENTE
# ==============================================================================
export PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset PYTHONHOME
module purge

# ==============================================================================
# 2. LOCALIZAÇÃO DINÂMICA DO REPOSITÓRIO
# ==============================================================================
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
echo ">> [SISTEMA] Diretório de execução do projeto: $(pwd)"

# ==============================================================================
# 3. ATIVAÇÃO DO CONDA E FORÇAMENTO DO PYTHON 3.11
# ==============================================================================
MINICONDA_DIR="/mnt/beegfs/scratch/jaribeiro/miniconda3"
source $MINICONDA_DIR/etc/profile.d/conda.sh

ENV_NAME="thumbnail_dl_env"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo ">> [SISTEMA] Criando novo ambiente Conda ($ENV_NAME) com Python 3.11..."
    conda create -y -n $ENV_NAME python=3.11
fi
conda activate $ENV_NAME

# FORÇA O PATH PARA O CONDA PYTHON 3.11
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON_BIN="$CONDA_PREFIX/bin/python"

echo ">> [VERIFICAÇÃO DE INTERPRETADOR]"
echo ">> Binário Python: $(which python)"
$PYTHON_BIN --version

# ==============================================================================
# 4. INSTALAÇÃO OTIMIZADA DAS DEPENDÊNCIAS COM SUPORTE A CUDA
# ==============================================================================
if [ ! -f "$CONDA_PREFIX/.instalacao_ok" ]; then
    echo ">> [SISTEMA] Primeira execução detectada. Instalando dependências no Python 3.11 do Conda..."
    $PYTHON_BIN -m pip install --upgrade pip
    
    # PyTorch com CUDA 12.4
    $PYTHON_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    
    # Bibliotecas de ciência de dados do projeto
    $PYTHON_BIN -m pip install scikit-learn pandas matplotlib seaborn huggingface_hub tqdm requests pillow tabulate
    
    touch "$CONDA_PREFIX/.instalacao_ok"
else
    echo ">> [SISTEMA] Dependências já instaladas no ambiente $ENV_NAME."
fi

# ==============================================================================
# 5. CARREGAMENTO DOS COMPILADORES CUDA DO CLUSTER
# ==============================================================================
echo ">> [SISTEMA] Carregando compiladores CUDA 12.6 do NPAD..."
module load compilers/nvidia/cuda/12.6

# ==============================================================================
# 6. DIAGNÓSTICO DE HARDWARE
# ==============================================================================
echo ">> ================= DIAGNÓSTICO DE GPU ================="
nvidia-smi
$PYTHON_BIN -c "import torch; print(f'>> PyTorch CUDA Ativo: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NENHUMA (Fallback CPU)\"}')"
echo ">> ======================================================"

# ==============================================================================
# 7. EXECUÇÃO DOS PIPELINES DO PROJETO
# ==============================================================================
echo ">> [1/3] Gerando Análise Exploratória e Gráficos..."
$PYTHON_BIN -u src/eda_report.py

echo ">> [2/3] Executando Pipeline de Classificação (5 Abordagens)..."
$PYTHON_BIN -u src/train_classification.py

if [ -f "src/train_regression.py" ]; then
    echo ">> [3/3] Executando Pipeline de Regressão (5 Abordagens)..."
    $PYTHON_BIN -u src/train_regression.py
fi

echo ">> [SUCESSO] Pipeline concluído com sucesso no NPAD!"
