#!/bin/bash
#SBATCH --job-name=ThumbnailPredictor
#SBATCH --output=slurm_thumbnail_%j.log
#SBATCH --error=slurm_thumbnail_%j.err
#SBATCH --partition=gpu-4-a100       # Para V100, altere para: --partition=gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-user=jacson.arruda.123@ufrn.edu.br
#SBATCH --mail-type=ALL

# Otimização para bibliotecas de processamento multithread
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ==============================================================================
# 1. BLINDAGEM DO AMBIENTE (Isola o nó de login do cluster)
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
# 3. ATIVAÇÃO DO CONDA E CRIAÇÃO DO AMBIENTE
# ==============================================================================
MINICONDA_DIR="/mnt/beegfs/scratch/jaribeiro/miniconda3"
source $MINICONDA_DIR/etc/profile.d/conda.sh

ENV_NAME="thumbnail_dl_env"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo ">> [SISTEMA] Criando novo ambiente Conda ($ENV_NAME) com Python 3.11..."
    conda create -y -n $ENV_NAME python=3.11
fi
conda activate $ENV_NAME

# ==============================================================================
# 4. INSTALAÇÃO OTIMIZADA DAS DEPENDÊNCIAS COM SUPORTE A CUDA
# ==============================================================================
if [ ! -f "$CONDA_PREFIX/.instalacao_ok" ]; then
    echo ">> [SISTEMA] Primeira execução detectada. Instalando dependências com suporte a GPU..."
    python3 -m pip install --upgrade pip
    
    # PyTorch com CUDA 12.4 para compatibilidade total com A100 / V100
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    
    # Bibliotecas de ciência de dados do projeto
    python3 -m pip install scikit-learn pandas matplotlib seaborn huggingface_hub tqdm requests pillow tabulate
    
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
python3 -c "import torch; print(f'>> PyTorch CUDA Ativo: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NENHUMA (Fallback CPU)\"}')"
echo ">> ======================================================"

# ==============================================================================
# 7. EXECUÇÃO DOS PIPELINES DO PROJETO
# ==============================================================================
echo ">> [1/3] Gerando Análise Exploratória e Gráficos..."
python3 -u src/eda_report.py

echo ">> [2/3] Executando Pipeline de Classificação (5 Abordagens)..."
python3 -u src/train_classification.py

if [ -f "src/train_regression.py" ]; then
    echo ">> [3/3] Executando Pipeline de Regressão (5 Abordagens)..."
    python3 -u src/train_regression.py
fi

echo ">> [SUCESSO] Pipeline concluído com sucesso no NPAD!"
