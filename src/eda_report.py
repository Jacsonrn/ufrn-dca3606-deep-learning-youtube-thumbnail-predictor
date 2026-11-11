"""
Módulo de Análise Exploratória (EDA) e Visualização de Pré-Processamento.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Gera as figuras exigidas para os slides 2 e 3 da apresentação:
1. Painel de Amostras de Thumbnails com rótulos e views.
2. Histograma da distribuição de views (Original vs. Log10).
3. Gráfico de barras do balanceamento de classes entre partições.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import prepare_dataset, RAW_THUMBNAILS_DIR
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def plot_sample_thumbnails(df_samples, output_path: Path):
    """Gera uma grade com 8 thumbnails aleatórias com seus metadados."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, (_, row) in enumerate(df_samples.iterrows()):
        img_path = RAW_THUMBNAILS_DIR / f"{row['Video ID']}.jpg"
        ax = axes[i]
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
        ax.axis("off")
        title = (row["Title"][:35] + "...") if len(row["Title"]) > 35 else row["Title"]
        views = f"{int(row['Views']):,} views".replace(",", ".")
        cls_name = "Alto Engajamento" if row["Views"] >= 610055 else "Baixo Engajamento"
        color = "darkgreen" if cls_name == "Alto Engajamento" else "darkred"
        ax.set_title(f"{title}\n{views} | {cls_name}", fontsize=9, fontweight="bold", color=color)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Grade de thumbnails salva em: {output_path}")


def plot_target_distribution(y_views: np.ndarray, median_val: float, output_path: Path):
    """Gera o comparativo da distribuição original (cauda longa) vs. Log10."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Escala Original (Cauda Longa)
    sns.histplot(y_views, bins=40, kde=True, ax=ax1, color="#2b5c8f")
    ax1.axvline(median_val, color="red", linestyle="--", linewidth=2, label=f"Mediana Treino ({median_val:,.0f})")
    ax1.set_title("Distribuição Original de Visualizações (Cauda Longa)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Views", fontsize=10)
    ax1.set_ylabel("Frequência", fontsize=10)
    ax1.legend()

    # Escala Logarítmica (Estabilizada)
    log_views = np.log10(y_views + 1)
    log_median = np.log10(median_val + 1)
    sns.histplot(log_views, bins=40, kde=True, ax=ax2, color="#2ca02c")
    ax2.axvline(log_median, color="red", linestyle="--", linewidth=2, label=f"Mediana log10 ({log_median:.2f})")
    ax2.set_title("Distribuição em Escala Logarítmica log10(Views)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("log10(Views + 1)", fontsize=10)
    ax2.set_ylabel("Frequência", fontsize=10)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Gráfico de distribuição de views salvo em: {output_path}")


def plot_class_balance(data: dict, output_path: Path):
    """Gera gráfico de barras com o balanceamento de classes em Treino, Val e Teste."""
    splits = ["Treino", "Validação", "Teste"]
    class_0_counts = [
        np.sum(data["y_class_train"] == 0),
        np.sum(data["y_class_val"] == 0),
        np.sum(data["y_class_test"] == 0),
    ]
    class_1_counts = [
        np.sum(data["y_class_train"] == 1),
        np.sum(data["y_class_val"] == 1),
        np.sum(data["y_class_test"] == 1),
    ]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, class_0_counts, width, label="Classe 0 (Baixo Engajamento)", color="#d95f02")
    rects2 = ax.bar(x + width/2, class_1_counts, width, label="Classe 1 (Alto Engajamento)", color="#1b9e77")

    ax.set_ylabel("Número de Amostras", fontsize=10)
    ax.set_title("Equilíbrio de Classes nas Partições (Sem Vazamento de Dados)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=10)
    ax.legend()

    # Rótulos nas barras
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Gráfico de equilíbrio de classes salvo em: {output_path}")


def generate_eda_figures():
    """Orquestra a criação de todos os gráficos visuais de EDA."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    loaders, data = prepare_dataset(batch_size=32)
    
    # 1. Carregar metadados para buscar títulos das amostras
    import pandas as pd
    from huggingface_hub import hf_hub_download
    csv_path = hf_hub_download(repo_id="jason1966/advaypatil_youtube-statistics", filename="videos-stats.csv", repo_type="dataset")
    df = pd.read_csv(csv_path).dropna(subset=["Video ID", "Views"])
    
    # Selecionar 8 amostras variadas (4 virais e 4 de baixo engajamento)
    median_val = data["median_views_train"]
    high_samples = df[df["Views"] >= median_val].sample(4, random_state=42)
    low_samples = df[df["Views"] < median_val].sample(4, random_state=42)
    samples_combined = pd.concat([high_samples, low_samples]).sample(frac=1, random_state=42)
    
    # Gerar gráficos
    plot_sample_thumbnails(samples_combined, FIGURES_DIR / "sample_thumbnails_grid.png")
    all_views = np.concatenate([data["y_views_train"], data["y_views_val"], data["y_views_test"]])
    plot_target_distribution(all_views, median_val, FIGURES_DIR / "target_distribution_views.png")
    plot_class_balance(data, FIGURES_DIR / "class_balance_splits.png")
    print("\n[OK] Todos os gráficos de EDA e Pré-Processamento foram gerados com sucesso!")


if __name__ == "__main__":
    generate_eda_figures()
