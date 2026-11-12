"""
Script de Consolidação e Comparação de Desempenho Multi-Resolução.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Projeto 2ª Unidade - Previsor de Thumbnails com Deep Learning

Compara:
- Baseline Local: 64x64 (CPU)
- Alta Resolução Nativa: 480x360 (A100)
Gera tabela comparativa consolidada e gráfico de barras agrupadas.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def generate_comparison():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_64 = REPORTS_DIR / "classification_metrics_64x64.csv"
    csv_480 = REPORTS_DIR / "classification_metrics_480x360.csv"
    
    if not csv_480.exists():
        # Fallback para pasta reports_gpu_fase_1 se ainda não copiado
        csv_480 = PROJECT_ROOT / "reports_gpu_fase_1" / "classification_metrics_480x360.csv"
        
    if not csv_64.exists() or not csv_480.exists():
        raise FileNotFoundError(f"Arquivos de métricas não localizados em {REPORTS_DIR}")
        
    df_64 = pd.read_csv(csv_64, index_col=0)
    df_480 = pd.read_csv(csv_480, index_col=0)
    
    order = [
        "MLP / DNN",
        "CNN",
        "ResNet",
        "Clássico com entrada original",
        "Clássico treinado com embeddings do AE"
    ]
    df_64 = df_64.loc[order]
    df_480 = df_480.loc[order]
    
    # Montar DataFrame Comparativo Consolidado
    comp_data = []
    for model in order:
        acc_64 = df_64.loc[model, "Acurácia"]
        acc_480 = df_480.loc[model, "Acurácia"]
        f1_64 = df_64.loc[model, "Macro-F1"]
        f1_480 = df_480.loc[model, "Macro-F1"]
        rec_64 = df_64.loc[model, "Recall"]
        rec_480 = df_480.loc[model, "Recall"]
        prec_64 = df_64.loc[model, "Precisão"]
        prec_480 = df_480.loc[model, "Precisão"]
        
        comp_data.append({
            "Modelo": model,
            "Acurácia (64x64)": f"{acc_64 * 100:.2f}%",
            "Acurácia (480x360)": f"{acc_480 * 100:.2f}%",
            "Δ Acurácia": f"{(acc_480 - acc_64) * 100:+.2f}%",
            "Macro-F1 (64x64)": f"{f1_64 * 100:.2f}%",
            "Macro-F1 (480x360)": f"{f1_480 * 100:.2f}%",
            "Δ Macro-F1": f"{(f1_480 - f1_64) * 100:+.2f}%",
            "Recall (64x64)": f"{rec_64 * 100:.2f}%",
            "Recall (480x360)": f"{rec_480 * 100:.2f}%",
            "Precisão (64x64)": f"{prec_64 * 100:.2f}%",
            "Precisão (480x360)": f"{prec_480 * 100:.2f}%",
        })
        
    df_comp = pd.DataFrame(comp_data).set_index("Modelo")
    out_csv = REPORTS_DIR / "comparison_classification_64x64_vs_480x360.csv"
    df_comp.to_csv(out_csv)
    
    print("\n================ TABELA COMPARATIVA: 64x64 (CPU) vs 480x360 (A100) ================")
    try:
        print(df_comp.to_markdown())
    except Exception:
        print(df_comp.to_string())
    print(f" -> Tabela salva em: {out_csv}")
    
    # Gráfico Comparativo de Barras Agrupadas
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(order))
    width = 0.35
    
    # Nomes mais curtos para caber no eixo X
    short_names = ["MLP / DNN", "CNN 2D", "ResNet 2D", "RF (Pixels)", "RF (Autoencoder)"]
    
    # Painel 1: Acurácia
    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, df_64["Acurácia"] * 100, width, label="64x64 (CPU)", color="#4A90E2")
    rects2 = ax1.bar(x + width/2, df_480["Acurácia"] * 100, width, label="480x360 (A100)", color="#2ECC71")
    ax1.set_title("Comparação de Acurácia (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Acurácia (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax1.set_ylim(35, 75)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    
    # Anotar valores nas barras
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    # Painel 2: Macro-F1
    ax2 = axes[1]
    rects3 = ax2.bar(x - width/2, df_64["Macro-F1"] * 100, width, label="64x64 (CPU)", color="#E67E22")
    rects4 = ax2.bar(x + width/2, df_480["Macro-F1"] * 100, width, label="480x360 (A100)", color="#9B59B6")
    ax2.set_title("Comparação de Macro-F1 Score (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Macro-F1 (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax2.set_ylim(30, 75)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    plt.suptitle("Impacto da Resolução Espacial e Aceleração por Hardware no Desempenho", fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    out_fig = FIGURES_DIR / "comparison_resolution_cpu_vs_a100.png"
    fig.savefig(out_fig, dpi=300)
    plt.close()
    print(f" -> Gráfico comparativo salvo em: {out_fig}")


if __name__ == "__main__":
    generate_comparison()
