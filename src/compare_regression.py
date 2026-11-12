"""
Script de Consolidação e Comparação de Desempenho Multi-Resolução em Regressão.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Projeto 2ª Unidade - Previsor de Thumbnails com Deep Learning

Compara:
- Baseline Local: 64x64 (CPU)
- Alta Resolução Nativa: 480x360 (A100)
Gera tabela comparativa consolidada e gráfico de barras agrupadas de R² e MAE.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def generate_regression_comparison():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_64 = REPORTS_DIR / "regression_metrics_64x64.csv"
    csv_480 = REPORTS_DIR / "regression_metrics_480x360.csv"
    
    if not csv_480.exists():
        csv_480 = PROJECT_ROOT / "reports_gpu_fase_2" / "regression_metrics_480x360.csv"
        
    if not csv_64.exists() or not csv_480.exists():
        raise FileNotFoundError(f"Arquivos de métricas de regressão não localizados em {REPORTS_DIR}")
        
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
    
    comp_data = []
    for model in order:
        r2_64 = df_64.loc[model, "R² (Log10)"]
        r2_480 = df_480.loc[model, "R² (Log10)"]
        mae_64 = df_64.loc[model, "MAE (Log10)"]
        mae_480 = df_480.loc[model, "MAE (Log10)"]
        rmse_64 = df_64.loc[model, "RMSE (Log10)"]
        rmse_480 = df_480.loc[model, "RMSE (Log10)"]
        
        comp_data.append({
            "Modelo": model,
            "R² Log (64x64)": f"{r2_64 * 100:+.2f}%",
            "R² Log (480x360)": f"{r2_480 * 100:+.2f}%",
            "Δ R² Log": f"{(r2_480 - r2_64) * 100:+.2f}%",
            "MAE Log (64x64)": f"{mae_64:.4f}",
            "MAE Log (480x360)": f"{mae_480:.4f}",
            "Δ MAE Log": f"{(mae_480 - mae_64):+.4f}",
            "RMSE Log (64x64)": f"{rmse_64:.4f}",
            "RMSE Log (480x360)": f"{rmse_480:.4f}",
        })
        
    df_comp = pd.DataFrame(comp_data).set_index("Modelo")
    out_csv = REPORTS_DIR / "comparison_regression_64x64_vs_480x360.csv"
    df_comp.to_csv(out_csv)
    
    print("\n================ TABELA COMPARATIVA DE REGRESSÃO: 64x64 (CPU) vs 480x360 (A100) ================")
    try:
        print(df_comp.to_markdown())
    except Exception:
        print(df_comp.to_string())
    print(f" -> Tabela salva em: {out_csv}")
    
    # Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(order))
    width = 0.35
    short_names = ["MLP / DNN", "CNN 2D", "ResNet 2D", "RF (Pixels)", "RF (Autoencoder)"]
    
    # Painel 1: R² em Log10 (%)
    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, df_64["R² (Log10)"] * 100, width, label="64x64 (CPU)", color="#4A90E2")
    rects2 = ax1.bar(x + width/2, df_480["R² (Log10)"] * 100, width, label="480x360 (A100)", color="#2ECC71")
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_title("Coeficiente de Determinação R² em Escala Log10 (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("R² Log10 (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax1.set_ylim(-14, 11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    
    for rect in rects1:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        y_pos = h + 0.3 if h >= 0 else h - 0.7
        ax1.annotate(f"{h:+.1f}%", xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                     ha="center", va=va, fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        y_pos = h + 0.3 if h >= 0 else h - 0.7
        ax1.annotate(f"{h:+.1f}%", xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                     ha="center", va=va, fontsize=9, fontweight="bold")
                     
    # Painel 2: MAE em Log10 (Menor é Melhor)
    ax2 = axes[1]
    rects3 = ax2.bar(x - width/2, df_64["MAE (Log10)"], width, label="64x64 (CPU)", color="#E67E22")
    rects4 = ax2.bar(x + width/2, df_480["MAE (Log10)"], width, label="480x360 (A100)", color="#9B59B6")
    ax2.set_title("Erro Médio Absoluto MAE em Escala Log10 (Menor = Melhor)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("MAE Log10 (dex)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax2.set_ylim(0.7, 1.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    plt.suptitle("Impacto da Resolução Nativa na Regressão Contínua (Visualizações)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    out_fig = FIGURES_DIR / "comparison_regression_resolution_cpu_vs_a100.png"
    fig.savefig(out_fig, dpi=300)
    plt.close()
    print(f" -> Gráfico comparativo de regressão salvo em: {out_fig}")


if __name__ == "__main__":
    generate_regression_comparison()
