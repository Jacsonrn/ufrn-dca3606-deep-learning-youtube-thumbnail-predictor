"""
Análise do Espaço Latente (Embeddings) e Reconstrução Visual do Autoencoder.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Projeto 2ª Unidade - Dia 3 (Slide 19 do Roteiro Oficial)
"""

from pathlib import Path
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures import ConvAutoencoder
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEVICE = torch.device("cpu")  # Análise determinística e compatível localmente


def run_embedding_analysis():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "dataset_cache_64x64.npz"
    ae_path = MODELS_DIR / "autoencoder.pt"

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache não encontrado em {cache_path}")
    if not ae_path.exists():
        raise FileNotFoundError(f"Pesos do Autoencoder não encontrados em {ae_path}")

    print("Carregando tensores de teste e modelo Autoencoder...")
    data = np.load(cache_path)
    X_train = data["X_train"]
    y_class_train = data["y_class_train"]
    y_views_train = data["y_views_train"]

    X_test = data["X_test"]
    y_class_test = data["y_class_test"]
    y_views_test = data["y_views_test"]
    y_log_test = np.log10(y_views_test + 1.0)
    y_log_train = np.log10(y_views_train + 1.0)

    # 1. Carregar Autoencoder (64x64)
    model = ConvAutoencoder(latent_dim=64, target_shape=(64, 64)).to(DEVICE)
    model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    model.eval()

    # =========================================================================
    # PARTE 1: Amostras de Reconstrução Visual (Original vs Reconstruída)
    # =========================================================================
    print("Gerando painel comparativo de reconstrução visual (Original vs Reconstruída)...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=8, replace=False)
    sample_imgs = X_test[sample_indices]

    # Formato PyTorch: (N, C, H, W)
    tensors = torch.from_numpy(sample_imgs).permute(0, 3, 1, 2).float().to(DEVICE)
    with torch.no_grad():
        recon_tensors, _ = model(tensors)
        recon_imgs = recon_tensors.permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(20, 5.5))
    for i in range(8):
        orig = sample_imgs[i]
        rec = np.clip(recon_imgs[i], 0.0, 1.0)
        mse_pixel = np.mean((orig - rec) ** 2)

        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"Original #{i+1}", fontsize=10, fontweight="bold")
        axes[0, i].axis("off")

        axes[1, i].imshow(rec)
        axes[1, i].set_title(f"Reconst.\nMSE: {mse_pixel:.4f}", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle(
        "Autoencoder Convolucional 2D — Reconstrução de Thumbnails de Teste (64x64)\n"
        "Gargalo Latente: d = 12.288 pixels -> k = 64 dimensões (Taxa de Compressão: 192:1)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    recon_fig_path = FIGURES_DIR / "autoencoder_reconstruction_samples.png"
    plt.savefig(recon_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Salvo: {recon_fig_path.name}")

    # =========================================================================
    # PARTE 2: Extração de Embeddings e Projeções 2D (PCA e t-SNE)
    # =========================================================================
    print("Extraindo embeddings latentes Z para o conjunto de teste...")
    test_tensor_all = torch.from_numpy(X_test).permute(0, 3, 1, 2).float().to(DEVICE)
    with torch.no_grad():
        Z_test = model.encode(test_tensor_all).cpu().numpy()

    train_tensor_all = torch.from_numpy(X_train).permute(0, 3, 1, 2).float().to(DEVICE)
    with torch.no_grad():
        Z_train = model.encode(train_tensor_all).cpu().numpy()

    print("Calculando projeções PCA (linear) e t-SNE (não-linear)...")
    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(Z_test)
    var_exp = pca.explained_variance_ratio_ * 100

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    Z_tsne = tsne.fit_transform(Z_test)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot A: PCA
    scatter_pca = axes[0].scatter(
        Z_pca[:, 0], Z_pca[:, 1],
        c=y_class_test, cmap="coolwarm", alpha=0.8, edgecolors="k", linewidth=0.5, s=45
    )
    axes[0].set_title(f"Projeção PCA 2D (Linear)\nVar. Explicada: {var_exp[0]:.1f}% (PC1) + {var_exp[1]:.1f}% (PC2) = {sum(var_exp):.1f}%", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Componente Principal 1")
    axes[0].set_ylabel("Componente Principal 2")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    cbar1 = plt.colorbar(scatter_pca, ax=axes[0], ticks=[0, 1])
    cbar1.ax.set_yticklabels(["Baixo Engajamento", "Alto Engajamento"])

    # Subplot B: t-SNE
    scatter_tsne = axes[1].scatter(
        Z_tsne[:, 0], Z_tsne[:, 1],
        c=y_log_test, cmap="viridis", alpha=0.85, edgecolors="k", linewidth=0.5, s=45
    )
    axes[1].set_title("Projeção t-SNE 2D (Não-Linear, Perplexity=30)\nGradiente Contínuo de log10(Views)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimensão 1")
    axes[1].set_ylabel("t-SNE Dimensão 2")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    cbar2 = plt.colorbar(scatter_tsne, ax=axes[1])
    cbar2.set_label("log10(Views + 1)")

    fig.suptitle(
        "Mapeamento Bidimensional do Espaço Latente Z (k = 64) — Conjunto de Teste (N = 255)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    latent_fig_path = FIGURES_DIR / "latent_space_tsne_pca.png"
    plt.savefig(latent_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Salvo: {latent_fig_path.name}")

    # =========================================================================
    # PARTE 3: Tabela Oficial do Slide 19 (Efeito da Representação Aprendida)
    # =========================================================================
    print("Calculando tempos e consolidando tabela do Slide 19...")
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # Medir RF Classificador sobre X
    t0 = time.time()
    rf_c_x = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1)
    rf_c_x.fit(X_train_flat, y_class_train)
    t_train_c_x = time.time() - t0
    acc_c_x = accuracy_score(y_class_test, rf_c_x.predict(X_test_flat)) * 100

    # Medir RF Classificador sobre Z
    t0 = time.time()
    rf_c_z = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1)
    rf_c_z.fit(Z_train, y_class_train)
    t_train_c_z = time.time() - t0
    acc_c_z = accuracy_score(y_class_test, rf_c_z.predict(Z_test)) * 100

    # Medir RF Regressor sobre X
    t0 = time.time()
    rf_r_x = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1)
    rf_r_x.fit(X_train_flat, y_log_train)
    t_train_r_x = time.time() - t0
    r2_r_x = r2_score(y_log_test, rf_r_x.predict(X_test_flat)) * 100

    # Medir RF Regressor sobre Z
    t0 = time.time()
    rf_r_z = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1)
    rf_r_z.fit(Z_train, y_log_train)
    t_train_r_z = time.time() - t0
    r2_r_z = r2_score(y_log_test, rf_r_z.predict(Z_test)) * 100

    table_data = [
        {
            "Tarefa": "Classificação",
            "Representação": "Atributos Originais (X)",
            "Dimensão": "d = 12.288",
            "Taxa de Compressão": "1:1",
            "Métrica Principal": f"Acurácia: {acc_c_x:.2f}%",
            "Tempo Treino (s)": f"{t_train_c_x:.2f}s"
        },
        {
            "Tarefa": "Classificação",
            "Representação": "Embedding do AE (Z)",
            "Dimensão": "k = 64",
            "Taxa de Compressão": "192:1 (99.48% menor)",
            "Métrica Principal": f"Acurácia: {acc_c_z:.2f}%",
            "Tempo Treino (s)": f"{t_train_c_z:.2f}s"
        },
        {
            "Tarefa": "Regressão",
            "Representação": "Atributos Originais (X)",
            "Dimensão": "d = 12.288",
            "Taxa de Compressão": "1:1",
            "Métrica Principal": f"R²_log: {r2_r_x:.2f}%",
            "Tempo Treino (s)": f"{t_train_r_x:.2f}s"
        },
        {
            "Tarefa": "Regressão",
            "Representação": "Embedding do AE (Z)",
            "Dimensão": "k = 64",
            "Taxa de Compressão": "192:1 (99.48% menor)",
            "Métrica Principal": f"R²_log: {r2_r_z:.2f}%",
            "Tempo Treino (s)": f"{t_train_r_z:.2f}s"
        }
    ]

    df_effect = pd.DataFrame(table_data)
    csv_path = REPORTS_DIR / "embedding_representation_effect.csv"
    df_effect.to_csv(csv_path, index=False)
    print(f" -> Tabela do Slide 19 salva em: {csv_path.name}")
    print("\n" + df_effect.to_string(index=False))


if __name__ == "__main__":
    run_embedding_analysis()
