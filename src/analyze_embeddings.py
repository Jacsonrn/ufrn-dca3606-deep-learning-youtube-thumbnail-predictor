"""
Análise do Espaço Latente (Embeddings) e Reconstrução Visual do Autoencoder.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Projeto 2ª Unidade - Dia 3 (Slide 19 do Roteiro Oficial)
Suporta execução em resolução 64x64 ou Nativa 480x360.
"""

from pathlib import Path
import sys
import time
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures import ConvAutoencoder
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_embedding_analysis(res: str = "480x360"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = DATA_DIR / f"dataset_cache_{res}.npz"
    suffix = f"_{res}"
    ae_path = MODELS_DIR / f"autoencoder{suffix}.pt"

    # Fallback transparente caso a resolução nativa ainda não tenha sido gerada no ambiente atual
    if not cache_path.exists() or not ae_path.exists():
        if res == "480x360":
            print(f">> [AVISO] Cache ou modelo para {res} não encontrado. Tentando fallback para 64x64...")
            fallback_cache = DATA_DIR / "dataset_cache_64x64.npz"
            fallback_ae = MODELS_DIR / "autoencoder_64x64.pt" if (MODELS_DIR / "autoencoder_64x64.pt").exists() else MODELS_DIR / "autoencoder.pt"
            if fallback_cache.exists() and fallback_ae.exists():
                cache_path = fallback_cache
                ae_path = fallback_ae
                suffix = "_64x64"
                res = "64x64"
            else:
                raise FileNotFoundError(f"Nem os arquivos de 480x360 nem o fallback 64x64 foram encontrados em {DATA_DIR} e {MODELS_DIR}.")
        else:
            raise FileNotFoundError(f"Arquivos não encontrados para resolução {res}.")

    print(f">> Carregando dataset cache ({res}) de {cache_path.name}...")
    print(f">> Carregando pesos do Autoencoder de {ae_path.name}...")
    data = np.load(cache_path)
    X_train = data["X_train"]
    y_class_train = data["y_class_train"]
    y_views_train = data["y_views_train"]

    X_test = data["X_test"]
    y_class_test = data["y_class_test"]
    y_views_test = data["y_views_test"]
    y_log_test = np.log10(y_views_test + 1.0)
    y_log_train = np.log10(y_views_train + 1.0)

    # Dimensões dinâmicas
    target_h, target_w = X_train.shape[1], X_train.shape[2]
    d_input = target_h * target_w * 3
    compression_ratio = d_input / 64.0
    compression_pct = (1.0 - (64.0 / d_input)) * 100.0

    print(f">> Dimensão espacial: {target_w}x{target_h}x3 (d = {d_input:,} pixels) -> Latente k = 64")
    print(f">> Razão de Compressão de Shannon: {compression_ratio:.1f}:1 ({compression_pct:.3f}% de redução)")

    # 1. Carregar Autoencoder dinâmico
    model = ConvAutoencoder(latent_dim=64, target_shape=(target_h, target_w)).to(DEVICE)
    model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    model.eval()

    # =========================================================================
    # PARTE 1: Amostras de Reconstrução Visual (Original vs Reconstruída)
    # =========================================================================
    print("Gerando painel comparativo de reconstrução visual em alta resolução...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=8, replace=False)
    sample_imgs = X_test[sample_indices]

    tensors = torch.from_numpy(sample_imgs).permute(0, 3, 1, 2).float().to(DEVICE)
    with torch.no_grad():
        recon_tensors, _ = model(tensors)
        recon_imgs = recon_tensors.permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(22, 6))
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
        f"Autoencoder Convolucional 2D — Reconstrução de Thumbnails ({target_w}x{target_h})\n"
        f"Gargalo de Informação de Shannon: d = {d_input:,} pixels -> k = 64 dimensões (Compressão: {compression_ratio:.0f}:1 | {compression_pct:.2f}% de redução)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    recon_fig_res = FIGURES_DIR / f"autoencoder_reconstruction_samples{suffix}.png"
    recon_fig_default = FIGURES_DIR / "autoencoder_reconstruction_samples.png"
    plt.savefig(recon_fig_res, dpi=300, bbox_inches="tight")
    plt.savefig(recon_fig_default, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Painel de reconstrução salvo: {recon_fig_res.name}")

    # =========================================================================
    # PARTE 2: Extração de Embeddings e Projeções 2D (PCA e t-SNE)
    # =========================================================================
    print("Extraindo embeddings latentes Z para os conjuntos de treino e teste...")
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

    scatter_pca = axes[0].scatter(
        Z_pca[:, 0], Z_pca[:, 1],
        c=y_class_test, cmap="coolwarm", alpha=0.8, edgecolors="k", linewidth=0.5, s=45
    )
    axes[0].set_title(f"Projeção PCA 2D (Linear) — Resolução {target_w}x{target_h}\nVar. Explicada: {var_exp[0]:.1f}% (PC1) + {var_exp[1]:.1f}% (PC2) = {sum(var_exp):.1f}%", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Componente Principal 1")
    axes[0].set_ylabel("Componente Principal 2")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    cbar1 = plt.colorbar(scatter_pca, ax=axes[0], ticks=[0, 1])
    cbar1.ax.set_yticklabels(["Baixo Engajamento", "Alto Engajamento"])

    scatter_tsne = axes[1].scatter(
        Z_tsne[:, 0], Z_tsne[:, 1],
        c=y_log_test, cmap="viridis", alpha=0.85, edgecolors="k", linewidth=0.5, s=45
    )
    axes[1].set_title(f"Projeção t-SNE 2D (Não-Linear) — Resolução {target_w}x{target_h}\nGradiente Contínuo de log10(Views)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimensão 1")
    axes[1].set_ylabel("t-SNE Dimensão 2")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    cbar2 = plt.colorbar(scatter_tsne, ax=axes[1])
    cbar2.set_label("log10(Views + 1)")

    fig.suptitle(
        f"Mapeamento Bidimensional do Espaço Latente Z (k = 64) — Conjunto de Teste (N = {len(X_test)})",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    latent_fig_res = FIGURES_DIR / f"latent_space_tsne_pca{suffix}.png"
    latent_fig_default = FIGURES_DIR / "latent_space_tsne_pca.png"
    plt.savefig(latent_fig_res, dpi=300, bbox_inches="tight")
    plt.savefig(latent_fig_default, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Projeções latentes salvas: {latent_fig_res.name}")

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
            "Representação": f"Pixels Originais ({target_w}x{target_h})",
            "Dimensão": f"d = {d_input:,}",
            "Taxa de Compressão": "1:1",
            "Métrica Principal": f"Acurácia: {acc_c_x:.2f}%",
            "Tempo Treino (s)": f"{t_train_c_x:.2f}s"
        },
        {
            "Tarefa": "Classificação",
            "Representação": "Embedding do AE (Z)",
            "Dimensão": "k = 64",
            "Taxa de Compressão": f"{compression_ratio:.0f}:1 ({compression_pct:.2f}% menor)",
            "Métrica Principal": f"Acurácia: {acc_c_z:.2f}%",
            "Tempo Treino (s)": f"{t_train_c_z:.2f}s"
        },
        {
            "Tarefa": "Regressão",
            "Representação": f"Pixels Originais ({target_w}x{target_h})",
            "Dimensão": f"d = {d_input:,}",
            "Taxa de Compressão": "1:1",
            "Métrica Principal": f"R²_log: {r2_r_x:.2f}%",
            "Tempo Treino (s)": f"{t_train_r_x:.2f}s"
        },
        {
            "Tarefa": "Regressão",
            "Representação": "Embedding do AE (Z)",
            "Dimensão": "k = 64",
            "Taxa de Compressão": f"{compression_ratio:.0f}:1 ({compression_pct:.2f}% menor)",
            "Métrica Principal": f"R²_log: {r2_r_z:.2f}%",
            "Tempo Treino (s)": f"{t_train_r_z:.2f}s"
        }
    ]

    df_effect = pd.DataFrame(table_data)
    csv_res = REPORTS_DIR / f"embedding_representation_effect{suffix}.csv"
    csv_default = REPORTS_DIR / "embedding_representation_effect.csv"
    df_effect.to_csv(csv_res, index=False)
    df_effect.to_csv(csv_default, index=False)
    print(f" -> Tabela salva em: {csv_res.name}")
    print("\n" + df_effect.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise do Espaço Latente e Reconstrução Visual")
    parser.add_argument("--res", type=str, default="480x360", choices=["64x64", "480x360"],
                        help="Resolução para carregar e analisar (padrão: 480x360)")
    args = parser.parse_args()
    run_embedding_analysis(res=args.res)
