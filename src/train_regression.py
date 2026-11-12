"""
Pipeline de Regressão Contínua das 5 Abordagens Obrigatórias.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Projeto 2ª Unidade - Previsor de Thumbnails com Deep Learning

Executa:
- Abordagem 1: MLP / DNN Profunda Regressora
- Abordagem 2: CNN Convolucional 2D Regressora
- Abordagem 3: ResNet 2D Regressora com Blocos Residuais
- Abordagem 4: Autoencoder Convolucional + Random Forest Regressor (espaço latente Z)
- Abordagem 5: Random Forest Regressor nos Pixels Brutos (Flatten)
Gera tabela comparativa de métricas (MAE, RMSE, R² na escala real e log) e gráficos de diagnóstico.
"""

import sys
from pathlib import Path
import copy
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import prepare_dataset
from src.architectures import (
    MLPRegressor, CNNRegressor, ResNetRegressor, ConvAutoencoder
)
from src.train_classification import train_autoencoder, extract_embeddings

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_regression_loaders(
    data: dict, scaler: StandardScaler, batch_size: int = 32
) -> Dict[str, DataLoader]:
    """Cria DataLoaders com alvo contínuo logarítmico padronizado."""
    loaders = {}
    for split in ["train", "val", "test"]:
        x_tensor = torch.tensor(data[f"X_{split}"], dtype=torch.float32).permute(0, 3, 1, 2)
        y_log = np.log10(data[f"y_views_{split}"] + 1.0)
        
        if split == "train":
            y_norm = scaler.fit_transform(y_log.reshape(-1, 1)).flatten()
        else:
            y_norm = scaler.transform(y_log.reshape(-1, 1)).flatten()
            
        y_tensor = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(x_tensor, y_tensor)
        shuffle = True if split == "train" else False
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    return loaders


def train_deep_regressor(
    model: nn.Module,
    loaders: dict,
    epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 5,
    model_name: str = "model"
) -> Tuple[nn.Module, dict]:
    """Treina redes neurais regressoras com perda MSE e Early Stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n---> Treinando {model_name}...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, total = 0.0, 0
        for x_b, y_b in loaders["train"]:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(y_b)
            total += len(y_b)
            
        train_loss = running_loss / total
        
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for x_b, y_b in loaders["val"]:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                out = model(x_b)
                loss = criterion(out, y_b)
                val_loss += loss.item() * len(y_b)
                val_total += len(y_b)
                
        val_loss /= val_total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Época {epoch:02d}/{epochs} | Treino MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f" -> Early stopping na época {epoch} (Melhor Val MSE: {best_val_loss:.4f})")
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, history


def evaluate_regression(
    y_true_raw: np.ndarray, preds_raw: np.ndarray,
    y_true_log: np.ndarray, preds_log: np.ndarray
) -> dict:
    """Calcula métricas formais na escala real e na escala logarítmica."""
    return {
        "MAE (Views)": mean_absolute_error(y_true_raw, preds_raw),
        "RMSE (Views)": np.sqrt(mean_squared_error(y_true_raw, preds_raw)),
        "R² (Views)": r2_score(y_true_raw, preds_raw),
        "MAE (Log10)": mean_absolute_error(y_true_log, preds_log),
        "RMSE (Log10)": np.sqrt(mean_squared_error(y_true_log, preds_log)),
        "R² (Log10)": r2_score(y_true_log, preds_log),
    }


def plot_real_vs_predicted(predictions_dict: dict, y_test_log: np.ndarray, output_path: Path):
    """Plota gráficos de dispersão Real vs. Previsto com a reta ideal y = x."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.8))
    
    for ax, (name, (preds_raw, preds_log)) in zip(axes, predictions_dict.items()):
        r2_log = r2_score(y_test_log, preds_log)
        mae_log = mean_absolute_error(y_test_log, preds_log)
        
        ax.scatter(y_test_log, preds_log, alpha=0.45, color="#1f77b4", edgecolors="none")
        # Linha ideal de 45 graus
        min_val = min(y_test_log.min(), preds_log.min())
        max_val = max(y_test_log.max(), preds_log.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal (y=x)")
        
        ax.set_title(f"{name}\nR²(Log): {r2_log:.3f} | MAE: {mae_log:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Real log10(views+1)", fontsize=10)
        ax.set_ylabel("Previsto log10(views+1)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)
        
    plt.suptitle("Regressão: Visualizações Reais vs. Predições dos Modelos (Escala Log10)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Gráficos de dispersão salvos em: {output_path}")


def plot_residuals_distribution(predictions_dict: dict, y_test_log: np.ndarray, output_path: Path):
    """Plota a distribuição e densidade dos resíduos de predição."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ["#4A90E2", "#2ECC71", "#E67E22", "#9B59B6", "#E74C3C"]
    for (name, (_, preds_log)), color in zip(predictions_dict.items(), colors):
        residuals = preds_log - y_test_log
        sns.kdeplot(residuals, label=name, color=color, linewidth=1.8, ax=ax)
        
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Erro Zero")
    ax.set_title("Distribuição dos Resíduos de Regressão [Previsto - Real] (Escala Log10)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Erro Residual log10(views)", fontsize=11)
    ax.set_ylabel("Densidade", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Distribuição de resíduos salva em: {output_path}")


def run_regression_pipeline(
    res: str = "480x360",
    device_arg: str = "auto",
    batch_size: int = 32,
    epochs: int = 25
):
    """Executa as 5 abordagens de regressão contínua no conjunto de dados."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    global DEVICE
    if device_arg == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(device_arg)
        
    device_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU Nativa"
    print(f" -> Dispositivo de computação ativo: {DEVICE} ({device_name})")
    
    target_size = (64, 64) if res == "64x64" else (480, 360)
    suffix = f"_{target_size[0]}x{target_size[1]}" if res != "64x64" else "_64x64"
    print(f" -> Configuração Regressão: Resolução {target_size[0]}x{target_size[1]} | Batch {batch_size} | Épocas {epochs}")
    
    # 1. Carregamento dos dados base
    raw_loaders, data = prepare_dataset(batch_size=batch_size, target_size=target_size)
    
    # 2. Preparação do alvo com log10 e StandardScaler anti-vazamento
    scaler = StandardScaler()
    reg_loaders = create_regression_loaders(data, scaler, batch_size=batch_size)
    
    y_test_raw = data["y_views_test"]
    y_test_log = np.log10(y_test_raw + 1.0)
    
    y_train_raw = data["y_views_train"]
    y_train_log = np.log10(y_train_raw + 1.0)
    y_train_norm = scaler.transform(y_train_log.reshape(-1, 1)).flatten()
    
    results = {}
    preds_dict = {}
    
    # -------------------------------------------------------------
    # Abordagem 5: Random Forest Regressor nos Pixels Brutos
    # -------------------------------------------------------------
    print("\n[Abordagem 5] Treinando Random Forest Regressor nos Pixels Brutos (Flatten)...")
    X_train_flat = data["X_train"].reshape(len(data["X_train"]), -1)
    X_test_flat = data["X_test"].reshape(len(data["X_test"]), -1)
    
    rf_raw = RandomForestRegressor(
        n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf_raw.fit(X_train_flat, y_train_log)
    preds_log_rf_raw = rf_raw.predict(X_test_flat)
    preds_raw_rf_raw = np.maximum(0.0, 10 ** preds_log_rf_raw - 1.0)
    
    results["Clássico com entrada original"] = evaluate_regression(
        y_test_raw, preds_raw_rf_raw, y_test_log, preds_log_rf_raw
    )
    preds_dict["Clássico com entrada original"] = (preds_raw_rf_raw, preds_log_rf_raw)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Abordagem 4: Autoencoder Convolucional + Random Forest Regressor
    # -------------------------------------------------------------
    target_h, target_w = data["X_train"].shape[1], data["X_train"].shape[2]
    print(f"\n[Abordagem 4] Autoencoder Convolucional + Random Forest Regressor ({target_w}x{target_h})...")
    ae_path = MODELS_DIR / f"autoencoder{suffix}.pt"
    ae_model = ConvAutoencoder(latent_dim=64, target_shape=(target_h, target_w)).to(DEVICE)
    if ae_path.exists():
        print(f" -> Carregando pesos salvos de {ae_path.name}...")
        ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    else:
        ae_model, _ = train_autoencoder(ae_model, raw_loaders, epochs=min(epochs, 20), lr=1e-3)
        torch.save(ae_model.state_dict(), ae_path)
        
    embeddings = extract_embeddings(ae_model, raw_loaders)
    rf_ae = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_ae.fit(embeddings["train"], y_train_log)
    preds_log_rf_ae = rf_ae.predict(embeddings["test"])
    preds_raw_rf_ae = np.maximum(0.0, 10 ** preds_log_rf_ae - 1.0)
    
    results["Clássico treinado com embeddings do AE"] = evaluate_regression(
        y_test_raw, preds_raw_rf_ae, y_test_log, preds_log_rf_ae
    )
    preds_dict["Clássico treinado com embeddings do AE"] = (preds_raw_rf_ae, preds_log_rf_ae)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Abordagem 1: MLP Regressor
    # -------------------------------------------------------------
    print(f"\n[Abordagem 1] MLP / DNN Regressora ({target_w}x{target_h})...")
    mlp_path = MODELS_DIR / f"mlp_regressor{suffix}.pt"
    input_dim = target_h * target_w * 3
    mlp_model = MLPRegressor(input_dim=input_dim).to(DEVICE)
    if mlp_path.exists():
        print(f" -> Carregando pesos salvos de {mlp_path.name}...")
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
    else:
        mlp_model, _ = train_deep_regressor(mlp_model, reg_loaders, epochs=epochs, lr=1e-3, model_name="MLP Regressora")
        torch.save(mlp_model.state_dict(), mlp_path)
        
    mlp_model.eval()
    with torch.no_grad():
        preds_mlp_norm = []
        for x_b, _ in reg_loaders["test"]:
            preds_mlp_norm.extend(mlp_model(x_b.to(DEVICE)).cpu().numpy().flatten())
    preds_mlp_log = scaler.inverse_transform(np.array(preds_mlp_norm).reshape(-1, 1)).flatten()
    preds_mlp_raw = np.maximum(0.0, 10 ** preds_mlp_log - 1.0)
    results["MLP / DNN"] = evaluate_regression(y_test_raw, preds_mlp_raw, y_test_log, preds_mlp_log)
    preds_dict["MLP / DNN"] = (preds_mlp_raw, preds_mlp_log)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Abordagem 2: CNN 2D Regressora
    # -------------------------------------------------------------
    print(f"\n[Abordagem 2] CNN 2D Regressora ({target_w}x{target_h})...")
    cnn_path = MODELS_DIR / f"cnn_regressor{suffix}.pt"
    cnn_model = CNNRegressor().to(DEVICE)
    if cnn_path.exists():
        print(f" -> Carregando pesos salvos de {cnn_path.name}...")
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    else:
        cnn_model, _ = train_deep_regressor(cnn_model, reg_loaders, epochs=epochs, lr=1e-3, model_name="CNN 2D Regressora")
        torch.save(cnn_model.state_dict(), cnn_path)
        
    cnn_model.eval()
    with torch.no_grad():
        preds_cnn_norm = []
        for x_b, _ in reg_loaders["test"]:
            preds_cnn_norm.extend(cnn_model(x_b.to(DEVICE)).cpu().numpy().flatten())
    preds_cnn_log = scaler.inverse_transform(np.array(preds_cnn_norm).reshape(-1, 1)).flatten()
    preds_cnn_raw = np.maximum(0.0, 10 ** preds_cnn_log - 1.0)
    results["CNN"] = evaluate_regression(y_test_raw, preds_cnn_raw, y_test_log, preds_cnn_log)
    preds_dict["CNN"] = (preds_cnn_raw, preds_cnn_log)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Abordagem 3: ResNet 2D Regressora
    # -------------------------------------------------------------
    print(f"\n[Abordagem 3] ResNet 2D Regressora ({target_w}x{target_h})...")
    resnet_path = MODELS_DIR / f"resnet_regressor{suffix}.pt"
    resnet_model = ResNetRegressor().to(DEVICE)
    if resnet_path.exists():
        print(f" -> Carregando pesos salvos de {resnet_path.name}...")
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
    else:
        resnet_model, _ = train_deep_regressor(resnet_model, reg_loaders, epochs=epochs, lr=1e-3, model_name="ResNet 2D Regressora")
        torch.save(resnet_model.state_dict(), resnet_path)
        
    resnet_model.eval()
    with torch.no_grad():
        preds_resnet_norm = []
        for x_b, _ in reg_loaders["test"]:
            preds_resnet_norm.extend(resnet_model(x_b.to(DEVICE)).cpu().numpy().flatten())
    preds_resnet_log = scaler.inverse_transform(np.array(preds_resnet_norm).reshape(-1, 1)).flatten()
    preds_resnet_raw = np.maximum(0.0, 10 ** preds_resnet_log - 1.0)
    results["ResNet"] = evaluate_regression(y_test_raw, preds_resnet_raw, y_test_log, preds_resnet_log)
    preds_dict["ResNet"] = (preds_resnet_raw, preds_resnet_log)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Consolidação dos Resultados e Exportação
    # -------------------------------------------------------------
    order = [
        "MLP / DNN",
        "CNN",
        "ResNet",
        "Clássico com entrada original",
        "Clássico treinado com embeddings do AE"
    ]
    df_results = pd.DataFrame(results).T.loc[order]
    print("\n================ TABELA COMPARATIVA DE REGRESSÃO ================")
    try:
        print(df_results.to_markdown())
    except Exception:
        print(df_results.to_string())
        
    csv_out = REPORTS_DIR / f"regression_metrics{suffix}.csv"
    df_results.to_csv(csv_out)
    df_results.to_csv(REPORTS_DIR / "regression_metrics.csv")
    print(f" -> Tabela salva em: {csv_out}")
    
    # Plots
    fig_scatter = FIGURES_DIR / f"regression_real_vs_predicted{suffix}.png"
    plot_real_vs_predicted(preds_dict, y_test_log, fig_scatter)
    plot_real_vs_predicted(preds_dict, y_test_log, FIGURES_DIR / "regression_real_vs_predicted.png")
    
    fig_residuals = FIGURES_DIR / f"regression_residuals_distribution{suffix}.png"
    plot_residuals_distribution(preds_dict, y_test_log, fig_residuals)
    plot_residuals_distribution(preds_dict, y_test_log, FIGURES_DIR / "regression_residuals_distribution.png")
    
    return df_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline de Regressão - 5 Abordagens de Deep Learning")
    parser.add_argument("--res", type=str, default="480x360", choices=["64x64", "480x360"],
                        help="Resolução das thumbnails (padrão: 480x360)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Dispositivo de computação (padrão: auto)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Tamanho do batch (padrão: 32)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Número máximo de épocas (padrão: 25)")
    args = parser.parse_args()
    
    run_regression_pipeline(
        res=args.res,
        device_arg=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
