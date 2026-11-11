"""
Pipeline de Classificação das 5 Abordagens Obrigatórias.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Projeto 2ª Unidade - Previsor de Thumbnails com Deep Learning

Executa:
- Abordagem 1: MLP / DNN Profunda
- Abordagem 2: CNN Convolucional 2D
- Abordagem 3: ResNet com Blocos Residuais
- Abordagem 4: Autoencoder Convolucional + Random Forest (espaço latente Z)
- Abordagem 5: Random Forest Puro (pixels brutos achatados X)
Gera tabela de métricas (Acurácia, Precisão, Recall, Macro-F1) e Matrizes de Confusão.
"""

import sys
from pathlib import Path
import copy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import prepare_dataset
from src.architectures import MLPClassifier, CNNClassifier, ResNetClassifier, ConvAutoencoder


MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_deep_model(
    model: nn.Module,
    loaders: dict,
    epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 5,
    model_name: str = "model"
) -> Tuple[nn.Module, dict]:
    """Treina redes supervisionadas com Adam, CrossEntropy e Early Stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    print(f"\n---> Treinando {model_name}...")
    for epoch in range(1, epochs + 1):
        # Fase de Treino
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x_b, y_class_b, _ in loaders["train"]:
            x_b, y_class_b = x_b.to(DEVICE), y_class_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_class_b)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(y_class_b)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y_class_b).sum().item()
            total += len(y_class_b)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Fase de Validação
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_b, y_class_b, _ in loaders["val"]:
                x_b, y_class_b = x_b.to(DEVICE), y_class_b.to(DEVICE)
                out = model(x_b)
                loss = criterion(out, y_class_b)
                val_loss += loss.item() * len(y_class_b)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == y_class_b).sum().item()
                val_total += len(y_class_b)
                
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Época {epoch:02d}/{epochs} | Treino Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f" -> Early stopping na época {epoch} (Melhor Val Loss: {best_val_loss:.4f})")
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, history


def train_autoencoder(
    model: ConvAutoencoder, loaders: dict, epochs: int = 20, lr: float = 1e-3, patience: int = 5
) -> Tuple[ConvAutoencoder, dict]:
    """Treina o Autoencoder Convolucional de forma NÃO SUPERVISIONADA (MSE Loss)."""
    print("\n---> Treinando Autoencoder Convolucional (Não Supervisionado)...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, total_t = 0.0, 0
        for x_b, _, _ in loaders["train"]:
            x_b = x_b.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(x_b)
            loss = criterion(recon, x_b)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(x_b)
            total_t += len(x_b)
            
        train_loss = t_loss / total_t
        
        model.eval()
        v_loss, total_v = 0.0, 0
        with torch.no_grad():
            for x_b, _, _ in loaders["val"]:
                x_b = x_b.to(DEVICE)
                recon, _ = model(x_b)
                loss = criterion(recon, x_b)
                v_loss += loss.item() * len(x_b)
                total_v += len(x_b)
                
        val_loss = v_loss / total_v
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Época {epoch:02d}/{epochs} | Treino Reconstrução MSE: {train_loss:.5f} | Val Reconstrução MSE: {val_loss:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f" -> Autoencoder Early Stopping na época {epoch} (Melhor Val Loss: {best_loss:.5f})")
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, history


def extract_embeddings(model: ConvAutoencoder, loaders: dict) -> dict:
    """Extrai o espaço latente Z para Treino, Validação e Teste."""
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            z_list = []
            for x_b, _, _ in loaders[split]:
                x_b = x_b.to(DEVICE)
                z = model.encode(x_b)
                z_list.append(z.cpu().numpy())
            embeddings[split] = np.concatenate(z_list, axis=0)
    return embeddings


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula as métricas obrigatórias de classificação."""
    return {
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_combined_confusion_matrices(cms: dict, output_path: Path):
    """Gera figura combinada com as 5 Matrizes de Confusão lado a lado."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    labels = ["Baixo", "Alto"]
    
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Previsto", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)
        
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Matrizes de Confusão salvas em: {output_path}")


def run_classification_pipeline(
    res: str = "480x360",
    device_arg: str = "auto",
    batch_size: int = 32,
    epochs: int = 25
):
    """Executa as 5 abordagens de classificação no conjunto de dados parametrizado."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Resolução do dispositivo de computação
    if device_arg == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device_arg)
        
    global DEVICE
    DEVICE = dev
    device_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU Nativa"
    print(f" -> Dispositivo de computação ativo: {DEVICE} ({device_name})")
    
    # Resolução do target size
    if res == "64x64":
        target_size = (64, 64)
        suffix = "_64x64"
    else:
        target_size = (480, 360)
        suffix = "_480x360"
        
    print(f" -> Configuração: Resolução {target_size[0]}x{target_size[1]} | Batch Size {batch_size} | Épocas {epochs}")
    loaders, data = prepare_dataset(batch_size=batch_size, target_size=target_size)
    y_test = data["y_class_test"]
    
    results = {}
    confusion_matrices = {}
    
    # -------------------------------------------------------------
    # Abordagem 5: Random Forest Puro nos Pixels Brutos (Flatten)
    # -------------------------------------------------------------
    print("\n[Abordagem 5] Treinando Random Forest nos Pixels Brutos (Flatten)...")
    X_train_flat = data["X_train"].reshape(len(data["X_train"]), -1)
    X_test_flat = data["X_test"].reshape(len(data["X_test"]), -1)
    
    rf_raw = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_raw.fit(X_train_flat, data["y_class_train"])
    y_pred_rf_raw = rf_raw.predict(X_test_flat)
    
    results["Clássico com entrada original"] = evaluate_predictions(y_test, y_pred_rf_raw)
    confusion_matrices["Clássico com entrada original"] = confusion_matrix(y_test, y_pred_rf_raw)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Abordagem 4: Autoencoder Convolucional + Random Forest (Z)
    # -------------------------------------------------------------
    target_h, target_w = data["X_train"].shape[1], data["X_train"].shape[2]
    print(f"\n[Abordagem 4] Autoencoder Convolucional + Random Forest ({target_w}x{target_h})...")
    ae_path = MODELS_DIR / f"autoencoder{suffix}.pt"
    ae_model = ConvAutoencoder(latent_dim=64, target_shape=(target_h, target_w)).to(DEVICE)
    if ae_path.exists():
        print(f" -> Carregando pesos salvos de {ae_path.name}...")
        ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    else:
        ae_model, _ = train_autoencoder(ae_model, loaders, epochs=min(epochs, 20), lr=1e-3)
        torch.save(ae_model.state_dict(), ae_path)
    
    embeddings = extract_embeddings(ae_model, loaders)
    rf_ae = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_ae.fit(embeddings["train"], data["y_class_train"])
    y_pred_rf_ae = rf_ae.predict(embeddings["test"])
    
    results["Clássico treinado com embeddings do AE"] = evaluate_predictions(y_test, y_pred_rf_ae)
    confusion_matrices["Clássico treinado com embeddings do AE"] = confusion_matrix(y_test, y_pred_rf_ae)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Abordagem 1: MLP / DNN Profunda
    # -------------------------------------------------------------
    print(f"\n[Abordagem 1] MLP / DNN Profunda ({target_w}x{target_h})...")
    mlp_path = MODELS_DIR / f"mlp_classifier{suffix}.pt"
    input_dim = data["X_train"].shape[1] * data["X_train"].shape[2] * data["X_train"].shape[3]
    mlp_model = MLPClassifier(input_dim=input_dim).to(DEVICE)
    if mlp_path.exists():
        print(f" -> Carregando pesos salvos de {mlp_path.name}...")
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
    else:
        mlp_model, _ = train_deep_model(mlp_model, loaders, epochs=25, lr=1e-3, model_name="MLP / DNN")
        torch.save(mlp_model.state_dict(), mlp_path)
    
    mlp_model.eval()
    with torch.no_grad():
        preds_mlp = []
        for x_b, _, _ in loaders["test"]:
            preds_mlp.extend(torch.argmax(mlp_model(x_b.to(DEVICE)), dim=1).cpu().numpy())
    results["MLP / DNN"] = evaluate_predictions(y_test, np.array(preds_mlp))
    confusion_matrices["MLP / DNN"] = confusion_matrix(y_test, preds_mlp)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Abordagem 2: CNN 2D
    # -------------------------------------------------------------
    print(f"\n[Abordagem 2] CNN 2D ({target_w}x{target_h})...")
    cnn_path = MODELS_DIR / f"cnn_classifier{suffix}.pt"
    cnn_model = CNNClassifier().to(DEVICE)
    if cnn_path.exists():
        print(f" -> Carregando pesos salvos de {cnn_path.name}...")
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    else:
        cnn_model, _ = train_deep_model(cnn_model, loaders, epochs=25, lr=1e-3, model_name="CNN 2D")
        torch.save(cnn_model.state_dict(), cnn_path)
    
    cnn_model.eval()
    with torch.no_grad():
        preds_cnn = []
        for x_b, _, _ in loaders["test"]:
            preds_cnn.extend(torch.argmax(cnn_model(x_b.to(DEVICE)), dim=1).cpu().numpy())
    results["CNN"] = evaluate_predictions(y_test, np.array(preds_cnn))
    confusion_matrices["CNN"] = confusion_matrix(y_test, preds_cnn)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Abordagem 3: ResNet 2D
    # -------------------------------------------------------------
    print(f"\n[Abordagem 3] ResNet 2D ({target_w}x{target_h})...")
    resnet_path = MODELS_DIR / f"resnet_classifier{suffix}.pt"
    resnet_model = ResNetClassifier().to(DEVICE)
    if resnet_path.exists():
        print(f" -> Carregando pesos salvos de {resnet_path.name}...")
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
    else:
        resnet_model, _ = train_deep_model(resnet_model, loaders, epochs=25, lr=1e-3, model_name="ResNet 2D")
        torch.save(resnet_model.state_dict(), resnet_path)
    
    resnet_model.eval()
    with torch.no_grad():
        preds_resnet = []
        for x_b, _, _ in loaders["test"]:
            preds_resnet.extend(torch.argmax(resnet_model(x_b.to(DEVICE)), dim=1).cpu().numpy())
    results["ResNet"] = evaluate_predictions(y_test, np.array(preds_resnet))
    confusion_matrices["ResNet"] = confusion_matrix(y_test, preds_resnet)
    print(" -> Concluído!")
    
    # -------------------------------------------------------------
    # Consolidação e Exportação dos Resultados
    # -------------------------------------------------------------
    order = [
        "MLP / DNN",
        "CNN",
        "ResNet",
        "Clássico com entrada original",
        "Clássico treinado com embeddings do AE"
    ]
    df_results = pd.DataFrame(results).T.loc[order][["Acurácia", "Precisão", "Recall", "Macro-F1"]]
    print("\n================ TABELA COMPARATIVA DE CLASSIFICAÇÃO ================")
    try:
        print(df_results.to_markdown())
    except Exception:
        print(df_results.to_string())
    
    # Salvar tabela comparativa em CSV
    csv_out = PROJECT_ROOT / "reports" / f"classification_metrics{suffix}.csv"
    df_results.to_csv(csv_out)
    if suffix:
        df_results.to_csv(PROJECT_ROOT / "reports" / "classification_metrics.csv")
    print(f" -> Tabela salva em: {csv_out}")
    
    # Salvar Matrizes de Confusão
    ordered_cms = {k: confusion_matrices[k] for k in order}
    fig_out = FIGURES_DIR / f"confusion_matrices_classification{suffix}.png"
    plot_combined_confusion_matrices(ordered_cms, fig_out)
    if suffix:
        plot_combined_confusion_matrices(ordered_cms, FIGURES_DIR / "confusion_matrices_classification.png")
    return df_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline de Classificação - 5 Abordagens de Deep Learning")
    parser.add_argument("--res", type=str, default="480x360", choices=["64x64", "480x360"],
                        help="Resolução das thumbnails (padrão: 480x360)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Dispositivo de computação (padrão: auto)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Tamanho do batch (padrão: 32)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Número máximo de épocas (padrão: 25)")
    args = parser.parse_args()
    
    run_classification_pipeline(
        res=args.res,
        device_arg=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
