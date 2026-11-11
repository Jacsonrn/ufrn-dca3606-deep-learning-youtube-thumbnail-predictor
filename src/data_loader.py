"""
Módulo de Ingestão e Pré-Processamento de Dados de Thumbnails do YouTube.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Projeto 2ª Unidade - Previsor de Thumbnails com Deep Learning

Garante:
1. Download concorrente e cache local das imagens do YouTube.
2. Redimensionamento padronizado para 64x64x3 RGB normalizado em [0, 1].
3. Divisão estrita em Treino (70%), Validação (15%) e Teste (15%).
4. Binarização anti-vazamento (mediana calculada EXCLUSIVAMENTE no Treino).
5. Fornecimento de DataLoaders PyTorch e matrizes NumPy achatadas (flatten).
"""

import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_THUMBNAILS_DIR = DATA_DIR / "raw" / "thumbnails"
PROCESSED_DIR = DATA_DIR / "processed"


def download_metadata() -> pd.DataFrame:
    """Baixa o CSV de estatísticas do YouTube do Hugging Face Hub."""
    print("[1/5] Baixando metadados do Hugging Face Hub...")
    csv_path = hf_hub_download(
        repo_id="jason1966/advaypatil_youtube-statistics",
        filename="videos-stats.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(csv_path)
    # Remove entradas com valores nulos essenciais
    df = df.dropna(subset=["Video ID", "Views"]).copy()
    df["Views"] = df["Views"].astype(np.float64)
    df = df.drop_duplicates(subset=["Video ID"]).reset_index(drop=True)
    print(f" -> {len(df)} registros de vídeos válidos carregados.")
    return df


def download_single_thumbnail(video_id: str, output_path: Path) -> bool:
    """Baixa uma thumbnail individual da CDN oficial do YouTube."""
    if output_path.exists() and output_path.stat().st_size > 1000:
        return True
    
    url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                content = resp.read()
                # Verifica se não é a imagem placeholder de erro do YouTube (~1097 bytes)
                if len(content) > 1500:
                    output_path.write_bytes(content)
                    return True
    except Exception:
        pass
    return False


def fetch_all_thumbnails(df: pd.DataFrame, max_workers: int = 16) -> pd.DataFrame:
    """Executa o download concorrente das thumbnails e filtra vídeos indisponíveis."""
    RAW_THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[2/5] Verificando e baixando thumbnails para {len(df)} vídeos (threads={max_workers})...")
    
    valid_indices = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_single_thumbnail,
                row["Video ID"],
                RAW_THUMBNAILS_DIR / f"{row['Video ID']}.jpg",
            ): idx
            for idx, row in df.iterrows()
        }
        
        for future in tqdm(futures, desc="Baixando imagens"):
            idx = futures[future]
            if future.result():
                valid_indices.append(idx)
                
    filtered_df = df.loc[valid_indices].reset_index(drop=True)
    print(f" -> {len(filtered_df)} thumbnails íntegras disponíveis em disco.")
    return filtered_df


def process_images_and_targets(
    df: pd.DataFrame, target_size: Tuple[int, int] = (480, 360)
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Lê as imagens, preserva a resolução nativa máxima (480x360) RGB e normaliza pixels em [0, 1]."""
    print(f"[3/5] Processando imagens em alta resolução nativa {target_size} RGB...")
    images = []
    views = []
    kept_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processando tensores"):
        img_path = RAW_THUMBNAILS_DIR / f"{row['Video ID']}.jpg"
        try:
            with Image.open(img_path) as img:
                if img.size == target_size:
                    img_rgb = img.convert("RGB")
                else:
                    img_rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
                arr = np.array(img_rgb, dtype=np.float32) / 255.0  # Normalização [0, 1]
                images.append(arr)
                views.append(row["Views"])
                kept_rows.append(row)
        except Exception:
            continue

    X = np.stack(images, axis=0)  # Formato: (N, H, W, 3)
    y_views = np.array(views, dtype=np.float32)
    clean_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    print(f" -> Tensor gerado: X={X.shape}, y_views={y_views.shape}")
    return X, y_views, clean_df


def split_data_anti_leakage(
    X: np.ndarray, y_views: np.ndarray, random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Realiza a divisão em Treino (70%), Validação (15%) e Teste (15%)
    calculando a mediana de classificação EXCLUSIVAMENTE sobre o Treino.
    """
    print("[4/5] Executando partição Treino (70%) / Val (15%) / Teste (15%) sem vazamento...")
    np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)
    
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Cálculo da mediana EXCLUSIVAMENTE nos dados de treino
    median_views_train = float(np.median(y_views[train_idx]))
    print(f" -> Mediana de views do Treino (limiar de corte): {median_views_train:,.0f} views")
    
    # Binarização usando estritamente o limiar de treino
    y_class_train = (y_views[train_idx] >= median_views_train).astype(np.int64)
    y_class_val = (y_views[val_idx] >= median_views_train).astype(np.int64)
    y_class_test = (y_views[test_idx] >= median_views_train).astype(np.int64)
    
    return {
        "X_train": X[train_idx],
        "y_views_train": y_views[train_idx],
        "y_class_train": y_class_train,
        "X_val": X[val_idx],
        "y_views_val": y_views[val_idx],
        "y_class_val": y_class_val,
        "X_test": X[test_idx],
        "y_views_test": y_views[test_idx],
        "y_class_test": y_class_test,
        "median_views_train": median_views_train,
    }


def get_dataloaders(
    data: Dict[str, np.ndarray], batch_size: int = 32
) -> Dict[str, DataLoader]:
    """
    Converte tensores no padrão PyTorch (N, C, H, W) e retorna DataLoaders.
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        # PyTorch espera canais primeiro: (N, H, W, C) -> (N, C, H, W)
        x_tensor = torch.tensor(data[f"X_{split}"], dtype=torch.float32).permute(0, 3, 1, 2)
        y_class_tensor = torch.tensor(data[f"y_class_{split}"], dtype=torch.long)
        y_views_tensor = torch.tensor(data[f"y_views_{split}"], dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(x_tensor, y_class_tensor, y_views_tensor)
        shuffle = True if split == "train" else False
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    return loaders


def prepare_dataset(
    batch_size: int = 32, target_size: Tuple[int, int] = (480, 360)
) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray]]:
    """Função principal que orquestra todo o pipeline com cache rápido em disco."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PROCESSED_DIR / f"dataset_cache_{target_size[0]}x{target_size[1]}.npz"

    if cache_file.exists():
        print(f" -> Carregando dataset em alta resolução do cache: {cache_file.name}...")
        cached = np.load(cache_file)
        data = {k: cached[k] for k in cached.files}
        data["median_views_train"] = float(data["median_views_train"])
        loaders = get_dataloaders(data, batch_size=batch_size)
        print("[5/5] Pipeline de dados concluído com sucesso (via cache)!")
        return loaders, data

    df = download_metadata()
    df_thumbnails = fetch_all_thumbnails(df)
    X, y_views, clean_df = process_images_and_targets(df_thumbnails, target_size=target_size)
    data = split_data_anti_leakage(X, y_views)
    
    # Salva cache compactado para acelerar execuções futuras
    np.savez_compressed(cache_file, **data)
    print(f" -> Cache salvo em: {cache_file}")

    loaders = get_dataloaders(data, batch_size=batch_size)
    print("[5/5] Pipeline de dados concluído com sucesso!")
    return loaders, data


if __name__ == "__main__":
    loaders, data = prepare_dataset()
    print("\n--- Resumo das Partições Geradas ---")
    for split in ["train", "val", "test"]:
        print(f"Partição {split.upper():5s}: {len(data[f'X_{split}'])} amostras | "
              f"Classes: {np.bincount(data[f'y_class_{split}'])}")
