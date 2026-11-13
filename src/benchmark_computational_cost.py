"""
Benchmark Formal de Custo Computacional, Parâmetros e Latência de Inferência.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Projeto 2ª Unidade - Dia 3 (Slide 14 do Roteiro Oficial)
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.architectures import (
    MLPClassifier, CNNClassifier, ResNetClassifier, ConvAutoencoder,
    MLPRegressor, CNNRegressor, ResNetRegressor
)
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def measure_latency_device(model: torch.nn.Module, sample_batch: torch.Tensor, device: torch.device, runs: int = 30) -> float:
    """Mede a latência média (ms) para processar 100 imagens com warm-up e sincronização explícita."""
    model = model.to(device)
    batch = sample_batch.to(device)
    model.eval()
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        # Warm-up de 5 iterações para estabilizar kernels e caches
        for _ in range(5):
            _ = model(batch)
        if is_cuda:
            torch.cuda.synchronize()

        latencies = []
        for _ in range(runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if is_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies))


def run_computational_benchmark():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Local"
    print(f"Iniciando Benchmark de Custo Computacional (Slide 14)...")
    print(f" -> Hardware Detectado: {device_name} (CUDA Ativo: {has_cuda})")

    # Lote sintético padronizado de 100 thumbnails (64x64x3)
    batch_64 = torch.randn(100, 3, 64, 64)

    models_to_test = [
        ("MLPClassifier (64x64)", MLPClassifier(input_dim=64*64*3, num_classes=2), batch_64, "Classificação"),
        ("CNNClassifier (64x64)", CNNClassifier(in_channels=3, num_classes=2), batch_64, "Classificação"),
        ("ResNetClassifier (64x64)", ResNetClassifier(in_channels=3, num_classes=2), batch_64, "Classificação"),
        ("ConvAutoencoder (64x64)", ConvAutoencoder(latent_dim=64, target_shape=(64, 64)), batch_64, "Não-Supervisionado"),
        ("MLPRegressor (64x64)", MLPRegressor(input_dim=64*64*3), batch_64, "Regressão"),
        ("CNNRegressor (64x64)", CNNRegressor(in_channels=3), batch_64, "Regressão"),
        ("ResNetRegressor (64x64)", ResNetRegressor(in_channels=3), batch_64, "Regressão"),
    ]

    results = []
    for name, model, batch, task in models_to_test:
        params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        # 1. Medição em CPU
        latency_cpu = measure_latency_device(model, batch, torch.device("cpu"), runs=30)
        fps_cpu = (100.0 / (latency_cpu / 1000.0))

        # 2. Medição em GPU (quando disponível)
        if has_cuda:
            latency_gpu = measure_latency_device(model, batch, torch.device("cuda"), runs=30)
            fps_gpu = (100.0 / (latency_gpu / 1000.0))
            speedup = latency_cpu / max(latency_gpu, 1e-4)
        else:
            latency_gpu = latency_cpu
            fps_gpu = fps_cpu
            speedup = 1.0

        results.append({
            "Modelo": name,
            "Tarefa": task,
            "Parâmetros Treináveis": params_trainable,
            "Tamanho Memória (MB)": round(model_size_mb, 2),
            "Latência CPU (ms)": round(latency_cpu, 2),
            "Throughput CPU (FPS)": round(fps_cpu, 1),
            "Latência A100 (ms)": round(latency_gpu, 2) if has_cuda else "N/A (CPU)",
            "Throughput A100 (FPS)": round(fps_gpu, 1) if has_cuda else "N/A (CPU)",
            "Speedup Hardware": f"{speedup:.1f}x" if has_cuda else "1.0x"
        })

    df = pd.DataFrame(results)
    csv_path = REPORTS_DIR / "computational_cost_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f" -> Tabela salva em: {csv_path.name}")
    print("\n" + df.to_string(index=False))

    # Gráfico de Barras: Parâmetros vs Latência
    fig, ax1 = plt.subplots(figsize=(12, 6))

    indices = np.arange(len(df))
    width = 0.38

    color1 = "#1f77b4"
    ax1.set_xlabel("Arquitetura Neural", fontweight="bold")
    ax1.set_ylabel("Parâmetros Treináveis (milhares)", color=color1, fontweight="bold")
    bars1 = ax1.bar(indices - width/2, df["Parâmetros Treináveis"] / 1000.0, width, label="Parâmetros (k)", color=color1, alpha=0.85)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(indices)
    ax1.set_xticklabels([m.split(" ")[0] for m in df["Modelo"]], rotation=25, ha="right", fontsize=9)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("Latência para 100 Imagens (ms)", color=color2, fontweight="bold")
    bars2 = ax2.bar(indices + width/2, df["Latência CPU (ms)"], width, label="Latência CPU (ms)", color=color2, alpha=0.85)
    ax2.tick_params(axis="y", labelcolor=color2)

    title_hardware = f"Hardware: {device_name}" if has_cuda else "Ambiente: CPU Local (Baseline)"
    plt.title(f"Comparativo de Custo Computacional: Parâmetros vs. Latência de Inferência\n({title_hardware} — Lote de 100 Imagens)", fontweight="bold", fontsize=12)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "computational_cost_vs_accuracy.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Gráfico salvo: {fig_path.name}")


if __name__ == "__main__":
    run_computational_benchmark()
