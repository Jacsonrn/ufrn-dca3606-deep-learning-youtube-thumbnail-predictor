"""
Definições das Arquiteturas Neurais em PyTorch.
Disciplina: DCA3606 - Inteligência Artificial (DCA/UFRN)
Projeto 2ª Unidade - Previsor de Thumbnails com Deep Learning

Modelos:
1. MLPClassifier: Rede densa profunda supervisionada com Dropout e BatchNorm.
2. CNNClassifier: Rede convolucional profunda com pooling e blocos Conv2d.
3. ResNetClassifier: Rede convolucional com blocos residuais e skip connections.
4. ConvAutoencoder: Autoencoder convolucional para compressão e geração de embeddings Z.
"""

from typing import Tuple
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Abordagem 1: MLP / DNN Profunda para classificação binária."""
    def __init__(self, input_dim: int = 64 * 64 * 3, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNClassifier(nn.Module):
    """Abordagem 2: Rede Convolucional 2D supervisionada."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Bloco 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Bloco 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Bloco 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


class ResidualBlock(nn.Module):
    """Bloco Residual básico com Skip Connection F(x) + x."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip Connection direta
        return self.relu(out)


class ResNetClassifier(nn.Module):
    """Abordagem 3: Rede ResNet 2D supervisionada com blocos residuais."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # Camada de entrada
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Estágios com blocos residuais
        self.res1 = ResidualBlock(32)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.res3 = ResidualBlock(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.pool(x)
        return self.classifier(x)


class ConvAutoencoder(nn.Module):
    """
    Abordagem 4: Autoencoder Convolucional para compressão não supervisionada.
    Comprime imagens 64x64x3 em um vetor latente compacto Z (k=64 dimensões).
    """
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # Encoder: 64x64x3 -> 8x8x128 -> latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_latent = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Decoder: latent_dim -> 8x8x128 -> 64x64x3
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.Sigmoid(),  # Força os pixels reconstruídos em [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder_conv(x)
        flat = self.flatten(feat)
        return self.fc_latent(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decode(z)
        x = x.view(-1, 128, 8, 8)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
