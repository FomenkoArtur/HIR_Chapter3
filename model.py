"""
Архитектура свёрточной нейронной сети регулятора (п. 2.2.3).

По сравнению с исходным вариантом:
  - больше каналов (32/64) и глубина (3 свёрточных слоя);
  - BatchNorm1d после каждой свёртки (стабилизирует обучение);
  - AdaptiveAvgPool1d(8) вместо двух MaxPool1d (не теряем временное разрешение
    и получаем независимость от WINDOW);
  - fc1 увеличен с 32 до 128 нейронов.
Это нужно, чтобы сеть могла различить, какой из 8 показателей вышел за
границы, и выдать u только на соответствующую зону (адресное воздействие).
"""

import torch.nn as nn
from config import N_FEATURES, N_ZONES, WINDOW


class CNNController(nn.Module):
    def __init__(self, window_size=WINDOW, in_channels=N_FEATURES, n_outputs=N_ZONES):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.c2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.c3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(64 * 8, 128)
        self.fc2 = nn.Linear(128, n_outputs)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn1(self.c1(x)))
        x = self.relu(self.bn2(self.c2(x)))
        x = self.relu(self.bn3(self.c3(x)))
        x = self.pool(x).flatten(1)
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x))