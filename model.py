"""
Архитектура свёрточной нейронной сети регулятора (п. 2.2.3).
"""

import torch.nn as nn
from config import WINDOW


class CNNController(nn.Module):
    """
    Свёрточная нейронная сеть для регулирования технологического процесса.
    """

    def __init__(self, window_size=WINDOW):
        super().__init__()

        self.c1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.p1 = nn.MaxPool1d(2)

        self.c2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(32 * (window_size // 4), 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.p1(self.relu(self.c1(x)))
        x = self.p2(self.relu(self.c2(x)))
        x = self.relu(self.fc1(x.flatten(1)))
        u = self.tanh(self.fc2(x))
        return u