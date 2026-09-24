"""
Архитектура свёрточной нейронной сети регулятора (п. 2.2.3).
"""

import torch.nn as nn
from config import WINDOW


class CNNController(nn.Module):
    """
    Свёрточная нейронная сеть для регулирования технологического процесса.

    Архитектура (п. 2.2.3):
    - Два свёрточных слоя (16 и 32 фильтра длиной 5 и 3 отсчётов)
    - ReLU активация и субдискретизация между слоями
    - Один полносвязный слой из 32 нейронов
    - Выходной нейрон с активацией tanh
    """

    def __init__(self, window_size=WINDOW):
        super().__init__()

        # Свёрточный слой 1: 16 фильтров длиной 5
        self.c1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.p1 = nn.MaxPool1d(2)

        # Свёрточный слой 2: 32 фильтра длиной 3
        self.c2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool1d(2)

        # Полносвязный слой: 32 нейрона
        # Размерность: 32 * (window_size // 4) после двух MaxPool
        self.fc1 = nn.Linear(32 * (window_size // 4), 32)

        # Выходной слой: 1 нейрон
        self.fc2 = nn.Linear(32, 1)

        # Активации
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Прямой проход сети.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, window_size)
            Входное окно рассогласования

        Returns
        -------
        u : torch.Tensor, shape (batch, 1)
            Управляющее воздействие в диапазоне [-1, 1]
        """
        # Свёртки с субдискретизацией
        x = self.p1(self.relu(self.c1(x)))
        x = self.p2(self.relu(self.c2(x)))

        # Полносвязные слои
        x = self.relu(self.fc1(x.flatten(1)))
        u = self.tanh(self.fc2(x))

        return u