"""
Цифровой двойник технологического процесса и калибровка контрольных карт.
Реализует формулы (3), (5), (6) из текста работы.
"""

import numpy as np
from config import (N_PARAMS, A_BASE, SIG_BASE, N_CALIB, A2, D4, D3,
                    A_CUR, SIG_CUR, C_RAW_FACTOR, G)


class Plant:
    """Цифровой двойник технологического процесса участка металлообработки."""

    def __init__(self, seed=None):
        """
        Инициализация и калибровка контрольных карт.
        """
        if seed is not None:
            np.random.seed(seed)

        self._calibrate_control_charts()
        self._compute_weights()

        self.C_RAW = C_RAW_FACTOR * self.S
        self.G = G

    def _calibrate_control_charts(self):
        """
        Калибровка контрольных карт Шухарта по формулам (5) и (6).
        """
        base = np.zeros((N_CALIB, N_PARAMS))
        for t in range(1, N_CALIB):
            base[t] = A_BASE * base[t-1] + np.random.normal(0, SIG_BASE, N_PARAMS)

        mr_bar = np.abs(np.diff(base, axis=0)).mean(axis=0)
        self.x_bar = base.mean(axis=0)
        self.UCL = self.x_bar + A2 * mr_bar
        self.LCL = self.x_bar - A2 * mr_bar
        self.S = self.UCL - self.LCL

    def _compute_weights(self):
        """
        Расчёт весовых коэффициентов по формуле (11).
        """
        from config import W_OPS, N_GRP

        self.w = np.zeros(N_PARAMS)
        for k in range(len(W_OPS)):
            s = int(N_GRP[:k].sum())
            self.w[s:s+N_GRP[k]] = W_OPS[k] / N_GRP[k]

        assert np.isclose(self.w.sum(), 1.0), f"Сумма весов {self.w.sum()} != 1.0"

    def normalize(self, x_raw):
        """Нормализация по формуле (3)."""
        return (x_raw - self.LCL) / self.S

    def compute_super_criterion(self, x_norm):
        """Критериальная свёртка по формуле (4)."""
        return float(self.w @ x_norm)

    def compute_error(self, x_raw):
        """Расчёт рассогласования по формуле (1)."""
        x_norm = self.normalize(x_raw)
        K = self.compute_super_criterion(x_norm)
        eps = self.G - K
        return eps, K

    def evolve(self, L_prev, rate, u, noise):
        """
        Эволюция состояния объекта на один такт.
        Состояние возвращается к центру калибровки x_bar.
        """
        return self.x_bar + A_CUR * (L_prev - self.x_bar) + rate - self.C_RAW * u + noise


# Глобальный экземпляр
plant = Plant(seed=42)