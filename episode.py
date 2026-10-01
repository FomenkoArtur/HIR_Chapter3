"""
Единый прогон эпизода: объект, политика управления, история состояний.
"""

from dataclasses import dataclass

import numpy as np
import torch

from config import G, KD, KI, KP, N_PARAMS, N_ZONES, PID_WINDOW, SIG_CUR, T_START, T_TOTAL, WINDOW
from plant import plant
from scenarios import get_disturbance_profile


def build_features(eps_window, n_window):
    """
    Входной тензор регулятора (N_FEATURES x WINDOW): окно рассогласования,
    окна нормализованных критериев N_i и окна их весовых вкладов w_i * N_i.
    """
    eps = np.asarray(eps_window, dtype=np.float32)
    n = np.asarray(n_window, dtype=np.float32)
    return np.vstack([eps[None, :], n.T, (n * plant.w.astype(np.float32)).T])


@dataclass
class Episode:
    X: np.ndarray
    N: np.ndarray
    K: np.ndarray
    eps: np.ndarray
    u: np.ndarray


class NullPolicy:
    def reset(self):
        pass

    def act(self, eps, x_norm):
        return np.zeros(N_ZONES)


class PIDPolicy:
    def reset(self):
        self.e_prev = np.zeros(N_ZONES)
        self.window = []

    def act(self, eps, x_norm):
        e = plant.zone_criteria(x_norm) - G
        self.window.append(e)
        if len(self.window) > PID_WINDOW:
            self.window.pop(0)
        u = KP * e + KD * (e - self.e_prev) + KI * np.sum(self.window, axis=0)
        self.e_prev = e
        return np.clip(u, -1.0, 1.0)


class CNNPolicy:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def reset(self):
        self.eps = []
        self.n = []

    def act(self, eps, x_norm):
        self.eps.append(eps)
        self.n.append(x_norm)
        if len(self.eps) > WINDOW:
            self.eps.pop(0)
            self.n.pop(0)
        if len(self.eps) < WINDOW:
            return np.zeros(N_ZONES)
        features = torch.from_numpy(build_features(self.eps, self.n)).unsqueeze(0)
        with torch.no_grad():
            return self.model(features).squeeze(0).numpy().astype(float)


def run_episode(scenario, policy, seed=None, scale=1.0, t0=T_START, profile=None):
    if seed is not None:
        np.random.seed(seed)

    rate, jump = profile if profile is not None else get_disturbance_profile(scenario, t0, scale)
    state = plant.x_bar.copy()
    policy.reset()
    u = np.zeros(N_ZONES)

    X, N, K, E, U = [], [], [], [], []
    for t in range(T_TOTAL):
        if t == t0:
            state = state + jump
        if t > 0:
            noise = np.random.normal(0, SIG_CUR, N_PARAMS)
            state = plant.evolve(state, rate[t], u, noise)

        x_norm = plant.normalize(state)
        k = plant.compute_super_criterion(x_norm)
        eps = plant.G - k
        u = policy.act(eps, x_norm)

        X.append(state.copy())
        N.append(x_norm)
        K.append(k)
        E.append(eps)
        U.append(u)

    return Episode(np.array(X), np.array(N), np.array(K), np.array(E), np.array(U))
