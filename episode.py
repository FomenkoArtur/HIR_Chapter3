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
    eps = np.asarray(eps_window, dtype=np.float32)
    n = np.asarray(n_window, dtype=np.float32)
    return np.vstack([
        eps[None, :] / 0.5,
        ((n - 0.5) * 2.0).T,
        (n * plant.w.astype(np.float32)).T * 4.0,
    ])


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

        i_term = np.clip(KI * np.sum(self.window, axis=0), -0.4, 0.4)
        u_raw = KP * e + KD * (e - self.e_prev) + i_term
        u = np.clip(u_raw, -1.0, 1.0)

        saturated_high = (u_raw > 1.0) & (e > 0)
        saturated_low = (u_raw < -1.0) & (e < 0)
        if np.any(saturated_high) or np.any(saturated_low):
            self.window[-1] = np.where(saturated_high | saturated_low, 0.0, self.window[-1])

        self.e_prev = e
        return u


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
            u = self.model(features).squeeze(0).numpy().astype(float)

        return u


def run_episode(scenario, policy, seed=None, scale=1.0, t0=T_START, profile=None, noise_scale=1.0):
    if seed is not None:
        np.random.seed(seed)

    rate, jump = profile if profile is not None else get_disturbance_profile(scenario, t0, scale)
    state = plant.x_bar.copy()
    normalizer = plant.make_normalizer()
    policy.reset()
    u = np.zeros(N_ZONES)

    X, N, K, E, U = [], [], [], [], []
    for t in range(T_TOTAL):
        if t == t0:
            state = state + jump
        if t > 0:
            noise = np.random.normal(0, SIG_CUR * noise_scale, N_PARAMS)
            state = plant.evolve(state, rate[t], u, noise)

        x_norm = normalizer.update(state)
        k = plant.compute_super_criterion(x_norm)
        eps = plant.G - k
        u = policy.act(eps, x_norm)

        X.append(state.copy())
        N.append(x_norm)
        K.append(k)
        E.append(eps)
        U.append(u)

    return Episode(np.array(X), np.array(N), np.array(K), np.array(E), np.array(U))