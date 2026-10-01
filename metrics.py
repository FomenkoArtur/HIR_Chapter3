"""
Расчёт метрик качества переходного процесса (формулы 15, 16).
"""

import numpy as np
from config import T_OBS, DELTA, N_CONS, IDLE_THRESHOLD


def compute_metrics(eps_history, u_history, t0):
    """
    Вычисление метрик качества переходного процесса.
    """
    eps_obs = eps_history[t0:t0+T_OBS]
    u_obs = u_history[t0:t0+T_OBS]

    iae = float(np.abs(eps_obs).sum())

    ts = T_OBS
    for t in range(len(eps_obs) - N_CONS + 1):
        if np.all(np.abs(eps_obs[t:t+N_CONS]) <= DELTA):
            ts = t
            break

    u_abs = np.abs(u_obs)
    if u_abs.ndim > 1:
        u_abs = u_abs.max(axis=1)
    idle_rate = float(np.mean(u_abs > IDLE_THRESHOLD))

    return ts, iae, idle_rate


def out_of_control_rate(n_history, t0):
    """
    Доля тактов горизонта наблюдения, в которых хотя бы один
    нормализованный показатель N_i вышел за контрольные границы [0, 1].
    """
    n_obs = n_history[t0:t0+T_OBS]
    outside = (n_obs < 0.0) | (n_obs > 1.0)
    return float(np.mean(outside.any(axis=1)))