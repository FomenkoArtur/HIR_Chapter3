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

    idle_rate = float(np.mean(np.abs(u_obs) > IDLE_THRESHOLD))

    return ts, iae, idle_rate