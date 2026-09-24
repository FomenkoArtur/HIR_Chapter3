"""
Расчёт метрик качества переходного процесса (формулы 15, 16).
"""

import numpy as np
from config import T_OBS, DELTA, N_CONS, IDLE_THRESHOLD


def compute_metrics(eps_history, u_history, t0):
    """
    Вычисление метрик качества переходного процесса.

    Parameters
    ----------
    eps_history : array_like
        История рассогласований
    u_history : array_like
        История управляющих воздействий
    t0 : int
        Момент внесения возмущения

    Returns
    -------
    ts : int
        Время переходного процесса (тактов)
    iae : float
        Интегральная абсолютная ошибка
    idle_rate : float
        Доля тактов с ложным срабатыванием
    """
    # Горизонт наблюдения от момента t0
    eps_obs = eps_history[t0:t0 + T_OBS]
    u_obs = u_history[t0:t0 + T_OBS]

    # IAE (формула 16)
    iae = float(np.abs(eps_obs).sum())

    # Ts (формула 15): первый такт, начиная с которого |eps| <= DELTA
    # на протяжении N_CONS тактов подряд
    ts = T_OBS
    for t in range(len(eps_obs) - N_CONS + 1):
        if np.all(np.abs(eps_obs[t:t + N_CONS]) <= DELTA):
            ts = t
            break

    # Доля тактов с ложным срабатыванием
    idle_rate = float(np.mean(np.abs(u_obs) > IDLE_THRESHOLD))

    return ts, iae, idle_rate