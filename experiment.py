"""
Проведение вычислительного эксперимента (п. 3.3.1).
"""

import numpy as np
import pandas as pd
import torch

from config import (T_TOTAL, T_START, N_REAL, SCENARIOS,
                    SEED_EXPERIMENT_START, SEED_FALSE_ALARM_START, SIG_CUR, N_PARAMS)
from plant import plant
from scenarios import get_disturbance_profile, get_scenario_description
from metrics import compute_metrics


def run_experiment_episode(scenario, mode, seed=None, model=None):
    """
    Прогон одного эпизода эксперимента.
    """
    if seed is not None:
        np.random.seed(seed)

    rate, jump = get_disturbance_profile(scenario, T_START, scale=1.0)

    # Начальное состояние
    L = plant.x_bar.copy()

    K_history = []
    eps_history = []
    u_history = []

    buf_e = []
    u = 0.0

    for t in range(T_TOTAL):
        # Внесение скачка
        if t == T_START and scenario != 'S0':
            L = L + jump

        # Эволюция состояния
        if t > 0:
            # ИСПРАВЛЕНО: использование калиброванного уровня шума SIG_CUR
            noise = np.random.normal(0, SIG_CUR, N_PARAMS)
            L = plant.evolve(L, rate[t], u, noise)

        # Расчёт рассогласования
        eps, K = plant.compute_error(L)

        # Скользящее окно
        buf_e.append(eps)
        if len(buf_e) > 10:  # WINDOW = 10
            buf_e.pop(0)

        # Формирование управления
        if mode == 'cnn' and len(buf_e) == 10 and model is not None:
            model.eval()
            with torch.no_grad():
                # ИСПРАВЛЕНО: подаём окно eps напрямую, без переворота знака
                xt = torch.tensor([buf_e], dtype=torch.float32).view(1, 1, 10)
                u = float(model(xt).item())
        else:
            u = 0.0

        K_history.append(K)
        eps_history.append(eps)
        u_history.append(u)

    return np.array(K_history), np.array(eps_history), np.array(u_history)


def run_full_experiment(model):
    """
    Проведение полного вычислительного эксперимента.
    """
    print("Вычислительный эксперимент: 30 реализаций на сценарий...")

    results = []

    # Эксперименты для S1, S2, S3
    for scenario in ['S1', 'S2', 'S3']:
        ts_list = []
        iae_no_ctrl_list = []
        iae_ctrl_list = []

        for rep in range(N_REAL):
            # ИСПРАВЛЕНО: детерминированный расчёт сида для 100% воспроизводимости результатов
            seed = SEED_EXPERIMENT_START + 1000 * {'S1': 1, 'S2': 2, 'S3': 3}[scenario] + rep

            # Без регулятора
            _, eps_no_ctrl, _ = run_experiment_episode(scenario, 'none', seed=seed)
            _, iae_no_ctrl, _ = compute_metrics(eps_no_ctrl, np.zeros(T_TOTAL), T_START)

            # С регулятором
            _, eps_ctrl, u_ctrl = run_experiment_episode(scenario, 'cnn', seed=seed, model=model)
            ts_ctrl, iae_ctrl, _ = compute_metrics(eps_ctrl, u_ctrl, T_START)

            ts_list.append(ts_ctrl)
            iae_no_ctrl_list.append(iae_no_ctrl)
            iae_ctrl_list.append(iae_ctrl)

        # Средние значения
        mean_ts = round(np.mean(ts_list))
        mean_iae_no = round(np.mean(iae_no_ctrl_list), 1)
        mean_iae_ctrl = round(np.mean(iae_ctrl_list), 1)
        reduction = round(100 * (1 - mean_iae_ctrl / mean_iae_no))

        description = get_scenario_description(scenario)
        results.append([scenario, description, mean_ts, mean_iae_no, mean_iae_ctrl, reduction])

    # Оценка ложных срабатываний в S0
    false_alarm_rates = []
    for rep in range(N_REAL):
        seed = SEED_FALSE_ALARM_START + rep
        _, eps_s0, u_s0 = run_experiment_episode('S0', 'cnn', seed=seed, model=model)
        _, _, idle_rate = compute_metrics(eps_s0, u_s0, T_START)
        false_alarm_rates.append(idle_rate)

    mean_false_alarm = 100 * np.mean(false_alarm_rates)

    # Формирование DataFrame
    df = pd.DataFrame(
        results,
        columns=['Сценарий', 'Особая причина изменчивости', 'Ts, тактов (с регулятором)',
                 'IAE без регулятора', 'IAE с регулятором', 'Снижение IAE, %']
    )

    # Добавление строки со средними значениями
    means = df.iloc[:, 2:].mean().round(1)
    df.loc[len(df)] = ['Среднее', '-', means.iloc[0], means.iloc[1], means.iloc[2], int(means.iloc[3])]

    return df, mean_false_alarm