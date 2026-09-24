"""
Сценарии особых причин изменчивости (п. 3.2.1).
"""

import numpy as np
from config import N_PARAMS, T_TOTAL, SCENARIOS
from plant import plant


def get_disturbance_profile(scenario, t0=20, scale=1.0):
    """
    Формирование профиля возмущений для заданного сценария.
    """
    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)

    # S0 - контрольный сценарий без особой причины
    if scenario == 'S0':
        return rate * plant.S, jump * plant.S

    config = SCENARIOS[scenario]

    # Тренд (rate)
    if config['rate_param'] is not None:
        rate[t0:, config['rate_param']] = config['rate_value'] * scale

    # Скачок (jump)
    if config['jump_param'] is not None:
        jump[config['jump_param']] = config['jump_value'] * scale

    # Перевод в сырые единицы через диапазон нормализации
    return rate * plant.S, jump * plant.S


def get_scenario_description(scenario):
    """Получение текстового описания сценария."""
    if scenario == 'S0':
        return 'Без особой причины (контрольный сценарий)'
    return SCENARIOS[scenario]['description']