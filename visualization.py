"""
Визуализация результатов эксперимента.
"""

import matplotlib.pyplot as plt
from config import G, DELTA, T_START, SEED
from experiment import run_experiment_episode


def plot_transition_process(model, scenario='S1'):
    """
    Построение графика переходного процесса (Рисунок 5).

    Parameters
    ----------
    model : CNNController
        Обученная модель
    scenario : str, optional
        Сценарий для визуализации (по умолчанию 'S1')
    """
    # Прогоны для построения графика
    K_no_ctrl, _, _ = run_experiment_episode(scenario, 'none', seed=SEED)
    K_ctrl, _, _ = run_experiment_episode(scenario, 'cnn', seed=SEED, model=model)

    # Построение графика
    plt.figure(figsize=(10, 6))

    plt.plot(K_no_ctrl, color='red', alpha=0.8, label='Без регулятора')
    plt.plot(K_ctrl, color='blue', lw=2, label='С регулятором (CNN)')

    # Программное задание
    plt.axhline(G, color='k', alpha=0.3, label=f'Программное задание g = {G}')

    # Допустимая полоса
    plt.axhline(G + DELTA, color='green', ls='--', alpha=0.6, label=f'Допустимая полоса +/- {DELTA}')
    plt.axhline(G - DELTA, color='green', ls='--', alpha=0.6)

    # Момент внесения возмущения
    plt.axvline(T_START, color='gray', ls=':', label='Момент возникновения особой причины')

    # Оформление
    plt.title(f'Рисунок 5 - Переходный процесс суперкритерия K(t) (сценарий {scenario})')
    plt.xlabel('Такт моделирования (t)')
    plt.ylabel('Суперкритерий K(t)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()