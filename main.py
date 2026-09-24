"""
Главный скрипт для проведения вычислительного эксперимента.
Апробация контура автоматического управления (Глава 3).
"""

import numpy as np
import torch

from config import SEED
from trainer import train_model
from experiment import run_full_experiment
from visualization import plot_transition_process


def main():
    """Основная функция эксперимента."""
    print("=" * 70)
    print("АПРОБАЦИЯ КОНТУРА АВТОМАТИЧЕСКОГО УПРАВЛЕНИЯ")
    print("=" * 70)

    # Установка сидов для воспроизводимости
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Обучение модели
    model = train_model()

    # Проведение эксперимента
    results_df, false_alarm_rate = run_full_experiment(model)

    # Вывод результатов
    print("\n" + "=" * 70)
    print("Таблица 1 - Результаты апробации (среднее по 30 реализациям)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nS0: доля тактов с ложным срабатыванием регулятора: {false_alarm_rate:.2f}%")

    # Построение графика
    print("\nПостроение графика переходного процесса...")
    plot_transition_process(model, scenario='S1')

    print("\nЭксперимент завершён успешно!")


if __name__ == '__main__':
    main()