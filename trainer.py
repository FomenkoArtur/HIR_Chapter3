"""
Генерация обучающей выборки и обучение CNN.
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from config import (WINDOW, T_TOTAL, KP, KD, KI, BATCH_SIZE, MAX_EPOCHS,
                    PATIENCE, VAL_SPLIT, LR, SEED_TEACHER_START, SIG_CUR, N_PARAMS)
from plant import plant
from scenarios import get_disturbance_profile
from model import CNNController


def run_episode_with_teacher(scenario, seed=None, t0=20, scale=1.0):
    """
    Прогон эпизода с ПИД-учителем для генерации обучающих данных.
    """
    if seed is not None:
        np.random.seed(seed)

    rate, jump = get_disturbance_profile(scenario, t0, scale)

    # Начальное состояние в центре поля допуска
    L = plant.x_bar.copy()

    errors = []
    controls = []
    buf_e = []
    e_prev = 0.0
    u = 0.0

    for t in range(T_TOTAL):
        # Внесение скачка в момент t0
        if t == t0 and scenario != 'S0':
            L = L + jump

        # Эволюция состояния
        if t > 0:
            # ИСПРАВЛЕНО: использование калиброванного уровня шума SIG_CUR
            noise = np.random.normal(0, SIG_CUR, N_PARAMS)
            L = plant.evolve(L, rate[t], u, noise)

        # Расчёт рассогласования
        eps, K = plant.compute_error(L)

        # ИСПРАВЛЕНО: учитель работает по отрицательной обратной связи через e = K - G
        e = K - plant.G

        # Скользящее окно (сохраняем eps для формирования входных данных сети)
        buf_e.append(eps)
        if len(buf_e) > WINDOW:
            buf_e.pop(0)

        # ПИД-регулятор (учитель)
        # sum(buf_e) содержит eps, поэтому для суммы e используем -sum(buf_e)
        u = float(np.clip(KP*e + KD*(e - e_prev) - KI*sum(buf_e), -1, 1))
        e_prev = e

        errors.append(eps)
        controls.append(u)

    return errors, controls


def generate_training_data():
    """
    Генерация обучающей выборки пар "окно eps - целевое u".
    """
    print("Формирование выборки (учитель на том же объекте)...")

    Xs, Ys = [], []
    ep = 0

    # Прогоны для всех сценариев (включая S0)
    for scenario in ['S0', 'S1', 'S2', 'S3']:
        for rep in range(3):  # 3 реализации на сценарий
            ep += 1
            errors, controls = run_episode_with_teacher(
                scenario,
                seed=SEED_TEACHER_START + ep,
                t0=np.random.randint(15, 30),
                scale=np.random.uniform(0.7, 1.3)
            )

            # Формирование пар "окно - целевое u"
            for t in range(WINDOW, T_TOTAL):
                Xs.append(errors[t-WINDOW:t])
                Ys.append(controls[t])

    # Преобразование в тензоры
    X = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(np.array(Ys), dtype=torch.float32).unsqueeze(1)

    # Разделение на обучающую и проверочную выборки
    n_total = len(X)
    n_train = int((1 - VAL_SPLIT) * n_total)
    idx = torch.randperm(n_total)

    X_train = X[idx[:n_train]]
    Y_train = Y[idx[:n_train]]
    X_val = X[idx[n_train:]]
    Y_val = Y[idx[n_train:]]

    print(f"Объём выборки: {n_total} пар (обучение {n_train} / проверка {n_total - n_train})")

    return X_train, Y_train, X_val, Y_val


def train_model():
    """
    Обучение CNN с ранним остановом.
    """
    # Генерация данных
    X_train, Y_train, X_val, Y_val = generate_training_data()

    # DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Инициализация модели
    model = CNNController()
    optimizer = optim.Adam(model.parameters(), lr=LR) if LR else optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # Обучение с ранним остановом
    print(f"Обучение CNN: Adam (lr по умолчанию), MSE, ранний останов patience={PATIENCE}...")

    best_val_loss = float('inf')
    bad_epochs = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        # Обучение на батчах
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Проверка на валидации
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), Y_val).item()

        # Ранний останов
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad_epochs = 0
            best_state = deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Ранний останов на эпохе {epoch+1}, val MSE = {best_val_loss:.6f}")
                break

    # Восстановление лучшей модели
    if best_state is not None:
        model.load_state_dict(best_state)

    print("Обучение завершено.")
    return model