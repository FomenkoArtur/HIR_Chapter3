"""
Генерация обучающей выборки и обучение CNN.
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from config import (WINDOW, T_TOTAL, BATCH_SIZE, MAX_EPOCHS, PATIENCE, VAL_SPLIT,
                    LR, SEED_TEACHER_START, N_PARAMS, SCENARIOS)
from episode import PIDPolicy, build_features, run_episode
from scenarios import custom_profile, get_disturbance_profile
from model import CNNController


def training_profiles():
    """
    Профили возмущений обучающей выборки: тренды и скачки обоих знаков
    на каждый из показателей, плюс прогоны без особой причины.
    """
    trend = SCENARIOS['S1']['rate_value']
    jump = abs(SCENARIOS['S2']['jump_value'])

    profiles = [('S4', None)] * 4
    for param in range(N_PARAMS):
        for kind, value in (('trend', trend), ('trend', -trend), ('jump', jump), ('jump', -jump)):
            profiles.append((kind, (param, kind, value)))
    return profiles


def generate_training_data():
    """
    Генерация обучающей выборки пар "окно признаков - целевое воздействие по зонам".
    """
    print("Формирование выборки (учитель на том же объекте)...")

    Xs, Ys = [], []

    for ep, (name, spec) in enumerate(training_profiles(), start=1):
        t0 = np.random.randint(15, 30)
        scale = np.random.uniform(0.7, 1.3)

        if spec is None:
            profile = get_disturbance_profile('S4', t0, scale)
        else:
            param, kind, value = spec
            profile = custom_profile(param, kind, value * scale, t0)

        episode = run_episode(name, PIDPolicy(), seed=SEED_TEACHER_START + ep, t0=t0, profile=profile)

        for t in range(WINDOW - 1, T_TOTAL):
            window = slice(t - WINDOW + 1, t + 1)
            Xs.append(build_features(episode.eps[window], episode.N[window]))
            Ys.append(episode.u[t])

    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    Y = torch.tensor(np.array(Ys), dtype=torch.float32)

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
    X_train, Y_train, X_val, Y_val = generate_training_data()

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CNNController()
    optimizer = optim.Adam(model.parameters(), lr=LR) if LR else optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    print(f"Обучение CNN: Adam (lr по умолчанию), MSE, ранний останов patience={PATIENCE}...")

    best_val_loss = float('inf')
    bad_epochs = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), Y_val).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad_epochs = 0
            best_state = deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Ранний останов на эпохе {epoch+1}, val MSE = {best_val_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Обучение завершено.")
    return model