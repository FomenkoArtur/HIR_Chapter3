"""
Генерация выборок и обучение CNN.
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from config import (WINDOW, T_TOTAL, BATCH_SIZE, MAX_EPOCHS, PATIENCE, VAL_SPLIT,
                    LR, SEED_TEACHER_START, SEED_CRITICAL_START, N_PARAMS, SCENARIOS,
                    CRITICAL_SCENARIOS)
from episode import PIDPolicy, build_features, run_episode
from scenarios import critical_profile, custom_profile, get_disturbance_profile
from model import CNNController


def weighted_mse(pred, target):
    err = (pred - target) ** 2
    active = (target.abs().max(dim=1, keepdim=True).values > 0.1).float()
    w = 1.0 + 5.0 * active
    return (err * w).mean()


def training_profiles():
    trend = SCENARIOS['S1']['rate_value']
    jump = abs(SCENARIOS['S2']['jump_value'])

    profiles = [('S4', None)] * 4
    for param in range(N_PARAMS):
        for kind, value in (('trend', trend), ('trend', -trend),
                            ('jump', jump), ('jump', -jump)):
            profiles.append((kind, (param, kind, value)))

    for p1, p2 in [(0, 4), (1, 5), (2, 7), (3, 6)]:
        for sign in (1, -1):
            profiles.append(('jump2', ((p1, p2), 'jump2', sign * jump * 0.8)))
    return profiles


def window_pairs(episode):
    Xs, Ys = [], []
    for t in range(WINDOW - 1, T_TOTAL):
        window = slice(t - WINDOW + 1, t + 1)
        Xs.append(build_features(episode.eps[window], episode.N[window]))
        Ys.append(episode.u[t])
    return Xs, Ys


def generate_train_set():
    print("Формирование обучающей выборки...")
    Xs, Ys = [], []

    for ep, (name, spec) in enumerate(training_profiles(), start=1):
        t0 = np.random.randint(WINDOW + 5, WINDOW + 20)
        scale = np.random.uniform(0.7, 1.3)

        if spec is None:
            profile = get_disturbance_profile('S4', t0, scale)
        else:
            param, kind, value = spec
            if kind == 'jump2':
                p1, p2 = param
                rate1, jump1 = custom_profile(p1, 'jump', value * scale, t0)
                rate2, jump2 = custom_profile(p2, 'jump', value * scale, t0)
                profile = (rate1 + rate2, jump1 + jump2)
            else:
                profile = custom_profile(param, kind, value * scale, t0)

        episode = run_episode(name, PIDPolicy(), seed=SEED_TEACHER_START + ep,
                              t0=t0, profile=profile)
        x, y = window_pairs(episode)
        Xs += x
        Ys += y

    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    Y = torch.tensor(np.array(Ys), dtype=torch.float32)
    return X, Y


def generate_critical_set():
    print("Формирование тестовой выборки...")
    Xs, Ys, groups = [], [], []

    for ep, (name, spec) in enumerate(CRITICAL_SCENARIOS.items(), start=1):
        episode = run_episode(name, PIDPolicy(), seed=SEED_CRITICAL_START + ep, t0=spec['t0'],
                              profile=critical_profile(name),
                              noise_scale=spec.get('noise_scale', 1.0))
        x, y = window_pairs(episode)
        Xs += x
        Ys += y
        groups += [name] * len(x)

    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    Y = torch.tensor(np.array(Ys), dtype=torch.float32)
    return X, Y, np.array(groups)


def check_sets(X_train, Y_train, X_test, Y_test):
    train_seeds = set(SEED_TEACHER_START + i for i in range(1, len(training_profiles()) + 1))
    test_seeds = set(SEED_CRITICAL_START + i for i in range(1, len(CRITICAL_SCENARIOS) + 1))
    assert not (train_seeds & test_seeds), "Зёрна пересекаются"

    train_keys = {row.tobytes() for row in X_train.numpy()}
    duplicates = sum(row.tobytes() in train_keys for row in X_test.numpy())
    assert duplicates == 0, f"Пересечений окон: {duplicates}"

    eps_train = X_train[:, 0, :].abs().max(dim=1).values
    eps_test = X_test[:, 0, :].abs().max(dim=1).values
    eps_max_train = eps_train.max().item()
    outside = (eps_test > eps_max_train).float().mean().item()
    sat_train = (Y_train.abs().max(dim=1).values > 0.99).float().mean().item()
    sat_test = (Y_test.abs().max(dim=1).values > 0.99).float().mean().item()

    n_train, n_test = len(X_train), len(X_test)
    print(f"Выборки: обучение {n_train} / тест {n_test}")
    print(f"  max|eps|: обучение {eps_max_train:.3f}, тест {eps_test.max().item():.3f}; вне обучения: {outside:.1%}")
    print(f"  насыщение (|u| > 0.99): обучение {sat_train:.1%}, тест {sat_test:.1%}")


def build_datasets():
    X_train, Y_train = generate_train_set()
    X_test, Y_test, groups = generate_critical_set()
    check_sets(X_train, Y_train, X_test, Y_test)
    return X_train, Y_train, X_test, Y_test, groups


def generate_training_data():
    X_train, Y_train, X_test, Y_test, _ = build_datasets()
    return X_train, Y_train, X_test, Y_test


def mse_by_scenario(model, X_test, Y_test, groups):
    model.eval()
    with torch.no_grad():
        err = ((model(X_test) - Y_test) ** 2).mean(dim=1).numpy()
    return {name: float(err[groups == name].mean()) for name in CRITICAL_SCENARIOS}


def weights_fingerprint(model):
    return sum(p.detach().abs().sum().item() for p in model.parameters())


def train_model():
    X_train, Y_train, X_val, Y_val, groups = build_datasets()

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True
    )

    model = CNNController()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    print("Обучение CNN...")

    best_val_loss = float('inf')
    bad_epochs = 0
    best_state = None
    first_by_scenario = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = weighted_mse(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(X_train)

        before = weights_fingerprint(model)
        model.eval()
        with torch.no_grad():
            val_loss = weighted_mse(model(X_val), Y_val).item()
        assert weights_fingerprint(model) == before
        scheduler.step(val_loss)

        if epoch == 0:
            first_by_scenario = mse_by_scenario(model, X_val, Y_val, groups)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  эпоха {epoch + 1}: train = {train_loss:.6f}, val = {val_loss:.6f}, lr = {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad_epochs = 0
            best_state = deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Ранний останов на эпохе {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_by_scenario = mse_by_scenario(model, X_val, Y_val, groups)
    print("\nMSE на критических ситуациях:")
    for name, spec in CRITICAL_SCENARIOS.items():
        first, final = first_by_scenario[name], final_by_scenario[name]
        print(f"  {name}: {first:.6f} -> {final:.6f} ({final / first - 1:+.0%})")

    print("Обучение завершено.")
    return model