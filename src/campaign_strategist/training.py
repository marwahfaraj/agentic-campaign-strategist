"""Leakage-safe training helpers for notebooks 03 and 04."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import CAMPAIGN_CLASSES
from .model import JourneyLSTM


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    groups: np.ndarray


@dataclass
class ScaledArrays:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def grouped_train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 7,
) -> SplitData:
    """Split by household so overlapping windows cannot leak across folds."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss.split(x, y, groups=groups))

    relative_val = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=random_state)
    train_rel, val_rel = next(
        gss_val.split(x[train_val_idx], y[train_val_idx], groups=groups[train_val_idx])
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    if set(groups[train_idx]) & set(groups[test_idx]):
        raise RuntimeError("Household leakage between train and test.")
    if set(groups[train_idx]) & set(groups[val_idx]):
        raise RuntimeError("Household leakage between train and validation.")
    if set(groups[val_idx]) & set(groups[test_idx]):
        raise RuntimeError("Household leakage between validation and test.")

    return SplitData(
        train_idx=np.asarray(train_idx),
        val_idx=np.asarray(val_idx),
        test_idx=np.asarray(test_idx),
        groups=groups,
    )


def fit_scaler_on_train(x: np.ndarray, train_idx: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    n_features = x.shape[-1]
    scaler.fit(x[train_idx].reshape(-1, n_features))
    return scaler


def transform_sequences(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n_features = x.shape[-1]
    return scaler.transform(x.reshape(-1, n_features)).reshape(x.shape).astype(np.float32)


def scale_splits(x: np.ndarray, y: np.ndarray, split: SplitData) -> ScaledArrays:
    scaler = fit_scaler_on_train(x, split.train_idx)
    return ScaledArrays(
        x_train=transform_sequences(x[split.train_idx], scaler),
        x_val=transform_sequences(x[split.val_idx], scaler),
        x_test=transform_sequences(x[split.test_idx], scaler),
        y_train=y[split.train_idx],
        y_val=y[split.val_idx],
        y_test=y[split.test_idx],
        scaler=scaler,
    )


def class_weight_tensor(y_train: np.ndarray, n_classes: int = len(CAMPAIGN_CLASSES)) -> torch.Tensor:
    """Mild inverse-frequency weights (sqrt) so majority classes are not ignored."""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / np.sqrt(counts)
    weights = inv * (n_classes / inv.sum())
    return torch.tensor(weights, dtype=torch.float32)


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_size: int = 96,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    epochs: int = 40,
    batch_size: int = 128,
    patience: int = 7,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    random_state: int = 7,
) -> tuple[JourneyLSTM, list[dict[str, float]]]:
    """Train an LSTM with early stopping on validation macro-F1."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = JourneyLSTM(
        n_features=x_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weight_tensor(y_train),
        label_smoothing=label_smoothing,
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    x_val_t = torch.tensor(x_val)
    y_val_t = torch.tensor(y_val)

    history: list[dict[str, float]] = []
    best_state = None
    best_f1 = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = float(loss_fn(val_logits, y_val_t).item())
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
        )

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_labels(model: JourneyLSTM, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).argmax(dim=1).cpu().numpy()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    present = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=present,
            target_names=[CAMPAIGN_CLASSES[i] for i in present],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CAMPAIGN_CLASSES)))
        ).tolist(),
    }
