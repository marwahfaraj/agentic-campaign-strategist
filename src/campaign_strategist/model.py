from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import CAMPAIGN_CLASSES


class JourneyLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 48, n_classes: int = len(CAMPAIGN_CLASSES)):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


@dataclass
class TrainingResult:
    model: JourneyLSTM
    metrics: dict[str, object]


def train_journey_model(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 6,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_size: int = 48,
    random_state: int = 7,
) -> TrainingResult:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = JourneyLSTM(n_features=x.shape[-1], hidden_size=hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
        predictions = logits.argmax(dim=1).cpu().numpy()

    present_labels = sorted(np.unique(np.concatenate([y_test, predictions])).tolist())
    report = classification_report(
        y_test,
        predictions,
        labels=present_labels,
        target_names=[CAMPAIGN_CLASSES[idx] for idx in present_labels],
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "history": history,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=list(range(len(CAMPAIGN_CLASSES)))).tolist(),
        "y_test": y_test.tolist(),
        "y_pred": predictions.tolist(),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    return TrainingResult(model=model, metrics=metrics)


def predict_probabilities(model: JourneyLSTM, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities


def save_model(model: JourneyLSTM, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str, n_features: int) -> JourneyLSTM:
    model = JourneyLSTM(n_features=n_features)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
