from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm.auto import trange

from classifier_model import TimeDependentMNISTClassifier


def load_mnist_classifier_data(batch_size=128, train=True, num_workers=0):
    data = datasets.MNIST(
        root="data",
        train=train,
        download=True,
        transform=ToTensor(),
    )

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
    )
    return loader, data


def evaluate_classifier_accuracy(model, data_loader, device="cpu"):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            t = torch.zeros(x.shape[0], device=device, dtype=x.dtype)

            logits = model(x, t)
            loss = criterion(logits, y)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * x.shape[0]
            correct += (preds == y).sum().item()
            total += x.shape[0]

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def train_mnist_classifier(
    checkpoint_path="checkpoints/classifier/classifier_checkpoint.pth",
    batch_size=128,
    n_epochs=10,
    learning_rate=1e-3,
    device="cpu",
):
    checkpoint_path = Path(checkpoint_path)

    train_loader, _ = load_mnist_classifier_data(batch_size=batch_size, train=True)
    test_loader, _ = load_mnist_classifier_data(batch_size=batch_size, train=False)

    model = TimeDependentMNISTClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    epoch_bar = trange(n_epochs)
    for _ in epoch_bar:
        model.train()

        total = 0
        total_loss = 0.0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            t = torch.zeros(x.shape[0], device=device, dtype=x.dtype)

            logits = model(x, t)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * x.shape[0]
            correct += (preds == y).sum().item()
            total += x.shape[0]

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate_classifier_accuracy(
            model=model,
            data_loader=test_loader,
            device=device,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        epoch_bar.set_description(
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Mejor accuracy de validación: {best_acc:.4f}")
    print(f"Checkpoint guardado en: {checkpoint_path}")

    return model


def load_classifier_from_checkpoint(
    checkpoint_path="checkpoints/classifier/classifier_checkpoint.pth",
    device="cpu",
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")

    model = TimeDependentMNISTClassifier().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Checkpoint cargado desde: {checkpoint_path}")
    return model


def get_or_train_mnist_classifier(
    checkpoint_path="checkpoints/classifier/classifier_checkpoint.pth",
    batch_size=128,
    n_epochs=10,
    learning_rate=1e-3,
    device="cpu",
    force_retrain=False,
):
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists() and not force_retrain:
        model = TimeDependentMNISTClassifier().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        print(f"Checkpoint encontrado. Cargando clasificador desde: {checkpoint_path}")
        return model

    print("Checkpoint no encontrado o force_retrain=True. Entrenando clasificador...")
    return train_mnist_classifier(
        checkpoint_path=str(checkpoint_path),
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device,
    )