import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from train import IrisNet

def main():
    torch.manual_seed(42)

    data = load_iris()
    X, y = data.data.astype(np.float32), data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 300
    history_epochs = []
    history_loss = []
    history_acc = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_out = model(X_test_t)
            acc = (test_out.argmax(1) == y_test_t).float().mean().item()

        history_epochs.append(epoch + 1)
        history_loss.append(loss.item())
        history_acc.append(acc)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | loss: {loss.item():.4f} | test_acc: {acc:.4f}")

    torch.save(model.state_dict(), "iris_model.pt")

    with open("training_history.json", "w") as f:
        json.dump({
            "epochs": history_epochs,
            "train_loss": history_loss,
            "test_accuracy": history_acc
        }, f)

    print("Historial guardado en training_history.json")

if __name__ == "__main__":
    main()
