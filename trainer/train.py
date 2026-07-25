import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

torch.manual_seed(42)

class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 6)
        self.fc3 = nn.Linear(6, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
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
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(X_test_t)
                acc = (test_out.argmax(1) == y_test_t).float().mean().item()
            print(f"Epoch {epoch+1:3d} | loss: {loss.item():.4f} | test_acc: {acc:.4f}")

    torch.save(model.state_dict(), "iris_model.pt")
    print("Modelo guardado en iris_model.pt")
    return model

if __name__ == "__main__":
    main()