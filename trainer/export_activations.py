import torch
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from train import IrisNet

def export_activations(model_path="iris_model.pt", out_path="activations.json", n_samples=10):
    model = IrisNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    data = load_iris()
    X, y = data.data.astype(np.float32), data.target

    # Mismo scaler que en train.py (fit sobre todo el dataset para reproducibilidad simple)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Elegimos n_samples indices distribuidos entre las 3 clases
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_scaled), size=n_samples, replace=False)

    samples = []
    with torch.no_grad():
        for idx in indices:
            x = torch.tensor(X_scaled[idx]).unsqueeze(0)  # [1,4]

            # Forward manual capa por capa para capturar activaciones intermedias
            h0 = x.squeeze(0)                       # input (4)
            z1 = model.fc1(x)
            h1 = torch.relu(z1).squeeze(0)          # tras fc1 + ReLU (8)
            z2 = model.fc2(torch.relu(z1))
            h2 = torch.relu(z2).squeeze(0)          # tras fc2 + ReLU (6)
            z3 = model.fc3(torch.relu(z2))
            h3 = z3.squeeze(0)                      # logits finales (3)

            predicted = int(torch.argmax(h3).item())

            samples.append({
                "input": X[idx].tolist(),           # valores originales (sin escalar), para mostrar
                "true_label": int(y[idx]),
                "predicted_label": predicted,
                "activations": [
                    h0.tolist(),
                    h1.tolist(),
                    h2.tolist(),
                    h3.tolist()
                ]
            })

    with open(out_path, "w") as f:
        json.dump({"samples": samples}, f, indent=2)

    print(f"Exportado {len(samples)} samples a {out_path}")

if __name__ == "__main__":
    export_activations()
