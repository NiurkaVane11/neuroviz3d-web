import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- Ajustá esto para que coincida EXACTO con la clase de train.py ---
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 6)
        self.fc3 = nn.Linear(6, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
# ----------------------------------------------------------------------

model = IrisNet()
model.load_state_dict(torch.load("iris_model.pt", map_location="cpu"))
model.eval()

iris = load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = iris.target

linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
hidden_layer = linear_layers[-2]

captured = {}
def hook(module, inp, out):
    captured["hidden"] = torch.relu(out).detach()

h = hidden_layer.register_forward_hook(hook)
with torch.no_grad():
    logits = model(X)
h.remove()

hidden6 = captured["hidden"].numpy()
predicted = logits.argmax(dim=1).numpy()

def normalize_to_scene(coords, target_extent=3.5):
    coords = coords - coords.mean(axis=0)
    max_abs = np.abs(coords).max()
    if max_abs < 1e-8:
        max_abs = 1.0
    return coords * (target_extent / max_abs)

pca = PCA(n_components=3, random_state=42)
pca_coords = normalize_to_scene(pca.fit_transform(hidden6))

tsne = TSNE(n_components=3, random_state=42, perplexity=30, init="pca")
tsne_coords = normalize_to_scene(tsne.fit_transform(hidden6))

samples = []
for i in range(len(X)):
    samples.append({
        "index": i,
        "label": int(y[i]),
        "predicted": int(predicted[i]),
        "pca": [float(v) for v in pca_coords[i]],
        "tsne": [float(v) for v in tsne_coords[i]],
    })

out = {
    "numSamples": len(samples),
    "numClasses": 3,
    "samples": samples,
}

with open("embeddings.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"Exportado embeddings.json con {len(samples)} muestras")
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
