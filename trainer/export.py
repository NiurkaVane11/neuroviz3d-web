import torch
import json
from train import IrisNet

def export_weights(model_path="iris_model.pt", out_path="network.json"):
    model = IrisNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    layers = []
    for name, layer in [("fc1", model.fc1), ("fc2", model.fc2), ("fc3", model.fc3)]:
        W = layer.weight.detach().numpy()  # shape [out, in]
        b = layer.bias.detach().numpy()    # shape [out]
        layers.append({
            "name": name,
            "in_features": int(W.shape[1]),
            "out_features": int(W.shape[0]),
            "weights": W.tolist(),  # [out][in]
            "bias": b.tolist()
        })

    architecture = {
        "layer_sizes": [4, 8, 6, 3],
        "layers": layers
    }

    with open(out_path, "w") as f:
        json.dump(architecture, f, indent=2)

    print(f"Exportado a {out_path}")

if __name__ == "__main__":
    export_weights()