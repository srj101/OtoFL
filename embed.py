"""Image embeddings for a second, feature-based near-duplicate check (ResNet-50 ImageNet + DINOv2 ViT-B/14)."""
import sys, json, numpy as np, torch, torch.nn.functional as F
cache, out = sys.argv[1], sys.argv[2]
X = torch.load(f"{cache}/images.pt"); dev = "cuda" if torch.cuda.is_available() else "cpu"
M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
def run(model, res):
    model.eval().to(dev); outs = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            x = X[i:i + 64].to(dev).float() / 255.0
            x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
            f = model((x - M) / S); outs.append(F.normalize(f.flatten(1), dim=1).cpu())
    return torch.cat(outs).numpy().astype(np.float32)
import torchvision.models as tv
r = tv.resnet50(weights="IMAGENET1K_V2"); r.fc = torch.nn.Identity()
E = {"resnet50": run(r, 224)}
try:
    d = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14"); E["dinov2_vitb14"] = run(d, 224)
except Exception as e:
    print("dinov2 failed:", e)
np.savez_compressed(out, **E); print({k: v.shape for k, v in E.items()})
