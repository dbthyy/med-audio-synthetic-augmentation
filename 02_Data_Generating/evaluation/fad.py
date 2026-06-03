import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from torchvision.models import resnet18, ResNet18_Weights

from config import DEVICE

class ResNetEmbeddingExtractor:

    def __init__(self, device: torch.device = DEVICE) -> None:
        weights    = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)

        # Adapt conv1: 3-channel → 1-channel (average over RGB)
        old_weight = base_model.conv1.weight.data
        new_weight = old_weight.mean(dim=1, keepdim=True)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base_model.conv1.weight.data = new_weight

        # Drop classification head; keep everything up to global avg-pool
        self.net    = nn.Sequential(*list(base_model.children())[:-1])
        self.device = device
        self.net.to(device).eval()

    @torch.no_grad()
    def extract(self, x: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        embs: list[np.ndarray] = []
        for i in range(0, x.size(0), batch_size):
            emb = self.net(x[i : i + batch_size].to(self.device)).flatten(1)
            embs.append(emb.cpu().numpy())
        return np.concatenate(embs, axis=0)

def _stats(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(emb, axis=0), np.cov(emb, rowvar=False)


def _frechet_distance(
    mu1: np.ndarray,
    s1:  np.ndarray,
    mu2: np.ndarray,
    s2:  np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Numerically stable Fréchet distance between two Gaussians."""
    diff    = mu1 - mu2
    covmean, _ = sqrtm(s1.dot(s2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    if not np.isfinite(covmean).all():
        reg     = np.eye(s1.shape[0]) * eps
        covmean = sqrtm((s1 + reg).dot(s2 + reg))

    return float(diff @ diff + np.trace(s1 + s2 - 2.0 * covmean))


def compute_fad(
    real_tensor: torch.Tensor,
    fake_tensor: torch.Tensor,
    extractor:   ResNetEmbeddingExtractor,
) -> float:

    mu_r, s_r = _stats(extractor.extract(real_tensor))
    mu_f, s_f = _stats(extractor.extract(fake_tensor))
    return _frechet_distance(mu_r, s_r, mu_f, s_f)
