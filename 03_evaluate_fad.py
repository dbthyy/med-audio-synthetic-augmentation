"""
03_evaluate_fad.py
==================
Tính Fréchet Audio Distance (FAD) giữa dữ liệu thật (COVID+)
và dữ liệu sinh từ VAE, WGAN-GP, Diffusion.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import resnet18, ResNet18_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. RESNET FEATURE EXTRACTOR ───────────────────────────────────────────────
class ResNetEmbeddingExtractor:
    """
    ResNet18 làm feature extractor, output vector 512-dim.

    [FIX] conv1 thay thế: thay vì init ngẫu nhiên 1-channel conv,
    ta average trọng số 3-channel gốc về 1 channel:
        W_new = mean(W_pretrained, dim=1, keepdim=True)
    Cách này giữ lại các pattern đã học từ ImageNet ở tầng đầu,
    phù hợp hơn khi input là mel-spectrogram grayscale.
    """
    def __init__(self):
        weights    = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)

        # Lấy trọng số conv1 gốc (64, 3, 7, 7) → average về (64, 1, 7, 7)
        old_weight = base_model.conv1.weight.data          # (64, 3, 7, 7)
        new_weight = old_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)

        # Tạo conv1 mới 1-channel và gán trọng số đã average
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                     stride=2, padding=3, bias=False)
        base_model.conv1.weight.data = new_weight

        # Bỏ lớp fc để lấy embedding 512-dim
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_extractor.to(DEVICE)
        self.feature_extractor.eval()

    @torch.no_grad()
    def extract(self, x_tensor: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """Trích xuất embeddings theo batch, tránh OOM."""
        all_emb = []
        for i in range(0, x_tensor.size(0), batch_size):
            batch = x_tensor[i : i + batch_size].to(DEVICE)
            emb   = self.feature_extractor(batch).flatten(1)
            all_emb.append(emb.cpu().numpy())
        return np.concatenate(all_emb, axis=0)


# ── 2. FRÉCHET DISTANCE ───────────────────────────────────────────────────────
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """
    d² = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^0.5)
    Giữ nguyên từ file cũ — đúng về mặt toán học.
    """
    mu1    = np.atleast_1d(mu1);    mu2    = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff   = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        offset  = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_stats(embeddings: np.ndarray):
    return np.mean(embeddings, axis=0), np.cov(embeddings, rowvar=False)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists("dataset.pt"):
        raise FileNotFoundError("Không tìm thấy dataset.pt. Chạy file 01 trước.")

    data       = torch.load("dataset.pt", map_location="cpu")
    covid_mask = data["y_train"] == 1
    real_all   = data["X_train"][covid_mask]

    # Chuẩn hóa shape về (N, 1, H, W)
    if real_all.ndim == 4 and real_all.shape[-1] == 1:
        real_all = real_all.permute(0, 3, 1, 2)

    extractor = ResNetEmbeddingExtractor()
    print("Extracting features from real COVID+ samples...")
    emb_real  = extractor.extract(real_all)
    mu_real, sigma_real = compute_stats(emb_real)

    # ── [NEW] FAD baseline: Real vs Real (upper-half vs lower-half) ──────────
    # Đây là lower bound — cho biết FAD tối thiểu mà một mô hình sinh
    # "hoàn hảo" cũng không thể thấp hơn nhiều.
    half       = len(real_all) // 2
    emb_r1     = extractor.extract(real_all[:half])
    emb_r2     = extractor.extract(real_all[half:])
    mu_r1, sigma_r1 = compute_stats(emb_r1)
    mu_r2, sigma_r2 = compute_stats(emb_r2)
    fad_baseline = calculate_frechet_distance(mu_r1, sigma_r1, mu_r2, sigma_r2)

    # ── Load generated samples ───────────────────────────────────────────────
    def load_samples(path):
        if os.path.exists(path):
            s = torch.load(path, map_location="cpu")
            if s.ndim == 4 and s.shape[-1] == 1:
                s = s.permute(0, 3, 1, 2)
            return s
        return None

    vae_samples  = load_samples("vae_samples.pt")
    gan_samples  = load_samples("gan_samples.pt")
    diff_samples = load_samples("diffusion_samples.pt")

    # ── Compute FAD ──────────────────────────────────────────────────────────
    results = {}

    if vae_samples is not None:
        emb = extractor.extract(vae_samples)
        mu, sigma = compute_stats(emb)
        results["VAE"] = calculate_frechet_distance(mu_real, sigma_real, mu, sigma)

    if gan_samples is not None:
        emb = extractor.extract(gan_samples)
        mu, sigma = compute_stats(emb)
        results["WGAN-GP"] = calculate_frechet_distance(mu_real, sigma_real, mu, sigma)

    if diff_samples is not None:
        emb = extractor.extract(diff_samples)
        mu, sigma = compute_stats(emb)
        results["Diffusion"] = calculate_frechet_distance(mu_real, sigma_real, mu, sigma)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'FAD EVALUATION RESULTS':^60}")
    print("=" * 60)
    print(f"  [Reference] FAD (Real vs Real split): {fad_baseline:.4f}")
    print(f"  (Lower bound — FAD tốt nhất có thể đạt)")
    print("-" * 60)

    if results:
        for name, fad in sorted(results.items(), key=lambda x: x[1]):
            gap = fad - fad_baseline
            print(f"  FAD (Real vs {name:<12}): {fad:.4f}  [+{gap:.4f} vs baseline]")
    else:
        print("  Chưa có sample file. Chạy file 02 trước.")

    print("=" * 60)
    print("  (*) FAD thấp hơn = dữ liệu sinh gần thật hơn.")
    print("  (*) FAD (Real vs Real) là điểm tham chiếu lý tưởng.")
