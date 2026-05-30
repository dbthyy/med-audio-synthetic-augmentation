"""
04_classifier.py
================
Huấn luyện và đánh giá CNN classifier trên 4 kịch bản dữ liệu:
  1. Baseline — chỉ dữ liệu thật
  2. Real + VAE augmented
  3. Real + WGAN-GP augmented
  4. Real + Diffusion augmented

"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-3          # Adam lr=0.001 theo bài báo Section 2.3
PATIENCE   = 15            # Early stopping patience theo bài báo
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True


# ── 1. DATASETS ───────────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y.long() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.shape[-1] == 1:
            x = x.permute(2, 0, 1)    # (H, W, 1) → (1, H, W)
        return x, self.y[idx]


class SyntheticDataset(Dataset):
    """Dữ liệu sinh — tự động gán nhãn COVID+ (1)."""
    def __init__(self, X):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = torch.ones(len(self.X), dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def make_weighted_loader(dataset, batch_size=BATCH_SIZE):
    """
    [NEW] WeightedRandomSampler để cân bằng class trong mỗi batch.
    Đặc biệt quan trọng khi thêm synthetic COVID+ samples vào tập train.
    """
    labels = []
    for _, y in dataset:
        labels.append(int(y))
    labels = np.array(labels)

    class_counts = np.bincount(labels)
    weights      = 1.0 / class_counts[labels]
    sampler      = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# ── 2. CNN FROM SCRATCH ───────────────────────────────────────────────────────
class ConvBlock(nn.Sequential):
    """Conv2D → BN → ReLU → MaxPool (đúng theo bài báo Section 2.3)."""
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )


class AudioCNN(nn.Module):
    """
    [FIX] CNN train from scratch — đúng bài báo Section 2.3.
    4 ConvBlocks: 32 → 64 → 128 → 256 filters, 3×3 kernel.
    Trained from random initialization (không dùng pretrained weights).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ── 3. TRAIN & EVALUATE ───────────────────────────────────────────────────────
def train_and_evaluate(train_loader, val_loader, test_loader,
                       scenario_name: str) -> dict:
    """
    Train CNN from scratch với:
    - Adam optimizer, lr=1e-3 (bài báo Section 2.3)
    - CosineAnnealingLR (bài báo Section 2.3)
    - Early stopping theo val macro F1, patience=15 (bài báo Section 2.3)
    - Lưu model checkpoint tốt nhất
    - Metric: macro F1 + AUROC (bài báo Section 2.6)
    """
    print(f"\nTraining: {scenario_name}")
    model     = AudioCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1  = -1.0
    best_wts     = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # ── Validate (macro F1) ──────────────────────────────────────────────
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                _, preds = torch.max(model(inputs), 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        # ── Early stopping theo val macro F1 (bài báo Section 2.3) ──────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_wts    = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}/{EPOCHS} | val_macro_F1={val_f1:.4f} "
                  f"| best={best_val_f1:.4f} | patience={patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(best_wts)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(DEVICE)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Macro F1 + AUROC (đúng metric của bài báo Section 2.6)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    auroc    = roc_auc_score(all_labels, all_probs)

    print(f"  Result → macro F1: {macro_f1:.4f} | AUROC: {auroc:.4f}")

    # Lưu model để file 05 dùng cho ensemble
    model_path = f"model_{scenario_name.replace(' ', '_')}.pth"
    torch.save(model.state_dict(), model_path)

    return {
        "macro_F1":   round(macro_f1, 4),
        "AUROC":      round(auroc, 4),
        "model_path": model_path,
        "probs":      all_probs,    # cần cho ensemble
        "labels":     all_labels,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists("dataset.pt"):
        raise FileNotFoundError("Không tìm thấy dataset.pt. Chạy file 01 trước.")

    data = torch.load("dataset.pt", map_location="cpu")
    real_train_ds = AudioDataset(data["X_train"], data["y_train"])
    val_ds        = AudioDataset(data["X_val"],   data["y_val"])
    test_ds       = AudioDataset(data["X_test"],  data["y_test"])

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load synthetic samples
    def load_pt(path):
        return torch.load(path, map_location="cpu") if os.path.exists(path) else None

    vae_X  = load_pt("vae_samples.pt")
    gan_X  = load_pt("gan_samples.pt")
    diff_X = load_pt("diffusion_samples.pt")

    # Định nghĩa các kịch bản
    scenarios = {
        "Baseline": make_weighted_loader(real_train_ds),
    }
    if vae_X  is not None:
        scenarios["Real+VAE"]      = make_weighted_loader(
            ConcatDataset([real_train_ds, SyntheticDataset(vae_X)]))
    if gan_X  is not None:
        scenarios["Real+WGAN-GP"]  = make_weighted_loader(
            ConcatDataset([real_train_ds, SyntheticDataset(gan_X)]))
    if diff_X is not None:
        scenarios["Real+Diffusion"] = make_weighted_loader(
            ConcatDataset([real_train_ds, SyntheticDataset(diff_X)]))

    # Chạy tất cả kịch bản
    results = {}
    for name, train_loader in scenarios.items():
        results[name] = train_and_evaluate(
            train_loader, val_loader, test_loader, name)

    # In bảng tổng hợp
    print("\n" + "=" * 55)
    print(f"{'KẾT QUẢ THỰC NGHIỆM':^55}")
    print("=" * 55)
    print(f"{'Kịch bản':<22} | {'macro F1':>8} | {'AUROC':>7}")
    print("-" * 55)
    baseline_f1 = results.get("Baseline", {}).get("macro_F1", 0)
    for name, m in results.items():
        delta = m["macro_F1"] - baseline_f1
        sign  = "+" if delta >= 0 else ""
        print(f"{name:<22} | {m['macro_F1']:>8.4f} | {m['AUROC']:>7.4f}  "
              f"({sign}{delta:.3f})")
    print("=" * 55)
    print(f"  Bài báo gốc: Baseline F1=0.645 | AUROC=0.745")

    # Lưu kết quả (file 05 sẽ đọc để làm ensemble)
    save_data = {k: {"macro_F1": v["macro_F1"], "AUROC": v["AUROC"],
                     "model_path": v["model_path"]}
                 for k, v in results.items()}
    with open("results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("\n[Save] Results saved → results.json")
    print("[Save] Model checkpoints saved → model_*.pth")
