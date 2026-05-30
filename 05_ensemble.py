"""
05_ensemble.py
==============
Ensemble 4 models (Baseline, VAE, WGAN-GP, Diffusion) bằng cách
average predicted probabilities → final prediction.

Reproduce kết quả bài báo: Ensemble F1=0.664, AUROC=0.761
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. DATASET ────────────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y.long() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.permute(2, 0, 1)   # (H,W,1) → (1,H,W)
        return x, self.y[idx]


# ── 2. CNN ARCHITECTURE (phải khớp với 04_classifier.py) ─────────────────────
class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

class AudioCNN(nn.Module):
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
        return self.classifier(self.pool(self.features(x)).flatten(1))


# ── 3. LOAD MODEL ─────────────────────────────────────────────────────────────
def load_model(path: str, device=DEVICE) -> AudioCNN:
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ── 4. GET PROBABILITIES ──────────────────────────────────────────────────────
@torch.no_grad()
def get_probs(model: AudioCNN, loader: DataLoader, device=DEVICE):
    """Trả về (probs_covid, true_labels) — dùng cho ensemble."""
    all_probs, all_labels = [], []
    for inputs, labels in loader:
        inputs  = inputs.to(device)
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


# ── 5. ENSEMBLE PREDICTION ────────────────────────────────────────────────────
def ensemble_predict(prob_list: list[np.ndarray], threshold: float = 0.5):
    """
    Equation (1) trong bài báo:
        P_ensemble(COVID+) = (1/N) * Σ p_i(COVID+)
    """
    avg_probs = np.mean(np.stack(prob_list, axis=0), axis=0)
    preds     = (avg_probs >= threshold).astype(int)
    return avg_probs, preds


# ── 6. METRICS ────────────────────────────────────────────────────────────────
def compute_metrics(labels, preds, probs) -> dict:
    return {
        "macro_F1": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "AUROC":    round(roc_auc_score(labels, probs), 4),
        "cm":       confusion_matrix(labels, preds),
        "report":   classification_report(labels, preds,
                                          target_names=["Healthy", "COVID+"],
                                          zero_division=0),
    }


# ── 7. VISUALIZATION ─────────────────────────────────────────────────────────
SCENARIO_NAMES = ["Baseline", "Real+VAE", "Real+WGAN-GP", "Real+Diffusion", "Ensemble"]
PAPER_F1       = [0.645, 0.646, 0.609, 0.644, 0.664]
PAPER_AUROC    = [0.745, 0.748, 0.726, 0.746, 0.761]

def plot_results(all_results: dict, labels, prob_dict: dict, save_dir: str = "."):
    """
    3 biểu đồ:
      (A) F1 + AUROC bar chart so sánh paper vs reproduced
      (B) Confusion matrix của Ensemble
      (C) ROC curves tất cả models
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Palette ---
    COLOR_PAPER = "#B0BEC5"
    COLOR_REPRO = "#1565C0"
    COLOR_ENS   = "#E53935"
    COLORS_ROC  = ["#455A64", "#1E88E5", "#43A047", "#FB8C00", COLOR_ENS]

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.40, wspace=0.35,
                            left=0.06, right=0.97,
                            top=0.91, bottom=0.08)

    # ── (A1) F1 Bar Chart ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    scenarios   = list(all_results.keys())
    repro_f1    = [all_results[s]["macro_F1"] for s in scenarios]
    # match paper order
    paper_f1_matched = []
    for s in scenarios:
        idx = SCENARIO_NAMES.index(s) if s in SCENARIO_NAMES else -1
        paper_f1_matched.append(PAPER_F1[idx] if idx >= 0 else None)

    x    = np.arange(len(scenarios))
    w    = 0.35
    bars_paper  = ax1.bar(x - w/2, paper_f1_matched, w,
                           label="Paper", color=COLOR_PAPER, edgecolor="white", linewidth=0.8)
    bars_repro  = ax1.bar(x + w/2, repro_f1, w,
                           label="Reproduced", color=[
                               COLOR_ENS if s == "Ensemble" else COLOR_REPRO
                               for s in scenarios
                           ], edgecolor="white", linewidth=0.8)

    for bar in bars_repro:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.set_ylim(0.55, 0.72)
    ax1.set_ylabel("Macro-averaged F1 Score", fontsize=11)
    ax1.set_title("(A) F1 Score — Paper vs Reproduced", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(fontsize=10)
    ax1.axhline(0.645, color="#B71C1C", linestyle="--", linewidth=1.0, alpha=0.6,
                label="Baseline (paper)")
    ax1.set_facecolor("#FFFFFF")
    ax1.spines[["top","right"]].set_visible(False)

    # ── (A2) AUROC Bar Chart ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    repro_auroc = [all_results[s]["AUROC"] for s in scenarios]
    paper_auroc_matched = []
    for s in scenarios:
        idx = SCENARIO_NAMES.index(s) if s in SCENARIO_NAMES else -1
        paper_auroc_matched.append(PAPER_AUROC[idx] if idx >= 0 else None)

    ax2.bar(x - w/2, paper_auroc_matched, w, label="Paper", color=COLOR_PAPER,
            edgecolor="white")
    ax2.bar(x + w/2, repro_auroc, w, label="Reproduced",
            color=[COLOR_ENS if s == "Ensemble" else COLOR_REPRO for s in scenarios],
            edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=8, rotation=20, ha="right")
    ax2.set_ylim(0.68, 0.80)
    ax2.set_ylabel("AUROC", fontsize=11)
    ax2.set_title("(B) AUROC", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(fontsize=9)
    ax2.set_facecolor("#FFFFFF")
    ax2.spines[["top","right"]].set_visible(False)

    # ── (B) Confusion Matrix — Ensemble ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if "Ensemble" in all_results:
        cm = all_results["Ensemble"]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Healthy", "COVID+"],
                    yticklabels=["Healthy", "COVID+"],
                    ax=ax3, cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
        ax3.set_xlabel("Predicted", fontsize=10)
        ax3.set_ylabel("True", fontsize=10)
        ax3.set_title("(C) Confusion Matrix\n(Ensemble)", fontsize=12, fontweight="bold")

    # ── (C) ROC Curves ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.plot([0,1],[0,1], "k--", linewidth=0.8, alpha=0.5, label="Random")
    for i, (name, probs) in enumerate(prob_dict.items()):
        if name in all_results:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc_val = all_results[name]["AUROC"]
            lw  = 2.5 if name == "Ensemble" else 1.5
            col = COLOR_ENS if name == "Ensemble" else COLORS_ROC[i % len(COLORS_ROC)]
            ax4.plot(fpr, tpr, color=col, linewidth=lw,
                     label=f"{name} (AUC={auc_val:.3f})")

    ax4.set_xlabel("False Positive Rate", fontsize=11)
    ax4.set_ylabel("True Positive Rate", fontsize=11)
    ax4.set_title("(D) ROC Curves — All Models", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9, loc="lower right")
    ax4.set_facecolor("#FFFFFF")
    ax4.spines[["top","right"]].set_visible(False)

    fig.suptitle("Synthetic Data Augmentation — Reproduction Results",
                 fontsize=15, fontweight="bold", y=0.97)

    out_path = os.path.join(save_dir, "ensemble_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Plot] Saved → {out_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Load test set ─────────────────────────────────────────────────────────
    if not os.path.exists("dataset.pt"):
        raise FileNotFoundError("Không tìm thấy dataset.pt. Chạy file 01 trước.")

    data       = torch.load("dataset.pt", map_location="cpu", weights_only=False)
    test_ds    = AudioDataset(data["X_test"], data["y_test"])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    true_labels = data["y_test"].numpy()

    # ── Load results.json từ file 04 ─────────────────────────────────────────
    if not os.path.exists("results.json"):
        raise FileNotFoundError("Không tìm thấy results.json. Chạy file 04 trước.")

    with open("results.json") as f:
        results_04 = json.load(f)

    # ── Xác định model paths ──────────────────────────────────────────────────
    model_map = {
        "Baseline":       "model_Baseline.pth",
        "Real+VAE":       "model_Real+VAE.pth",
        "Real+WGAN-GP":   "model_Real+WGAN-GP.pth",
        "Real+Diffusion": "model_Real+Diffusion.pth",
    }

    # ── Thu thập probabilities từng model ────────────────────────────────────
    all_probs_list = []
    prob_dict      = {}
    all_results    = {}

    print("\n" + "=" * 60)
    print(f"{'INDIVIDUAL MODEL RESULTS':^60}")
    print("=" * 60)

    for name, path in model_map.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {name}: {path} không tồn tại.")
            continue

        model = load_model(path)
        probs, _ = get_probs(model, test_loader)
        preds    = (probs >= 0.5).astype(int)
        metrics  = compute_metrics(true_labels, preds, probs)

        all_probs_list.append(probs)
        prob_dict[name]   = probs
        all_results[name] = metrics

        print(f"\n  [{name}]")
        print(f"    macro F1 : {metrics['macro_F1']:.4f}")
        print(f"    AUROC    : {metrics['AUROC']:.4f}")
        print(f"\n{metrics['report']}")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    if len(all_probs_list) > 1:
        print("\n" + "=" * 60)
        print(f"{'ENSEMBLE RESULTS':^60}")
        print("=" * 60)

        ens_probs, ens_preds = ensemble_predict(all_probs_list)
        ens_metrics = compute_metrics(true_labels, ens_preds, ens_probs)

        prob_dict["Ensemble"]   = ens_probs
        all_results["Ensemble"] = ens_metrics

        print(f"\n  [Ensemble — Equation (1) in paper]")
        print(f"    macro F1 : {ens_metrics['macro_F1']:.4f}  (paper: 0.664)")
        print(f"    AUROC    : {ens_metrics['AUROC']:.4f}  (paper: 0.761)")
        print(f"\n{ens_metrics['report']}")

    # ── Bảng tổng hợp so sánh với bài báo ────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'FULL COMPARISON — Reproduced vs Paper':^65}")
    print("=" * 65)
    print(f"{'Scenario':<22} | {'Repro F1':>8} | {'Paper F1':>8} | {'Δ F1':>6} | {'AUROC':>6}")
    print("-" * 65)

    paper_lookup = dict(zip(SCENARIO_NAMES, zip(PAPER_F1, PAPER_AUROC)))
    for name, m in all_results.items():
        pf1, _ = paper_lookup.get(name, (None, None))
        delta   = f"{m['macro_F1'] - pf1:+.3f}" if pf1 else "  N/A"
        pf1_str = f"{pf1:.3f}" if pf1 else "  N/A"
        print(f"{name:<22} | {m['macro_F1']:>8.4f} | {pf1_str:>8} | {delta:>6} | {m['AUROC']:>6.4f}")

    print("=" * 65)

    # ── Lưu kết quả ensemble ──────────────────────────────────────────────────
    save_data = {
        name: {"macro_F1": m["macro_F1"], "AUROC": m["AUROC"]}
        for name, m in all_results.items()
    }
    with open("ensemble_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("\n[Save] ensemble_results.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(all_results, true_labels, prob_dict, save_dir=".")
