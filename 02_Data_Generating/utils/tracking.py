import json
import os
import torch.optim as optim

class MetricsLogger:

    def __init__(
        self,
        mode:        str,
        model_tag:   str,
        metrics_dir: str = "metrics",
    ) -> None:
        os.makedirs(metrics_dir, exist_ok=True)
        self.path = os.path.join(metrics_dir, f"{mode}_{model_tag}.jsonl")

    def log(self, **kwargs) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(kwargs) + "\n")

    def load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


class FADAdaptiveTracker:

    def __init__(self, patience: int = 2) -> None:
        self.history:          list[tuple[int, float]] = []
        self.patience:         int   = patience
        self.adaptations_done: int   = 0
        self.best_epoch:       int | None = None
        self.best_fad_val:     float = float("inf")

    def update(self, epoch: int, fad: float) -> bool:
        if fad < self.best_fad_val:
            self.best_fad_val = fad
            self.best_epoch   = epoch

        self.history.append((epoch, fad))

        if len(self.history) < self.patience + 1:
            return False

        recent      = [f for _, f in self.history[-(self.patience + 1):]]
        best_before = min(recent[:-1])
        return recent[-1] >= best_before   # plateau or worse

    def best_fad(self) -> float:
        return self.best_fad_val

    def last_fad(self) -> float:
        return self.history[-1][1] if self.history else float("inf")


def _adapt_vae(
    cfg:       dict,
    tracker:   FADAdaptiveTracker,
    optimizer: optim.Optimizer,
) -> dict:
    new_beta = max(cfg["vae_beta"] / 2.0, cfg["vae_beta_min"])
    if new_beta == cfg["vae_beta"]:
        print(f"  [Adapt-VAE] β already at floor ({cfg['vae_beta_min']}), skipping.")
        return cfg
    cfg = {**cfg, "vae_beta": new_beta}
    tracker.adaptations_done += 1
    print(f"  [Adapt-VAE] FAD plateau → β {new_beta * 2:.4f} → {new_beta:.4f}")
    return cfg


def _adapt_wgan(
    cfg:   dict,
    tracker: FADAdaptiveTracker,
    opt_G: optim.Optimizer,
    opt_D: optim.Optimizer,
) -> dict:
    new_lr = min(cfg["wgan_lr"] * 1.2, cfg["wgan_lr_max"])
    if new_lr == cfg["wgan_lr"]:
        print(f"  [Adapt-GAN] LR already at ceiling ({cfg['wgan_lr_max']}), skipping.")
        return cfg
    cfg = {**cfg, "wgan_lr": new_lr}
    for pg in opt_G.param_groups:
        pg["lr"] = new_lr
    for pg in opt_D.param_groups:
        pg["lr"] = new_lr
    tracker.adaptations_done += 1
    print(f"  [Adapt-GAN] FAD plateau → LR bumped to {new_lr:.2e}")
    return cfg


def _adapt_diffusion(
    cfg:       dict,
    tracker:   FADAdaptiveTracker,
    optimizer: optim.Optimizer,
) -> dict:
    new_lr = cfg["diff_lr"] * 0.7
    if new_lr < 1e-6:
        print(f"  [Adapt-Diff] LR already very small ({cfg['diff_lr']:.2e}), skipping.")
        return cfg
    cfg = {**cfg, "diff_lr": new_lr}
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr
    tracker.adaptations_done += 1
    print(f"  [Adapt-Diff] FAD plateau → LR reduced to {new_lr:.2e}")
    return cfg