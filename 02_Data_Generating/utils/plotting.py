import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


_STYLE: dict[str, dict] = {
    "reproduce": dict(color="#2196F3", ls="-",  label_suffix=" (reproduce)"),
    "improve":   dict(color="#F44336", ls="--", label_suffix=" (improve)"),
}

_MODEL_SPECS: dict[str, dict] = {
    "vae": {
        "title":     "VAE",
        "loss_keys": [
            ("loss",  "Total loss"),
            ("recon", "Recon loss"),
            ("kl",    "KL loss"),
        ],
        "extra_key": ("beta", "β (KL weight)", "right"),
    },
    "wgan": {
        "title":     "WGAN-GP",
        "loss_keys": [
            ("d_loss", "Critic loss (D)"),
            ("g_loss", "Generator loss (G)"),
        ],
        "extra_key": None,
    },
    "diffusion": {
        "title":     "Diffusion",
        "loss_keys": [("loss", "MSE loss")],
        "extra_key": None,
    },
}

def plot_metrics(
    modes:       list[str],
    metrics_dir: str = "metrics",
    out_dir:     str = "plots",
    eval_every:  int = 25,
) -> None:

    os.makedirs(out_dir, exist_ok=True)

    for model_tag, spec in _MODEL_SPECS.items():
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 8),
            gridspec_kw={"height_ratios": [3, 2]},
        )
        ax_loss, ax_fad = axes
        has_data = False

        for mode in modes:
            path    = os.path.join(metrics_dir, f"{mode}_{model_tag}.jsonl")
            records = _load_jsonl(path)
            if not records:
                print(f"[Plot] No data: {path} — skipping.")
                continue

            has_data = True
            st       = _STYLE[mode]
            epochs   = [r["epoch"] for r in records]

            for i, (key, label) in enumerate(spec["loss_keys"]):
                vals = [r.get(key) for r in records]
                if all(v is None for v in vals):
                    continue
                xs = [e for e, v in zip(epochs, vals) if v is not None]
                ys = [v for v in vals if v is not None]
                ax_loss.plot(
                    xs, ys,
                    color=st["color"],
                    ls=["-", "--", ":"][i % 3],
                    alpha=0.85,
                    label=f"{label}{st['label_suffix']}",
                )

            if spec["extra_key"] and mode == "improve":
                key, label, _ = spec["extra_key"]
                vals = [r.get(key) for r in records]
                if any(v is not None for v in vals):
                    ax2 = ax_loss.twinx()
                    xs  = [e for e, v in zip(epochs, vals) if v is not None]
                    ys  = [v for v in vals if v is not None]
                    ax2.plot(xs, ys, color="#FF9800", ls=":", alpha=0.7, label=label)
                    ax2.set_ylabel(label, color="#FF9800")
                    ax2.tick_params(axis="y", labelcolor="#FF9800")
                    ax2.legend(loc="upper right", fontsize=8)

            fad_records = [r for r in records if r.get("fad") is not None]
            if fad_records:
                fx = [r["epoch"] for r in fad_records]
                fy = [r["fad"]   for r in fad_records]
                ax_fad.plot(
                    fx, fy,
                    color=st["color"], ls=st["ls"],
                    marker="o", markersize=4,
                    label=f"FAD{st['label_suffix']}",
                )
                best_idx = int(np.argmin(fy))
                ax_fad.annotate(
                    f"  best={fy[best_idx]:.1f}",
                    xy=(fx[best_idx], fy[best_idx]),
                    fontsize=8, color=st["color"],
                )

        if not has_data:
            plt.close(fig)
            continue

        ax_loss.set_title(f"{spec['title']} — Training Metrics", fontsize=13)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=8, loc="upper right")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax_fad.set_title(
            f"{spec['title']} — FAD (every {eval_every} epochs)", fontsize=11,
        )
        ax_fad.set_xlabel("Epoch")
        ax_fad.set_ylabel("FAD ↓")
        ax_fad.legend(fontsize=8, loc="upper right")
        ax_fad.grid(True, alpha=0.3)
        ax_fad.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout(pad=2.0)
        out_path = os.path.join(out_dir, f"{model_tag}_metrics.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Plot] Saved → {out_path}")