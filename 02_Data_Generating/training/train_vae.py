
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DEVICE
from evaluation.fad import ResNetEmbeddingExtractor, compute_fad
from models.vae import VAE
from utils.tracking import (
    FADAdaptiveTracker,
    MetricsLogger,
    _adapt_vae,
)


def train_vae(
    model:      VAE,
    loader:     DataLoader,
    real_covid: torch.Tensor,
    extractor:  ResNetEmbeddingExtractor,
    cfg:        dict,
    start_epoch: int               = 0,
    ckpt_dir:   str                = "checkpoints",
    device:     torch.device       = DEVICE,
    adaptive:   bool               = False,
    logger:     MetricsLogger | None = None,
) -> VAE:

    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["vae_lr"])

    if start_epoch > 0:
        ck_path = os.path.join(ckpt_dir, f"vae_epoch_{start_epoch}.pth")
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
                print(f"[VAE] Optimizer restored from {ck_path}")

    epochs     = cfg["vae_epochs"]
    beta       = cfg["vae_beta"]
    save_every = cfg["save_every"]
    eval_every = cfg["eval_every"]
    tracker    = FADAdaptiveTracker(patience=2) if adaptive else None
    best_fad_so_far = float('inf')
    best_epoch_so_far = start_epoch

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        total = recon_t = kl_t = 0.0

        for x, _ in tqdm(loader, desc=f"VAE {epoch}/{epochs}", leave=False):
            x = x.to(device)
            opt.zero_grad()
            recon, mu, lv = model(x)
            loss, rl, kl  = model.loss(recon, x, mu, lv, beta)
            loss.backward()
            opt.step()
            total   += loss.item()
            recon_t += rl.item()
            kl_t    += kl.item()

        n = len(loader)
        log_base = (
            f"[VAE] E{epoch:03d}  loss={total/n:.4f}  "
            f"recon={recon_t/n:.4f}  kl={kl_t/n:.4f}"
        )

        if epoch % eval_every == 0:
            fake = model.generate(real_covid.size(0), device)
            fad  = compute_fad(real_covid, fake, extractor)
            print(f"{log_base}  FAD={fad:.2f}  β={beta:.4f}")
            if logger:
                logger.log(
                    epoch=epoch, loss=total/n, recon=recon_t/n,
                    kl=kl_t/n, fad=fad, beta=beta,
                )

            if fad < best_fad_so_far:
                best_fad_so_far = fad
                best_epoch_so_far = epoch
                best_path = os.path.join(ckpt_dir, "vae_best.pth")
                model.save(best_path, epoch, opt)
                print(f"  [VAE] ★ New best FAD={fad:.2f} @ epoch {epoch} → {best_path}")

            if adaptive and tracker is not None:
                if tracker.update(epoch, fad):
                    cfg  = _adapt_vae(cfg, tracker, opt)
                    beta = cfg["vae_beta"]
        else:
            print(log_base)
            if logger:
                logger.log(epoch=epoch, loss=total/n, recon=recon_t/n,
                        kl=kl_t/n, beta=beta)

        if epoch % save_every == 0:
            ck_path = os.path.join(ckpt_dir, f"vae_epoch_{epoch}.pth")
            model.save(ck_path, epoch, opt)
            print(f"[VAE] Checkpoint → {ck_path}")

    if best_fad_so_far < float('inf'):
        print(f"[VAE] Best FAD achieved: {best_fad_so_far:.2f} @ epoch {best_epoch_so_far}")
    else:
        print("[VAE] Warning: No FAD evaluation occurred during this run.")

    if adaptive and tracker is not None:
        print(f"[VAE] Adaptive summary: {tracker.adaptations_done} adaptation(s) fired.")
    print('='*40 + '\n')

    return model
