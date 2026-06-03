import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DEVICE
from evaluation.fad import ResNetEmbeddingExtractor, compute_fad
from models.diffusion import Diffusion
from utils.tracking import FADAdaptiveTracker, MetricsLogger, _adapt_diffusion

def train_diffusion(
    model:      Diffusion,
    loader:     DataLoader,
    real_covid: torch.Tensor,
    extractor:  ResNetEmbeddingExtractor,
    cfg:        dict,
    start_epoch: int               = 0,
    ckpt_dir:   str                = "checkpoints",
    device:     torch.device       = DEVICE,
    adaptive:   bool               = False,
    logger:     MetricsLogger | None = None,
) -> Diffusion:

    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(device)
    opt = AdamW(model.parameters(), lr=cfg["diff_lr"], weight_decay=0.01)

    if start_epoch > 0:
        ck_path = os.path.join(ckpt_dir, f"diffusion_epoch_{start_epoch}.pth")
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
                print(f"[Diff] Optimizer restored from {ck_path}")

    epochs     = cfg["diff_epochs"]
    mini_bs    = cfg["diff_mini_bs"]
    save_every = cfg["save_every"]
    eval_every = cfg["eval_every"]
    tracker    = FADAdaptiveTracker(patience=2) if adaptive else None
    best_fad_so_far = float('inf')
    best_epoch_so_far = start_epoch

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        total = 0.0

        for x, _ in tqdm(loader, desc=f"Diff {epoch}/{epochs}", leave=False):
            x = x.to(device)
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            opt.step()
            total += loss.item()

        n = len(loader)
        log_base = f"[Diff] E{epoch:03d}  loss={total/n:.4f}"

        if epoch % eval_every == 0:
            fake   = model.generate(real_covid.size(0), device, mini_bs=mini_bs)
            fad    = compute_fad(real_covid, fake, extractor)
            cur_lr = opt.param_groups[0]["lr"]
            print(f"{log_base}  FAD={fad:.2f}  lr={cur_lr:.2e}")

            if logger:
                logger.log(epoch=epoch, loss=total/n, fad=fad, lr=cur_lr)

            if fad < best_fad_so_far:
                best_fad_so_far = fad
                best_epoch_so_far = epoch
                best_path = os.path.join(ckpt_dir, "diffusion_best.pth")
                model.save(best_path, epoch, opt)
                print(
                    f"  [Diff] ★ New best FAD={fad:.2f} "
                    f"@ epoch {epoch} → {best_path}"
                )

            if adaptive and tracker is not None:
                if tracker.update(epoch, fad):
                    cfg = _adapt_diffusion(cfg, tracker, opt)
        else:
            print(log_base)
            if logger:
                logger.log(epoch=epoch, loss=total/n,
                           lr=opt.param_groups[0]["lr"])

        if epoch % save_every == 0:
            ck_path = os.path.join(ckpt_dir, f"diffusion_epoch_{epoch}.pth")
            model.save(ck_path, epoch, opt)
            print(f"[Diff] Checkpoint → {ck_path}")

    if best_fad_so_far < float('inf'):
        print(f"[Diff] Best FAD achieved: {best_fad_so_far:.2f} @ epoch {best_epoch_so_far}")
    else:
        print("[Diff] Warning: No FAD evaluation occurred during this run.")

    if adaptive and tracker is not None:
        print(f"[Diff] Adaptive summary: {tracker.adaptations_done} adaptation(s) fired.")
    print('='*40 + '\n')

    return model
