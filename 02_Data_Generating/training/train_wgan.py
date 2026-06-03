import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import DEVICE
from evaluation.fad import ResNetEmbeddingExtractor, compute_fad
from models.wgan import Discriminator, Generator, _gradient_penalty, generate_gan
from utils.tracking import FADAdaptiveTracker, MetricsLogger, _adapt_wgan


def train_wgan_gp(
    generator:  Generator,
    critic:     Discriminator,
    loader:     DataLoader,
    real_covid: torch.Tensor,
    extractor:  ResNetEmbeddingExtractor,
    cfg:        dict,
    start_epoch: int               = 0,
    ckpt_dir:   str                = "checkpoints",
    device:     torch.device       = DEVICE,
    adaptive:   bool               = False,
    logger:     MetricsLogger | None = None,
) -> tuple[Generator, Discriminator]:

    os.makedirs(ckpt_dir, exist_ok=True)
    generator.to(device)
    critic.to(device)
    opt_G = optim.RMSprop(generator.parameters(), lr=cfg["wgan_lr"])
    opt_D = optim.RMSprop(critic.parameters(),    lr=cfg["wgan_lr"])

    if start_epoch > 0:
        ck_path = os.path.join(ckpt_dir, f"wgan_epoch_{start_epoch}.pth")
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            if "opt_G" in ck:
                opt_G.load_state_dict(ck["opt_G"])
            if "opt_D" in ck:
                opt_D.load_state_dict(ck["opt_D"])
            print(f"[GAN] Optimizers restored from {ck_path}")

    epochs     = cfg["wgan_epochs"]
    n_critic   = cfg["wgan_n_critic"]
    lambda_gp  = cfg["wgan_lambda_gp"]
    save_every = cfg["save_every"]
    eval_every = cfg["eval_every"]
    tracker    = FADAdaptiveTracker(patience=2) if adaptive else None
    best_fad_so_far = float('inf')
    best_epoch_so_far = start_epoch
    
    d_loss = g_loss = torch.tensor(0.0)

    for epoch in range(start_epoch + 1, epochs + 1):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            bs   = real.size(0)

            z    = torch.randn(bs, generator.latent_dim, device=device)
            fake = generator(z).detach()
            gp   = _gradient_penalty(critic, real, fake, device)
            d_loss = critic(fake).mean() - critic(real).mean() + lambda_gp * gp
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            if i % n_critic == 0:
                z      = torch.randn(bs, generator.latent_dim, device=device)
                g_loss = -critic(generator(z)).mean()
                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()

        log_base = (
            f"[GAN] E{epoch:03d}  "
            f"D={d_loss.item():.4f}  G={g_loss.item():.4f}"
        )

        if epoch % eval_every == 0:
            fake   = generate_gan(generator, real_covid.size(0), device)
            fad    = compute_fad(real_covid, fake, extractor)
            cur_lr = opt_G.param_groups[0]["lr"]
            print(f"{log_base}  FAD={fad:.2f}  lr={cur_lr:.2e}")
            if logger:
                logger.log(epoch=epoch, d_loss=d_loss.item(),
                           g_loss=g_loss.item(), fad=fad, lr=cur_lr)

            if fad < best_fad_so_far:
                best_fad_so_far = fad
                best_epoch_so_far = epoch
                best_path = os.path.join(ckpt_dir, "wgan_best.pth")
                torch.save(
                    {
                        "gen":   generator.state_dict(),
                        "disc":  critic.state_dict(),
                        "opt_G": opt_G.state_dict(),
                        "opt_D": opt_D.state_dict(),
                        "epoch": epoch,
                    },
                    best_path,
                )
                print(
                    f"  [GAN] ★ New best FAD={fad:.2f} "
                    f"@ epoch {epoch} → {best_path}"
                )

            if adaptive and tracker is not None:
                if tracker.update(epoch, fad):
                    cfg = _adapt_wgan(cfg, tracker, opt_G, opt_D)
        else:
            print(log_base)
            if logger:
                logger.log(epoch=epoch, d_loss=d_loss.item(),
                           g_loss=g_loss.item(),
                           lr=opt_G.param_groups[0]["lr"])

        if epoch % save_every == 0:
            ck_path = os.path.join(ckpt_dir, f"wgan_epoch_{epoch}.pth")
            torch.save(
                {
                    "gen":   generator.state_dict(),
                    "disc":  critic.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                    "epoch": epoch,
                },
                ck_path,
            )
            print(f"[GAN] Checkpoint → {ck_path}")

    print(f"[GAN] Best internal metrics achieved: FAD={best_fad_so_far:.2f} @ epoch {best_epoch_so_far}")
    if adaptive and tracker is not None:
        print(f"[GAN] Adaptive adjustments fired: {tracker.adaptations_done} times.")

    return generator, critic
