import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DEVICE, build_run_cfg
from data.datasets import AudioDataset
from evaluation.fad import ResNetEmbeddingExtractor
from models.diffusion import Diffusion, UNet
from models.vae import VAE
from models.wgan import Discriminator, Generator, generate_gan
from training.train_diffusion import train_diffusion
from training.train_vae import train_vae
from training.train_wgan import train_wgan_gp
from utils.plotting import plot_metrics
from utils.tracking import MetricsLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generative audio training pipeline")
    p.add_argument(
        "--mode",
        choices=["reproduce", "improve", "all"],
        default="reproduce",
        help="Training mode. 'all' is only valid with --plot.",
    )
    p.add_argument(
        "--model",
        choices=["vae", "wgan", "diffusion", "all"],
        default="all",
        help="Which model(s) to train/generate.",
    )
    p.add_argument("--resume",        default=None,          help="Checkpoint path to resume from.")
    p.add_argument("--generate_only", action="store_true",   help="Skip training; generate only.")
    p.add_argument("--ckpt_dir",      default="checkpoints", help="Checkpoint root directory.")
    p.add_argument("--metrics_dir",   default="metrics",     help="Directory for JSONL metrics.")
    p.add_argument("--plot",          action="store_true",   help="Read metrics and export PNGs.")
    p.add_argument("--plot_dir",      default="plots",       help="Directory for plot output PNGs.")
    p.add_argument("--dataset",       default="dataset.pt",  help="Path to dataset .pt file.")
    return p.parse_args()


def _banner(mode: str, model: str, adaptive: bool, cfg: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Mode    : {mode.upper()}")
    print(f"  Model   : {model}")
    print(f"  Adaptive: {adaptive}")
    print(
        f"  VAE β={cfg['vae_beta']}  |  GAN base={cfg['wgan_gen_base']}  "
        f"|  Diff mini_bs={cfg['diff_mini_bs']}"
    )
    print(f"  eval_every={cfg['eval_every']} epochs")
    print(f"  Generate from: BEST checkpoint")
    print(f"{'='*60}\n")


def _load_best_or_warn(path: str, label: str) -> bool:
    if os.path.exists(path):
        return True
    print(f"[{label}] Best checkpoint not found, generating from last epoch.")
    return False


def main() -> None:
    args = parse_args()

    if args.plot:
        modes = ["reproduce", "improve"] if args.mode == "all" else [args.mode]
        print(f"[Plot] Reading metrics from '{args.metrics_dir}/', modes={modes}")
        plot_metrics(
            modes,
            metrics_dir=args.metrics_dir,
            out_dir=args.plot_dir,
            eval_every=25,
        )
        return

    if args.mode == "all":
        raise SystemExit(
            "--mode all is only valid with --plot. "
            "For training, choose reproduce or improve."
        )
    if args.resume and args.model == "all":
        raise SystemExit(
            "When using --resume, specify a single model via --model "
            "(vae | wgan | diffusion)."
        )

    args.ckpt_dir = os.path.join(args.ckpt_dir, args.mode)
    cfg           = build_run_cfg(args.mode)
    adaptive      = (args.mode == "improve")
    prefix        = args.mode

    _banner(args.mode, args.model, adaptive, cfg)

    data         = torch.load(args.dataset, map_location="cpu", weights_only=False)
    covid_mask   = data["y_train"] == 1
    covid_X      = data["X_train"][covid_mask]
    covid_y      = data["y_train"][covid_mask]
    covid_ds     = AudioDataset(covid_X, covid_y)
    covid_loader = DataLoader(covid_ds, BATCH_SIZE, shuffle=True)

    real_covid_4d = (
        covid_X.permute(0, 3, 1, 2) if covid_X.shape[-1] == 1 else covid_X
    )
    n_generate = int(covid_mask.sum().item() * cfg["aug_ratio"])

    print(f"[Info] COVID+ samples : {covid_mask.sum().item()}")
    print(f"[Info] Will generate  : {n_generate} synthetic samples per model")
    print(f"[Info] Device         : {DEVICE}\n")

    print("[Info] Initialising FAD extractor (ResNet-18) …")
    extractor = ResNetEmbeddingExtractor(DEVICE)
    print("[Info] FAD extractor ready.\n")

    run_vae  = args.model in ("vae",  "all")
    run_wgan = args.model in ("wgan", "all")
    run_diff = args.model in ("diffusion", "all")


    if run_vae:
        print("=" * 40, "\n  VAE\n" + "=" * 40)
        vae         = VAE().to(DEVICE)
        start_epoch = 0
        vae_logger  = MetricsLogger(args.mode, "vae", args.metrics_dir)

        if args.resume and args.model == "vae":
            start_epoch = vae.load(args.resume, device=DEVICE)

        if not args.generate_only:
            train_vae(
                vae, covid_loader, real_covid_4d, extractor, cfg,
                start_epoch=start_epoch, ckpt_dir=args.ckpt_dir,
                adaptive=adaptive, logger=vae_logger,
            )
            vae.save(os.path.join(args.ckpt_dir, f"vae_final_{args.mode}.pth"))

        best_path = os.path.join(args.ckpt_dir, "vae_best.pth")
        if _load_best_or_warn(best_path, "VAE"):
            ep = vae.load(best_path, device=DEVICE)
            print(f"[VAE] Generating from BEST checkpoint (epoch {ep})")

        vae_samples = vae.generate(n_generate, DEVICE)
        out_path    = f"{prefix}_vae_samples.pt"
        torch.save(vae_samples, out_path)
        print(f"[VAE] Generated {vae_samples.shape[0]} samples → {out_path}\n")


    if run_wgan:
        print("=" * 40, "\n  WGAN-GP\n" + "=" * 40)
        gen         = Generator(base=cfg["wgan_gen_base"]).to(DEVICE)
        disc        = Discriminator(base=cfg["wgan_disc_base"]).to(DEVICE)
        start_epoch = 0
        wgan_logger = MetricsLogger(args.mode, "wgan", args.metrics_dir)

        if args.resume and args.model == "wgan":
            ck = torch.load(args.resume, map_location=DEVICE, weights_only=False)
            gen.load_state_dict(ck["gen"])
            disc.load_state_dict(ck["disc"])
            start_epoch = ck.get("epoch", 0)
            print(f"[GAN] Resumed from epoch {start_epoch}")

        if not args.generate_only:
            train_wgan_gp(
                gen, disc, covid_loader, real_covid_4d, extractor, cfg,
                start_epoch=start_epoch, ckpt_dir=args.ckpt_dir,
                adaptive=adaptive, logger=wgan_logger,
            )
            torch.save(
                gen.state_dict(),
                os.path.join(args.ckpt_dir, f"wgan_gen_final_{args.mode}.pth"),
            )

        best_path = os.path.join(args.ckpt_dir, "wgan_best.pth")
        if _load_best_or_warn(best_path, "GAN"):
            ck = torch.load(best_path, map_location=DEVICE, weights_only=False)
            gen.load_state_dict(ck["gen"])
            print(f"[GAN] Generating from BEST checkpoint (epoch {ck.get('epoch')})")

        gan_samples = generate_gan(gen, n_generate, DEVICE)
        out_path    = f"{prefix}_gan_samples.pt"
        torch.save(gan_samples, out_path)
        print(f"[GAN] Generated {gan_samples.shape[0]} samples → {out_path}\n")


    if run_diff:
        print("=" * 40, "\n  Diffusion\n" + "=" * 40)
        diff        = Diffusion(UNet()).to(DEVICE)
        start_epoch = 0
        diff_logger = MetricsLogger(args.mode, "diffusion", args.metrics_dir)

        if args.resume and args.model == "diffusion":
            start_epoch = diff.load(args.resume, device=DEVICE)

        if not args.generate_only:
            train_diffusion(
                diff, covid_loader, real_covid_4d, extractor, cfg,
                start_epoch=start_epoch, ckpt_dir=args.ckpt_dir,
                adaptive=adaptive, logger=diff_logger,
            )
            diff.save(os.path.join(args.ckpt_dir, f"diffusion_final_{args.mode}.pth"))

        best_path = os.path.join(args.ckpt_dir, "diffusion_best.pth")
        if _load_best_or_warn(best_path, "Diff"):
            ep = diff.load(best_path, device=DEVICE)
            print(f"[Diff] Generating from BEST checkpoint (epoch {ep})")

        diff_samples = diff.generate(n_generate, DEVICE, mini_bs=cfg["diff_mini_bs"])
        out_path     = f"{prefix}_diffusion_samples.pt"
        torch.save(diff_samples, out_path)
        print(f"[Diff] Generated {diff_samples.shape[0]} samples → {out_path}\n")

    print("[Done] All requested models processed.")


if __name__ == "__main__":
    main()
