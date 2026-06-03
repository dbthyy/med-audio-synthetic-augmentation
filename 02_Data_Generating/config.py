import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM: int = 128
MEL_H: int      = 128
MEL_W: int      = 94
BATCH_SIZE: int = 32

_CFG_TABLE: dict = {
    # VAE
    "reproduce_vae_beta":  0.1,
    "improve_vae_beta":    0.05,
    "vae_beta_min":        0.001,
    "vae_lr":              1e-4,
    "vae_epochs":          200,

    # WGAN-GP
    "reproduce_wgan_base": 64,
    "improve_wgan_base":   128,
    "wgan_lr":             5e-5,
    "wgan_lr_max":         2e-4,
    "wgan_epochs":         400,
    "wgan_n_critic":       5,
    "wgan_lambda_gp":      10,

    # Diffusion
    "diff_lr":             1e-4,
    "diff_epochs":         400,
    "diff_mini_bs":        8,
    "diff_save_every":     50,
    "diff_eval_every":     50,

    # Shared
    "aug_ratio":           0.5,
    "save_every":          25,
    "eval_every":          25,
}

def build_run_cfg(mode: str) -> dict:
    """Return a flat config dict for the requested training mode."""
    improve = (mode == "improve")
    return {
        "vae_beta":       _CFG_TABLE["improve_vae_beta"]   if improve else _CFG_TABLE["reproduce_vae_beta"],
        "vae_beta_min":   _CFG_TABLE["vae_beta_min"],
        "vae_lr":         _CFG_TABLE["vae_lr"],
        "vae_epochs":     _CFG_TABLE["vae_epochs"],
        "wgan_gen_base":  _CFG_TABLE["improve_wgan_base"]  if improve else _CFG_TABLE["reproduce_wgan_base"],
        "wgan_disc_base": _CFG_TABLE["improve_wgan_base"]  if improve else _CFG_TABLE["reproduce_wgan_base"],
        "wgan_lr":        _CFG_TABLE["wgan_lr"],
        "wgan_lr_max":    _CFG_TABLE["wgan_lr_max"],
        "wgan_epochs":    _CFG_TABLE["wgan_epochs"],
        "wgan_n_critic":  _CFG_TABLE["wgan_n_critic"],
        "wgan_lambda_gp": _CFG_TABLE["wgan_lambda_gp"],
        "diff_lr":        _CFG_TABLE["diff_lr"],
        "diff_epochs":    _CFG_TABLE["diff_epochs"],
        "diff_mini_bs":   _CFG_TABLE["diff_mini_bs"],
        "aug_ratio":      _CFG_TABLE["aug_ratio"],
        "save_every":     _CFG_TABLE["save_every"],
        "eval_every":     _CFG_TABLE["eval_every"],
        "diff_save_every": _CFG_TABLE["diff_save_every"],
        "diff_eval_every": _CFG_TABLE["diff_eval_every"],
    }
