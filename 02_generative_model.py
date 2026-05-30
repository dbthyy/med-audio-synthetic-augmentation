"""
02_generative_models.py  (v2 — resume-aware)
=============================================
Định nghĩa và training 3 mô hình sinh:
  • VAE       — Variational Autoencoder
  • WGAN-GP   — Wasserstein GAN + Gradient Penalty
  • Diffusion — DDPM với U-Net backbone

Cách dùng
---------
# Chạy toàn bộ (train cả 3, không resume)
python 02_generative_model.py

# Chỉ train một model
python 02_generative_model.py --model vae
python 02_generative_model.py --model wgan
python 02_generative_model.py --model diffusion

# Resume từ checkpoint cụ thể
python 02_generative_model.py --model vae       --resume vae_epoch_200.pth
python 02_generative_model.py --model wgan      --resume wgan_epoch_300.pth
python 02_generative_model.py --model diffusion --resume diffusion_epoch_200.pth

# Chỉ generate samples (không train lại)
python 02_generative_model.py --model vae       --generate_only --resume vae_epoch_200.pth
python 02_generative_model.py --model wgan      --generate_only --resume wgan_epoch_300.pth
python 02_generative_model.py --model diffusion --generate_only --resume diffusion_epoch_200.pth
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm.notebook import tqdm

# ─────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────
LATENT_DIM  = 128
MEL_H       = 128
MEL_W       = 94
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAE_EPOCHS       = 200
WGAN_EPOCHS      = 300
DIFFUSION_EPOCHS = 400
SAVE_EVERY       = 50


# ─────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y.long() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].permute(2, 0, 1), self.y[idx]   # (H,W,1)→(1,H,W)


class SyntheticDataset(Dataset):
    def __init__(self, X: torch.Tensor):
        self.X = X
        self.y = torch.ones(len(X), dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_augmented_loader(real_dataset: AudioDataset,
                          fake_X: torch.Tensor,
                          batch_size: int = BATCH_SIZE) -> DataLoader:
    synthetic = SyntheticDataset(fake_X)
    combined  = ConcatDataset([real_dataset, synthetic])
    return DataLoader(combined, batch_size=batch_size, shuffle=True)


# ─────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────
class _VAEConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
        )


class _VAEDeconvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(out_c, out_c, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
        )


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.blocks = nn.ModuleList([
            _VAEConvBlock(1, 64), _VAEConvBlock(64, 128),
            _VAEConvBlock(128, 256), _VAEConvBlock(256, 512),
        ])
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_lv = nn.Linear(512, latent_dim)

    def forward(self, x):
        for b in self.blocks: x = b(x)
        x = self.pool(x).flatten(1)
        return self.fc_mu(x), self.fc_lv(x)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 6)
        self.blocks = nn.ModuleList([
            _VAEDeconvBlock(512, 256), _VAEDeconvBlock(256, 128),
            _VAEDeconvBlock(128, 64),  _VAEDeconvBlock(64, 32),
        ])
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1,  3, padding=1), nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d((MEL_H, MEL_W))

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 8, 6)
        for b in self.blocks: x = b(x)
        return self.pool(self.final(x))


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder    = VAEEncoder(latent_dim)
        self.decoder    = VAEDecoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def forward(self, x):
        mu, lv = self.encoder(x)
        return self.decoder(self.reparameterize(mu, lv)), mu, lv

    def loss(self, recon, x, mu, lv, beta=0.1):
        recon_l = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl_l    = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
        return recon_l + beta * kl_l, recon_l, kl_l

    @torch.no_grad()
    def generate(self, n: int, device=DEVICE) -> torch.Tensor:
        self.eval()
        z   = torch.randn(n, self.latent_dim, device=device)
        out = self.decoder(z).cpu()
        self.train()
        return out   # (N, 1, H, W)

    def save(self, path, epoch=None, opt=None):
        s = {"enc": self.encoder.state_dict(), "dec": self.decoder.state_dict()}
        if epoch is not None: s["epoch"] = epoch
        if opt   is not None: s["opt"]   = opt.state_dict()
        torch.save(s, path)

    def load(self, path, device=DEVICE, opt=None):
        ck = torch.load(path, map_location=device)
        self.encoder.load_state_dict(ck["enc"])
        self.decoder.load_state_dict(ck["dec"])
        if opt and "opt" in ck: opt.load_state_dict(ck["opt"])
        self.to(device)
        epoch = ck.get("epoch", 0)
        print(f"[VAE] Resumed from epoch {epoch}")
        return epoch


def train_vae(model: VAE, loader: DataLoader,
              start_epoch: int = 0,
              epochs: int = VAE_EPOCHS,
              lr: float = 1e-4, beta: float = 0.1,
              device=DEVICE, save_every: int = SAVE_EVERY) -> VAE:
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # restore optimizer if resuming
    if start_epoch > 0:
        ckpt_path = f"vae_epoch_{start_epoch}.pth"
        ck = torch.load(ckpt_path, map_location=device)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            print(f"[VAE] Optimizer restored from {ckpt_path}")

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        total = recon_t = kl_t = 0

        for (x, _) in tqdm(loader, desc=f"VAE {epoch}/{epochs}", leave=False):
            x = x.to(device)
            opt.zero_grad()
            recon, mu, lv = model(x)
            loss, rl, kl  = model.loss(recon, x, mu, lv, beta)
            loss.backward(); opt.step()
            total += loss.item(); recon_t += rl.item(); kl_t += kl.item()

        n = len(loader)
        print(f"[VAE] E{epoch:03d} loss={total/n:.4f} recon={recon_t/n:.4f} kl={kl_t/n:.4f}")

        if epoch % save_every == 0:
            model.save(f"vae_epoch_{epoch}.pth", epoch, opt)
            print(f"[VAE] Checkpoint saved → vae_epoch_{epoch}.pth")

    return model


# ─────────────────────────────────────────────
# WGAN-GP
# ─────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, base=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base * 8 * 8 * 6),
            nn.BatchNorm1d(base * 8 * 8 * 6),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList([
            self._blk(base * 8, base * 4), self._blk(base * 4, base * 2),
            self._blk(base * 2, base),     self._blk(base, base // 2),
        ])
        self.out        = nn.Sequential(nn.Conv2d(base // 2, 1, 3, padding=1), nn.Tanh())
        self.latent_dim = latent_dim

    @staticmethod
    def _blk(ic, oc):
        return nn.Sequential(
            nn.ConvTranspose2d(ic, oc, 4, stride=2, padding=1),
            nn.BatchNorm2d(oc), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 8, 6)
        for b in self.blocks: x = b(x)
        return self.out(x)[:, :, :MEL_H, :MEL_W]


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            self._blk(1,       base),
            self._blk(base,    base * 2),
            self._blk(base*2,  base * 4),
            self._blk(base*4,  base * 8),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(base * 8, 1)

    @staticmethod
    def _blk(ic, oc):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ic, oc, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(oc, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.fc(self.pool(x).flatten(1))


def _gradient_penalty(critic, real, fake, device):
    alpha  = torch.rand(real.size(0), 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_int  = critic(interp)
    grad   = torch.autograd.grad(
        d_int, interp,
        grad_outputs=torch.ones_like(d_int),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(generator: Generator, critic: Discriminator,
                  loader: DataLoader,
                  start_epoch: int = 0,
                  epochs: int = WGAN_EPOCHS,
                  lr: float = 5e-5, lambda_gp: int = 10,
                  n_critic: int = 5,
                  device=DEVICE, save_every: int = SAVE_EVERY):
    generator.to(device); critic.to(device)
    opt_G = optim.RMSprop(generator.parameters(), lr=lr)
    opt_D = optim.RMSprop(critic.parameters(),    lr=lr)

    # restore optimizers if resuming
    if start_epoch > 0:
        ckpt_path = f"wgan_epoch_{start_epoch}.pth"
        ck = torch.load(ckpt_path, map_location=device)
        if "opt_G" in ck: opt_G.load_state_dict(ck["opt_G"])
        if "opt_D" in ck: opt_D.load_state_dict(ck["opt_D"])
        print(f"[GAN] Optimizers restored from {ckpt_path}")

    d_loss = g_loss = torch.tensor(0.0)

    for epoch in range(start_epoch + 1, epochs + 1):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            bs   = real.size(0)

            # ── Critic step ──
            z    = torch.randn(bs, generator.latent_dim, device=device)
            fake = generator(z).detach()
            gp   = _gradient_penalty(critic, real, fake, device)
            d_loss = critic(fake).mean() - critic(real).mean() + lambda_gp * gp
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ── Generator step ──
            if i % n_critic == 0:
                z      = torch.randn(bs, generator.latent_dim, device=device)
                g_loss = -critic(generator(z)).mean()
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        print(f"[GAN] E{epoch:03d} D={d_loss.item():.4f} G={g_loss.item():.4f}")

        if epoch % save_every == 0:
            torch.save({
                "gen":   generator.state_dict(),
                "disc":  critic.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "epoch": epoch,
            }, f"wgan_epoch_{epoch}.pth")
            print(f"[GAN] Checkpoint saved → wgan_epoch_{epoch}.pth")

    return generator, critic


@torch.no_grad()
def generate_gan(generator: Generator, n: int, device=DEVICE) -> torch.Tensor:
    generator.eval()
    out = generator(torch.randn(n, generator.latent_dim, device=device)).cpu()
    generator.train()
    return out   # (N, 1, H, W)


# ─────────────────────────────────────────────
# DIFFUSION (DDPM + U-Net)
# ─────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SelfAttention2d(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.hd    = c // heads
        self.scale = self.hd ** -0.5
        self.qkv   = nn.Conv2d(c, c * 3, 1)
        self.proj  = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = [t.view(B, self.heads, self.hd, H * W).transpose(-2, -1) for t in qkv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out  = (attn.softmax(-1) @ v).transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(out)


class UNetBlock(nn.Module):
    def __init__(self, ic, oc, t_dim, use_attn=False):
        super().__init__()
        self.t_proj = nn.Linear(t_dim, oc)
        self.c1     = nn.Conv2d(ic, oc, 3, padding=1)
        self.n1     = nn.GroupNorm(8, oc)
        self.c2     = nn.Conv2d(oc, oc, 3, padding=1)
        self.n2     = nn.GroupNorm(8, oc)
        self.act    = nn.SiLU()
        self.skip   = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.attn   = SelfAttention2d(oc) if use_attn else None

    def forward(self, x, t_emb):
        h = self.act(self.n1(self.c1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.act(self.n2(self.c2(h)))
        if self.attn: h = self.attn(h)
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, base=64, t_dim=256):
        super().__init__()
        self.t_mlp = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        self.in_conv = nn.Conv2d(in_c, base, 3, padding=1)
        ch = [base, base*2, base*4, base*8, base*8]

        self.enc  = nn.ModuleList([
            UNetBlock(ch[i-1] if i else base, ch[i], t_dim, use_attn=(i == 4))
            for i in range(5)
        ])
        self.down = nn.ModuleList([
            nn.Conv2d(ch[i], ch[i], 3, stride=2, padding=1) for i in range(4)
        ])

        rch      = list(reversed(ch))
        self.dec = nn.ModuleList()
        self.up  = nn.ModuleList()
        for i in range(5):
            ic = rch[i] if i == 0 else rch[i] + rch[i-1]
            self.dec.append(UNetBlock(ic, rch[i], t_dim, use_attn=(i == 0)))
            if i < 4:
                self.up.append(nn.Conv2d(rch[i], rch[i], 3, padding=1))

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, out_c, 3, padding=1),
        )
        self._sizes = []

    def forward(self, x, t):
        t_emb = self.t_mlp(t)
        x     = self.in_conv(x)
        skips = []
        self._sizes = []

        for i, blk in enumerate(self.enc):
            x = blk(x, t_emb)
            skips.append(x)
            self._sizes.append(x.shape[2:])
            if i < 4: x = self.down[i](x)

        x = self.dec[0](x, t_emb)
        for i in range(1, 5):
            x  = F.interpolate(x, size=self._sizes[4 - i], mode="nearest")
            x  = self.up[i-1](x)
            sk = skips[4 - i]
            if x.shape[2:] != sk.shape[2:]:
                x = F.interpolate(x, size=sk.shape[2:], mode="nearest")
            x = torch.cat([x, sk], dim=1)
            x = self.dec[i](x, t_emb)

        return self.out_conv(x)


class Diffusion(nn.Module):
    def __init__(self, unet: UNet, T=1000, b_start=1e-4, b_end=0.02):
        super().__init__()
        self.unet = unet
        self.T    = T
        betas     = torch.linspace(b_start, b_end, T)
        alphas    = 1.0 - betas
        acp       = torch.cumprod(alphas, 0)
        self.register_buffer("betas",        betas)
        self.register_buffer("acp",          acp)
        self.register_buffer("sqrt_acp",     acp.sqrt())
        self.register_buffer("sqrt_1m_acp",  (1 - acp).sqrt())

    def _extract(self, a, t, shape):
        return a.gather(-1, t).reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def forward(self, x0):
        t     = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        xt    = (self._extract(self.sqrt_acp, t, x0.shape) * x0 +
                 self._extract(self.sqrt_1m_acp, t, x0.shape) * noise)
        return F.mse_loss(self.unet(xt, t), noise)

    @torch.no_grad()
    def generate(self, n: int, device=DEVICE, mini_bs=8) -> torch.Tensor:
        self.unet.eval()
        all_out = []
        for s in range(0, n, mini_bs):
            bs  = min(mini_bs, n - s)
            img = torch.randn(bs, 1, MEL_H, MEL_W, device=device)
            for i in reversed(range(self.T)):
                t        = torch.full((bs,), i, device=device, dtype=torch.long)
                betas_t  = self._extract(self.betas,                t, img.shape)
                s1m_acp  = self._extract(self.sqrt_1m_acp,          t, img.shape)
                sqrt_a   = self._extract((1 - self.betas).sqrt(),    t, img.shape)
                pred     = self.unet(img, t)
                img      = (1 / sqrt_a) * (img - (betas_t / s1m_acp) * pred)
                if i > 0:
                    img = img + betas_t.sqrt() * torch.randn_like(img)
                if i % 200 == 0:
                    torch.cuda.empty_cache()
            img = (img.clamp(-1, 1) + 1) / 2
            all_out.append(img.cpu())
        self.unet.train()
        return torch.cat(all_out, 0)

    def save(self, path, epoch=None, opt=None):
        s = {"unet": self.unet.state_dict()}
        if epoch is not None: s["epoch"] = epoch
        if opt   is not None: s["opt"]   = opt.state_dict()
        torch.save(s, path)

    def load(self, path, device=DEVICE, opt=None):
        ck = torch.load(path, map_location=device)
        self.unet.load_state_dict(ck["unet"])
        if opt and "opt" in ck: opt.load_state_dict(ck["opt"])
        epoch = ck.get("epoch", 0)
        print(f"[Diff] Resumed from epoch {epoch}")
        return epoch


def train_diffusion(model: Diffusion, loader: DataLoader,
                    start_epoch: int = 0,
                    epochs: int = DIFFUSION_EPOCHS,
                    lr: float = 1e-4, w_decay: float = 0.01,
                    device=DEVICE, save_every: int = SAVE_EVERY) -> Diffusion:
    model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=w_decay)

    # restore optimizer if resuming
    if start_epoch > 0:
        ckpt_path = f"diffusion_epoch_{start_epoch}.pth"
        ck = torch.load(ckpt_path, map_location=device)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            print(f"[Diff] Optimizer restored from {ckpt_path}")

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        total = 0
        for (x, _) in tqdm(loader, desc=f"Diff {epoch}/{epochs}", leave=True):
            x = x.to(device)
            opt.zero_grad()
            loss = model(x)
            loss.backward(); opt.step()
            total += loss.item()
        print(f"[Diff] E{epoch:03d} loss={total/len(loader):.4f}")

        if epoch % save_every == 0:
            model.save(f"diffusion_epoch_{epoch}.pth", epoch, opt)
            print(f"[Diff] Checkpoint saved → diffusion_epoch_{epoch}.pth")

    return model


# ─────────────────────────────────────────────
# HELPERS — load models từ checkpoint
# ─────────────────────────────────────────────
def load_vae(path: str, device=DEVICE) -> VAE:
    model = VAE().to(device)
    model.load(path, device=device)
    return model


def load_wgan(path: str, device=DEVICE):
    ck  = torch.load(path, map_location=device)
    gen = Generator().to(device)
    gen.load_state_dict(ck["gen"])
    disc = Discriminator().to(device)
    disc.load_state_dict(ck["disc"])
    epoch = ck.get("epoch", 0)
    print(f"[GAN] Loaded from epoch {epoch}")
    return gen, disc, epoch


def load_diffusion(path: str, device=DEVICE) -> tuple:
    model = Diffusion(UNet()).to(device)
    epoch = model.load(path, device=device)
    return model, epoch


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  choices=["vae", "wgan", "diffusion", "all"],
                   default="all", help="Which model to train")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from  e.g. diffusion_epoch_200.pth")
    p.add_argument("--generate_only", action="store_true",
                   help="Skip training, only generate samples from checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Load dataset ──
    data        = torch.load("dataset.pt", map_location="cpu", weights_only=False)
    full_train  = AudioDataset(data["X_train"], data["y_train"])
    covid_mask  = data["y_train"] == 1
    covid_X     = data["X_train"][covid_mask]
    covid_y     = data["y_train"][covid_mask]
    covid_ds    = AudioDataset(covid_X, covid_y)
    covid_loader = DataLoader(covid_ds, BATCH_SIZE, shuffle=True)

    n_generate  = int(covid_mask.sum().item() * 0.5)
    print(f"[Info] Will generate {n_generate} synthetic COVID samples per model")
    print(f"[Info] Running on {DEVICE}")

    run_vae  = args.model in ("vae",  "all")
    run_wgan = args.model in ("wgan", "all")
    run_diff = args.model in ("diffusion", "all")

    # ══════════════════════════════════════════
    # VAE
    # ══════════════════════════════════════════
    if run_vae:
        print("\n=== VAE ===")
        vae         = VAE().to(DEVICE)
        start_epoch = 0

        if args.resume and args.model in ("vae", "all"):
            resume_path = args.resume if args.model == "vae" else f"vae_epoch_{VAE_EPOCHS}.pth"
            start_epoch = vae.load(resume_path, device=DEVICE)

        if not args.generate_only:
            train_vae(vae, covid_loader,
                      start_epoch=start_epoch,
                      epochs=VAE_EPOCHS)
            vae.save("vae_final.pth")

        vae_samples = vae.generate(n_generate, DEVICE)
        torch.save(vae_samples, "vae_samples.pt")
        print(f"[VAE] Generated {vae_samples.shape[0]} samples → vae_samples.pt")

    # ══════════════════════════════════════════
    # WGAN-GP
    # ══════════════════════════════════════════
    if run_wgan:
        print("\n=== WGAN-GP ===")
        gen         = Generator().to(DEVICE)
        disc        = Discriminator().to(DEVICE)
        start_epoch = 0

        if args.resume and args.model in ("wgan", "all"):
            resume_path = args.resume if args.model == "wgan" else f"wgan_epoch_{WGAN_EPOCHS}.pth"
            ck          = torch.load(resume_path, map_location=DEVICE)
            gen.load_state_dict(ck["gen"])
            disc.load_state_dict(ck["disc"])
            start_epoch = ck.get("epoch", 0)
            print(f"[GAN] Resumed from epoch {start_epoch}")

        if not args.generate_only:
            gen, disc = train_wgan_gp(gen, disc, covid_loader,
                                      start_epoch=start_epoch,
                                      epochs=WGAN_EPOCHS)
            torch.save(gen.state_dict(), "wgan_gen_final.pth")

        gan_samples = generate_gan(gen, n_generate, DEVICE)
        torch.save(gan_samples, "gan_samples.pt")
        print(f"[GAN] Generated {gan_samples.shape[0]} samples → gan_samples.pt")

    # ══════════════════════════════════════════
    # DIFFUSION
    # ══════════════════════════════════════════
    if run_diff:
        print("\n=== Diffusion ===")
        diff        = Diffusion(UNet()).to(DEVICE)
        start_epoch = 0

        if args.resume and args.model in ("diffusion", "all"):
            resume_path = args.resume if args.model == "diffusion" else None
            if resume_path:
                start_epoch = diff.load(resume_path, device=DEVICE)

        if not args.generate_only:
            train_diffusion(diff, covid_loader,
                            start_epoch=start_epoch,
                            epochs=DIFFUSION_EPOCHS)
            diff.save("diffusion_final.pth")

        diff_samples = diff.generate(n_generate, DEVICE)
        torch.save(diff_samples, "diffusion_samples.pt")
        print(f"[Diff] Generated {diff_samples.shape[0]} samples → diffusion_samples.pt")

    print("\n[Done] All requested models processed.")