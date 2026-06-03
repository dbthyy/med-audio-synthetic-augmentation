import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, LATENT_DIM, MEL_H, MEL_W

class _VAEConvBlock(nn.Sequential):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__(
            nn.Conv2d(in_c,  out_c, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
        )


class _VAEDeconvBlock(nn.Sequential):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(in_c,  out_c, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(out_c, out_c, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True),
        )


class VAEEncoder(nn.Module):

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _VAEConvBlock(1,   64),
            _VAEConvBlock(64,  128),
            _VAEConvBlock(128, 256),
            _VAEConvBlock(256, 512),
        ])
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_lv = nn.Linear(512, latent_dim)   # log-variance

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).flatten(1)
        return self.fc_mu(x), self.fc_lv(x)


class VAEDecoder(nn.Module):

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 6)
        self.blocks = nn.ModuleList([
            _VAEDeconvBlock(512, 256),
            _VAEDeconvBlock(256, 128),
            _VAEDeconvBlock(128, 64),
            _VAEDeconvBlock(64,  32),
        ])
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,  1, 3, padding=1), nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d((MEL_H, MEL_W))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 512, 8, 6)
        for block in self.blocks:
            x = block(x)
        return self.pool(self.final(x))


class VAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder    = VAEEncoder(latent_dim)
        self.decoder    = VAEDecoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def forward(self, x: torch.Tensor):
        mu, lv = self.encoder(x)
        return self.decoder(self.reparameterize(mu, lv)), mu, lv

    def loss(
        self,
        recon: torch.Tensor,
        x:     torch.Tensor,
        mu:    torch.Tensor,
        lv:    torch.Tensor,
        beta:  float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_l = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl_l    = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
        return recon_l + beta * kl_l, recon_l, kl_l

    @torch.no_grad()
    def generate(self, n: int, device: torch.device = DEVICE) -> torch.Tensor:
        """Sample *n* mel-spectrograms from the prior N(0, I)."""
        self.eval()
        z   = torch.randn(n, self.latent_dim, device=device)
        out = self.decoder(z).cpu()
        self.train()
        return out

    def save(self, path: str, epoch: int | None = None, opt=None) -> None:
        state: dict = {
            "enc": self.encoder.state_dict(),
            "dec": self.decoder.state_dict(),
        }
        if epoch is not None:
            state["epoch"] = epoch
        if opt is not None:
            state["opt"] = opt.state_dict()
        torch.save(state, path)

    def load(
        self,
        path:   str,
        device: torch.device = DEVICE,
        opt=None,
    ) -> int:
        ck = torch.load(path, map_location=device, weights_only=False)
        self.encoder.load_state_dict(ck["enc"])
        self.decoder.load_state_dict(ck["dec"])
        if opt and "opt" in ck:
            opt.load_state_dict(ck["opt"])
        self.to(device)
        epoch = ck.get("epoch", 0)
        print(f"[VAE] Resumed from epoch {epoch}")
        return epoch
