import torch
import torch.nn as nn
from config import DEVICE, LATENT_DIM, MEL_H, MEL_W

class Generator(nn.Module):

    def __init__(self, latent_dim: int = LATENT_DIM, base: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base * 8 * 8 * 6),
            nn.BatchNorm1d(base * 8 * 8 * 6),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList([
            self._upblock(base * 8, base * 4),
            self._upblock(base * 4, base * 2),
            self._upblock(base * 2, base),
            self._upblock(base,     base // 2),
        ])
        self.out = nn.Sequential(
            nn.Conv2d(base // 2, 1, 3, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _upblock(ic: int, oc: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(ic, oc, 4, stride=2, padding=1),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.size(0), -1, 8, 6)
        for block in self.blocks:
            x = block(x)
        # Crop to exact mel shape
        return self.out(x)[:, :, :MEL_H, :MEL_W]


class Discriminator(nn.Module):

    def __init__(self, base: int = 64) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            self._downblock(1,        base),
            self._downblock(base,     base * 2),
            self._downblock(base * 2, base * 4),
            self._downblock(base * 4, base * 8),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(base * 8, 1)

    @staticmethod
    def _downblock(ic: int, oc: int) -> nn.Sequential:
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ic, oc, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(oc, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.fc(self.pool(x).flatten(1))

def _gradient_penalty(
    critic: Discriminator,
    real:   torch.Tensor,
    fake:   torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    alpha  = torch.rand(real.size(0), 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_int  = critic(interp)
    grad   = torch.autograd.grad(
        d_int, interp,
        grad_outputs=torch.ones_like(d_int),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

@torch.no_grad()
def generate_gan(
    generator: Generator,
    n:         int,
    device:    torch.device = DEVICE,
) -> torch.Tensor:
    generator.eval()
    z   = torch.randn(n, generator.latent_dim, device=device)
    out = generator(z).cpu()
    generator.train()
    return out
