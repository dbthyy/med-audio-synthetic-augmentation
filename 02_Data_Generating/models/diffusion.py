import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, MEL_H, MEL_W

class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        freq   = math.log(10_000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -freq)
        emb    = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SelfAttention2d(nn.Module):
    def __init__(self, c: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.hd    = c // heads
        self.scale = self.hd ** -0.5
        self.qkv   = nn.Conv2d(c, c * 3, 1)
        self.proj  = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv        = self.qkv(x).chunk(3, dim=1)
        q, k, v    = [
            t.view(B, self.heads, self.hd, H * W).transpose(-2, -1)
            for t in qkv
        ]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out  = (attn.softmax(-1) @ v).transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(out)


class UNetBlock(nn.Module):
    def __init__(
        self,
        ic:       int,
        oc:       int,
        t_dim:    int,
        use_attn: bool = False,
    ) -> None:
        super().__init__()
        self.t_proj = nn.Linear(t_dim, oc)
        self.c1     = nn.Conv2d(ic, oc, 3, padding=1)
        self.n1     = nn.GroupNorm(8, oc)
        self.c2     = nn.Conv2d(oc, oc, 3, padding=1)
        self.n2     = nn.GroupNorm(8, oc)
        self.act    = nn.SiLU()
        self.skip   = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.attn   = SelfAttention2d(oc) if use_attn else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.n1(self.c1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.act(self.n2(self.c2(h)))
        if self.attn is not None:
            h = self.attn(h)
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_c:  int = 1,
        out_c: int = 1,
        base:  int = 64,
        t_dim: int = 256,
    ) -> None:
        super().__init__()

        # Time MLP
        self.t_mlp = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        self.in_conv = nn.Conv2d(in_c, base, 3, padding=1)
        ch = [base, base*2, base*4, base*8, base*8]

        # Encoder
        self.enc  = nn.ModuleList([
            UNetBlock(ch[i-1] if i else base, ch[i], t_dim, use_attn=(i >= 3))
            for i in range(5)
        ])
        self.down = nn.ModuleList([
            nn.Conv2d(ch[i], ch[i], 3, stride=2, padding=1) for i in range(4)
        ])

        # Decoder (reverse channel list)
        rch      = list(reversed(ch))
        self.dec = nn.ModuleList()
        self.up  = nn.ModuleList()
        for i in range(5):
            ic = rch[i] if i == 0 else rch[i] + rch[i - 1]
            self.dec.append(UNetBlock(ic, rch[i], t_dim, use_attn=(i <= 1)))
            if i < 4:
                self.up.append(nn.Conv2d(rch[i], rch[i], 3, padding=1))

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, out_c, 3, padding=1),
        )
        self._sizes: list = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(t)
        x     = self.in_conv(x)
        skips = []
        self._sizes = []

        for i, blk in enumerate(self.enc):
            x = blk(x, t_emb)
            skips.append(x)
            self._sizes.append(x.shape[2:])
            if i < 4:
                x = self.down[i](x)

        x = self.dec[0](x, t_emb)
        for i in range(1, 5):
            x  = F.interpolate(x, size=self._sizes[4 - i], mode="nearest")
            x  = self.up[i - 1](x)
            sk = skips[4 - i]
            if x.shape[2:] != sk.shape[2:]:
                x = F.interpolate(x, size=sk.shape[2:], mode="nearest")
            x = torch.cat([x, sk], dim=1)
            x = self.dec[i](x, t_emb)

        return self.out_conv(x)


class Diffusion(nn.Module):
    """
    DDPM wrapper around a UNet.

    Parameters
    ----------
    unet    : UNet instance
    T       : total diffusion timesteps
    b_start : starting β value
    b_end   : ending β value
    """

    def __init__(
        self,
        unet:    UNet,
        T:       int   = 1000,
        b_start: float = 1e-4,
        b_end:   float = 0.02,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.T    = T

        betas = torch.linspace(b_start, b_end, T)
        alphas  = 1.0 - betas
        acp     = torch.cumprod(alphas, 0)

        self.register_buffer("betas",       betas)
        self.register_buffer("acp",         acp)
        self.register_buffer("sqrt_acp",    acp.sqrt())
        self.register_buffer("sqrt_1m_acp", (1 - acp).sqrt())


    def _extract(
        self,
        a:     torch.Tensor,
        t:     torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        return a.gather(-1, t).reshape(t.shape[0], *([1] * (len(shape) - 1)))


    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Sample a random timestep, add noise, predict it, return MSE loss.
        """
        t     = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        xt    = (
            self._extract(self.sqrt_acp,    t, x0.shape) * x0 +
            self._extract(self.sqrt_1m_acp, t, x0.shape) * noise
        )
        return F.mse_loss(self.unet(xt, t), noise)


    @torch.no_grad()
    def generate(
        self,
        n:       int,
        device:  torch.device = DEVICE,
        mini_bs: int          = 8,
    ) -> torch.Tensor:
        
        self.unet.eval()
        all_out: list[torch.Tensor] = []

        for s in range(0, n, mini_bs):
            bs  = min(mini_bs, n - s)
            img = torch.randn(bs, 1, MEL_H, MEL_W, device=device)

            for i in reversed(range(self.T)):
                t       = torch.full((bs,), i, device=device, dtype=torch.long)
                beta_t  = self._extract(self.betas,                t, img.shape)
                s1m_acp = self._extract(self.sqrt_1m_acp,          t, img.shape)
                sqrt_a  = self._extract((1 - self.betas).sqrt(),    t, img.shape)

                pred = self.unet(img, t)
                img  = (1 / sqrt_a) * (img - (beta_t / s1m_acp) * pred)
                if i > 0:
                    img = img + beta_t.sqrt() * torch.randn_like(img)
                if i % 200 == 0:
                    torch.cuda.empty_cache()

            img = (img.clamp(-1, 1) + 1) / 2
            all_out.append(img.cpu())

        self.unet.train()
        return torch.cat(all_out, 0)

    def save(self, path: str, epoch: int | None = None, opt=None) -> None:
        state: dict = {"unet": self.unet.state_dict()}
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
        self.unet.load_state_dict(ck["unet"])
        if opt and "opt" in ck:
            opt.load_state_dict(ck["opt"])
        epoch = ck.get("epoch", 0)
        print(f"[Diff] Resumed from epoch {epoch}")
        return epoch
