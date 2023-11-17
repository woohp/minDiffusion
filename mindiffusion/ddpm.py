import torch
import torch.nn as nn
from torch import Tensor


class DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, beta1: float, beta2: float, n_T: int):
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        betas = torch.linspace(beta1, beta2, n_T + 1, dtype=torch.float)

        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)

        self.sqrt_betas: list[float] = torch.sqrt(betas).tolist()
        self.sqrt_alphas_bar: Tensor
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.sqrt_one_minus_alphas_bar: Tensor
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1 - alphas_bar))
        self.one_over_sqrt_alphas: list[float] = torch.rsqrt(alphas).tolist()
        self.one_minus_alphas_over_sqrt_one_minus_alphas_bar: list[float] = (
            (1 - alphas) / torch.sqrt(1 - alphas_bar)
        ).tolist()

        self.n_T = n_T

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],), device=x.device)
        # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrt_alphas_bar[_ts, None, None, None] * x
            + self.sqrt_one_minus_alphas_bar[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.eps_model(x_t, _ts / self.n_T), eps

    def sample(self, n_sample: int, size: tuple[int, ...], device) -> Tensor:
        x_i = torch.randn(n_sample, *size, device=device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z: Tensor | int = torch.randn(n_sample, *size, device=device) if i > 1 else 0
            eps = self.eps_model(x_i, torch.tensor(i / self.n_T, device=device).expand(n_sample, 1))
            x_i = (
                self.one_over_sqrt_alphas[i] * (x_i - eps * self.one_minus_alphas_over_sqrt_one_minus_alphas_bar[i])
                + self.sqrt_betas[i] * z
            )

        return x_i
