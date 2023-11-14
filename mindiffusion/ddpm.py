import torch
import torch.nn as nn
from torch import Tensor


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        beta1, beta2 = betas
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = torch.linspace(beta1, beta2, n_T + 1, dtype=torch.float)
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = torch.rsqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        self.alpha_t: Tensor
        self.oneover_sqrta: Tensor  # 1/\sqrt{\alpha_t}
        self.sqrt_beta_t: Tensor  # \sqrt{\beta_t}
        self.alphabar_t: Tensor  # \bar{\alpha_t}
        self.sqrtab: Tensor  # \sqrt{\bar{\alpha_t}}
        self.sqrtmab: Tensor  # \sqrt{1-\bar{\alpha_t}}
        self.mab_over_sqrtmab: Tensor  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}

        self.register_buffer("alpha_t", alpha_t)  # \alpha_t
        self.register_buffer("oneover_sqrta", oneover_sqrta)  # 1/\sqrt{\alpha_t}
        self.register_buffer("sqrt_beta_t", sqrt_beta_t)  # \sqrt{\beta_t}
        self.register_buffer("alphabar_t", alphabar_t)  # \bar{\alpha_t}
        self.register_buffer("sqrtab", sqrtab)  # \sqrt{\bar{\alpha_t}}
        self.register_buffer("sqrtmab", sqrtmab)  # \sqrt{1-\bar{\alpha_t}}
        self.register_buffer("mab_over_sqrtmab", mab_over_sqrtmab_inv)  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: Tensor, eps: Tensor) -> Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],), device=x.device)
        # t ~ Uniform(0, n_T)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.eps_model(x_t, _ts / self.n_T)

    def sample(self, n_sample: int, size: tuple[int, ...]) -> Tensor:
        device = self.alpha_t.device
        x_i = torch.randn(n_sample, *size, device=device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size, device=device) if i > 1 else 0
            eps = self.eps_model(x_i, torch.tensor(i / self.n_T, device=device).expand(n_sample, 1))
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

        return x_i
