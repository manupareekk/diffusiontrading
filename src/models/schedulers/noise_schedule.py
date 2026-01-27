"""
Noise schedules for diffusion models.

Defines how noise is added during the forward diffusion process.
The schedule determines the variance of noise at each timestep.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        self._betas: Optional[torch.Tensor] = None
        self._alphas: Optional[torch.Tensor] = None
        self._alphas_cumprod: Optional[torch.Tensor] = None

    @abstractmethod
    def _compute_betas(self) -> torch.Tensor:
        """Compute beta values for each timestep."""
        pass

    def get_betas(self) -> torch.Tensor:
        """Get beta schedule (variance at each step)."""
        if self._betas is None:
            self._betas = self._compute_betas()
        return self._betas

    def get_alphas(self) -> torch.Tensor:
        """Get alpha schedule (1 - beta)."""
        if self._alphas is None:
            self._alphas = 1.0 - self.get_betas()
        return self._alphas

    def get_alphas_cumprod(self) -> torch.Tensor:
        """Get cumulative product of alphas."""
        if self._alphas_cumprod is None:
            self._alphas_cumprod = torch.cumprod(self.get_alphas(), dim=0)
        return self._alphas_cumprod

    def get_sqrt_alphas_cumprod(self) -> torch.Tensor:
        """Get sqrt of cumulative product of alphas."""
        return torch.sqrt(self.get_alphas_cumprod())

    def get_sqrt_one_minus_alphas_cumprod(self) -> torch.Tensor:
        """Get sqrt of (1 - cumulative product of alphas)."""
        return torch.sqrt(1.0 - self.get_alphas_cumprod())

    def get_posterior_variance(self) -> torch.Tensor:
        """
        Compute posterior variance for reverse process.

        beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        """
        alphas_cumprod = self.get_alphas_cumprod()
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        betas = self.get_betas()

        return betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def to(self, device: torch.device) -> "NoiseSchedule":
        """Move all tensors to specified device."""
        if self._betas is not None:
            self._betas = self._betas.to(device)
        if self._alphas is not None:
            self._alphas = self._alphas.to(device)
        if self._alphas_cumprod is not None:
            self._alphas_cumprod = self._alphas_cumprod.to(device)
        return self


class LinearNoiseSchedule(NoiseSchedule):
    """
    Linear noise schedule.

    Beta increases linearly from beta_start to beta_end.
    This is the original schedule used in DDPM (Ho et al., 2020).
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end

    def _compute_betas(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)


class CosineNoiseSchedule(NoiseSchedule):
    """
    Cosine noise schedule.

    Provides a smoother noise schedule that often works better
    for images and other data. From Nichol & Dhariwal (2021).

    The schedule is designed so that alpha_bar follows a cosine curve.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        s: float = 0.008,  # Small offset to prevent beta from being too small
        max_beta: float = 0.999,
    ):
        super().__init__(num_timesteps)
        self.s = s
        self.max_beta = max_beta

    def _compute_betas(self) -> torch.Tensor:
        steps = self.num_timesteps + 1
        t = torch.linspace(0, self.num_timesteps, steps)

        # Cosine schedule for alpha_bar
        f_t = torch.cos(((t / self.num_timesteps) + self.s) / (1 + self.s) * np.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]

        # Compute betas from alphas_cumprod
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        # Clip betas to reasonable range
        return torch.clamp(betas, 0.0001, self.max_beta)


class SigmoidNoiseSchedule(NoiseSchedule):
    """
    Sigmoid noise schedule.

    Another smooth schedule option that can be tuned with start/end parameters.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        start: float = -3,
        end: float = 3,
        tau: float = 1.0,
    ):
        super().__init__(num_timesteps)
        self.start = start
        self.end = end
        self.tau = tau

    def _compute_betas(self) -> torch.Tensor:
        t = torch.linspace(0, 1, self.num_timesteps)
        v_start = torch.sigmoid(torch.tensor(self.start / self.tau))
        v_end = torch.sigmoid(torch.tensor(self.end / self.tau))

        alphas_cumprod = torch.sigmoid((self.start + (self.end - self.start) * t) / self.tau)
        alphas_cumprod = (v_end - alphas_cumprod) / (v_end - v_start)

        # Compute betas
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        betas = 1 - alphas_cumprod / alphas_cumprod_prev

        return torch.clamp(betas, 0.0001, 0.999)


class QuadraticNoiseSchedule(NoiseSchedule):
    """
    Quadratic noise schedule.

    Beta increases quadratically, providing a slower start
    and faster end compared to linear.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end

    def _compute_betas(self) -> torch.Tensor:
        return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_timesteps) ** 2


class NoiseScheduleModule(torch.nn.Module):
    """
    PyTorch module wrapper for noise schedules.

    Registers schedule parameters as buffers so they move with the model
    to the correct device automatically.
    """

    def __init__(self, schedule: NoiseSchedule):
        """
        Initialize the module.

        Args:
            schedule: A NoiseSchedule instance
        """
        super().__init__()
        self.num_timesteps = schedule.num_timesteps

        # Precompute and register all schedule parameters as buffers
        betas = schedule.get_betas()
        alphas = schedule.get_alphas()
        alphas_cumprod = schedule.get_alphas_cumprod()
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # For forward diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # For reverse diffusion p(x_{t-1} | x_t)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance for reverse process
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))

        # Posterior mean coefficients
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def extract(self, tensor: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """
        Extract values from tensor at timesteps t and reshape for broadcasting.

        Args:
            tensor: 1D tensor of values indexed by timestep
            t: Batch of timestep indices, shape (batch,)
            shape: Target shape for broadcasting

        Returns:
            Extracted values reshaped to (batch, 1, 1, ...) for broadcasting
        """
        batch_size = t.shape[0]
        out = tensor.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


def create_noise_schedule(
    schedule_type: str,
    num_timesteps: int = 1000,
    as_module: bool = False,
    **kwargs
) -> NoiseSchedule:
    """
    Factory function to create noise schedules.

    Args:
        schedule_type: Type of schedule ('linear', 'cosine', 'sigmoid', 'quadratic')
        num_timesteps: Number of diffusion timesteps
        as_module: If True, wrap in NoiseScheduleModule for device handling
        **kwargs: Additional arguments for the specific schedule

    Returns:
        NoiseSchedule instance (or NoiseScheduleModule if as_module=True)
    """
    schedules = {
        "linear": LinearNoiseSchedule,
        "cosine": CosineNoiseSchedule,
        "sigmoid": SigmoidNoiseSchedule,
        "quadratic": QuadraticNoiseSchedule,
    }

    if schedule_type not in schedules:
        raise ValueError(f"Unknown schedule type: {schedule_type}. Available: {list(schedules.keys())}")

    schedule = schedules[schedule_type](num_timesteps=num_timesteps, **kwargs)

    if as_module:
        return NoiseScheduleModule(schedule)

    return schedule
