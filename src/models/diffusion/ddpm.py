"""
Denoising Diffusion Probabilistic Model (DDPM) implementation.

This is the core diffusion model for financial time series prediction.
Based on Ho et al. (2020) "Denoising Diffusion Probabilistic Models".
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schedulers.noise_schedule import NoiseSchedule, LinearNoiseSchedule


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.

    Forward process:
        q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)

    Reverse process (learned):
        p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)

    For financial time series, we operate on log returns rather than
    prices to ensure stationarity.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        noise_schedule: NoiseSchedule = None,
        num_timesteps: int = 1000,
        prediction_type: str = "epsilon",  # "epsilon", "x_start", or "v"
        loss_type: str = "mse",  # "mse" or "huber"
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        """
        Initialize DDPM.

        Args:
            denoiser: Neural network that predicts noise (or x_0 or v)
            noise_schedule: Noise schedule (default: linear)
            num_timesteps: Number of diffusion timesteps
            prediction_type: What the model predicts ("epsilon", "x_start", "v")
            loss_type: Loss function type
            clip_denoised: Whether to clip denoised values
            clip_range: Range for clipping
        """
        super().__init__()

        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range

        # Setup noise schedule
        if noise_schedule is None:
            noise_schedule = LinearNoiseSchedule(num_timesteps)
        else:
            # If schedule provided, use its timesteps
            num_timesteps = noise_schedule.num_timesteps
            
        self.noise_schedule = noise_schedule
        self.num_timesteps = num_timesteps

        # Register schedule parameters as buffers
        self._register_schedule_buffers()

    def _register_schedule_buffers(self):
        """Register noise schedule parameters as buffers for proper device handling."""
        betas = self.noise_schedule.get_betas()
        alphas = self.noise_schedule.get_alphas()
        alphas_cumprod = self.noise_schedule.get_alphas_cumprod()
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20))
        )

        # Posterior mean coefficients
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from a at indices t, broadcasting to x_shape."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t from q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_start: Clean data x_0, shape (batch, seq_len, features)
            t: Timesteps, shape (batch,)
            noise: Optional pre-generated noise

        Returns:
            Tuple of (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        x_t = sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

        return x_t, noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        """
        sqrt_recip_alphas_cumprod = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise

    def predict_noise_from_start(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_start: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from x_t and x_0."""
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        return (x_t - sqrt_alpha_cumprod * x_start) / sqrt_one_minus_alpha_cumprod

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).

        Returns mean and log variance.
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_variance

    def model_predictions(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model predictions for both noise and x_start.

        Returns:
            Tuple of (predicted_noise, predicted_x_start)
        """
        model_output = self.denoiser(x_t, t, condition)

        if self.prediction_type == "epsilon":
            pred_noise = model_output
            pred_x_start = self.predict_start_from_noise(x_t, t, pred_noise)

        elif self.prediction_type == "x_start":
            pred_x_start = model_output
            pred_noise = self.predict_noise_from_start(x_t, t, pred_x_start)

        elif self.prediction_type == "v":
            # v-prediction: v = sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x_0
            sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alpha_cumprod = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            pred_x_start = sqrt_alpha_cumprod * x_t - sqrt_one_minus_alpha_cumprod * model_output
            pred_noise = sqrt_one_minus_alpha_cumprod * x_t + sqrt_alpha_cumprod * model_output

        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Optionally clip x_start
        if self.clip_denoised:
            pred_x_start = pred_x_start.clamp(*self.clip_range)

        return pred_noise, pred_x_start

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).

        Returns:
            Tuple of (mean, variance, log_variance)
        """
        _, pred_x_start = self.model_predictions(x_t, t, condition)

        # Get posterior statistics
        model_mean, model_log_variance = self.q_posterior(pred_x_start, x_t, t)
        model_variance = torch.exp(model_log_variance)

        return model_mean, model_variance, model_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: sample x_{t-1} from p(x_{t-1} | x_t).
        """
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t, condition)

        noise = torch.randn_like(x_t)

        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using the full reverse diffusion process.

        Args:
            shape: Output shape (batch_size, seq_len, features)
            condition: Optional conditioning tensor
            return_trajectory: If True, return all intermediate steps

        Returns:
            Generated samples
        """
        device = next(self.parameters()).device
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        trajectory = [x] if return_trajectory else None

        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, condition)

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return torch.stack(trajectory, dim=1)

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling for faster inference.

        Args:
            shape: Output shape
            condition: Optional conditioning
            num_inference_steps: Number of denoising steps (can be less than training steps)
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)

        Returns:
            Generated samples
        """
        device = next(self.parameters()).device
        batch_size = shape[0]

        # Compute timesteps for DDIM
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, self.num_timesteps, step_ratio, device=device)
        timesteps = torch.flip(timesteps, [0])

        # Start from noise
        x = torch.randn(shape, device=device)

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Get predictions
            pred_noise, pred_x_start = self.model_predictions(x, t_batch, condition)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = torch.tensor(0, device=device)

            # Get alpha values
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            # DDIM update
            sigma = eta * torch.sqrt(
                (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) *
                (1 - alpha_cumprod / alpha_cumprod_prev)
            )

            pred_dir = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * pred_noise
            noise = torch.randn_like(x) if eta > 0 and t_prev > 0 else 0

            x = torch.sqrt(alpha_cumprod_prev) * pred_x_start + pred_dir + sigma * noise

        return x

    def compute_loss(
        self,
        x_start: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            x_start: Clean data
            condition: Optional conditioning
            noise: Optional pre-generated noise

        Returns:
            Loss value
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get noisy samples
        x_t, _ = self.q_sample(x_start, t, noise)

        # Get model prediction
        model_output = self.denoiser(x_t, t, condition)

        # Compute target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x_start":
            target = x_start
        elif self.prediction_type == "v":
            sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alpha_cumprod = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            target = sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * x_start
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Compute loss
        if self.loss_type == "mse":
            loss = F.mse_loss(model_output, target)
        elif self.loss_type == "huber":
            loss = F.huber_loss(model_output, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def forward(
        self,
        x_start: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass computes the training loss.

        Args:
            x_start: Clean data of shape (batch, seq_len, features)
            condition: Optional conditioning

        Returns:
            Loss value
        """
        return self.compute_loss(x_start, condition)
