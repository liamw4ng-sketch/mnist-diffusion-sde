# -*- coding: utf-8 -*-
"""
Group 09
Authors: Geer Wang, Yijun Wang

euler_maruyama.py
"""

from __future__ import annotations
from typing import Callable, Union
import torch
from torch import Tensor


def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    seed: Union[int, None] = None,
) -> tuple[Tensor, Tensor]:
    """
    Euler-Maruyama integrator (approximate).

    Args:
        x_0: Initial images of shape
            (batch_size, n_channels, image_height, image_width)
        t_0: Initial time.
        t_end: End of the integration interval.
        n_steps: Number of integration steps.
        drift_coefficient: Function f(x(t), t) defining the drift term.
        diffusion_coefficient: Function g(t) defining the diffusion term.
        seed: Random seed for reproducibility.

    Returns:
        times: Time grid.
        x_t: Simulated trajectories with shape
             (*x_0.shape, n_steps + 1)
    """
    device = x_0.device
    dtype = x_0.dtype

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=dtype)
    dt = times[1] - times[0]

    x_t = torch.empty(
        (*x_0.shape, len(times)),
        dtype=dtype,
        device=device,
    )
    x_t[..., 0] = x_0

    z = torch.randn_like(x_t)
    z[..., -1] = 0.0  # no extra noise in the final stored step

    for n, t in enumerate(times[:-1]):
        batch_t = torch.full((x_0.shape[0],), float(t), device=device, dtype=dtype)

        x_t[..., n + 1] = (
            x_t[..., n]
            + drift_coefficient(x_t[..., n], batch_t) * dt
            + diffusion_coefficient(batch_t).view(-1, 1, 1, 1)
            * torch.sqrt(torch.abs(dt))
            * z[..., n]
        )

    return times, x_t