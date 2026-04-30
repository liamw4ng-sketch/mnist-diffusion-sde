# -*- coding: utf-8 -*-
"""
Group 09
Authors: Geer Wang, Yijun Wang

ODE.py
"""

import torch


def ode_sampler(
    score_model,
    diffusion_process,
    scheme=None,   # compatibilidad
    beta_t=None,   # compatibilidad
    n_steps=1000,
    n_images=4,
    T=1.0,
    eps=1.0e-3,
    image_shape=(1, 28, 28),
    device="cpu",
):
    """
    Probability Flow ODE:

        dx/dt = f(x,t) - 0.5 g(t)^2 s_theta(x,t)

    Se discretiza con Euler explícito:
        x_{k+1} = x_k + drift(x_k, t_k) * dt
    """
    score_model.eval()

    with torch.no_grad():
        # Evitar extremos exactos por estabilidad numérica
        t0 = T - eps
        t_end = eps

        # Inicialización:
        # x(t0) ~ N(0, sigma(t0)^2 I)
        sigma_T = diffusion_process.sigma_t(
            torch.tensor([t0], device=device, dtype=torch.float32)
        ).item()

        x = sigma_T * torch.randn(n_images, *image_shape, device=device)

        # Mallado temporal decreciente
        times = torch.linspace(t0, t_end, n_steps + 1, device=device, dtype=torch.float32)
        dt = times[1] - times[0]  # negativo

        x_trajectory = torch.empty((*x.shape, n_steps + 1), device=device, dtype=x.dtype)
        x_trajectory[..., 0] = x

        for i, t in enumerate(times[:-1]):
            t_batch = torch.full((n_images,), float(t), device=device)

            f_t = diffusion_process.drift_coefficient(x, t_batch)
            g_t = diffusion_process.diffusion_coefficient(t_batch).view(-1, 1, 1, 1)
            score = score_model(x, t_batch)

            # Probability Flow ODE drift
            drift = f_t - 0.5 * (g_t ** 2) * score

            # Euler explícito para ODE
            x = x + drift * dt

            x_trajectory[..., i + 1] = x

    return times, x_trajectory