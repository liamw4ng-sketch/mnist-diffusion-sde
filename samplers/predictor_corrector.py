# -*- coding: utf-8 -*-
"""
Group 09
Authors: Geer Wang, Yijun Wang

predictor_corrector.py
"""

import torch


def pc_sampler(
    score_model,
    diffusion_process,
    scheme=None,   # compatibilidad
    beta_t=None,   # compatibilidad
    n_steps=500,
    n_corrector_steps=1,
    step_size=0.01,
    n_images=4,
    T=1.0,
    eps=1.0e-3,
    image_shape=(1, 28, 28),
    device="cpu",
):
    """
    Predictor-Corrector sampler.

    Corrector:
        paso de Langevin:
            x <- x + eta * score(x,t) + sqrt(2 eta) * z

    Predictor:
        un paso de Euler-Maruyama sobre la reverse SDE:
            dx = [f(x,t) - g(t)^2 s_theta(x,t)] dt + g(t) dW
    """
    score_model.eval()

    with torch.no_grad():
        # Evitar extremos exactos por estabilidad numérica
        t0 = T - eps
        t_end = eps

        sigma_T = diffusion_process.sigma_t(
            torch.tensor([t0], device=device, dtype=torch.float32)
        ).item()

        # Inicialización desde la gaussiana final
        x = sigma_T * torch.randn(n_images, *image_shape, device=device)

        # Mallado temporal decreciente
        times = torch.linspace(t0, t_end, n_steps + 1, device=device, dtype=torch.float32)

        # Guardar trayectoria
        x_trajectory = torch.empty((*x.shape, n_steps + 1), device=device, dtype=x.dtype)
        x_trajectory[..., 0] = x

        step_tensor = torch.tensor(step_size, device=device, dtype=torch.float32)
        for i in range(n_steps):
                    t = times[i]
                    t_next = times[i + 1]
                    dt = t_next - t  # negativo
                    h = -dt          # positivo

                    t_batch = torch.full((n_images,), float(t), device=device)
                    g_t = diffusion_process.diffusion_coefficient(t_batch).view(-1, 1, 1, 1)

                    # ------------------
                    # CORRECTOR (Langevin)
                    # ------------------
                    for _ in range(n_corrector_steps):
                        z = torch.randn_like(x)
                        score = score_model(x, t_batch)
                        x = x + step_tensor * score + torch.sqrt(2.0 * step_tensor) * z

                    # ------------------
                    # PREDICTOR (reverse SDE)
                    # ------------------
                    score = score_model(x, t_batch)
                    f_t = diffusion_process.drift_coefficient(x, t_batch)
                    reverse_drift = f_t - (g_t ** 2) * score

                    z = torch.randn_like(x)
                    x_mean = x + reverse_drift * dt
                    x = x_mean + g_t * torch.sqrt(torch.tensor(h, device=device)) * z

                    x_trajectory[..., i + 1] = x

    return times, x_trajectory