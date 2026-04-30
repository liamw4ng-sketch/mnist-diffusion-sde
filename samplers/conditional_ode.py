from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def classifier_guidance_gradient(
    classifier_model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    y_target: torch.Tensor,
    grad_clip_value: Optional[float] = 1.0,
) -> torch.Tensor:
    x_in = x_t.detach().clone().requires_grad_(True)
    logits = classifier_model(x_in, t)
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs[torch.arange(x_t.shape[0], device=x_t.device), y_target].sum()

    grad = torch.autograd.grad(selected, x_in, create_graph=False, retain_graph=False)[0]
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    if grad_clip_value is not None:
        grad = torch.clamp(grad, -grad_clip_value, grad_clip_value)

    return grad.detach()


def conditional_ode_sampler(
    score_model,
    classifier_model,
    diffusion_process,
    y_target: torch.Tensor,
    n_images: Optional[int] = None,
    n_steps: int = 1000,
    T: float = 1.0,
    eps: float = 1.0e-3,
    image_shape=(1, 28, 28),
    device: str = "cpu",
    guidance_scale: float = 1.0,
    grad_clip_value: Optional[float] = 1.0,
    seed: Optional[int] = None,
):
    """
    Probability Flow ODE condicional para VP.

    El mismo archivo sirve para VP-linear y VP-cosine.
    """
    score_model.eval()
    classifier_model.eval()

    y_target = y_target.to(device).long()
    if n_images is None:
        n_images = y_target.shape[0]
    if y_target.shape[0] != n_images:
        raise ValueError("n_images debe coincidir con la longitud de y_target")

    if seed is not None:
        torch.manual_seed(seed)

    t0 = T - eps
    t_end = eps

    sigma_T = diffusion_process.sigma_t(
        torch.tensor([t0], device=device, dtype=torch.float32)
    ).item()

    x = sigma_T * torch.randn(n_images, *image_shape, device=device)

    times = torch.linspace(t0, t_end, n_steps + 1, device=device, dtype=x.dtype)
    dt = times[1] - times[0]

    x_trajectory = torch.empty((*x.shape, n_steps + 1), device=device, dtype=x.dtype)
    x_trajectory[..., 0] = x

    for i, t in enumerate(times[:-1]):
        t_batch = torch.full((n_images,), float(t), device=device, dtype=x.dtype)

        score = score_model(x, t_batch)
        f_t = diffusion_process.drift_coefficient(x, t_batch)
        g_t = diffusion_process.diffusion_coefficient(t_batch).view(-1, 1, 1, 1)

        if guidance_scale != 0.0:
            grad_cls = classifier_guidance_gradient(
                classifier_model=classifier_model,
                x_t=x,
                t=t_batch,
                y_target=y_target,
                grad_clip_value=grad_clip_value,
            )
        else:
            grad_cls = torch.zeros_like(x)

        guided_score = score + guidance_scale * grad_cls
        drift = f_t - 0.5 * (g_t ** 2) * guided_score

        x = x + drift * dt
        x_trajectory[..., i + 1] = x.detach()

    return times, x_trajectory


def generate_digit_class_ode(
    digit: int,
    n_images: int,
    score_model,
    classifier_model,
    diffusion_process,
    n_steps: int = 1000,
    T: float = 1.0,
    eps: float = 1.0e-3,
    image_shape=(1, 28, 28),
    device: str = "cpu",
    guidance_scale: float = 1.0,
    grad_clip_value: Optional[float] = 1.0,
    seed: Optional[int] = None,
):
    y_target = torch.full((n_images,), digit, device=device, dtype=torch.long)
    return conditional_ode_sampler(
        score_model=score_model,
        classifier_model=classifier_model,
        diffusion_process=diffusion_process,
        y_target=y_target,
        n_images=n_images,
        n_steps=n_steps,
        T=T,
        eps=eps,
        image_shape=image_shape,
        device=device,
        guidance_scale=guidance_scale,
        grad_clip_value=grad_clip_value,
        seed=seed,
    )
