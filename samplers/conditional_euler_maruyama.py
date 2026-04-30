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


def backward_drift_conditional(
    x_t: torch.Tensor,
    t: torch.Tensor,
    score_model,
    classifier_model,
    y_target: torch.Tensor,
    diffusion_process,
    guidance_scale: float = 1.0,
    grad_clip_value: Optional[float] = 1.0,
) -> torch.Tensor:
    score = score_model(x_t, t)
    f_t = diffusion_process.drift_coefficient(x_t, t)
    g_t = diffusion_process.diffusion_coefficient(t).view(-1, 1, 1, 1)

    if guidance_scale != 0.0:
        grad_cls = classifier_guidance_gradient(
            classifier_model=classifier_model,
            x_t=x_t,
            t=t,
            y_target=y_target,
            grad_clip_value=grad_clip_value,
        )
    else:
        grad_cls = torch.zeros_like(x_t)

    guided_score = score + guidance_scale * grad_cls
    return f_t - (g_t ** 2) * guided_score


def conditional_euler_maruyama_sampler(
    score_model,
    classifier_model,
    diffusion_process,
    y_target: torch.Tensor,
    n_images: Optional[int] = None,
    n_steps: int = 500,
    T: float = 1.0,
    image_shape=(1, 28, 28),
    device: str = "cpu",
    t_end: float = 1.0e-3,
    guidance_scale: float = 1.0,
    grad_clip_value: Optional[float] = 1.0,
    seed: Optional[int] = None,
):
    """
    Euler-Maruyama condicional para VP.

    El mismo archivo sirve para schedule="linear" y schedule="cosine".
    El schedule ya queda incorporado dentro de diffusion_process.
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

    sigma_T = diffusion_process.sigma_t(
        torch.tensor([T], device=device, dtype=torch.float32)
    ).item()

    x = sigma_T * torch.randn(n_images, *image_shape, device=device)

    times = torch.linspace(T, t_end, n_steps + 1, device=device, dtype=x.dtype)
    dt = times[1] - times[0]

    x_trajectory = torch.empty((*x.shape, n_steps + 1), device=device, dtype=x.dtype)
    x_trajectory[..., 0] = x

    z = torch.randn_like(x_trajectory)
    z[..., -1] = 0.0

    for n, t in enumerate(times[:-1]):
        batch_t = torch.full((n_images,), float(t), device=device, dtype=x.dtype)

        drift = backward_drift_conditional(
            x_t=x,
            t=batch_t,
            score_model=score_model,
            classifier_model=classifier_model,
            y_target=y_target,
            diffusion_process=diffusion_process,
            guidance_scale=guidance_scale,
            grad_clip_value=grad_clip_value,
        )

        g_t = diffusion_process.diffusion_coefficient(batch_t).view(-1, 1, 1, 1)
        x = x + drift * dt + g_t * torch.sqrt(torch.abs(dt)) * z[..., n]
        x_trajectory[..., n + 1] = x.detach()

    return times, x_trajectory


def generate_digit_class(
    digit: int,
    n_images: int,
    score_model,
    classifier_model,
    diffusion_process,
    n_steps: int = 500,
    T: float = 1.0,
    image_shape=(1, 28, 28),
    device: str = "cpu",
    t_end: float = 1.0e-3,
    guidance_scale: float = 1.0,
    grad_clip_value: Optional[float] = 1.0,
    seed: Optional[int] = None,
):
    y_target = torch.full((n_images,), digit, device=device, dtype=torch.long)
    return conditional_euler_maruyama_sampler(
        score_model=score_model,
        classifier_model=classifier_model,
        diffusion_process=diffusion_process,
        y_target=y_target,
        n_images=n_images,
        n_steps=n_steps,
        T=T,
        image_shape=image_shape,
        device=device,
        t_end=t_end,
        guidance_scale=guidance_scale,
        grad_clip_value=grad_clip_value,
        seed=seed,
    )
