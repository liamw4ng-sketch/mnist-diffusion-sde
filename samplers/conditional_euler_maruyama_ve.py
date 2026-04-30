from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# =========================================================
# CLASSIFIER GUIDANCE
# =========================================================

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

    selected = log_probs[
        torch.arange(x_t.shape[0], device=x_t.device),
        y_target,
    ].sum()

    grad = torch.autograd.grad(selected, x_in)[0]
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    if grad_clip_value is not None:
        grad = torch.clamp(grad, -grad_clip_value, grad_clip_value)

    return grad.detach()


# =========================================================
# BACKWARD DRIFT VE
# =========================================================

def backward_drift_conditional_ve(
    x_t,
    t,
    score_model,
    classifier_model,
    y_target,
    diffusion_process,
    guidance_scale=1.0,
    grad_clip_value=1.0,
):
    score = score_model(x_t, t)

    g_t = diffusion_process.diffusion_coefficient(t).view(-1, 1, 1, 1)

    if guidance_scale != 0.0:
        grad_cls = classifier_guidance_gradient(
            classifier_model,
            x_t,
            t,
            y_target,
            grad_clip_value,
        )
    else:
        grad_cls = torch.zeros_like(x_t)

    guided_score = score + guidance_scale * grad_cls

    # VE: drift = - g(t)^2 * score
    return - (g_t ** 2) * guided_score


# =========================================================
# SAMPLER
# =========================================================

def conditional_euler_maruyama_sampler_ve(
    score_model,
    classifier_model,
    diffusion_process,
    y_target: torch.Tensor,
    n_images=None,
    n_steps=500,
    T=1.0,
    t_end=1.0e-3,
    image_shape=(1, 28, 28),
    device="cpu",
    guidance_scale=1.0,
    grad_clip_value=1.0,
    seed=None,
):
    score_model.eval()
    classifier_model.eval()

    y_target = y_target.to(device).long()

    if n_images is None:
        n_images = y_target.shape[0]

    if seed is not None:
        torch.manual_seed(seed)

    sigma_T = diffusion_process.sigma_t(
        torch.tensor([T], device=device)
    ).item()

    x = sigma_T * torch.randn(n_images, *image_shape, device=device)

    times = torch.linspace(T, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_trajectory = torch.empty((*x.shape, n_steps + 1), device=device)
    x_trajectory[..., 0] = x

    z = torch.randn_like(x_trajectory)
    z[..., -1] = 0.0

    for i, t in enumerate(times[:-1]):
        t_batch = torch.full((n_images,), float(t), device=device)

        drift = backward_drift_conditional_ve(
            x,
            t_batch,
            score_model,
            classifier_model,
            y_target,
            diffusion_process,
            guidance_scale,
            grad_clip_value,
        )

        g_t = diffusion_process.diffusion_coefficient(t_batch).view(-1, 1, 1, 1)

        x = x + drift * dt + g_t * torch.sqrt(torch.abs(dt)) * z[..., i]
        x_trajectory[..., i + 1] = x.detach()

    return times, x_trajectory


# =========================================================
# HELPER
# =========================================================

def generate_digit_class_ve(
    digit,
    n_images,
    score_model,
    classifier_model,
    diffusion_process,
    **kwargs,
):
    y_target = torch.full((n_images,), digit, device=kwargs.get("device", "cpu"))
    return conditional_euler_maruyama_sampler_ve(
        score_model=score_model,
        classifier_model=classifier_model,
        diffusion_process=diffusion_process,
        y_target=y_target,
        n_images=n_images,
        **kwargs,
    )