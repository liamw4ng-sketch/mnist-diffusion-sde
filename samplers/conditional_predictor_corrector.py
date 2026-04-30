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


def conditional_pc_sampler(
    score_model,
    classifier_model,
    diffusion_process,
    y_target: torch.Tensor,
    n_images: Optional[int] = None,
    n_steps: int = 500,
    n_corrector_steps: int = 1,
    step_size: float = 0.01,
    T: float = 1.0,
    eps: float = 1.0e-3,
    image_shape=(1, 28, 28),
    device: str = "cpu",
    guidance_scale: float = 1.0,
    grad_clip_value: Optional[float] = 1.0,
    seed: Optional[int] = None,
):
    """
    Predictor-Corrector condicional para VP.

    El mismo archivo sirve para VP-linear y VP-cosine, porque el schedule ya
    está codificado en diffusion_process.
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
    x_trajectory = torch.empty((*x.shape, n_steps + 1), device=device, dtype=x.dtype)
    x_trajectory[..., 0] = x

    step_tensor = torch.tensor(step_size, device=device, dtype=x.dtype)

    for i in range(n_steps):
        t = times[i]
        t_next = times[i + 1]
        dt = t_next - t
        h = -dt

        t_batch = torch.full((n_images,), float(t), device=device, dtype=x.dtype)
        g_t = diffusion_process.diffusion_coefficient(t_batch).view(-1, 1, 1, 1)

        for _ in range(n_corrector_steps):
            z = torch.randn_like(x)
            score = score_model(x, t_batch)

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
            x = x + step_tensor * guided_score + torch.sqrt(2.0 * step_tensor) * z

        score = score_model(x, t_batch)
        f_t = diffusion_process.drift_coefficient(x, t_batch)

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
        reverse_drift = f_t - (g_t ** 2) * guided_score

        z = torch.randn_like(x)
        x_mean = x + reverse_drift * dt
        x = x_mean + g_t * torch.sqrt(torch.tensor(h, device=device, dtype=x.dtype)) * z

        x_trajectory[..., i + 1] = x.detach()

    return times, x_trajectory


def generate_digit_class_pc(
    digit: int,
    n_images: int,
    score_model,
    classifier_model,
    diffusion_process,
    n_steps: int = 500,
    n_corrector_steps: int = 1,
    step_size: float = 0.01,
    T: float = 1.0,
    eps: float = 1.0e-3,
    image_shape=(1, 28, 28),
    device: str = "cpu",
    guidance_scale: float = 1.0,
    grad_clip_value: Optional[float] = 1.0,
    seed: Optional[int] = None,
):
    y_target = torch.full((n_images,), digit, device=device, dtype=torch.long)
    return conditional_pc_sampler(
        score_model=score_model,
        classifier_model=classifier_model,
        diffusion_process=diffusion_process,
        y_target=y_target,
        n_images=n_images,
        n_steps=n_steps,
        n_corrector_steps=n_corrector_steps,
        step_size=step_size,
        T=T,
        eps=eps,
        image_shape=image_shape,
        device=device,
        guidance_scale=guidance_scale,
        grad_clip_value=grad_clip_value,
        seed=seed,
    )
