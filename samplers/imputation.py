from __future__ import annotations

import torch


# =========================================================
# MÁSCARA
# =========================================================

def create_center_mask(batch_size, image_shape, hole_size=10, device="cpu"):
    mask = torch.ones(batch_size, *image_shape, device=device)

    _, H, W = image_shape
    h_start = H // 2 - hole_size // 2
    h_end = h_start + hole_size
    w_start = W // 2 - hole_size // 2
    w_end = w_start + hole_size

    mask[:, :, h_start:h_end, w_start:w_end] = 0.0

    return mask


def apply_mask(x, mask, fill_value=0.0):
    return x * mask + fill_value * (1 - mask)


# =========================================================
# IMPUTATION SAMPLER
# =========================================================

def imputation_sampler(
    x_known,
    mask,
    diffusion_process,
    score_model,
    n_steps=500,
    T=1.0,
    t_end=1e-3,
    device="cpu",
    seed=None,
):
    score_model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    n_images = x_known.shape[0]

    sigma_T = diffusion_process.sigma_t(
        torch.tensor([T], device=device)
    ).item()

    x = sigma_T * torch.randn_like(x_known)

    times = torch.linspace(T, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_trajectory = torch.empty((*x.shape, n_steps + 1), device=device)
    x_trajectory[..., 0] = x

    for i, t in enumerate(times[:-1]):
        t_batch = torch.full((n_images,), float(t), device=device)

        score = score_model(x, t_batch)

        f_t = diffusion_process.drift_coefficient(x, t_batch)
        g_t = diffusion_process.diffusion_coefficient(t_batch).view(-1, 1, 1, 1)

        drift = f_t - (g_t ** 2) * score

        z = torch.randn_like(x)

        x = x + drift * dt + g_t * torch.sqrt(torch.abs(dt)) * z

        # 🔥 CLAVE: imponer píxeles conocidos
        x = x * (1 - mask) + x_known * mask

        x_trajectory[..., i + 1] = x.detach()

    return times, x_trajectory, x