# metrics/bpd.py

import math
import torch


def num_dimensions_from_shape(image_shape):
    D = 1
    for d in image_shape:
        D *= d
    return D


def bpd_from_log_prob(log_prob, image_shape):
    D = num_dimensions_from_shape(image_shape)
    return -log_prob / (D * math.log(2.0))


def _sum_flat(x):
    return x.reshape(x.shape[0], -1).sum(dim=1)


def _gaussian_log_prob_isotropic(x, sigma):
    """
    Log-probabilidad por muestra bajo N(0, sigma^2 I).
    """
    if isinstance(sigma, torch.Tensor):
        sigma = float(sigma.detach().cpu().item())

    D = x[0].numel()
    sigma2 = sigma ** 2
    quad = _sum_flat(x ** 2) / sigma2
    return -0.5 * (D * math.log(2.0 * math.pi * sigma2) + quad)


def _probability_flow_drift(x, t, score_model, diffusion_process):
    """
    Drift de la Probability Flow ODE:
        f_pf(x,t) = f(x,t) - 0.5 * g(t)^2 * s_theta(x,t)
    """
    f_t = diffusion_process.drift_coefficient(x, t)
    g_t = diffusion_process.diffusion_coefficient(t).view(-1, 1, 1, 1)
    score = score_model(x, t)
    return f_t - 0.5 * (g_t ** 2) * score


def _divergence_hutchinson(
    x,
    t,
    score_model,
    diffusion_process,
    noise_type="gaussian",
):
    """
    Estimador de Hutchinson para tr( d f_pf / d x ).
    Devuelve:
        - drift(x,t)
        - divergence estimate por muestra
    """
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)

        if noise_type == "gaussian":
            eps = torch.randn_like(x_req)
        elif noise_type == "rademacher":
            eps = torch.randint(
                low=0,
                high=2,
                size=x_req.shape,
                device=x_req.device,
            ).to(x_req.dtype)
            eps = eps * 2.0 - 1.0
        else:
            raise ValueError("noise_type debe ser 'gaussian' o 'rademacher'")

        drift = _probability_flow_drift(
            x=x_req,
            t=t,
            score_model=score_model,
            diffusion_process=diffusion_process,
        )

        inner = torch.sum(drift * eps)

        grad = torch.autograd.grad(
            outputs=inner,
            inputs=x_req,
            create_graph=False,
            retain_graph=False,
        )[0]

        div = _sum_flat(grad * eps)

    return drift.detach(), div.detach()


def estimate_log_prob_batch(
    x0,
    score_model,
    diffusion_process,
    T=1.0,
    eps=1.0e-3,
    n_steps=200,
    prior_std=None,
    noise_type="gaussian",
):
    """
    Estima log p(x0) integrando la Probability Flow ODE hacia delante,
    desde t = eps hasta t = T - eps.

    Esto lo hace coherente con el ODE sampler actualizado, que también evita
    los extremos exactos por estabilidad numérica.
    """
    if n_steps <= 0:
        raise ValueError("n_steps debe ser mayor que 0")
    if not (0.0 < eps < T):
        raise ValueError("Se requiere 0 < eps < T")

    device = x0.device
    dtype = x0.dtype

    score_model.eval()

    x = x0.detach().clone()
    delta_logp = torch.zeros(x.shape[0], device=device, dtype=dtype)

    t_start = float(eps)
    t_end = float(T - eps)

    times = torch.linspace(
        t_start,
        t_end,
        n_steps + 1,
        device=device,
        dtype=dtype,
    )

    for i in range(n_steps):
        t = times[i]
        t_next = times[i + 1]
        dt = t_next - t

        t_batch = torch.full(
            (x.shape[0],),
            float(t),
            device=device,
            dtype=dtype,
        )

        drift, div = _divergence_hutchinson(
            x=x,
            t=t_batch,
            score_model=score_model,
            diffusion_process=diffusion_process,
            noise_type=noise_type,
        )

        # Euler explícito en la ODE
        x = x + drift * dt

        # Ecuación de cambio de variables:
        # d/dt log p_t(x_t) = - div f_pf(x_t, t)
        # Entonces:
        # log p_0(x_0) = log p_T(x_T) + ∫ div f_pf dt
        delta_logp = delta_logp + div * dt

    if prior_std is None:
        prior_std = diffusion_process.sigma_t(
            torch.tensor([t_end], device=device, dtype=dtype)
        ).item()

    logp_T = _gaussian_log_prob_isotropic(x, sigma=prior_std)
    logp_0 = logp_T + delta_logp
    return logp_0


def average_bpd_from_log_probs(log_probs, image_shape):
    bpds = bpd_from_log_prob(log_probs, image_shape)
    return bpds.mean().item()


def uniform_dequantize(x, n_bits=8):
    """
    Convierte una imagen discreta en [0,1] a una versión dequantizada.
    """
    K = 2 ** n_bits
    x_int = torch.clamp((x * (K - 1)).round(), 0, K - 1)
    x_deq = (x_int + torch.rand_like(x)) / K
    return x_deq


def bpd_from_log_prob_discrete(log_prob_continuous, image_shape, n_bits=8):
    """
    Convierte la log-probabilidad continua de la imagen dequantizada
    en bits per dimension para imagen discreta.
    """
    D = num_dimensions_from_shape(image_shape)
    correction = D * math.log(2 ** n_bits)
    return -(log_prob_continuous - correction) / (D * math.log(2.0))


def evaluate_bpd(
    data_loader,
    score_model,
    diffusion_process,
    T=1.0,
    eps=1.0e-3,
    n_steps=1000,
    prior_std=None,
    noise_type="gaussian",
    device="cpu",
    n_bits=8,
):
    """
    Calcula el BPD medio del dataset usando likelihood por Probability Flow ODE
    con uniform dequantization.

    Esta versión está pensada para evaluación con ODE.
    """
    total_bpd = 0.0
    total_items = 0
    image_shape = None

    for batch in data_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        if image_shape is None:
            image_shape = tuple(x.shape[1:])

        x_deq = uniform_dequantize(x, n_bits=n_bits)

        log_probs = estimate_log_prob_batch(
            x0=x_deq,
            score_model=score_model,
            diffusion_process=diffusion_process,
            T=T,
            eps=eps,
            n_steps=n_steps,
            prior_std=prior_std,
            noise_type=noise_type,
        )

        bpd_batch = bpd_from_log_prob_discrete(
            log_probs,
            image_shape=image_shape,
            n_bits=n_bits,
        )

        total_bpd += bpd_batch.sum().item()
        total_items += x.shape[0]

    if total_items == 0:
        raise ValueError("El data_loader no contiene muestras")

    return total_bpd / total_items