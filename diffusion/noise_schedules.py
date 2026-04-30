# -*- coding: utf-8 -*-
"""
Group 09
Authors: Geer Wang, Yijun Wang

noise_schedules.py

Este archivo implementa los noise schedules del caso VP (Variance Preserving).

Teoría
------
En VP se define una función beta(t) que controla la deriva y la difusión:

    dx(t) = -1/2 beta(t) x(t) dt + sqrt(beta(t)) dW(t)

También es fundamental su integral acumulada:

    B(t) = ∫_0^t beta(s) ds

porque luego aparece en:

    mu_t(x_0) = exp(-B(t)/2) x_0
    sigma_t(t) = sqrt(1 - exp(-B(t)))

Schedules implementados
-----------------------
1) Linear:
       beta(t) = beta_min + (beta_max - beta_min) * t / T

   Integral:
       B(t) = beta_min * t + ((beta_max - beta_min)/(2T)) * t^2

2) Cosine:
       alpha_bar(t) =
           cos^2( (pi/2) * ((t/T + s)/(1+s)) ) /
           cos^2( (pi/2) * (s/(1+s)) )

       beta(t) = - d/dt log(alpha_bar(t))
               = pi / (T(1+s)) * tan( (pi/2) * ((t/T + s)/(1+s)) )

   Integral:
       B(t) = ∫_0^t beta(u) du = -log(alpha_bar(t))

Nota
----
Este archivo es para VP.
El caso VE exponencial NO se define aquí, sino en diffusion_model_general.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import math

# =========================================================
# AUXILIAR
# =========================================================

def _flatten_time(t):
    """
    Asegura que t tenga forma (batch,).
    """
    if t.ndim > 1:
        return t.view(t.shape[0])
    return t

# =========================================================
# 1) VERSIONES NUMPY (para visualización)
# =========================================================

def linear_beta_numpy(t, T, beta_min=1e-4, beta_max=20.0):
    """
    Schedule lineal:

        beta(t) = beta_min + (beta_max - beta_min) * t / T
    """
    return beta_min + (beta_max - beta_min) * (t / T)


def integrated_linear_beta_numpy(t, T, beta_min=0.1, beta_max=20.0):
    """
    Integral del schedule lineal:

        B(t) = ∫_0^t beta(s) ds
             = beta_min * t + ((beta_max - beta_min)/(2T)) * t^2
    """
    return beta_min * t + ((beta_max - beta_min) / (2.0 * T)) * (t ** 2)


def cosine_alpha_bar_numpy(t, T, s=0.008):
    """
    alpha_bar(t) del schedule cosine:

        alpha_bar(t) =
            cos^2( (pi/2) * ((t/T + s)/(1+s)) ) /
            cos^2( (pi/2) * (s/(1+s)) )

    Satisface:
        alpha_bar(0) = 1
    """
    num = np.cos((np.pi / 2.0) * ((t / T + s) / (1.0 + s))) ** 2
    den = np.cos((np.pi / 2.0) * (s / (1.0 + s))) ** 2
    return num / den


def cosine_beta_numpy(t, T, s=0.008, clip_max=0.999):
    """
    Schedule cosine continuo:

        beta(t) = - d/dt log(alpha_bar(t))
                = pi / (T(1+s)) * tan( (pi/2) * ((t/T + s)/(1+s)) )

    Se recorta por estabilidad numérica.
    """
    u = (np.pi / 2.0) * ((t / T + s) / (1.0 + s))
    beta = (np.pi / (T * (1.0 + s))) * np.tan(u)
    return np.clip(beta, 0.0, clip_max)


def integrated_cosine_beta_numpy(t, T, s=0.008, eps=1e-5):
    """
    Integral del schedule cosine:

        B(t) = ∫_0^t beta(u) du = -log(alpha_bar(t))

    donde alpha_bar(0) = 1.
    """
    alpha_bar = np.clip(cosine_alpha_bar_numpy(t, T, s=s), eps, 1.0)
    return -np.log(alpha_bar)


def plot_schedules(T=1000, beta_min=1e-4, beta_max=20.0, s=0.008):
    """
    Visualiza beta(t) para los schedules lineal y cosine.

    Nota:
    -----
    Esta parte es solo para representación gráfica.
    """
    n_points = 1000
    t_vals = np.linspace(0.0, T, n_points)

    beta_lin = linear_beta_numpy(t_vals, T, beta_min=beta_min, beta_max=beta_max)
    beta_cos = cosine_beta_numpy(t_vals, T, s=s)

    plt.figure(figsize=(10, 5))
    plt.plot(t_vals, beta_lin, label="Linear β(t)")
    plt.plot(t_vals, beta_cos, label="Cosine β(t)")
    plt.title("Noise schedules for VP")
    plt.xlabel("t")
    plt.ylabel("β(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# 2) VERSIONES TORCH (entrenamiento / sampling)
# =========================================================

def linear_beta_torch(t, T, beta_min=1e-4, beta_max=20.0):
    """
    Schedule lineal en PyTorch:

        beta(t) = beta_min + (beta_max - beta_min) * t / T
    """
    t = _flatten_time(t).float()
    return beta_min + (beta_max - beta_min) * (t / T)


def integrated_linear_beta_torch(t, T, beta_min=1e-4, beta_max=20.0):
    """
    Integral del schedule lineal:

        B(t) = ∫_0^t beta(s) ds
             = beta_min * t + ((beta_max - beta_min)/(2T)) * t^2
    """
    t = _flatten_time(t).float()
    return beta_min * t + ((beta_max - beta_min) / (2.0 * T)) * (t ** 2)


def cosine_alpha_bar_torch(t, T, s=0.008):
    """
    alpha_bar(t) del schedule cosine:

        alpha_bar(t) =
            cos^2( (pi/2) * ((t/T + s)/(1+s)) ) /
            cos^2( (pi/2) * (s/(1+s)) )

    con:
        alpha_bar(0) = 1
    """
    t = _flatten_time(t).float()

    u = (torch.pi / 2.0) * ((t / T + s) / (1.0 + s))
    u0 = (math.pi / 2.0) * (s / (1.0 + s))
    den = math.cos(u0) ** 2

    return (torch.cos(u) ** 2) / den


def integrated_cosine_beta_torch(t, T, s=0.008, eps=1e-5):
    """
    Integral del schedule cosine:

        B(t) = ∫_0^t beta(u) du
             = -log(alpha_bar(t))

    porque:
        beta(t) = - d/dt log(alpha_bar(t))
        alpha_bar(0) = 1
    """
    alpha_bar = torch.clamp(cosine_alpha_bar_torch(t, T, s=s), min=eps, max=1.0)
    return -torch.log(alpha_bar)


def cosine_beta_torch(t, T, s=0.008, clip_max=0.999):
    """
    Schedule cosine continuo:

        beta(t) = - d/dt log(alpha_bar(t))
                = pi / (T(1+s)) * tan( (pi/2) * ((t/T + s)/(1+s)) )

    Se recorta por estabilidad numérica.
    """
    t = _flatten_time(t).float()

    u = (torch.pi / 2.0) * ((t / T + s) / (1.0 + s))
    beta = (torch.pi / (T * (1.0 + s))) * torch.tan(u)

    return torch.clamp(beta, min=0.0, max=clip_max)


# =========================================================
# SELECTORES UNIFICADOS
# =========================================================

def get_beta_schedule_torch(
    t,
    T,
    schedule="linear",
    beta_min=0.1,
    beta_max=20.0,
    s=0.008,
):
    """
    Devuelve beta(t) para el schedule indicado.

    schedule:
        - "linear"
        - "cosine"
    """
    if schedule == "linear":
        return linear_beta_torch(t, T, beta_min=beta_min, beta_max=beta_max)
    elif schedule == "cosine":
        return cosine_beta_torch(t, T, s=s)
    else:
        raise ValueError("schedule debe ser 'linear' o 'cosine'")


def get_integrated_beta_schedule_torch(
    t,
    T,
    schedule="linear",
    beta_min=0.1,
    beta_max=20.0,
    s=0.008,
):
    """
    Devuelve la integral acumulada:

        B(t) = ∫_0^t beta(s) ds

    para el schedule indicado.
    """
    if schedule == "linear":
        return integrated_linear_beta_torch(t, T, beta_min=beta_min, beta_max=beta_max)
    elif schedule == "cosine":
        return integrated_cosine_beta_torch(t, T, s=s)
    else:
        raise ValueError("schedule debe ser 'linear' o 'cosine'")