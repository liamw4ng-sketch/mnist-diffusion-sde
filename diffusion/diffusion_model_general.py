# -*- coding: utf-8 -*-
"""
Group 09
Authors: Geer Wang, Yijun Wang

diffusion_model_general.py

Implementación general para:
- carga de datos MNIST,
- definición del proceso de difusión forward,
- entrenamiento del score model,
- generación de imágenes con la reverse SDE usando Euler-Maruyama.

Resumen teórico
---------------
Sea x(t) una imagen perturbada en el tiempo t.

Forward SDE general:
    dx(t) = f(x(t), t) dt + g(t) dW(t)

Reverse SDE:
    dx(t) = [ f(x(t), t) - g(t)^2 ∇_x log p_t(x(t)) ] dt + g(t) dW(t)

La red score_model aproxima:
    s_theta(x, t) ≈ ∇_x log p_t(x)

Casos implementados
-------------------
1) VE (Variance Exploding):
       dx(t) = g(t) dW(t)
       x_t | x_0 ~ N( x_0, sigma(t)^2 I )

   En esta práctica, VE se usa con sigma(t) exponencial:
       sigma(t) = sigma_min * (sigma_max / sigma_min)^(t / T)

   Y como:
       sigma(t)^2 = ∫_0^t g(s)^2 ds
   entonces:
       g(t)^2 = d/dt [sigma(t)^2] = 2 sigma(t) sigma'(t)

2) VP (Variance Preserving):
       dx(t) = -1/2 beta(t) x(t) dt + sqrt(beta(t)) dW(t)

   Definiendo:
       B(t) = ∫_0^t beta(s) ds

   Se tiene:
       x_t | x_0 ~ N( exp(-B(t)/2) x_0, (1 - exp(-B(t))) I )

   Por tanto:
       mu_t(x_0) = exp(-B(t)/2) x_0
       sigma_t(t) = sqrt(1 - exp(-B(t)))

Notas importantes del proyecto
------------------------------
- VE NO se compara con schedules linear/cosine.
- VE se usa en su versión estándar/exponencial.
- VP sí se compara con schedules linear y cosine.
- En el muestreo numérico se evita integrar exactamente en los extremos:
      t0 = T - eps
      t_end = eps
"""


import torch # tensoriales, GPU, redes, operaciones numéricas.
from torch.utils.data import DataLoader, Subset # para leer los datos por batches.
from torchvision import datasets # para cargar MNIST
from torchvision.transforms import ToTensor # trasnforma imagen a tensor en [0, 1]
from torch.optim import Adam # optimizador del entrenamiento
from functools import partial # sirve para fijar algunos argumentos de una funcion
from tqdm.auto import trange # barra de progreso 
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader, Subset, Dataset

try:
    from . import diffusion_process as dfp
    from .noise_schedules import (
        get_beta_schedule_torch,
        get_integrated_beta_schedule_torch,
    )
except ImportError:
    import diffusion_process as dfp
    from noise_schedules import (
        get_beta_schedule_torch,
        get_integrated_beta_schedule_torch,
    )


def _flatten_time(t):
    """
    Asegura que el tiempo t tenga forma (batch,).

    Si t llega con forma (batch, 1) o similar, lo aplana a un vector.
    Esto es útil porque funciones como beta_t(t), B_t(t), sigma_t(t) y g(t)
    esperan un vector de tiempos por batch.
    """
    if t.ndim > 1:
        return t.view(t.shape[0])
    return t

# =========================================================
# DATA
# =========================================================

def load_mnist_data(
    digit=None,        # si es None: usa todo MNIST; si es un entero: filtra solo ese dígito
    batch_size=32,     # número de imágenes por minibatch
    shuffle=True,      # si True, mezcla los datos en cada época
    num_workers=0,     # número de procesos de carga de datos
):
    """
    Carga el conjunto de entrenamiento de MNIST.

    Parámetros
    ----------
    digit : int o None
        Si es None, usa todo el dataset.
        Si es un entero (por ejemplo 7), usa solo ese dígito.
    batch_size : int
        Tamaño del minibatch.
    shuffle : bool
        Si True, baraja los datos en cada época.
    num_workers : int
        Número de workers del DataLoader.

    Devuelve
    --------
    data_loader : DataLoader
        DataLoader para entrenamiento.
    data : Dataset o Subset
        Dataset completo o subconjunto filtrado.
    """
    data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Si se quiere entrenar solo con un dígito, se filtra aquí
    if digit is not None:
        indices_digit = torch.where(data.targets == digit)[0]
        data = Subset(data, indices_digit)

    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return data_loader, data

# =========================================================
# DIFFUSION PROCESS
# =========================================================

def create_diffusion_process(
    scheme="vp",       # tipo de proceso forward: "ve" o "vp"
    schedule="linear", # schedule de beta(t) en VP: "linear" o "cosine"
    T=1.0,             # tiempo final del proceso forward
    beta_min=0.1,      # valor mínimo de beta(t) en VP
    beta_max=20.0,     # valor máximo de beta(t) en VP
    s=0.008,           # parámetro del schedule cosine de VP
    sigma_min=0.01,    # desviación típica mínima en VE
    sigma_max=50.0,    # desviación típica máxima en VE
):
    """
    Construye el proceso de difusión forward.

    Forward SDE general:
        dx(t) = f(x(t), t) dt + g(t) dW(t)

    Casos:
    ------
    VE:
        dx(t) = g(t) dW(t)
        x_t | x_0 ~ N( x_0, sigma(t)^2 I )

    VP:
        dx(t) = -1/2 beta(t) x(t) dt + sqrt(beta(t)) dW(t)

    Parámetros
    ----------
    scheme : str
        "ve" o "vp".
    schedule : str
        "linear" o "cosine". Solo se usa en VP.
    T : float
        Tiempo final del proceso.
    beta_min, beta_max : float
        Rango de beta(t) para VP.
    s : float
        Parámetro auxiliar del schedule cosine.
    sigma_min, sigma_max : float
        Rango de sigma(t) para VE.

    Devuelve
    --------
    diffusion_process : GaussianDiffussionProcess
        Objeto con drift, diffusion, mu_t y sigma_t.
    diffusion_coefficient : callable
        Función g(t).
    sigma_t : callable
        Función sigma_t(t).
    beta_t : callable o None
        Función beta(t) si scheme == "vp", None en VE.
    """
    scheme = scheme.lower()
    schedule = schedule.lower()

    # -----------------------------------------------------
    # Schedule beta(t) para VP
    # -----------------------------------------------------
    def beta_t(t):
        """
        Devuelve beta(t) en VP.

        Ejemplo conceptual:
            - schedule linear: beta(t) crece linealmente
            - schedule cosine: beta(t) sigue una ley coseno
        """
        return get_beta_schedule_torch(
            t=t,
            T=T,
            schedule=schedule,
            beta_min=beta_min,
            beta_max=beta_max,
            s=s,
        )

    def B_t(t):
        """
        Devuelve la integral acumulada:
            B(t) = ∫_0^t beta(s) ds

        Esta cantidad aparece en la solución cerrada de VP:
            mu_t(x_0) = exp(-B(t)/2) x_0
            sigma_t(t) = sqrt(1 - exp(-B(t)))
        """
        return get_integrated_beta_schedule_torch(
            t=t,
            T=T,
            schedule=schedule,
            beta_min=beta_min,
            beta_max=beta_max,
            s=s,
        )

    # -----------------------------------------------------
    # VE estándar: sigma(t) exponencial
    # -----------------------------------------------------
    def sigma_ve_t(t):
        """
        VE estándar con crecimiento exponencial de la desviación típica.

        Fórmula:
            sigma(t) = sigma_min * (sigma_max / sigma_min)^(t / T)

        Interpretación:
            - en t = 0, sigma(0) = sigma_min
            - en t = T, sigma(T) = sigma_max
        """
        t = _flatten_time(t).float()
        ratio = sigma_max / sigma_min
        return sigma_min * (ratio ** (t / T))

    def diffusion_ve_t(t):
        """
        Devuelve g(t) en VE.

        Como en VE:
            x_t | x_0 ~ N(x_0, sigma(t)^2 I)

        y además:
            sigma(t)^2 = ∫_0^t g(s)^2 ds

        entonces:
            g(t)^2 = d/dt [sigma(t)^2] = 2 sigma(t) sigma'(t)

        Para sigma(t) exponencial:
            sigma(t) = sigma_min * (sigma_max / sigma_min)^(t/T)

        se obtiene:
            sigma'(t) = sigma(t) * log(sigma_max / sigma_min) / T

        y por tanto:
            g(t)^2 = 2 sigma(t)^2 log(sigma_max / sigma_min) / T
            g(t)   = sigma(t) * sqrt(2 log(sigma_max / sigma_min) / T)
        """
        t = _flatten_time(t).float()
        ratio = torch.tensor(sigma_max / sigma_min, device=t.device, dtype=t.dtype)

        sigma_t_val = sigma_ve_t(t)
        c = 2.0 * torch.log(ratio) / T

        return sigma_t_val * torch.sqrt(torch.clamp(c, min=1.0e-12))

    # -----------------------------------------------------
    # VE
    # -----------------------------------------------------
    if scheme == "ve":
        # En VE:
        #   dx(t) = g(t) dW(t)
        # así que el drift es cero
        def drift_coefficient(x_t, t):
            return torch.zeros_like(x_t)

        def diffusion_coefficient(t):
            return diffusion_ve_t(t)

        # Media del proceso forward en VE:
        #   mu_t(x_0) = x_0
        def mu_t(x_0, t):
            return x_0

        # Desviación típica marginal en VE
        def sigma_t(t):
            return sigma_ve_t(t)

        diffusion_process = dfp.GaussianDiffussionProcess(
            drift_coefficient=drift_coefficient,
            diffusion_coefficient=diffusion_coefficient,
            mu_t=mu_t,
            sigma_t=sigma_t,
        )

        return diffusion_process, diffusion_coefficient, sigma_t, None

    # -----------------------------------------------------
    # VP
    # -----------------------------------------------------
    if scheme == "vp":
        # En VP:
        #   dx(t) = -1/2 beta(t) x(t) dt + sqrt(beta(t)) dW(t)
        #
        # Luego:
        #   f(x,t) = -1/2 beta(t) x
        #   g(t)   = sqrt(beta(t))
        def drift_coefficient(x_t, t):
            beta = beta_t(t).view(-1, 1, 1, 1)
            return -0.5 * beta * x_t

        def diffusion_coefficient(t):
            return torch.sqrt(torch.clamp(beta_t(t), min=1.0e-12))

        # Media de la transición forward en VP:
        #   mu_t(x_0) = exp(-B(t)/2) x_0
        def mu_t(x_0, t):
            B = B_t(t).view(-1, 1, 1, 1)
            return torch.exp(-0.5 * B) * x_0

        # Desviación típica marginal en VP:
        #   sigma_t(t) = sqrt(1 - exp(-B(t)))
        def sigma_t(t):
            B = B_t(t)
            return torch.sqrt(torch.clamp(1.0 - torch.exp(-B), min=1.0e-12))

        diffusion_process = dfp.GaussianDiffussionProcess(
            drift_coefficient=drift_coefficient,
            diffusion_coefficient=diffusion_coefficient,
            mu_t=mu_t,
            sigma_t=sigma_t,
        )

        return diffusion_process, diffusion_coefficient, sigma_t, beta_t

    raise ValueError("scheme debe ser 've' o 'vp'")

# =========================================================
# REVERSE DRIFT
# =========================================================

def backward_drift_coefficient(
    x_t,
    t,
    score_model,
    diffusion_process,
    scheme=None,                 # compatibilidad con notebooks anteriores
    beta_t=None,                 # compatibilidad con notebooks anteriores
    diffusion_coefficient=None,  # compatibilidad con notebooks anteriores
):
    """
    Drift de la reverse SDE.

    Fórmula:
        f_rev(x,t) = f(x,t) - g(t)^2 s_theta(x,t)

    donde:
        - f(x,t) es el drift del proceso forward
        - g(t) es el coeficiente de difusión
        - s_theta(x,t) ≈ ∇_x log p_t(x) es el score estimado por la red

    Parámetros
    ----------
    x_t : Tensor
        Estado actual del proceso en el tiempo t.
    t : Tensor
        Tiempo actual, típicamente con forma (batch,).
    score_model : nn.Module
        Red que aproxima el score.
    diffusion_process : GaussianDiffussionProcess
        Proceso forward con drift y difusión definidos.

    Devuelve
    --------
    Tensor
        Drift de la reverse SDE evaluado en (x_t, t).
    """
    score = score_model(x_t, t)  # s_theta(x_t, t)
    f_t = diffusion_process.drift_coefficient(x_t, t)  # f(x_t, t)
    g_t = diffusion_process.diffusion_coefficient(t).view(-1, 1, 1, 1)  # g(t)

    return f_t - (g_t ** 2) * score


# =========================================================
# TRAINING
# =========================================================

def train_model(
    data_loader,
    diffusion_process,
    score_model,
    n_epochs=30,                    # número de épocas de entrenamiento
    learning_rate=5e-4,             # tasa de aprendizaje del optimizador
    device="cpu",                   # dispositivo: "cpu" o "cuda"
    checkpoint_name="check_point.pth",  # ruta donde guardar el modelo entrenado
):
    """
    Entrena la red score_model minimizando la loss del proceso de difusión.

    La loss concreta está implementada dentro de:
        diffusion_process.loss_function(score_model, x)

    En términos teóricos, corresponde al entrenamiento del score:
        s_theta(x,t) ≈ ∇_x log p_t(x)

    normalmente mediante una variante de denoising score matching.

    Parámetros
    ----------
    data_loader : DataLoader
        Minibatches de imágenes reales.
    diffusion_process : GaussianDiffussionProcess
        Proceso forward que define la perturbación y la loss.
    score_model : nn.Module
        Red neuronal score.
    n_epochs : int
        Número de épocas.
    learning_rate : float
        Learning rate del optimizador Adam.
    device : str
        CPU o GPU.
    checkpoint_name : str
        Nombre del archivo donde se guardan los pesos entrenados.

    Devuelve
    --------
    score_model : nn.Module
        Red entrenada.
    """
    optimizer = Adam(score_model.parameters(), lr=learning_rate)

    score_model.train()
    epoch_bar = trange(n_epochs)

    for _ in epoch_bar:
        avg_loss = 0.0
        num_items = 0

        # Recorremos el dataset en minibatches
        for x, _ in data_loader:
            x = x.to(device)

            # La loss interna del diffusion_process:
            # - muestrea tiempos t
            # - muestrea ruido z
            # - construye x_t
            # - compara la salida del score model con el score teórico
            loss = diffusion_process.loss_function(score_model, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        epoch_bar.set_description(f"Average Loss: {avg_loss / num_items:.6f}")

    torch.save(score_model.state_dict(), checkpoint_name)
    return score_model


# =========================================================
# SAMPLING: EULER-MARUYAMA
# =========================================================

def generate_images_euler(
    score_model,
    diffusion_process,
    scheme=None,         # compatibilidad con notebooks anteriores
    beta_t=None,         # compatibilidad con notebooks anteriores
    n_images=3,          # número de imágenes a generar
    n_steps=750,         # número de pasos de Euler-Maruyama
    T=1.0,               # horizonte temporal teórico del proceso
    eps=1.0e-3,          # margen para evitar integrar exactamente en t=0 y t=T
    image_shape=(1, 28, 28),  # shape de una imagen MNIST: (canales, alto, ancho)
    device="cpu",        # dispositivo: "cpu" o "cuda"
):
    """
    Genera imágenes resolviendo la reverse SDE con Euler-Maruyama.

    Idea teórica
    -------------
    1) Inicializar desde ruido:
           x(T) ~ N(0, sigma(T)^2 I)

    2) Integrar la reverse SDE:
           dx(t) = [f(x,t) - g(t)^2 s_theta(x,t)] dt + g(t) dW(t)

    3) Usar Euler-Maruyama:
           x_{k+1} = x_k + a(x_k, t_k) dt + b(t_k) sqrt(|dt|) z_k

       donde:
           a(x,t) = f(x,t) - g(t)^2 s_theta(x,t)
           b(t)   = g(t)

    Nota numérica
    -------------
    Para evitar inestabilidades en los extremos, no integramos exactamente desde
    t = T hasta t = 0, sino desde:
        t0   = T - eps
        tend = eps

    Parámetros
    ----------
    score_model : nn.Module
        Red score entrenada.
    diffusion_process : GaussianDiffussionProcess
        Proceso forward con drift y difusión definidos.
    scheme : str o None
        Compatibilidad con código anterior.
    beta_t : callable o None
        Compatibilidad con código anterior.
    n_images : int
        Número de imágenes nuevas a generar.
    n_steps : int
        Número de pasos del integrador Euler-Maruyama.
    T : float
        Tiempo final teórico del proceso.
    eps : float
        Pequeño margen para evitar extremos.
    image_shape : tuple
        Forma de una imagen.
    device : str
        CPU o GPU.

    Devuelve
    --------
    synthetic_images_t : Tensor
        Trayectorias generadas con shape:
            (batch, channels, height, width, n_steps + 1)
    """
    score_model.eval()

    # No empezamos exactamente en T, sino en T - eps
    t0 = T - eps
    t_end = eps

    # Calculamos sigma(t0) para inicializar desde la gaussiana final
    sigma_T = diffusion_process.sigma_t(
        torch.tensor([t0], device=device, dtype=torch.float32)
    ).item()

    # Inicialización:
    #   x(t0) ~ N(0, sigma(t0)^2 I)
    image_T = sigma_T * torch.randn(n_images, *image_shape, device=device)

    with torch.no_grad():
        _, synthetic_images_t = dfp.euler_maruyama_integrator(
            image_T,
            t_0=t0,
            t_end=t_end,
            n_steps=n_steps,
            drift_coefficient=partial(
                backward_drift_coefficient,
                score_model=score_model,
                diffusion_process=diffusion_process,
                scheme=scheme,
                beta_t=beta_t,
                diffusion_coefficient=diffusion_process.diffusion_coefficient,
            ),
            diffusion_coefficient=diffusion_process.diffusion_coefficient,
        )

    return synthetic_images_t


class MNISTColorWrapper(Dataset):
    """
    Wrapper para convertir MNIST gris en versión RGB.
    
    mode:
    - "rgb_repeat": repite el canal gris 3 veces
    - "random_foreground": colorea el dígito con un color aleatorio fijo por imagen
    """
    def __init__(self, base_dataset, mode="rgb_repeat", seed=42):
        self.base_dataset = base_dataset
        self.mode = mode
        self.seed = seed

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]   # x: (1, 28, 28)

        if self.mode == "rgb_repeat":
            x = x.repeat(3, 1, 1)      # (3, 28, 28)

        elif self.mode == "random_foreground":
            g = torch.Generator()
            g.manual_seed(self.seed + idx)
            color = torch.rand(3, 1, 1, generator=g)   # color fijo por imagen
            x = x.repeat(3, 1, 1) * color

        else:
            raise ValueError("mode debe ser 'rgb_repeat' o 'random_foreground'")

        return x, y


def load_mnist_data_color(
    digit=None,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    color_mode="rgb_repeat",
):
    """
    Carga MNIST en formato RGB.
    """
    data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    if digit is not None:
        indices_digit = torch.where(data.targets == digit)[0]
        data = Subset(data, indices_digit)

    data = MNISTColorWrapper(data, mode=color_mode)

    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return data_loader, data