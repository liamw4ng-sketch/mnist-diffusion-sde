# diffusion_model.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import diffusion_process as dfp
from functools import partial
from score_model import ScoreNet
from torch.optim import Adam
import tqdm.notebook 
import matplotlib.pyplot as plt


# Función para cargar y preprocesar los datos
def load_mnist_data(digit=3, batch_size=32):
    data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
    indices_digit = torch.where(data.targets == digit)[0]
    data_train = Subset(data, indices_digit)
    
    data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    
    return data_loader, data_train


# Función para crear el proceso de difusións
def create_diffusion_process(sigma=25.0):
    def bm_drift_coefficient(x_t, t):
        return torch.zeros_like(x_t)
    
    def bm_diffusion_coefficient(t, sigma):
        return sigma ** t
    
    def bm_mu_t(x_0, t):
        return x_0
    
    def bm_sigma_t(t, sigma):
        return torch.sqrt(0.5 * (sigma ** (2 * t) - 1.0) / np.log(sigma))

    drift_coefficient = bm_drift_coefficient
    diffusion_coefficient = partial(bm_diffusion_coefficient, sigma=sigma)
    mu_t = bm_mu_t
    sigma_t = partial(bm_sigma_t, sigma=sigma)

    diffusion_process = dfp.GaussianDiffussionProcess(drift_coefficient, diffusion_coefficient, mu_t, sigma_t)

    return diffusion_process, diffusion_coefficient, sigma_t


# Función para calcular el drift (coeficiente de deriva) en el proceso inverso
def backward_drift_coefficient(x_t, t, diffusion_coefficient, score_model):
    """ El drift inverso se calcula como:
            b(x_t, t) = -g(t)^2 * score_model(x_t, t)
    """
    # Calcular el "score", que es el gradiente en cada paso de tiempo
    score = score_model(x_t, t)

    # Calculamos g(t)
    g_t = diffusion_coefficient(t)
    g_t = g_t.view(-1, 1, 1, 1)  # reshape

    # Calcular beta_t
    beta_t = g_t**2  # beta_t = g(t)^2

    # Calcular el drift inverso
    drift = -(beta_t * score)
    return drift


# Función para entrenar el modelo
def train_model(data_loader, diffusion_process, score_model, n_epochs=15, learning_rate=1e-3, device='cpu'):
    """
    Entrena el modelo ScoreNet usando el proceso de difusión.

    Parámetros:
    - data_loader: DataLoader de PyTorch que proporciona los datos en lotes.
    - diffusion_process: Proceso de difusión usado para calcular la pérdida.
    - score_model: Modelo de ScoreNet que será entrenado.
    - n_epochs: Número de épocas de entrenamiento (default: 15).
    - learning_rate: Tasa de aprendizaje para el optimizador (default: 1e-3).
    - device: Dispositivo donde se ejecutará el entrenamiento (default: "cpu").
    
    Retorna:
    - score_model: Modelo entrenado.
    """

    # Configurar el optimizador Adam
    optimizer = Adam(score_model.parameters(), lr=learning_rate)

    # Crear una barra de progreso para el entrenamiento
    tqdm_epoch = tqdm.notebook.trange(n_epochs)

    for epoch in tqdm_epoch:
        avg_loss = 0.0
        num_items = 0
        
        # Iterar sobre los lotes de datos
        for x, _ in data_loader:
            x = x.to(device)
            loss = diffusion_process.loss_function(score_model, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        # Mostrar la pérdida promedio en la barra de progreso
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

    # Guardar el modelo entrenado
    torch.save(score_model.state_dict(), 'check_point.pth')


# Función para generar imágenes con el modelo entrenado
def generate_images(score_model, diffusion_process, n_images=3, device='cpu'):
    T = 1.0
    image_T = torch.randn(n_images, 1, 28, 28, device=device)

    with torch.no_grad():
        times, synthetic_images_t = dfp.euler_maruyama_integrator(
            image_T,
            t_0=T,
            t_end=1.0e-3,
            n_steps=500,
            drift_coefficient=partial(
                backward_drift_coefficient, 
                diffusion_coefficient=diffusion_process.diffusion_coefficient, 
                score_model=score_model),
            diffusion_coefficient=diffusion_process.diffusion_coefficient,
        )

    return synthetic_images_t
