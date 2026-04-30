# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: <alberto.suarez@uam.es>
"""
# Load packages

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor



def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,  
    drift_coefficient: Callable[float, float],
    diffusion_coefficient: Callable[float],
    seed: Union[int, None] = None
) -> Tensor:
    """Euler-Maruyama integrator (approximate)

     Args:
        x_0: The initial images of dimensions 
            (batch_size, n_channels, image_height, image_width)
        t_0: float,
        t_end: endpoint of the integration interval    
        n_steps: number of integration steps 
        drift_coefficient: Function of :math`(x(t), t)` that defines the drift term            
        diffusion_coefficient: Function of :math`(t)` that defines the diffusion term  
        seed: Seed for the random number generator
        
    Returns:
        x_t: Trajectories that result from the integration of the SDE.
             The shape is (*np.shape(x_0), (n_steps + 1))
            
    Notes:
        The implementation is fully vectorized except for a loop over time.

    Examples:
        >>> import numpy as np
        >>> drift_coefficient = lambda x_t, t: - x_t
        >>> diffusion_coefficient = lambda t: torch.ones_like(t)
        >>> x_0 = torch.tensor(np.reshape(np.arange(120), (2, 3, 5, 4)))
        >>> t_0, t_end = 0.0, 3.0
        >>> n_steps = 6
        >>> times, x_t = euler_maruyama_integrator(
        ...     x_0, t_0, t_end, n_steps, drift_coefficient, diffusion_coefficient, 
        ... )
        >>> print(times)
        tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000])
        >>> print(np.shape(x_t))
        torch.Size([2, 3, 5, 4, 7])
    """
        
    device = x_0.device  #Asegura que los tensores esten en el mismo dispositivo (CPU o GPU)
    
    if seed is not None:
        torch.manual_seed(seed)
        
    # Crea un vector de tiempos equiespaciados desde t_0 hasta t_end
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]  # delta de tiempo
    
    # Preasigna el tensor para guardar todas las trayectorias x_t a lo largo del tiempo
    x_t = torch.tensor(
        np.empty((*np.shape(x_0), len(times))), 
        dtype=torch.float32,
        device=device,
    )
    
    x_t[..., 0] = x_0 # Inicializa con el valor inicial x_0, condicion inicial
    
    # Genera todos los valores de ruido Gaussiano que se usaran en la simulacion
    z = torch.randn_like(x_t)
    z[..., -1] = 0.0 # No noise injection in the last step

    
    for n, t in enumerate(times[:-1]): 
        t = torch.ones(x_0.shape[0], device=device) * t

        # Calcula el siguiente paso usando la formula de Euler-Maruyama:
        # x_{n+1} = x_n + f(x_n, t) * dt + g(t) * sqrt(dt) * z

        x_t[..., n + 1] = ( 
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t) * dt
            + diffusion_coefficient(t).view(-1, 1, 1, 1) 
              * np.sqrt(np.abs(dt)) 
              * z[..., n]
        )
        
    # Devuelve los tiempos y todas las trayectorias x_t
    return times, x_t
    
class DiffussionProcess:

    def __init__(
        self,          
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
     

        
class GaussianDiffussionProcess(DiffussionProcess):
    """    
    GaussianDiffusionProcess models an SDE with Gaussian noise,
    parameterized by time-dependent drift and diffusion coefficients.

    Args:
        drift_coefficient: Callable that defines the drift term f(x_t, t).
        diffusion_coefficient: Callable that defines the diffusion term g(t).
        mu_t: Mean of the transition kernel at time t.
        sigma_t: Standard deviation of the transition kernel at time t.
        
    
    Example 1:
        >>> mu, sigma = 1.5, 2.0
        >>> bm = GaussianDiffussionProcess(
        ...     drift_coefficient=lambda x_t, t: mu,
        ...     diffusion_coefficient=lambda t: sigma,
        ...     mu_t=lambda x_0, t: x_0 + mu*t,
        ...     sigma_t=lambda t: np.sqrt(2.0 * t),
        ... )
        >>> print(bm.drift_coefficient(x_t=3.0, t=10.0))
        1.5
        >>> print(bm.diffusion_coefficient(t=10.0))
        2.0
        >>> print(bm.mu_t(x_0=3.0, t=10.0), bm.sigma_t(t=10.0))
        18.0 4.47213595499958
        

    """
    kind = "Gaussian"
    
    def __init__(
        self,          
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
        mu_t: Callable[float, float] = lambda x_0, t: x_0,
        sigma_t: Callable[float] = lambda t: np.sqrt(t),
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
        self.mu_t = mu_t
        self.sigma_t = sigma_t
    

    def loss_function(
        self,
        score_model, 
        x_0: torch.Tensor, 
        eps: float = 1.0e-5,
    ):
        """The loss function for training score-based generative models.

          Args:
              score_model:  A PyTorch model instance that represents a 
                            time-dependent score-based model.
          x_0: A mini-batch of training data.    
          eps: A tolerance value for numerical stability.
        """
        
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps  
        
        # Generacion de ruido Zn Zn​~N(0,I)
        z = torch.randn_like(x_0, device=x_0.device)

        # Calcular media y desviación estándar en tiempo t
        mu_t = self.mu_t(x_0, t.view(-1, 1, 1, 1))
        sigma_t = self.sigma_t(t).view(-1, 1, 1, 1)

        # Calcular x_t = μ_t + σ_t * Z_m (datos perturbados en tiempo t)
        x_t = mu_t + sigma_t * z

        ## Get the score prediction s(x_t, t; theta) from the score_model
        score = score_model(x_t, t)
        
        #  Compute the loss term || sigma_t(t) * s(x_t, t; theta) + Z ||_2^2
        loss = torch.mean(torch.sum((sigma_t * score + z)**2, dim= tuple(range(1,x_0.ndim))))
        return loss



if __name__ == "__main__":
    import doctest
    doctest.testmod()
