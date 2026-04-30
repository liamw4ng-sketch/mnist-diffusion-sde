
import torch
import matplotlib.pyplot as plt

from classifier_model import TimeDependentMNISTClassifier
from score_model import ScoreNet
from diffusion_model_general import create_diffusion_process
from conditional_euler_maruyama import generate_digit_class


def show_generated_images(images: torch.Tensor, n_cols: int = 4):
    images = images.detach().cpu()
    n = images.shape[0]
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(images[i, 0], cmap="gray")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion_process, _, sigma_t, _ = create_diffusion_process(
        scheme="vp",
        schedule="linear",
        T=1.0,
        beta_min=0.1,
        beta_max=20.0,
    )

    score_model = ScoreNet(marginal_prob_std=sigma_t).to(device)
    score_model.load_state_dict(torch.load("check_point.pth", map_location=device))
    score_model.eval()

    classifier_model = TimeDependentMNISTClassifier().to(device)
    classifier_model.load_state_dict(torch.load("classifier_checkpoint.pth", map_location=device))
    classifier_model.eval()

    digit = 3
    _, trajectory = generate_digit_class(
        digit=digit,
        n_images=8,
        score_model=score_model,
        classifier_model=classifier_model,
        diffusion_process=diffusion_process,
        n_steps=500,
        T=1.0,
        image_shape=(1, 28, 28),
        device=device,
        guidance_scale=1.5,
    )

    final_images = trajectory[..., -1]
    show_generated_images(final_images)
