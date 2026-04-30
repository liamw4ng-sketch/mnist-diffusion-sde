
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import Adam
from tqdm.auto import trange

from classifier_model import TimeDependentMNISTClassifier
from diffusion_model_general import create_diffusion_process


def train_time_dependent_classifier(
    diffusion_process,
    n_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    checkpoint_name: str = "classifier_checkpoint.pth",
):
    """
    Entrena un clasificador dependiente del tiempo sobre MNIST ruidoso.
    """
    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    classifier = TimeDependentMNISTClassifier().to(device)
    optimizer = Adam(classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    classifier.train()
    epoch_bar = trange(n_epochs)

    for _ in epoch_bar:
        total_loss = 0.0
        total_items = 0
        total_correct = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            eps = 1.0e-5
            t = torch.rand(x.shape[0], device=device) * (1.0 - eps) + eps
            z = torch.randn_like(x)

            mu_t = diffusion_process.mu_t(x, t.view(-1, 1, 1, 1))
            sigma_t = diffusion_process.sigma_t(t).view(-1, 1, 1, 1)
            x_t = mu_t + sigma_t * z

            logits = classifier(x_t, t)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_items += x.shape[0]
            total_correct += (logits.argmax(dim=1) == y).sum().item()

        epoch_bar.set_description(
            f"Classifier Loss: {total_loss / total_items:.6f} | Acc: {total_correct / total_items:.4f}"
        )

    torch.save(classifier.state_dict(), checkpoint_name)
    return classifier


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion_process, _, _, _ = create_diffusion_process(
        scheme="vp",
        schedule="linear",
        T=1.0,
        beta_min=0.1,
        beta_max=20.0,
    )

    train_time_dependent_classifier(
        diffusion_process=diffusion_process,
        n_epochs=10,
        batch_size=128,
        learning_rate=1e-3,
        device=device,
        checkpoint_name="classifier_checkpoint.pth",
    )
