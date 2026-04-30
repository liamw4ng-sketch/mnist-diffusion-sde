# metrics/inception_score_mnist.py

import torch


def _get_probs(images, classifier, t_value=0.0, batch_size=64, device="cpu"):
    probs = []

    classifier.eval()

    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size].to(device)

            t = torch.full((batch.shape[0],), t_value, device=device)

            logits = classifier(batch, t)
            p = torch.softmax(logits, dim=1)

            probs.append(p)

    return torch.cat(probs, dim=0)


def calculate_inception_score_mnist(
    generated_images,
    classifier,
    batch_size=64,
    splits=10,
    device="cpu",
):
    probs = _get_probs(generated_images, classifier, batch_size=batch_size, device=device)

    N = probs.shape[0]
    splits = min(splits, N)

    scores = []

    for k in range(splits):
        part = probs[k * N // splits:(k + 1) * N // splits]

        p_y = part.mean(dim=0, keepdim=True)

        kl = part * (torch.log(part + 1e-12) - torch.log(p_y + 1e-12))
        kl_mean = kl.sum(dim=1).mean()

        scores.append(torch.exp(kl_mean))

    scores = torch.stack(scores)

    return float(scores.mean().item()), float(scores.std(unbiased=False).item())