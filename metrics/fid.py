# metrics/fid_mnist.py

import torch


def _get_features(images, classifier, t_value=0.0, batch_size=64, device="cpu"):
    features = []

    classifier.eval()

    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size].to(device)

            t = torch.full((batch.shape[0],), t_value, device=device)

            # coger features antes de la última capa
            h = classifier.conv1(batch)
            h = classifier.act(classifier.norm1(h + classifier.time1(classifier.embed(t))))

            h = classifier.conv2(h)
            h = classifier.act(classifier.norm2(h + classifier.time2(classifier.embed(t))))

            h = classifier.conv3(h)
            h = classifier.act(classifier.norm3(h + classifier.time3(classifier.embed(t))))

            h = classifier.pool(h).flatten(1)

            features.append(h)

    return torch.cat(features, dim=0)


def _mean_and_cov(features):
    mu = features.mean(dim=0)
    xc = features - mu
    cov = (xc.T @ xc) / (features.shape[0] - 1)
    return mu, cov


def _sqrtm_psd(matrix, eps=1e-12):
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T


def _trace_sqrt_product(cov1, cov2):
    sqrt_cov1 = _sqrtm_psd(cov1)
    middle = sqrt_cov1 @ cov2 @ sqrt_cov1
    sqrt_middle = _sqrtm_psd(middle)
    return torch.trace(sqrt_middle)


def calculate_fid_mnist(
    real_images,
    generated_images,
    classifier,
    batch_size=64,
    device="cpu",
):
    real_features = _get_features(real_images, classifier, batch_size=batch_size, device=device)
    gen_features = _get_features(generated_images, classifier, batch_size=batch_size, device=device)

    mu_r, cov_r = _mean_and_cov(real_features)
    mu_g, cov_g = _mean_and_cov(gen_features)

    mean_term = torch.sum((mu_r - mu_g) ** 2)
    trace_term = torch.trace(cov_r) + torch.trace(cov_g) - 2 * _trace_sqrt_product(cov_r, cov_g)

    return float((mean_term + trace_term).item())