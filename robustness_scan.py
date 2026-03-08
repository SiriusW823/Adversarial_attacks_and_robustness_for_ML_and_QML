from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

DEFAULT_EPS_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def _zero_model_grad(model: torch.nn.Module) -> None:
    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()


def _prepare_attack_tensors(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return images.detach().clone().to(dtype=torch.float32), labels.detach().to(dtype=torch.long)


def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Generate a batched FGSM adversarial example tensor for a classifier."""
    was_training = model.training
    model.eval()

    adv_images, labels = _prepare_attack_tensors(images, labels)
    adv_images.requires_grad_(True)

    _zero_model_grad(model)
    logits = model(adv_images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    perturbed = adv_images + eps * adv_images.grad.sign()
    perturbed = torch.clamp(perturbed, 0.0, 1.0).detach()

    if was_training:
        model.train()

    return perturbed


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 0.25,
    alpha: float = 0.04,
    steps: int = 10,
) -> torch.Tensor:
    """Run batched L-infinity PGD with step size ``alpha`` for ``steps`` updates."""
    was_training = model.training
    model.eval()

    original_images, labels = _prepare_attack_tensors(images, labels)
    adv_images = original_images.clone()

    for _ in range(steps):
        adv_images.requires_grad_()
        _zero_model_grad(model)

        logits = model(adv_images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            perturbation = torch.clamp(
                adv_images - original_images,
                min=-eps,
                max=eps,
            )
            adv_images = torch.clamp(original_images + perturbation, 0.0, 1.0)

        adv_images = adv_images.detach()

    if was_training:
        model.train()

    return adv_images


def evaluate_robustness_curve(
    model: torch.nn.Module,
    test_loader: Iterable[Sequence[torch.Tensor]],
    attack_type: str,
    eps_list: Sequence[float],
) -> list[float]:
    """Evaluate accuracy across ``eps_list`` for ``fgsm`` or ``pgd`` attacks."""
    attack_name = attack_type.lower()
    attack_map = {"fgsm": fgsm_attack, "pgd": pgd_attack}
    if attack_name not in attack_map:
        raise ValueError(
            f"Unsupported attack_type '{attack_type}'. Expected one of {tuple(attack_map)}."
        )

    attack_fn = attack_map[attack_name]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    accuracies: list[float] = []
    for eps in eps_list:
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            if eps == 0:
                eval_images = images
            else:
                eval_images = attack_fn(model, images, labels, eps=eps)

            with torch.no_grad():
                predictions = model(eval_images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracies.append(correct / total if total else 0.0)

    if was_training:
        model.train()

    return accuracies


def plot_robustness_decay_curve(
    eps_list: Sequence[float],
    ml_fgsm_curve: Sequence[float],
    ml_pgd_curve: Sequence[float],
    qml_fgsm_curve: Sequence[float],
    qml_pgd_curve: Sequence[float],
    ax: plt.Axes | None = None,
):
    """Plot ML/QML FGSM and PGD accuracy curves against epsilon."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    fig = ax.figure

    ax.plot(eps_list, ml_fgsm_curve, marker="o", label="ML-FGSM")
    ax.plot(eps_list, ml_pgd_curve, marker="s", label="ML-PGD")
    ax.plot(eps_list, qml_fgsm_curve, marker="^", label="QML-FGSM")
    ax.plot(eps_list, qml_pgd_curve, marker="d", label="QML-PGD")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness Decay Curve")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    return fig, ax
