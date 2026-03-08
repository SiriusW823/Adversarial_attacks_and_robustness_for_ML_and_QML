from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients


def _model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is None:
        return torch.device("cpu")
    return parameter.device


def _prepare_single_image(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    prepared = image.detach().clone().to(device)
    if prepared.ndim == 2:
        prepared = prepared.unsqueeze(0).unsqueeze(0)
    elif prepared.ndim == 3:
        prepared = prepared.unsqueeze(0)
    if prepared.ndim != 4:
        raise ValueError(
            "Expected clean_image / adv_image to have shape (H, W), (C, H, W), or (N, C, H, W)."
        )
    return prepared.requires_grad_(True)


def _as_image_array(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    return np.squeeze(image)


def _as_attribution_array(attribution: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(attribution, torch.Tensor):
        attribution = attribution.detach().cpu().numpy()
    attribution = np.asarray(attribution, dtype=np.float32)
    if attribution.ndim == 4:
        attribution = attribution[0]
    if attribution.ndim == 3:
        attribution = attribution.sum(axis=0)
    return np.squeeze(attribution)


def symmetric_attribution_limits(*attributions: torch.Tensor | np.ndarray) -> tuple[float, float]:
    """Return symmetric ``(vmin, vmax)`` limits for fair heatmap comparison."""
    max_abs = 0.0
    for attribution in attributions:
        array = _as_attribution_array(attribution)
        if array.size:
            max_abs = max(max_abs, float(np.max(np.abs(array))))
    if max_abs == 0.0:
        max_abs = 1e-8
    return -max_abs, max_abs


def generate_saliency_map(
    model: torch.nn.Module,
    clean_image: torch.Tensor,
    adv_image: torch.Tensor,
    target_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Integrated Gradients attributions for clean and adversarial images."""
    device = _model_device(model)
    clean_input = _prepare_single_image(clean_image, device)
    adv_input = _prepare_single_image(adv_image, device)

    ig = IntegratedGradients(model)
    baseline_clean = torch.zeros_like(clean_input)
    baseline_adv = torch.zeros_like(adv_input)

    was_training = model.training
    model.eval()
    try:
        clean_attr = ig.attribute(
            clean_input,
            baselines=baseline_clean,
            target=target_class,
            n_steps=30,
        ).detach()
        adv_attr = ig.attribute(
            adv_input,
            baselines=baseline_adv,
            target=target_class,
            n_steps=30,
        ).detach()
    finally:
        if was_training:
            model.train()

    return clean_attr.cpu(), adv_attr.cpu()


def plot_attribution_comparison(
    clean_image: torch.Tensor | np.ndarray,
    clean_attribution: torch.Tensor | np.ndarray,
    adv_image: torch.Tensor | np.ndarray,
    adv_attribution: torch.Tensor | np.ndarray,
    target_class: int,
    model_name: str = "",
    axes: Sequence[plt.Axes] | None = None,
    add_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot a 1x4 clean/adversarial image and attribution comparison."""
    clean_image_array = _as_image_array(clean_image)
    adv_image_array = _as_image_array(adv_image)
    clean_attr_array = _as_attribution_array(clean_attribution)
    adv_attr_array = _as_attribution_array(adv_attribution)

    if vmin is None or vmax is None:
        vmin, vmax = symmetric_attribution_limits(clean_attr_array, adv_attr_array)

    if axes is None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    else:
        axes = list(axes)
        if len(axes) != 4:
            raise ValueError("Expected exactly four matplotlib axes for the 1x4 comparison plot.")
        fig = axes[0].figure

    titles = [
        "Clean Image",
        "Clean Attribution",
        "Adversarial Image",
        "Adv Attribution",
    ]

    axes[0].imshow(clean_image_array, cmap="gray")
    axes[1].imshow(clean_attr_array, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[2].imshow(adv_image_array, cmap="gray")
    heatmap = axes[3].imshow(adv_attr_array, cmap="coolwarm", vmin=vmin, vmax=vmax)

    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.axis("off")

    prefix = f"{model_name} — " if model_name else ""
    axes[0].text(
        0.02,
        -0.1,
        f"{prefix}Target class: {target_class}",
        transform=axes[0].transAxes,
        fontsize=10,
        ha="left",
        va="top",
    )

    if add_colorbar:
        fig.colorbar(heatmap, ax=axes, fraction=0.03, pad=0.02, label="Integrated Gradients")

    return fig, axes
