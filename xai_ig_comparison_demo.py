from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from explainability import (
    generate_saliency_map,
    plot_attribution_comparison,
    symmetric_attribution_limits,
)
from quantum_ablation_study import (
    ClassicalCNN,
    CustomQuantumCNN,
    make_plusminus_dataset,
    train_model,
    train_test_split,
)
from robustness_scan import pgd_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Integrated Gradients heatmaps for ML and QML on the same sample."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for both models.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of plus/minus pairs.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Test sample index to visualize.")
    parser.add_argument("--attack-eps", type=float, default=0.15, help="PGD epsilon budget.")
    parser.add_argument("--attack-alpha", type=float, default=0.04, help="PGD step size.")
    parser.add_argument("--attack-steps", type=int, default=10, help="Number of PGD steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training and split.")
    parser.add_argument("--data-seed", type=int, default=42, help="Random seed for data generation.")
    parser.add_argument("--device", default="cpu", help="Torch device to use, e.g. cpu or cuda.")
    parser.add_argument(
        "--plot-path",
        default="integrated_gradients_ml_qml_comparison.png",
        help="Output path for the ML vs QML attribution figure.",
    )
    parser.add_argument("--show-plot", action="store_true", help="Display the figure interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y = make_plusminus_dataset(n_samples=args.n_samples, seed=args.data_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=args.seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

    ml_model = ClassicalCNN()
    qml_model = CustomQuantumCNN(embedding="angle", entangler="strong")

    train_model(ml_model, train_loader, epochs=args.epochs, device=args.device, verbose=False)
    train_model(qml_model, train_loader, epochs=args.epochs, device=args.device, verbose=False)

    sample_idx = args.sample_idx % len(X_test)
    clean_image = X_test[sample_idx : sample_idx + 1]
    target_tensor = y_test[sample_idx : sample_idx + 1]
    target_class = int(target_tensor.item())

    ml_adv_image = pgd_attack(
        ml_model,
        clean_image.to(args.device),
        target_tensor.to(args.device),
        eps=args.attack_eps,
        alpha=args.attack_alpha,
        steps=args.attack_steps,
    ).cpu()
    qml_adv_image = pgd_attack(
        qml_model,
        clean_image.to(args.device),
        target_tensor.to(args.device),
        eps=args.attack_eps,
        alpha=args.attack_alpha,
        steps=args.attack_steps,
    ).cpu()

    ml_clean_attr, ml_adv_attr = generate_saliency_map(
        ml_model,
        clean_image,
        ml_adv_image,
        target_class=target_class,
    )
    qml_clean_attr, qml_adv_attr = generate_saliency_map(
        qml_model,
        clean_image,
        qml_adv_image,
        target_class=target_class,
    )

    with torch.no_grad():
        ml_clean_pred = int(ml_model(clean_image.to(args.device)).argmax(dim=1).item())
        ml_adv_pred = int(ml_model(ml_adv_image.to(args.device)).argmax(dim=1).item())
        qml_clean_pred = int(qml_model(clean_image.to(args.device)).argmax(dim=1).item())
        qml_adv_pred = int(qml_model(qml_adv_image.to(args.device)).argmax(dim=1).item())

    print(f"Selected sample index: {sample_idx} | target class: {target_class}")
    print(f"ML predictions  - clean: {ml_clean_pred}, adversarial: {ml_adv_pred}")
    print(f"QML predictions - clean: {qml_clean_pred}, adversarial: {qml_adv_pred}")

    vmin, vmax = symmetric_attribution_limits(
        ml_clean_attr,
        ml_adv_attr,
        qml_clean_attr,
        qml_adv_attr,
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    plot_attribution_comparison(
        clean_image,
        ml_clean_attr,
        ml_adv_image,
        ml_adv_attr,
        target_class=target_class,
        model_name="ML",
        axes=axes[0],
        add_colorbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    plot_attribution_comparison(
        clean_image,
        qml_clean_attr,
        qml_adv_image,
        qml_adv_attr,
        target_class=target_class,
        model_name="QML",
        axes=axes[1],
        add_colorbar=False,
        vmin=vmin,
        vmax=vmax,
    )

    colorbar = fig.colorbar(
        axes[0, 3].images[0],
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label="Integrated Gradients",
    )
    colorbar.ax.tick_params(labelsize=9)
    fig.suptitle("Integrated Gradients Comparison for the Same Test Image", fontsize=16)
    plot_path = Path(args.plot_path)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved comparison figure to {plot_path.resolve()}")

    if args.show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
