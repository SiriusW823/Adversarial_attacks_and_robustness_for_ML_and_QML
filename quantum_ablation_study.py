from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from robustness_scan import pgd_attack

EmbeddingType = Literal["angle", "amplitude"]
EntanglerType = Literal["basic", "strong"]

DEFAULT_N_QUBITS = 4
DEFAULT_N_LAYERS = 2
DEFAULT_MEASUREMENT_WIRES = 2
AMPLITUDE_NORMALIZATION_EPS = 1e-12
RUNTIME_PACKAGES = ("numpy", "pennylane", "autoray", "torch", "captum")
GPU_QUANTUM_BACKENDS = ("lightning.gpu", "lightning.qubit", "default.qubit")
CPU_QUANTUM_BACKENDS = ("lightning.qubit", "default.qubit")


@dataclass(frozen=True)
class AblationResult:
    embedding: EmbeddingType
    entangler: EntanglerType
    clean_accuracy: float
    adversarial_accuracy: float

    @property
    def label(self) -> str:
        return f"{self.embedding.title()} + {self.entangler.title()}"


def resolve_torch_device(device: torch.device | str = "auto") -> torch.device:
    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = str(device).strip().lower()
        if requested == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(requested)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return resolved


def _quantum_backend_candidates(
    quantum_device: str,
    torch_device: torch.device,
) -> tuple[str, ...]:
    requested = quantum_device.strip().lower()
    if requested != "auto":
        return (requested,)
    return GPU_QUANTUM_BACKENDS if torch_device.type == "cuda" else CPU_QUANTUM_BACKENDS


def resolve_quantum_backend(
    n_qubits: int,
    quantum_device: str = "auto",
    torch_device: torch.device | str = "auto",
) -> tuple[qml.devices.Device, str]:
    resolved_torch_device = resolve_torch_device(torch_device)
    candidates = _quantum_backend_candidates(quantum_device, resolved_torch_device)
    errors: list[str] = []

    for backend_name in candidates:
        try:
            return qml.device(backend_name, wires=n_qubits), backend_name
        except Exception as exc:
            errors.append(f"{backend_name}: {type(exc).__name__}: {exc}")

    candidate_list = ", ".join(candidates)
    raise RuntimeError(
        f"Failed to initialize a PennyLane device from candidates [{candidate_list}]. "
        f"Errors: {' | '.join(errors)}"
    )


def resolve_model_device(
    model: nn.Module,
    device: torch.device | str = "auto",
) -> torch.device:
    requested_device = resolve_torch_device(device)
    model_device = getattr(model, "execution_device", requested_device)
    return resolve_torch_device(model_device)


def make_plusminus_dataset(
    n_samples: int = 200,
    img_size: int = 8,
    noise_std: float = 0.08,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n_samples):
        img_plus = np.zeros((img_size, img_size), dtype=np.float32)
        img_plus[img_size // 2, :] = 1
        img_plus[:, img_size // 2] = 1
        img_plus += noise_std * rng.standard_normal((img_size, img_size), dtype=np.float32)

        img_minus = np.zeros((img_size, img_size), dtype=np.float32)
        img_minus[img_size // 2, :] = 1
        img_minus += noise_std * rng.standard_normal((img_size, img_size), dtype=np.float32)

        X.append(img_plus)
        y.append(0)
        X.append(img_minus)
        y.append(1)

    X = np.clip(np.asarray(X, dtype=np.float32), 0.0, 1.0)
    return (
        torch.tensor(X, dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.asarray(y, dtype=np.int64), dtype=torch.long),
    )


def train_test_split(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=generator)
    n_train = int(train_ratio * len(X))
    indices_train = perm[:n_train]
    indices_test = perm[n_train:]
    return X[indices_train], X[indices_test], y[indices_train], y[indices_test]


class CNNBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)


class ClassicalCNN(nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.feature_dim = 32 * 2 * 2
        self.backbone = CNNBackbone()
        self.fc1 = nn.Linear(self.feature_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc_out = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


def get_embedding_feature_dim(embedding: EmbeddingType, n_qubits: int) -> int:
    """Return the classical feature size required by the chosen quantum embedding.

    Angle embedding applies one rotation angle per qubit, so it consumes ``n_qubits``
    features. Amplitude embedding prepares a full quantum state vector, which needs
    ``2**n_qubits`` amplitudes before optional zero padding.
    """
    return n_qubits if embedding == "angle" else 2**n_qubits


def get_weight_shape(
    entangler: EntanglerType,
    n_layers: int,
    n_qubits: int,
) -> tuple[int, ...]:
    if entangler == "basic":
        return qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)


def apply_embedding(
    embedding: EmbeddingType,
    inputs: torch.Tensor,
    wires: Iterable[int],
) -> None:
    inputs = inputs.to(dtype=torch.float32)
    if embedding == "angle":
        qml.AngleEmbedding(inputs, wires=wires)
        return

    qml.AmplitudeEmbedding(inputs, wires=wires, pad_with=0.0, normalize=True)


def apply_entangler(
    entangler: EntanglerType,
    weights: torch.Tensor,
    wires: Iterable[int],
) -> None:
    if entangler == "basic":
        qml.BasicEntanglerLayers(weights, wires=wires)
        return

    qml.StronglyEntanglingLayers(weights, wires=wires)


def build_quantum_torch_layer(
    embedding: EmbeddingType,
    entangler: EntanglerType,
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    measurement_wires: int = DEFAULT_MEASUREMENT_WIRES,
    quantum_device: str = "auto",
    torch_device: torch.device | str = "auto",
) -> qml.qnn.TorchLayer:
    dev, backend_name = resolve_quantum_backend(
        n_qubits=n_qubits,
        quantum_device=quantum_device,
        torch_device=torch_device,
    )
    wires = tuple(range(n_qubits))

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(inputs: torch.Tensor, weights: torch.Tensor):
        apply_embedding(embedding, inputs, wires)
        apply_entangler(entangler, weights, wires)
        return [qml.expval(qml.PauliZ(i)) for i in range(measurement_wires)]

    weight_shapes = {
        "weights": get_weight_shape(entangler=entangler, n_layers=n_layers, n_qubits=n_qubits)
    }
    layer = qml.qnn.TorchLayer(qnode, weight_shapes)
    layer.quantum_backend = backend_name
    return layer


class CustomQuantumCNN(nn.Module):
    def __init__(
        self,
        embedding: EmbeddingType,
        entangler: EntanglerType,
        n_qubits: int = DEFAULT_N_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
        n_classes: int = 2,
        quantum_device: str = "auto",
        torch_device: torch.device | str = "auto",
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.entangler = entangler
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.torch_device = resolve_torch_device(torch_device)
        self.quantum_device = quantum_device
        self.measurement_wires = min(DEFAULT_MEASUREMENT_WIRES, n_qubits)
        self.feature_dim = 32 * 2 * 2
        self.embedding_feature_dim = get_embedding_feature_dim(embedding, n_qubits)

        self.backbone = CNNBackbone()
        self.fc_feat = nn.Linear(self.feature_dim, self.embedding_feature_dim)
        self.quantum = build_quantum_torch_layer(
            embedding=embedding,
            entangler=entangler,
            n_qubits=n_qubits,
            n_layers=n_layers,
            measurement_wires=self.measurement_wires,
            quantum_device=quantum_device,
            torch_device=self.torch_device,
        )
        self.quantum_backend = self.quantum.quantum_backend
        self.execution_device = (
            self.torch_device
            if self.torch_device.type != "cuda" or self.quantum_backend == "lightning.gpu"
            else torch.device("cpu")
        )
        self.fc_out = nn.Linear(self.measurement_wires, n_classes)

    def _prepare_amplitude_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.real(x) if torch.is_complex(x) else x
        x = torch.nan_to_num(x.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        norms = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

        safe_norms = norms.clamp_min(AMPLITUDE_NORMALIZATION_EPS)
        normalized = x / safe_norms

        default_state = torch.zeros_like(normalized)
        default_state[:, 0] = 1.0
        valid_mask = norms > AMPLITUDE_NORMALIZATION_EPS
        return torch.where(valid_mask, normalized, default_state)

    def _prepare_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_feat(x).to(dtype=torch.float32)
        if self.embedding == "angle":
            return torch.tanh(x)
        # Amplitude embedding in modern PennyLane is stricter about real-valued, normalized inputs.
        return self._prepare_amplitude_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self._prepare_quantum_features(x)
        x = self.quantum(x)
        return self.fc_out(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 80,
    lr: float = 0.005,
    device: torch.device | str = "cpu",
    log_interval: int = 20,
    verbose: bool = True,
) -> nn.Module:
    device = resolve_model_device(model, device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1

        should_log = epoch == 0 or epoch + 1 == epochs or (epoch + 1) % log_interval == 0
        if verbose and batches and should_log:
            avg_loss = running_loss / batches
            print(f"  epoch {epoch + 1:>3}/{epochs}: loss={avg_loss:.4f}")

    return model


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device | str = "cpu",
) -> float:
    device = resolve_model_device(model, device)
    model.eval()
    predictions = model(X.to(device)).argmax(dim=1)
    return (predictions == y.to(device)).float().mean().item()


def run_architecture_ablation(
    attack_eps: float = 0.2,
    attack_alpha: float = 0.04,
    attack_steps: int = 10,
    n_samples: int = 200,
    batch_size: int = 32,
    epochs: int = 80,
    seed: int = 0,
    data_seed: int = 42,
    device: torch.device | str = "cpu",
    quantum_device: str = "auto",
    verbose: bool = True,
) -> list[AblationResult]:
    device = resolve_torch_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = make_plusminus_dataset(n_samples=n_samples, seed=data_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    results: list[AblationResult] = []
    architectures: list[tuple[EmbeddingType, EntanglerType]] = [
        ("angle", "basic"),
        ("angle", "strong"),
        ("amplitude", "basic"),
        ("amplitude", "strong"),
    ]

    for embedding, entangler in architectures:
        if verbose:
            print(f"Training architecture: embedding={embedding}, entangler={entangler}")
        model = CustomQuantumCNN(
            embedding=embedding,
            entangler=entangler,
            quantum_device=quantum_device,
            torch_device=device,
        )
        execution_device = resolve_model_device(model, device)
        if verbose:
            print(
                f"  using requested torch device={device}, quantum backend={model.quantum_backend}, "
                f"execution device={execution_device}"
            )
        train_model(model, train_loader, epochs=epochs, device=execution_device, verbose=verbose)

        clean_accuracy = evaluate_accuracy(model, X_test, y_test, device=execution_device)
        X_adv = pgd_attack(
            model,
            X_test.to(execution_device),
            y_test.to(execution_device),
            eps=attack_eps,
            alpha=attack_alpha,
            steps=attack_steps,
        )
        adversarial_accuracy = evaluate_accuracy(model, X_adv, y_test, device=execution_device)

        results.append(
            AblationResult(
                embedding=embedding,
                entangler=entangler,
                clean_accuracy=clean_accuracy,
                adversarial_accuracy=adversarial_accuracy,
            )
        )

    return results


def plot_ablation_results(
    results: list[AblationResult],
    attack_name: str = "PGD",
    attack_eps: float = 0.2,
    ax: plt.Axes | None = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    fig = ax.figure

    labels = [result.label for result in results]
    clean_scores = [result.clean_accuracy for result in results]
    adversarial_scores = [result.adversarial_accuracy for result in results]

    x = np.arange(len(labels))
    width = 0.36

    clean_bars = ax.bar(x - width / 2, clean_scores, width=width, label="Clean", color="#4C72B0")
    adv_bars = ax.bar(
        x + width / 2,
        adversarial_scores,
        width=width,
        label=f"{attack_name} (eps={attack_eps:.2f})",
        color="#DD8452",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Quantum Architecture Ablation Study")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    for bar_group in (clean_bars, adv_bars):
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    return fig, ax


def get_runtime_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def print_runtime_versions() -> None:
    print("Runtime dependency versions")
    for package_name in RUNTIME_PACKAGES:
        print(f"- {package_name}: {get_runtime_version(package_name)}")
    torch_device = resolve_torch_device("auto")
    print(f"- torch device selected: {torch_device}")
    if torch_device.type == "cuda":
        print(f"- cuda device name: {torch.cuda.get_device_name(torch_device)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 2x2 quantum architecture ablation study.")
    parser.add_argument("--attack-eps", type=float, default=0.2, help="PGD epsilon budget.")
    parser.add_argument("--attack-alpha", type=float, default=0.04, help="PGD step size.")
    parser.add_argument("--attack-steps", type=int, default=10, help="Number of PGD steps.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs per architecture.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of plus/minus pairs.")
    parser.add_argument("--seed", type=int, default=0, help="Model/data split seed.")
    parser.add_argument("--data-seed", type=int, default=42, help="Dataset generation seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use, e.g. 'auto', 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--quantum-device",
        default="auto",
        help="PennyLane device to use, e.g. 'auto', 'lightning.gpu', 'lightning.qubit', or 'default.qubit'.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-architecture training progress logs.",
    )
    parser.add_argument(
        "--plot-path",
        default="quantum_ablation_grouped_bar.png",
        help="Path to save the grouped bar chart.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_runtime_versions()
    resolved_device = resolve_torch_device(args.device)
    probe_model = CustomQuantumCNN(
        embedding="angle",
        entangler="strong",
        quantum_device=args.quantum_device,
        torch_device=resolved_device,
    )
    probe_execution_device = resolve_model_device(probe_model, resolved_device)
    print(
        f"Selected execution path: torch device={resolved_device}, "
        f"quantum backend={probe_model.quantum_backend}, execution device={probe_execution_device}"
    )
    results = run_architecture_ablation(
        attack_eps=args.attack_eps,
        attack_alpha=args.attack_alpha,
        attack_steps=args.attack_steps,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        data_seed=args.data_seed,
        device=resolved_device,
        quantum_device=args.quantum_device,
        verbose=not args.quiet,
    )

    print("Quantum architecture ablation results")
    for result in results:
        print(
            f"- {result.label:<20} | clean={result.clean_accuracy:.3f} "
            f"| adv={result.adversarial_accuracy:.3f}"
        )

    fig, _ = plot_ablation_results(results, attack_eps=args.attack_eps)
    plot_path = Path(args.plot_path)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved grouped bar chart to {plot_path.resolve()}")

    if args.show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
