import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.temporal_artithmetic import nlse, nlde

def _load_temporal_constants(constants_path: str, max_terms: int):
    data = torch.load(constants_path, weights_only=False)
    if "C_VALUES" not in data or "D_VALUES" not in data:
        raise RuntimeError(f"Missing C_VALUES/D_VALUES in {constants_path}.")
    if max_terms not in data["C_VALUES"] or max_terms not in data["D_VALUES"]:
        raise RuntimeError(f"Missing max_terms={max_terms} in {constants_path}.")

    c_vals = data["C_VALUES"][max_terms].clone().detach()
    d_vals = data["D_VALUES"][max_terms].clone().detach()

    e_map = data.get("E_VALUES", data["C_VALUES"])
    f_map = data.get("F_VALUES", data["D_VALUES"])
    if max_terms not in e_map or max_terms not in f_map:
        raise RuntimeError(f"Missing E/F (or fallback) for max_terms={max_terms} in {constants_path}.")

    e_vals = e_map[max_terms].clone().detach()
    f_vals = f_map[max_terms].clone().detach()
    return c_vals, d_vals, e_vals, f_vals


class TemporalDigitClassifier(nn.Module):
    """MNIST classifier in delay (temporal) space.

    Inputs are normalized to importance in (0,1], mapped to delay with ``-log``.
    Sums use nLSE, differences use nLDE; product/ratio of importances are delay add/subtract.
    Class logits are ``-output_beta * delay``; ``softmax`` yields probabilities.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 10,
        max_terms: int = 10,
        epsilon: float = 1e-9,
        output_beta: float = 1.0,
        constants_path: str = "constants/orig_constants.pt",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.epsilon = float(epsilon)
        self.output_beta = float(output_beta)

        c_vals, d_vals, e_vals, f_vals = _load_temporal_constants(constants_path, max_terms)
        self.register_buffer("C", c_vals)
        self.register_buffer("D", d_vals)
        self.register_buffer("E", e_vals)
        self.register_buffer("F", f_vals)

        # Delays near log(fan-in) so nLSE sums stay in a trainable range.
        target_w1 = math.log(input_dim)
        self.w1_delay = nn.Parameter(
            torch.empty(hidden_dim, input_dim).uniform_(target_w1 - 0.5, target_w1 + 0.5)
        )
        self.b1_delay = nn.Parameter(torch.zeros(hidden_dim))

        target_w2 = math.log(hidden_dim)
        self.w2_delay = nn.Parameter(
            torch.empty(num_classes, hidden_dim).uniform_(target_w2 - 0.5, target_w2 + 0.5)
        )
        self.b2_delay = nn.Parameter(torch.zeros(num_classes))

    def _temporal_add_pair(self, a_delay: torch.Tensor, b_delay: torch.Tensor) -> torch.Tensor:
        orig_shape = a_delay.shape
        out = nlse(
            a_delay.reshape(-1, 1),
            b_delay.reshape(-1, 1),
            self.C,
            self.D,
        )
        return out.reshape(orig_shape)

    def _temporal_sub_pair(self, a_delay: torch.Tensor, b_delay: torch.Tensor) -> torch.Tensor:
        orig_shape = a_delay.shape
        x_p = torch.minimum(a_delay, b_delay)
        y_p = torch.maximum(a_delay, b_delay)
        out = nlde(
            x_p.reshape(-1, 1),
            y_p.reshape(-1, 1),
            self.E,
            self.F,
        )
        return out.reshape(orig_shape)

    def _temporal_reduce_add(self, values_delay: torch.Tensor, dim: int) -> torch.Tensor:
        """Pairwise nLSE along *dim* (binary tree)."""
        values = values_delay.movedim(dim, -1)
        n = values.size(-1)
        if n == 0:
            raise ValueError("Cannot reduce empty tensor in temporal add.")

        while n > 1:
            remainder = None
            if n % 2 == 1:
                remainder = values[..., -1:]
                values = values[..., :-1]

            first = values[..., 0::2].contiguous()
            second = values[..., 1::2].contiguous()
            values = self._temporal_add_pair(first, second)

            if remainder is not None:
                values = torch.cat([values, remainder], dim=-1)
            n = values.size(-1)

        return values.squeeze(-1)

    @staticmethod
    def _importance_mul_to_delay(a_delay: torch.Tensor, b_delay: torch.Tensor) -> torch.Tensor:
        return a_delay + b_delay

    @staticmethod
    def _importance_div_to_delay(a_delay: torch.Tensor, b_delay: torch.Tensor) -> torch.Tensor:
        return a_delay - b_delay

    def _temporal_linear(
        self,
        x_delay: torch.Tensor,
        w_delay: torch.Tensor,
        b_delay: torch.Tensor,
    ) -> torch.Tensor:
        terms = self._importance_mul_to_delay(
            x_delay.unsqueeze(1),   # (B, 1, in_dim)
            w_delay.unsqueeze(0),   # (1, out_dim, in_dim)
        )

        summed = self._temporal_reduce_add(terms, dim=2)  # (B, out_dim)

        bias_expanded = b_delay.unsqueeze(0).expand_as(summed)
        return self._temporal_add_pair(summed, bias_expanded)

    def normalize_to_importance(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim > 2:
            x = x.reshape(x.size(0), -1)

        x = torch.abs(x)
        row_max = torch.amax(x, dim=1, keepdim=True)
        x = x / (row_max + self.epsilon)
        return torch.clamp(x, min=self.epsilon, max=1.0)

    def importance_to_delay(self, x_importance: torch.Tensor) -> torch.Tensor:
        x_importance = torch.clamp(x_importance, min=self.epsilon, max=1.0)
        return -torch.log(x_importance)

    @staticmethod
    def delay_to_importance(x_delay: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x_delay)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x_importance = self.normalize_to_importance(x)
        x_delay = self.importance_to_delay(x_importance)

        h_delay = self._temporal_linear(x_delay, self.w1_delay, self.b1_delay)

        h_sum_delay = self._temporal_reduce_add(h_delay, dim=1).unsqueeze(1)
        h_norm_delay = self._importance_div_to_delay(h_delay, h_sum_delay)

        out_delay = self._temporal_linear(h_norm_delay, self.w2_delay, self.b2_delay)

        logits = -out_delay * self.output_beta
        probs = torch.softmax(logits, dim=1)
        if return_logits:
            return probs, logits
        return probs


@torch.no_grad()
def visualize_predictions(
    model: TemporalDigitClassifier,
    device: torch.device,
    sample_count: int = 8,
    csv_path: str = "datasets/mnist/train.csv",
) -> None:
    """Plot random MNIST rows from ``csv_path`` with predictions."""
    data = pd.read_csv(csv_path)
    labels_np = data["label"].values
    pixels_np = data.drop(columns=["label"]).values.astype(np.float32) / 255.0

    sample_count = min(max(1, int(sample_count)), len(labels_np))
    indices = np.random.choice(len(labels_np), size=sample_count, replace=False)

    images_flat = torch.tensor(pixels_np[indices], dtype=torch.float32)
    labels = torch.tensor(labels_np[indices], dtype=torch.long)
    images_2d = images_flat.reshape(-1, 1, 28, 28)

    probs = model(images_2d.to(device))
    pred = torch.argmax(probs, dim=1).cpu()
    conf = torch.max(probs, dim=1).values.cpu()

    cols = min(4, sample_count)
    rows = (sample_count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    if hasattr(axes, "reshape"):
        axes = axes.reshape(-1)
    else:
        axes = [axes]

    for idx in range(rows * cols):
        ax = axes[idx]
        ax.axis("off")
        if idx >= sample_count:
            continue

        img = images_flat[idx].reshape(28, 28).numpy()
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(
            f"True: {labels[idx].item()} | Pred: {pred[idx].item()}\nConf: {conf[idx].item():.3f}"
        )

    plt.tight_layout()
    plt.show()


def train(
    model: TemporalDigitClassifier,
    device: torch.device,
    csv_path: str,
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 2e-2,
    val_split: float = 0.15,
) -> TemporalDigitClassifier:
    data = pd.read_csv(csv_path)
    labels_np = data["label"].values
    pixels_np = data.drop(columns=["label"]).values.astype(np.float32) / 255.0

    n = len(labels_np)
    perm = np.random.permutation(n)
    val_size = int(n * val_split)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    x_train = torch.tensor(pixels_np[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels_np[train_idx], dtype=torch.long)
    x_val = torch.tensor(pixels_np[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels_np[val_idx], dtype=torch.long)

    total_batches = (len(train_idx) + batch_size - 1) // batch_size
    print(f"\nTraining: {len(train_idx)} samples, {val_size} val samples, "
          f"{total_batches} batches/epoch, lr={learning_rate}")
    print("-" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    import time

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm_train = torch.randperm(len(x_train))
        epoch_loss = 0.0
        batches = 0
        epoch_start = time.time()

        for start in range(0, len(x_train), batch_size):
            idx = perm_train[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)

            _, logits = model(xb, return_logits=True)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

            if batches % 50 == 0 or batches == total_batches:
                elapsed = time.time() - epoch_start
                print(
                    f"\r  Epoch {epoch}/{num_epochs}  "
                    f"batch {batches}/{total_batches}  "
                    f"loss={epoch_loss / batches:.4f}  "
                    f"[{elapsed:.1f}s]",
                    end="", flush=True,
                )

        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = []
            for start in range(0, len(x_val), batch_size):
                xb = x_val[start : start + batch_size].to(device)
                val_probs.append(model(xb).cpu())
            val_probs = torch.cat(val_probs, dim=0)
            val_pred = torch.argmax(val_probs, dim=1)
            val_acc = (val_pred == y_val).float().mean().item() * 100

        epoch_time = time.time() - epoch_start
        print(
            f"\r  Epoch {epoch}/{num_epochs}  "
            f"loss={epoch_loss / batches:.4f}  "
            f"val_acc={val_acc:.2f}%  "
            f"[{epoch_time:.1f}s]"
        )

    print("-" * 60)
    print("Training complete.\n")
    return model


if __name__ == "__main__":
    device_type = "cpu"
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    device = torch.device(device_type)
    print("Device:", device_type)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "datasets", "mnist", "train.csv")

    model = TemporalDigitClassifier(
        constants_path=os.path.join(project_root, "constants", "orig_constants.pt"),
    ).to(device)

    choice = input("Train model before visualizing? [y/n]: ").strip().lower()
    if choice in ("y", "yes"):
        epochs = input("Number of epochs [20]: ").strip()
        epochs = int(epochs) if epochs else 20
        model = train(model, device, csv_path, num_epochs=epochs)

    model.eval()
    visualize_predictions(model=model, device=device, sample_count=8, csv_path=csv_path)
