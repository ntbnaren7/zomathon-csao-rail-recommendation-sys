"""
CSAO Reranker — Multi-task Deep & Cross Network with dual prediction heads.

Architecture:
- Input: concatenated [temporal, user, cart, candidate] feature vector
- Deep & Cross Network backbone
- Head 1: P(user adds this item) — primary acceptance prediction
- Head 2: P(user completes order) — C2O auxiliary prediction
- Combined score: α × P(accept) + (1-α) × P(complete)

Designed for RTX 4060 (8GB VRAM) — efficient architecture with mixed precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class CrossNetwork(nn.Module):
    """Cross Network layer — learns explicit feature interactions."""

    def __init__(self, input_dim: int, n_cross_layers: int = 3):
        super().__init__()
        self.n_layers = n_cross_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01)
            for _ in range(n_cross_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(n_cross_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for w, b in zip(self.weights, self.biases):
            # x_{l+1} = x_0 * (x_l^T * w_l) + b_l + x_l
            xw = torch.matmul(x, w)  # (batch, 1)
            x = x0 * xw + b + x
        return x


class DeepNetwork(nn.Module):
    """Deep (MLP) component of DCN-V2."""

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CSAOReranker(nn.Module):
    """
    Multi-task DCN-V2 Reranker for Cart Add-On recommendation.

    Two prediction heads:
    - Acceptance head: Will the user add this item?
    - C2O head: Will the user complete the order?
    """

    def __init__(self, input_dim: int, cross_layers: int = 3,
                 deep_dims: list = None, dropout: float = 0.2):
        super().__init__()

        if deep_dims is None:
            deep_dims = [256, 128, 64]

        # DCN-V2 components
        self.cross_net = CrossNetwork(input_dim, cross_layers)
        self.deep_net = DeepNetwork(input_dim, deep_dims, dropout)

        # Combined dimension = cross output + deep output
        combined_dim = input_dim + self.deep_net.output_dim

        # Shared representation layer
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Head 1: Acceptance prediction (will user add this item?)
        self.acceptance_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        # Head 2: C2O prediction (will user complete the order?)
        self.c2o_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            acceptance_logits: (batch_size, 1) — P(item added)
            c2o_logits: (batch_size, 1) — P(order completed)
        """
        # DCN-V2: parallel cross + deep
        cross_out = self.cross_net(x)
        deep_out = self.deep_net(x)

        # Concatenate and pass through shared layers
        combined = torch.cat([cross_out, deep_out], dim=1)
        shared = self.shared(combined)

        # Dual heads
        accept_logits = self.acceptance_head(shared)
        c2o_logits = self.c2o_head(shared)

        return accept_logits, c2o_logits

    def predict_score(self, x: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
        """
        Combined ranking score for inference.
        score = α × σ(accept_logits) + (1-α) × σ(c2o_logits)
        """
        accept_logits, c2o_logits = self.forward(x)
        accept_prob = torch.sigmoid(accept_logits)
        c2o_prob = torch.sigmoid(c2o_logits)
        return alpha * accept_prob + (1 - alpha) * c2o_prob


class MultiTaskLoss(nn.Module):
    """Combined loss with focal loss, label smoothing, and learnable task weights."""

    def __init__(self, alpha: float = 0.7, label_smoothing: float = 0.02):
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        # Learnable log-variance for uncertainty weighting (Kendall et al.)
        self.log_var_accept = nn.Parameter(torch.zeros(1))
        self.log_var_c2o = nn.Parameter(torch.zeros(1))

    def _smooth_bce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """BCE with label smoothing."""
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)

    def forward(self, accept_logits: torch.Tensor, c2o_logits: torch.Tensor,
                y_accept: torch.Tensor, y_c2o: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute multi-task loss with uncertainty weighting."""
        loss_accept = self._smooth_bce(accept_logits.squeeze(), y_accept)
        loss_c2o = self._smooth_bce(c2o_logits.squeeze(), y_c2o)

        # Uncertainty-weighted multi-task loss
        precision_accept = torch.exp(-self.log_var_accept)
        precision_c2o = torch.exp(-self.log_var_c2o)

        total_loss = (
            precision_accept * loss_accept + self.log_var_accept +
            precision_c2o * loss_c2o + self.log_var_c2o
        )

        metrics = {
            "loss_accept": loss_accept.item(),
            "loss_c2o": loss_c2o.item(),
            "loss_total": total_loss.item(),
            "weight_accept": precision_accept.item(),
            "weight_c2o": precision_c2o.item(),
        }

        return total_loss, metrics


if __name__ == "__main__":
    # Quick architecture test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Simulated feature vector dimensions
    # temporal(18) + user(9) + cart(18) + candidate(14) = 59
    INPUT_DIM = 59
    BATCH_SIZE = 256

    model = CSAOReranker(input_dim=INPUT_DIM).to(device)
    loss_fn = MultiTaskLoss().to(device)

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    x = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)
    y_accept = torch.randint(0, 2, (BATCH_SIZE,)).float().to(device)
    y_c2o = torch.randint(0, 2, (BATCH_SIZE,)).float().to(device)

    accept_logits, c2o_logits = model(x)
    print(f"Accept logits shape: {accept_logits.shape}")
    print(f"C2O logits shape: {c2o_logits.shape}")

    # Test loss
    loss, metrics = loss_fn(accept_logits, c2o_logits, y_accept, y_c2o)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Test inference score
    scores = model.predict_score(x)
    print(f"Scores shape: {scores.shape}, range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Latency benchmark
    import time
    model.eval()
    with torch.no_grad():
        # Single sample latency (simulates real-time inference)
        x_single = torch.randn(1, INPUT_DIM).to(device)
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = model.predict_score(x_single)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        print(f"\nSingle-sample latency: {np.median(times):.2f}ms (median)")
        print(f"  P50: {np.percentile(times, 50):.2f}ms")
        print(f"  P90: {np.percentile(times, 90):.2f}ms")
        print(f"  P99: {np.percentile(times, 99):.2f}ms")

        # Batch latency (100 candidates scored at once)
        x_batch = torch.randn(100, INPUT_DIM).to(device)
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = model.predict_score(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        print(f"\nBatch (100 candidates) latency: {np.median(times):.2f}ms")
