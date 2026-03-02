"""
CSAO Training Orchestrator — GPU-accelerated training with mixed precision.

Handles:
- Data loading with temporal train/val/test split
- Feature engineering pipeline execution
- Model training with mixed precision (AMP) on RTX 4060
- Early stopping + learning rate scheduling
- Metric tracking and model checkpointing
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import pickle

from features.feature_engineering import FeatureEngineer
from models.reranker import CSAOReranker, MultiTaskLoss


class CSAODataset(Dataset):
    """PyTorch dataset for CSAO training data."""

    def __init__(self, X: np.ndarray, y_accept: np.ndarray, y_c2o: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y_accept = torch.FloatTensor(y_accept)
        self.y_c2o = torch.FloatTensor(y_c2o)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_accept[idx], self.y_c2o[idx]


class Trainer:
    """Training orchestrator with GPU acceleration and mixed precision."""

    def __init__(self, data_dir: str = "data/generated",
                 model_dir: str = "models/checkpoints",
                 device: str = None, use_amp: bool = True):
        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"

        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"Mixed Precision (AMP): {self.use_amp}")

    def load_data(self) -> Tuple[pd.DataFrame, ...]:
        """Load generated datasets."""
        print("\nLoading data...")
        items_df = pd.read_csv(f"{self.data_dir}/items.csv")
        restaurants_df = pd.read_csv(f"{self.data_dir}/restaurants.csv")
        users_df = pd.read_csv(f"{self.data_dir}/users.csv")
        orders_df = pd.read_csv(f"{self.data_dir}/orders.csv")
        order_items_df = pd.read_csv(f"{self.data_dir}/order_items.csv")
        training_df = pd.read_csv(f"{self.data_dir}/training_data.csv")

        print(f"  Items: {len(items_df)}, Users: {len(users_df)}")
        print(f"  Orders: {len(orders_df)}, Training examples: {len(training_df)}")

        return items_df, restaurants_df, users_df, orders_df, order_items_df, training_df

    def temporal_split(self, training_df: pd.DataFrame, orders_df: pd.DataFrame,
                       train_ratio: float = 0.8, val_ratio: float = 0.1
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal train/val/test split — mimics production deployment.
        Train on older orders, test on newer orders.
        """
        print("\nPerforming temporal split...")

        # Get timestamp for each training example via order_id
        order_timestamps = orders_df.set_index("order_id")["timestamp"]
        training_df = training_df.copy()
        training_df["timestamp"] = training_df["order_id"].map(order_timestamps)
        training_df = training_df.dropna(subset=["timestamp"])
        training_df = training_df.sort_values("timestamp")

        n = len(training_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = training_df.iloc[:train_end]
        val_df = training_df.iloc[train_end:val_end]
        test_df = training_df.iloc[val_end:]

        print(f"  Train: {len(train_df)} ({len(train_df)/n:.0%})")
        print(f"  Val:   {len(val_df)} ({len(val_df)/n:.0%})")
        print(f"  Test:  {len(test_df)} ({len(test_df)/n:.0%})")
        print(f"  Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"  Test period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

        # Drop timestamp column (not a feature)
        for df in [train_df, val_df, test_df]:
            df.drop(columns=["timestamp"], inplace=True, errors='ignore')

        return train_df, val_df, test_df

    def build_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                       test_df: pd.DataFrame, items_df: pd.DataFrame,
                       users_df: pd.DataFrame, orders_df: pd.DataFrame,
                       order_items_df: pd.DataFrame
                       ) -> Tuple[Tuple[np.ndarray, ...], ...]:
        """Build feature matrices for train/val/test splits."""
        print("\nBuilding feature engineering pipeline...")
        fe = FeatureEngineer(items_df, orders_df, order_items_df)

        print("\nBuilding training features...")
        X_train, y_train, y_c2o_train = fe.build_training_features(train_df, users_df)

        print("\nBuilding validation features...")
        X_val, y_val, y_c2o_val = fe.build_training_features(val_df, users_df)

        print("\nBuilding test features...")
        X_test, y_test, y_c2o_test = fe.build_training_features(test_df, users_df)

        return (X_train, y_train, y_c2o_train), (X_val, y_val, y_c2o_val), (X_test, y_test, y_c2o_test)

    def train(self, n_epochs: int = 30, batch_size: int = 1024,
              lr: float = 1e-3, patience: int = 5):
        """Full training pipeline."""
        start_time = time.time()

        # Load data
        items_df, restaurants_df, users_df, orders_df, order_items_df, training_df = self.load_data()

        # Temporal split
        train_df, val_df, test_df = self.temporal_split(training_df, orders_df)

        # Build features
        train_data, val_data, test_data = self.build_features(
            train_df, val_df, test_df, items_df, users_df, orders_df, order_items_df
        )
        X_train, y_train, y_c2o_train = train_data
        X_val, y_val, y_c2o_val = val_data
        X_test, y_test, y_c2o_test = test_data

        # Save feature matrices for fast reloading
        np.savez_compressed(
            f"{self.model_dir}/features.npz",
            X_train=X_train, y_train=y_train, y_c2o_train=y_c2o_train,
            X_val=X_val, y_val=y_val, y_c2o_val=y_c2o_val,
            X_test=X_test, y_test=y_test, y_c2o_test=y_c2o_test,
        )
        print(f"\nFeature matrices saved to {self.model_dir}/features.npz")

        # StandardScaler normalization (fit on train, apply to all)
        print("\nNormalizing features with StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Save scaler for inference
        with open(f"{self.model_dir}/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Scaler saved (mean range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}])")

        # Create data loaders
        train_dataset = CSAODataset(X_train, y_train, y_c2o_train)
        val_dataset = CSAODataset(X_val, y_val, y_c2o_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False,
                                num_workers=0, pin_memory=True)

        # Initialize model
        input_dim = X_train.shape[1]
        print(f"\nInput dimension: {input_dim}")

        model = CSAOReranker(
            input_dim=input_dim,
            cross_layers=3,
            deep_dims=[256, 128, 64],
            dropout=0.2
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

        # Loss and optimizer
        loss_fn = MultiTaskLoss().to(self.device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=lr, weight_decay=1e-4
        )

        # Cosine annealing with linear warmup
        total_steps = n_epochs * len(train_loader)
        warmup_steps = int(0.05 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = GradScaler("cuda") if self.use_amp else None

        # Training loop
        best_val_auc = 0.0
        patience_counter = 0
        history = []

        print(f"\n{'='*60}")
        print(f"Training for {n_epochs} epochs")
        print(f"{'='*60}")

        for epoch in range(n_epochs):
            epoch_start = time.time()

            # Train
            model.train()
            train_losses = []
            for batch_x, batch_y, batch_c2o in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_c2o = batch_c2o.to(self.device)

                optimizer.zero_grad()

                if self.use_amp:
                    with autocast("cuda"):
                        accept_logits, c2o_logits = model(batch_x)
                        loss, metrics = loss_fn(accept_logits, c2o_logits, batch_y, batch_c2o)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    accept_logits, c2o_logits = model(batch_x)
                    loss, metrics = loss_fn(accept_logits, c2o_logits, batch_y, batch_c2o)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_losses.append(metrics)

                # Step LR scheduler per batch (for warmup + cosine)
                scheduler.step()

            val_metrics = self._evaluate(model, val_loader)

            # Log
            avg_train_loss = np.mean([m["loss_total"] for m in train_losses])
            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch+1:2d}/{n_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            # Early stopping / checkpointing
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "epoch": epoch + 1,
                    "val_auc": best_val_auc,
                }, f"{self.model_dir}/best_model.pt")
                print(f"  ✓ New best model saved (AUC: {best_val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Final evaluation on test set
        print(f"\n{'='*60}")
        print("Final Evaluation on Test Set")
        print(f"{'='*60}")

        # Load best model
        checkpoint = torch.load(f"{self.model_dir}/best_model.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

        test_dataset = CSAODataset(X_test, y_test, y_c2o_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)
        test_metrics = self._evaluate(model, test_loader)

        for k, v in test_metrics.items():
            print(f"  {k:20s}: {v:.4f}")

        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time/60:.1f} minutes")

        # Save training history
        pd.DataFrame(history).to_csv(f"{self.model_dir}/training_history.csv", index=False)

        return model, test_metrics, history

    def _evaluate(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a data loader."""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y, _ in data_loader:
                batch_x = batch_x.to(self.device)
                scores = model.predict_score(batch_x).cpu().numpy().flatten()
                all_preds.extend(scores)
                all_labels.extend(batch_y.numpy())

        preds = np.array(all_preds)
        labels = np.array(all_labels)

        # Compute metrics
        auc = roc_auc_score(labels, preds)
        binary_preds = (preds > 0.5).astype(int)
        accuracy = (binary_preds == labels).mean()

        # Precision@K and NDCG@K (computed per-group would be ideal,
        # but for aggregate metrics we use the full set)
        # Sort by predicted score descending
        sorted_idx = np.argsort(-preds)
        for k in [3, 5, 8]:
            top_k_idx = sorted_idx[:k * 100]  # Scale K for the full dataset
            top_k_labels = labels[top_k_idx]

        # NDCG on random groups of 20 (simulate recommendation lists)
        ndcg_scores = []
        n = len(preds)
        for i in range(0, min(n - 20, 5000), 20):
            chunk_labels = labels[i:i+20].reshape(1, -1)
            chunk_preds = preds[i:i+20].reshape(1, -1)
            if chunk_labels.sum() > 0:
                try:
                    ndcg_scores.append(ndcg_score(chunk_labels, chunk_preds, k=8))
                except Exception:
                    pass

        return {
            "auc": auc,
            "accuracy": accuracy,
            "ndcg_8": np.mean(ndcg_scores) if ndcg_scores else 0.0,
            "positive_rate": labels.mean(),
            "avg_score_pos": preds[labels == 1].mean() if (labels == 1).any() else 0,
            "avg_score_neg": preds[labels == 0].mean() if (labels == 0).any() else 0,
        }


if __name__ == "__main__":
    trainer = Trainer(data_dir="data/generated", model_dir="models/checkpoints")
    model, test_metrics, history = trainer.train(
        n_epochs=40,
        batch_size=512,
        lr=3e-4,
        patience=8
    )
