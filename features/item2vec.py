"""
Item2Vec — Learn item embeddings from order co-occurrence patterns.

Each order is treated as a "sentence" and items as "words".
Uses Skip-gram with negative sampling (like Word2Vec) to learn
dense embeddings that capture item-item relationships.

GPU-accelerated with PyTorch.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import time


class Item2VecDataset(Dataset):
    """Generate skip-gram pairs from order item sequences."""

    def __init__(self, order_items_df: pd.DataFrame, window_size: int = 5,
                 n_negatives: int = 5, n_items: int = 700):
        self.window_size = window_size
        self.n_negatives = n_negatives
        self.n_items = n_items

        # Build item index
        all_items = sorted(order_items_df["item_id"].unique())
        self.item2idx = {iid: i for i, iid in enumerate(all_items)}
        self.idx2item = {i: iid for iid, i in self.item2idx.items()}
        self.vocab_size = len(all_items)

        # Compute frequency for negative sampling (^0.75 smoothing)
        item_counts = order_items_df["item_id"].value_counts()
        freq = np.zeros(self.vocab_size)
        for iid, count in item_counts.items():
            if iid in self.item2idx:
                freq[self.item2idx[iid]] = count
        freq = freq ** 0.75
        self.neg_probs = freq / freq.sum()

        # Generate skip-gram pairs
        print("  Generating skip-gram pairs...")
        self.pairs = []  # (center, context) pairs
        for _, group in order_items_df.groupby("order_id"):
            items = [self.item2idx[iid] for iid in group["item_id"] if iid in self.item2idx]
            if len(items) < 2:
                continue
            # In food orders, all items co-occur — use full context window
            for i, center in enumerate(items):
                for j, context in enumerate(items):
                    if i != j:
                        self.pairs.append((center, context))

        print(f"  {len(self.pairs)} skip-gram pairs from {self.vocab_size} items")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        # Sample negatives
        negatives = np.random.choice(self.vocab_size, size=self.n_negatives,
                                     p=self.neg_probs, replace=False)
        return (torch.LongTensor([center]),
                torch.LongTensor([context]),
                torch.LongTensor(negatives))


class Item2VecModel(nn.Module):
    """Skip-gram with negative sampling for item embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

        # Xavier initialization
        nn.init.xavier_uniform_(self.center_embed.weight)
        nn.init.xavier_uniform_(self.context_embed.weight)

    def forward(self, center, context, negatives):
        """
        Skip-gram with negative sampling loss.
        center: (batch, 1)
        context: (batch, 1)
        negatives: (batch, n_neg)
        """
        center_emb = self.center_embed(center)      # (batch, 1, dim)
        context_emb = self.context_embed(context)    # (batch, 1, dim)
        neg_emb = self.context_embed(negatives)      # (batch, n_neg, dim)

        # Positive score: dot product
        pos_score = torch.bmm(center_emb, context_emb.transpose(1, 2)).squeeze()
        pos_loss = F.logsigmoid(pos_score)

        # Negative score
        neg_score = torch.bmm(center_emb, neg_emb.transpose(1, 2)).squeeze()
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self) -> np.ndarray:
        """Return averaged center+context embeddings."""
        with torch.no_grad():
            emb = (self.center_embed.weight + self.context_embed.weight) / 2
            return emb.cpu().numpy()


class Item2VecTrainer:
    """Train Item2Vec embeddings on order sequences."""

    def __init__(self, embed_dim: int = 32, n_negatives: int = 5):
        self.embed_dim = embed_dim
        self.n_negatives = n_negatives
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, order_items_df: pd.DataFrame, n_epochs: int = 20,
              batch_size: int = 2048, lr: float = 0.003,
              save_path: str = "models/checkpoints") -> Tuple[np.ndarray, Dict]:
        """Train embeddings and return embedding matrix + item2idx mapping."""
        print(f"\nTraining Item2Vec ({self.embed_dim}d embeddings)")
        print(f"  Device: {self.device}")

        # Build dataset
        dataset = Item2VecDataset(order_items_df, n_negatives=self.n_negatives)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)

        # Model
        model = Item2VecModel(dataset.vocab_size, self.embed_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Train
        start = time.time()
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            n_batches = 0
            for center, context, negatives in loader:
                center = center.to(self.device)
                context = context.to(self.device)
                negatives = negatives.to(self.device)

                loss = model(center, context, negatives)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}/{n_epochs} | Loss: {avg_loss:.4f}")

        train_time = time.time() - start
        print(f"  Training time: {train_time:.1f}s")

        # Extract embeddings
        embeddings = model.get_embeddings()
        print(f"  Embedding matrix: {embeddings.shape}")

        # Save
        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/item_embeddings.npy", embeddings)

        import json
        with open(f"{save_path}/item2idx.json", "w") as f:
            json.dump(dataset.item2idx, f)

        # Quick quality check: find nearest neighbors for a sample item
        print("\n  Nearest neighbor quality check:")
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        for sample_idx in [0, 10, 50]:
            if sample_idx < len(dataset.idx2item):
                sims = sim_matrix[sample_idx]
                top_5 = np.argsort(-sims)[1:6]
                item_name = dataset.idx2item[sample_idx]
                neighbors = [dataset.idx2item[i] for i in top_5]
                print(f"    {item_name} → {neighbors}")

        return embeddings, dataset.item2idx


if __name__ == "__main__":
    order_items_df = pd.read_csv("data/generated/order_items.csv")
    trainer = Item2VecTrainer(embed_dim=32, n_negatives=5)
    embeddings, item2idx = trainer.train(order_items_df, n_epochs=20)
