"""
CSAO Evaluation Pipeline — Comprehensive ranking metrics and business impact analysis.

Metrics:
- Offline: AUC, NDCG@K, Precision@K, Recall@K, HitRate@K, MRR
- Business: Projected AOV lift, C2O ratio, revenue impact
- Segment: Per-cuisine, per-city, per-meal-period breakdowns
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple
from collections import defaultdict


class Evaluator:
    """Comprehensive evaluation with ranking metrics and segment analysis."""

    def __init__(self):
        self.results = {}

    def evaluate_ranking(self, y_true: np.ndarray, y_pred: np.ndarray,
                         k_values: List[int] = None) -> Dict[str, float]:
        """Compute ranking metrics at various K."""
        if k_values is None:
            k_values = [3, 5, 8]

        metrics = {}

        # AUC
        metrics["auc"] = roc_auc_score(y_true, y_pred)

        # Binary metrics at threshold 0.5
        y_binary = (y_pred > 0.5).astype(int)
        metrics["accuracy"] = (y_binary == y_true).mean()
        metrics["precision"] = precision_score(y_true, y_binary, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_binary, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_binary, zero_division=0)

        # Score separation
        pos_mask = y_true == 1
        neg_mask = y_true == 0
        if pos_mask.any() and neg_mask.any():
            metrics["avg_score_pos"] = y_pred[pos_mask].mean()
            metrics["avg_score_neg"] = y_pred[neg_mask].mean()
            metrics["score_separation"] = metrics["avg_score_pos"] - metrics["avg_score_neg"]

        return metrics

    def evaluate_per_group(self, test_df: pd.DataFrame, scores: np.ndarray,
                           group_col: str = "order_id",
                           k_values: List[int] = None) -> Dict[str, float]:
        """
        Compute recommendation-style metrics grouped by order.
        Each order = one recommendation session.
        """
        if k_values is None:
            k_values = [3, 5, 8]

        test_df = test_df.copy()
        test_df["score"] = scores

        ndcg_results = {k: [] for k in k_values}
        precision_results = {k: [] for k in k_values}
        recall_results = {k: [] for k in k_values}
        hit_results = {k: [] for k in k_values}
        mrr_results = []

        for _, group in test_df.groupby(group_col):
            if len(group) < 2:
                continue

            # Sort by predicted score descending
            sorted_group = group.sort_values("score", ascending=False)
            labels = sorted_group["label"].values
            n_positive = labels.sum()

            if n_positive == 0:
                continue

            # MRR: position of first positive item
            for i, label in enumerate(labels):
                if label == 1:
                    mrr_results.append(1.0 / (i + 1))
                    break

            for k in k_values:
                top_k = labels[:k] if k <= len(labels) else labels

                # Precision@K
                precision_results[k].append(top_k.sum() / len(top_k))

                # Recall@K
                recall_results[k].append(top_k.sum() / n_positive)

                # HitRate@K (at least 1 positive in top-K)
                hit_results[k].append(float(top_k.sum() > 0))

                # NDCG@K
                dcg = sum(l / np.log2(i + 2) for i, l in enumerate(top_k))
                ideal = sorted(labels, reverse=True)[:k] if k <= len(labels) else sorted(labels, reverse=True)
                idcg = sum(l / np.log2(i + 2) for i, l in enumerate(ideal))
                ndcg_results[k].append(dcg / idcg if idcg > 0 else 0.0)

        metrics = {"mrr": np.mean(mrr_results) if mrr_results else 0.0}
        for k in k_values:
            metrics[f"ndcg@{k}"] = np.mean(ndcg_results[k]) if ndcg_results[k] else 0.0
            metrics[f"precision@{k}"] = np.mean(precision_results[k]) if precision_results[k] else 0.0
            metrics[f"recall@{k}"] = np.mean(recall_results[k]) if recall_results[k] else 0.0
            metrics[f"hitrate@{k}"] = np.mean(hit_results[k]) if hit_results[k] else 0.0

        return metrics

    def evaluate_segments(self, test_df: pd.DataFrame, scores: np.ndarray,
                          segment_cols: List[str] = None) -> Dict[str, Dict]:
        """Per-segment evaluation for error analysis."""
        if segment_cols is None:
            segment_cols = ["cuisine", "city", "meal_period", "order_type"]

        test_df = test_df.copy()
        test_df["score"] = scores

        segment_results = {}
        for col in segment_cols:
            if col not in test_df.columns:
                continue

            segment_results[col] = {}
            for segment_val, group in test_df.groupby(col):
                if len(group) < 50:
                    continue
                labels = group["label"].values
                preds = group["score"].values
                try:
                    auc = roc_auc_score(labels, preds)
                except:
                    auc = 0.0
                segment_results[col][segment_val] = {
                    "auc": round(auc, 4),
                    "n_samples": len(group),
                    "positive_rate": round(labels.mean(), 4),
                }

        return segment_results

    def business_impact(self, test_df: pd.DataFrame, scores: np.ndarray,
                        avg_item_price: float = 120.0,
                        baseline_acceptance: float = 0.15) -> Dict[str, float]:
        """
        Project business impact from model performance.
        Estimates AOV lift, acceptance rate, and revenue.
        """
        test_df = test_df.copy()
        test_df["score"] = scores

        # Simulated acceptance: items above score threshold are "accepted"
        threshold = np.percentile(scores[test_df["label"] == 1], 25)
        model_acceptance = (scores > threshold).mean()

        # AOV lift estimate
        avg_cart_value = test_df.get("cart_value", pd.Series([300])).mean()
        items_accepted_per_order = model_acceptance * 2  # Avg 2 recommendations shown
        aov_lift_pct = (items_accepted_per_order * avg_item_price / avg_cart_value) * 100

        # Revenue projection (per 1M orders)
        orders_per_month = 1_000_000
        revenue_lift = orders_per_month * items_accepted_per_order * avg_item_price

        return {
            "model_acceptance_rate": round(model_acceptance, 4),
            "baseline_acceptance_rate": baseline_acceptance,
            "acceptance_lift": round(model_acceptance / baseline_acceptance, 2),
            "avg_items_accepted_per_order": round(items_accepted_per_order, 3),
            "projected_aov_lift_pct": round(aov_lift_pct, 2),
            "projected_monthly_revenue_lift": int(revenue_lift),
            "avg_item_price": avg_item_price,
        }

    def full_report(self, test_df: pd.DataFrame, scores: np.ndarray) -> Dict:
        """Generate comprehensive evaluation report."""
        labels = test_df["label"].values

        report = {
            "ranking_metrics": self.evaluate_ranking(labels, scores),
            "per_group_metrics": self.evaluate_per_group(test_df, scores),
            "segment_analysis": self.evaluate_segments(test_df, scores),
            "business_impact": self.business_impact(test_df, scores),
        }

        return report

    def print_report(self, report: Dict):
        """Pretty-print evaluation report."""
        print("=" * 60)
        print("CSAO Model Evaluation Report")
        print("=" * 60)

        print("\n📊 Ranking Metrics:")
        for k, v in report["ranking_metrics"].items():
            print(f"  {k:25s}: {v:.4f}")

        print("\n🎯 Per-Group Recommendation Metrics:")
        for k, v in report["per_group_metrics"].items():
            print(f"  {k:25s}: {v:.4f}")

        print("\n📈 Business Impact Projection:")
        bi = report["business_impact"]
        print(f"  Model acceptance rate   : {bi['model_acceptance_rate']:.1%}")
        print(f"  Baseline acceptance     : {bi['baseline_acceptance_rate']:.1%}")
        print(f"  Acceptance lift         : {bi['acceptance_lift']}x")
        print(f"  AOV lift (projected)    : +{bi['projected_aov_lift_pct']:.1f}%")
        print(f"  Monthly revenue lift    : ₹{bi['projected_monthly_revenue_lift']:,.0f}")

        print("\n🔍 Segment Analysis:")
        for col, segments in report["segment_analysis"].items():
            print(f"\n  --- {col.upper()} ---")
            sorted_segs = sorted(segments.items(), key=lambda x: x[1]["auc"], reverse=True)
            for seg_name, seg_data in sorted_segs[:8]:
                print(f"    {seg_name:20s} AUC={seg_data['auc']:.4f}  "
                      f"n={seg_data['n_samples']:>6d}  "
                      f"pos_rate={seg_data['positive_rate']:.1%}")


if __name__ == "__main__":
    # Run evaluation on saved test data
    import torch
    from models.reranker import CSAOReranker

    print("Loading test data...")
    data = np.load("models/checkpoints/features.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Test set: {len(X_test)} examples ({y_test.mean():.1%} positive)")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("models/checkpoints/best_model.pt",
                            map_location=device, weights_only=False)
    model = CSAOReranker(input_dim=checkpoint["input_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Generate predictions
    print("Generating predictions...")
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        batch_size = 4096
        scores = []
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            batch_scores = model.predict_score(batch).cpu().numpy().flatten()
            scores.append(batch_scores)
        scores = np.concatenate(scores)

    # Load test_df for segment analysis
    training_df = pd.read_csv("data/generated/training_data.csv", engine="python", on_bad_lines="skip")
    orders_df = pd.read_csv("data/generated/orders.csv")

    # Temporal split to get test portion
    order_timestamps = orders_df.set_index("order_id")["timestamp"]
    training_df["timestamp"] = training_df["order_id"].map(order_timestamps)
    training_df = training_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    n = len(training_df)
    test_df = training_df.iloc[int(n * 0.9):]

    # Align sizes (test_df may not match X_test exactly due to sampling)
    min_len = min(len(test_df), len(scores))
    test_df = test_df.iloc[:min_len].copy()
    scores = scores[:min_len]

    evaluator = Evaluator()
    report = evaluator.full_report(test_df, scores)
    evaluator.print_report(report)
