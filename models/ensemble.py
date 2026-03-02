"""
XGBoost + DCN-V2 Ensemble — combines tree-based and neural approaches.

XGBoost excels at tabular feature interactions.
DCN-V2 excels at cross-feature learning.
Ensemble reduces variance and typically adds 2-4pp AUC.
"""

import os
import time
import pickle
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from models.reranker import CSAOReranker


class EnsembleTrainer:
    """Train XGBoost + DCN-V2 ensemble on pre-computed features."""

    def __init__(self, model_dir: str = "models/checkpoints"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_features(self):
        """Load pre-computed feature matrices."""
        print("Loading pre-computed features...")
        data = np.load(f"{self.model_dir}/features.npz")

        # Load scaler and normalize
        with open(f"{self.model_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        X_train = scaler.transform(data["X_train"])
        X_val = scaler.transform(data["X_val"])
        X_test = scaler.transform(data["X_test"])

        return (X_train, data["y_train"], data["y_c2o_train"],
                X_val, data["y_val"], data["y_c2o_val"],
                X_test, data["y_test"], data["y_c2o_test"])

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost with GPU acceleration."""
        print("\n" + "=" * 60)
        print("Training XGBoost (GPU-accelerated)")
        print("=" * 60)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 3.0,  # Handle class imbalance (1:3 ratio)
            "n_estimators": 500,
            "early_stopping_rounds": 20,
            "verbosity": 1,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        start = time.time()
        model = xgb.train(
            params, dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=params["early_stopping_rounds"],
            verbose_eval=50,
        )
        xgb_time = time.time() - start
        print(f"XGBoost training time: {xgb_time:.1f}s")

        # Save model
        model.save_model(f"{self.model_dir}/xgb_model.json")

        # Evaluate
        val_preds = model.predict(dval)
        val_auc = roc_auc_score(y_val, val_preds)
        print(f"XGBoost Val AUC: {val_auc:.4f}")

        # Feature importance
        importance = model.get_score(importance_type="gain")
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 features by gain:")
        for fname, score in sorted_imp:
            print(f"  {fname:15s}: {score:.1f}")

        return model, val_auc

    def get_dcn_scores(self, X, batch_size=4096):
        """Get DCN-V2 model predictions."""
        checkpoint = torch.load(f"{self.model_dir}/best_model.pt",
                                map_location=self.device, weights_only=False)
        model = CSAOReranker(input_dim=checkpoint["input_dim"]).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)
        scores = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                batch_scores = model.predict_score(batch).cpu().numpy().flatten()
                scores.append(batch_scores)

        return np.concatenate(scores)

    def find_best_alpha(self, dcn_scores, xgb_scores, y_true):
        """Grid search for optimal ensemble weight."""
        best_auc = 0
        best_alpha = 0.5
        for alpha in np.arange(0.0, 1.05, 0.05):
            combined = alpha * dcn_scores + (1 - alpha) * xgb_scores
            auc = roc_auc_score(y_true, combined)
            if auc > best_auc:
                best_auc = auc
                best_alpha = alpha
        return best_alpha, best_auc

    def train(self):
        """Full ensemble training pipeline."""
        (X_train, y_train, _, X_val, y_val, _,
         X_test, y_test, _) = self.load_features()

        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        print(f"Feature dim: {X_train.shape[1]}")

        # 1. Train XGBoost
        xgb_model, xgb_val_auc = self.train_xgboost(X_train, y_train, X_val, y_val)

        # 2. Get DCN-V2 scores
        print("\n" + "=" * 60)
        print("Getting DCN-V2 predictions")
        print("=" * 60)
        dcn_val = self.get_dcn_scores(X_val)
        dcn_test = self.get_dcn_scores(X_test)
        dcn_val_auc = roc_auc_score(y_val, dcn_val)
        print(f"DCN-V2 Val AUC: {dcn_val_auc:.4f}")

        # 3. XGBoost test scores
        dtest = xgb.DMatrix(X_test)
        xgb_test = xgb_model.predict(dtest)
        xgb_test_auc = roc_auc_score(y_test, xgb_test)

        dval = xgb.DMatrix(X_val)
        xgb_val = xgb_model.predict(dval)

        # 4. Find best ensemble weight on validation set
        print("\n" + "=" * 60)
        print("Finding optimal ensemble weights")
        print("=" * 60)
        best_alpha, best_val_auc = self.find_best_alpha(dcn_val, xgb_val, y_val)
        print(f"Best alpha (DCN weight): {best_alpha:.2f}")
        print(f"Best ensemble Val AUC:   {best_val_auc:.4f}")

        # 5. Final test evaluation
        print("\n" + "=" * 60)
        print("Final Test Evaluation")
        print("=" * 60)
        ensemble_test = best_alpha * dcn_test + (1 - best_alpha) * xgb_test
        ensemble_test_auc = roc_auc_score(y_test, ensemble_test)
        dcn_test_auc = roc_auc_score(y_test, dcn_test)

        print(f"{'Model':<20s} {'Test AUC':<12s}")
        print("-" * 32)
        print(f"{'DCN-V2 only':<20s} {dcn_test_auc:.4f}")
        print(f"{'XGBoost only':<20s} {xgb_test_auc:.4f}")
        print(f"{'Ensemble':<20s} {ensemble_test_auc:.4f}")
        print(f"\nEnsemble lift: +{(ensemble_test_auc - max(dcn_test_auc, xgb_test_auc)):.4f} over best single model")

        # Score separation
        pos_mask = y_test == 1
        neg_mask = y_test == 0
        print(f"\nScore separation:")
        print(f"  DCN-V2:   pos={dcn_test[pos_mask].mean():.4f}  neg={dcn_test[neg_mask].mean():.4f}  gap={dcn_test[pos_mask].mean() - dcn_test[neg_mask].mean():.4f}")
        print(f"  XGBoost:  pos={xgb_test[pos_mask].mean():.4f}  neg={xgb_test[neg_mask].mean():.4f}  gap={xgb_test[pos_mask].mean() - xgb_test[neg_mask].mean():.4f}")
        print(f"  Ensemble: pos={ensemble_test[pos_mask].mean():.4f}  neg={ensemble_test[neg_mask].mean():.4f}  gap={ensemble_test[pos_mask].mean() - ensemble_test[neg_mask].mean():.4f}")

        # Save ensemble config
        config = {
            "alpha": float(best_alpha),
            "dcn_test_auc": float(dcn_test_auc),
            "xgb_test_auc": float(xgb_test_auc),
            "ensemble_test_auc": float(ensemble_test_auc),
        }
        import json
        with open(f"{self.model_dir}/ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save ensemble test scores for evaluation
        np.savez_compressed(
            f"{self.model_dir}/ensemble_scores.npz",
            scores=ensemble_test,
            y_test=y_test,
        )

        return config


if __name__ == "__main__":
    trainer = EnsembleTrainer()
    config = trainer.train()
