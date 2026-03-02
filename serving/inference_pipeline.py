"""
CSAO Inference Pipeline — End-to-end recommendation generation.

Given a user, restaurant, and current cart, returns ranked add-on recommendations
with Meal DNA visualization, explanations, and latency tracking.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional

from data.menu_catalog import MenuCatalog
from features.meal_dna import MealDNAEncoder
from features.feature_engineering import FeatureEngineer
from models.reranker import CSAOReranker
from models.post_ranker import PostRanker


class InferencePipeline:
    """End-to-end CSAO recommendation inference."""

    def __init__(self, model_path: str = "models/checkpoints/best_model.pt",
                 data_dir: str = "data/generated",
                 device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load catalog and feature engineer
        self.catalog = MenuCatalog(seed=42)
        self.items_df = self.catalog.get_items_df()
        self.restaurants_df = self.catalog.get_restaurants_df()
        self.item_lookup = self.items_df.set_index("item_id").to_dict("index")
        self.meal_dna_encoder = MealDNAEncoder()
        self.post_ranker = PostRanker(exploration_rate=0.05)

        # Load order data for feature engineering (co-occurrence etc.)
        orders_df = None
        order_items_df = None
        if os.path.exists(f"{data_dir}/orders.csv"):
            orders_df = pd.read_csv(f"{data_dir}/orders.csv")
            order_items_df = pd.read_csv(f"{data_dir}/order_items.csv")

        self.feature_engineer = FeatureEngineer(
            self.items_df, orders_df, order_items_df
        )

        # Load model
        self.model = None
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            input_dim = checkpoint.get("input_dim", 59)
            self.model = CSAOReranker(input_dim=input_dim).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            print(f"Model loaded from {model_path} (AUC: {checkpoint.get('val_auc', 'N/A')})")
        else:
            print(f"WARNING: No model found at {model_path}, using fallback scoring")

    def recommend(self, user_id: str, restaurant_id: str,
                  cart_item_ids: List[str], hour: int = 19,
                  day_of_week: int = 3, meal_period: str = "dinner",
                  season: str = "winter", festival: str = "none",
                  is_weekend: bool = False, top_k: int = 8,
                  user_row: Dict = None) -> Dict:
        """
        Generate top-K add-on recommendations.

        Returns dict with:
        - recommendations: List of ranked items with scores and explanations
        - meal_dna: Current meal completion state
        - latency_ms: Inference time in milliseconds
        """
        start_time = time.perf_counter()

        # Get restaurant menu
        restaurant_menu = self.items_df[
            self.items_df["restaurant_id"] == restaurant_id
        ]
        restaurant = self.restaurants_df[
            self.restaurants_df["id"] == restaurant_id
        ]
        if restaurant.empty:
            return {"recommendations": [], "meal_dna": {}, "latency_ms": 0}

        cuisine = restaurant.iloc[0]["cuisine"]

        # Get cart items for Meal DNA
        cart_items = [self.item_lookup.get(iid, {}) for iid in cart_item_ids]
        cart_items = [it for it in cart_items if it]

        # Compute Meal DNA
        meal_dna = self.meal_dna_encoder.encode(cart_items, cuisine, meal_period)

        # Filter candidates: same restaurant, not in cart, veg constraints
        all_veg = all(it.get("is_veg", True) for it in cart_items) if cart_items else False
        candidates = restaurant_menu[
            ~restaurant_menu["item_id"].isin(set(cart_item_ids))
        ]
        if all_veg:
            candidates = candidates[candidates["is_veg"] == True]

        if candidates.empty:
            elapsed = (time.perf_counter() - start_time) * 1000
            return {"recommendations": [], "meal_dna": meal_dna, "latency_ms": elapsed}

        # Score candidates
        scored_candidates = []
        user_segment = user_row.get("segment", "mid") if user_row else "mid"

        if self.model is not None:
            # Build feature vectors for all candidates
            feature_vectors = []
            for _, cand in candidates.iterrows():
                feat = self.feature_engineer.build_feature_vector(
                    user_id=user_id,
                    cart_item_ids=cart_item_ids,
                    candidate_item_id=cand["item_id"],
                    cuisine=cuisine,
                    hour=hour, day_of_week=day_of_week,
                    meal_period=meal_period, season=season,
                    festival=festival, is_weekend=is_weekend,
                    user_row=user_row, user_segment=user_segment,
                )
                feature_vectors.append(feat)

            # Batch inference on GPU
            X = torch.FloatTensor(np.vstack(feature_vectors)).to(self.device)
            with torch.no_grad():
                scores = self.model.predict_score(X).cpu().numpy().flatten()

            for idx, (_, cand) in enumerate(candidates.iterrows()):
                scored_candidates.append({
                    **cand.to_dict(),
                    "score": float(scores[idx]),
                    "cooccurrence_score": self._get_cooc_score(cart_item_ids, cand["item_id"]),
                })
        else:
            # Fallback: popularity + gap-based scoring
            for _, cand in candidates.iterrows():
                gap = self.meal_dna_encoder.get_candidate_gap_score(meal_dna, cand["meal_role"])
                score = cand["popularity"] * 0.5 + gap * 0.5
                scored_candidates.append({
                    **cand.to_dict(),
                    "score": float(score),
                    "cooccurrence_score": 0.0,
                })

        # Sort by score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Post-ranking diversification
        diversified = self.post_ranker.diversify(scored_candidates, meal_dna, top_k)

        elapsed = (time.perf_counter() - start_time) * 1000

        return {
            "recommendations": diversified,
            "meal_dna": meal_dna,
            "meal_completion": self.meal_dna_encoder.get_completion_score(meal_dna),
            "missing_roles": self.meal_dna_encoder.get_missing_roles(meal_dna),
            "latency_ms": round(elapsed, 1),
            "n_candidates_scored": len(scored_candidates),
        }

    def _get_cooc_score(self, cart_item_ids: List[str], candidate_id: str) -> float:
        """Get co-occurrence score between cart items and candidate."""
        total = 0.0
        for cid in cart_item_ids:
            total += self.feature_engineer.cooccurrence.get((cid, candidate_id), 0)
        return min(total / 100.0, 1.0)

    def get_restaurant_menu(self, restaurant_id: str) -> List[Dict]:
        """Get full menu for a restaurant."""
        menu = self.items_df[self.items_df["restaurant_id"] == restaurant_id]
        return menu.to_dict("records")

    def get_restaurants(self, city: str = None) -> List[Dict]:
        """Get list of restaurants, optionally filtered by city."""
        rests = self.restaurants_df
        if city:
            rests = rests[rests["city"] == city]
        return rests.to_dict("records")
