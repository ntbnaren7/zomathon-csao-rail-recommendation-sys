"""
Feature Engineering Pipeline — Transforms raw data into model-ready features.

Handles:
- User features (RFM, preferences, segment encoding)
- Cart context features (Meal DNA, cart delta, price coherence)
- Temporal features (sin/cos encoded time, season, festival)
- Candidate item features (gap scores, co-occurrence, popularity)
"""

import json
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from features.meal_dna import MealDNAEncoder


class FeatureEngineer:
    """Transforms raw order data into features for model training and inference."""

    def __init__(self, items_df: pd.DataFrame, orders_df: pd.DataFrame = None,
                 order_items_df: pd.DataFrame = None,
                 embeddings_path: str = "models/checkpoints"):
        self.items_df = items_df
        self.orders_df = orders_df
        self.order_items_df = order_items_df
        self.meal_dna_encoder = MealDNAEncoder()

        # Pre-compute item lookup
        self.item_lookup = items_df.set_index("item_id").to_dict("index")

        # Pre-compute co-occurrence matrix if historical data available
        self.cooccurrence = {}
        if order_items_df is not None:
            self._build_cooccurrence(order_items_df)

        # Pre-compute user profiles if data available
        self.user_profiles = {}
        if orders_df is not None and order_items_df is not None:
            self._build_user_profiles(orders_df, order_items_df)

        # Load Item2Vec embeddings if available
        self.item_embeddings = None
        self.item2idx = {}
        emb_file = os.path.join(embeddings_path, "item_embeddings.npy")
        idx_file = os.path.join(embeddings_path, "item2idx.json")
        if os.path.exists(emb_file) and os.path.exists(idx_file):
            self.item_embeddings = np.load(emb_file)
            with open(idx_file, "r") as f:
                self.item2idx = json.load(f)
            # Pre-normalize embeddings for fast cosine similarity
            norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
            self.item_embeddings_normed = self.item_embeddings / np.maximum(norms, 1e-8)
            print(f"  Loaded Item2Vec embeddings: {self.item_embeddings.shape}")

    def _build_cooccurrence(self, order_items_df: pd.DataFrame):
        """Build item co-occurrence counts from historical orders."""
        print("  Building co-occurrence matrix...")
        for _, group in order_items_df.groupby("order_id"):
            items = list(group["item_id"])
            for i, a in enumerate(items):
                for j, b in enumerate(items):
                    if i != j:
                        key = (a, b)
                        self.cooccurrence[key] = self.cooccurrence.get(key, 0) + 1

    def _build_user_profiles(self, orders_df: pd.DataFrame,
                              order_items_df: pd.DataFrame):
        """Pre-compute per-user aggregate features."""
        print("  Building user profiles...")
        user_orders = orders_df.groupby("user_id")
        user_items = order_items_df.groupby("user_id")

        for user_id, user_group in user_orders:
            n_orders = len(user_group)
            self.user_profiles[user_id] = {
                "n_orders": n_orders,
                "avg_cart_value": user_group["cart_value"].mean(),
                "avg_cart_size": user_group["n_items"].mean(),
                "preferred_meal_period": user_group["meal_period"].mode().iloc[0] if not user_group["meal_period"].mode().empty else "dinner",
                "completion_rate": user_group["order_completed"].mean(),
                "preferred_cuisine": user_group["cuisine"].mode().iloc[0] if not user_group["cuisine"].mode().empty else "North Indian",
                "order_frequency": n_orders / max(1, (pd.to_datetime(user_group["timestamp"]).max() - pd.to_datetime(user_group["timestamp"]).min()).days),
            }

    # ─────────────────────────────────────────
    # Temporal Features
    # ─────────────────────────────────────────
    @staticmethod
    def encode_temporal(hour: int, day_of_week: int, meal_period: str,
                        season: str, festival: str, is_weekend: bool) -> np.ndarray:
        """
        Encode temporal context as feature vector.
        Uses sin/cos encoding for cyclical features.
        """
        # Sin/cos hour encoding (24-hour cycle)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Sin/cos day-of-week encoding (7-day cycle)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)

        # One-hot meal period
        meal_periods = ["breakfast", "lunch", "snack", "dinner", "late_night"]
        meal_vec = [1.0 if meal_period == mp else 0.0 for mp in meal_periods]

        # One-hot season
        seasons = ["summer", "monsoon", "autumn", "winter"]
        season_vec = [1.0 if season == s else 0.0 for s in seasons]

        # Festival flags
        festivals = ["none", "navratri", "diwali", "ramadan"]
        festival_vec = [1.0 if festival == f else 0.0 for f in festivals]

        weekend_flag = [1.0 if is_weekend else 0.0]

        return np.array(
            [hour_sin, hour_cos, dow_sin, dow_cos]
            + meal_vec + season_vec + festival_vec + weekend_flag,
            dtype=np.float32
        )

    # ─────────────────────────────────────────
    # User Features
    # ─────────────────────────────────────────
    def encode_user(self, user_id: str, user_row: Dict = None) -> np.ndarray:
        """Encode user features for model input."""
        profile = self.user_profiles.get(user_id, {})

        # Segment one-hot
        segments = ["budget", "mid", "premium"]
        segment = user_row.get("segment", "mid") if user_row else "mid"
        seg_vec = [1.0 if segment == s else 0.0 for s in segments]

        # Activity features
        n_orders = profile.get("n_orders", 0)
        avg_cart_val = profile.get("avg_cart_value", 250.0)
        avg_cart_size = profile.get("avg_cart_size", 2.5)
        completion_rate = profile.get("completion_rate", 0.85)
        order_freq = profile.get("order_frequency", 0.1)
        is_veg = float(user_row.get("is_veg", False)) if user_row else 0.0

        return np.array(
            seg_vec + [
                min(n_orders / 50.0, 1.0),  # Normalized order count
                avg_cart_val / 800.0,  # Normalized avg value
                avg_cart_size / 10.0,  # Normalized avg size
                completion_rate,
                min(order_freq, 1.0),
                is_veg,
            ],
            dtype=np.float32
        )

    # ─────────────────────────────────────────
    # Cart Context Features
    # ─────────────────────────────────────────
    def encode_cart(self, cart_item_ids: List[str], cuisine: str,
                    meal_period: str = "dinner",
                    order_type: str = "solo") -> np.ndarray:
        """
        Encode current cart state as feature vector.
        Includes Meal DNA, cart aggregates, category distribution, and order type.
        """
        cart_items = [self.item_lookup.get(iid, {}) for iid in cart_item_ids]
        cart_items = [it for it in cart_items if it]  # Filter missing

        if not cart_items:
            return np.zeros(25, dtype=np.float32)

        # Meal DNA (6 dims)
        meal_dna = self.meal_dna_encoder.encode_as_vector(cart_items, cuisine, meal_period)

        # Meal completion score (1 dim) — how complete is the cart already?
        meal_dna_dict = self.meal_dna_encoder.encode(cart_items, cuisine, meal_period)
        completion = self.meal_dna_encoder.get_completion_score(meal_dna_dict)

        # Cart aggregates
        prices = [it.get("price", 0) for it in cart_items]
        cart_value = sum(prices) / 1000.0  # Normalized
        cart_size = len(cart_items) / 10.0  # Normalized
        avg_price = np.mean(prices) / 500.0
        price_std = np.std(prices) / 200.0 if len(prices) > 1 else 0.0

        # Category distribution (6 dims for meal roles)
        role_counts = {r: 0 for r in self.meal_dna_encoder.roles}
        for it in cart_items:
            role = it.get("meal_role", "side")
            if role in role_counts:
                role_counts[role] += 1
        total = max(sum(role_counts.values()), 1)
        role_dist = [role_counts[r] / total for r in self.meal_dna_encoder.roles]

        # Veg ratio in cart
        veg_count = sum(1 for it in cart_items if it.get("is_veg", True))
        veg_ratio = veg_count / max(len(cart_items), 1)

        # Is cart all-veg? (hard constraint signal)
        all_veg = float(all(it.get("is_veg", True) for it in cart_items))

        # Order type one-hot (4 dims)
        order_types = ["solo", "pair", "group", "party"]
        otype_vec = [1.0 if order_type == ot else 0.0 for ot in order_types]

        return np.array(
            list(meal_dna) +  # 6
            [completion] +  # 1
            [cart_value, cart_size, avg_price, price_std] +  # 4
            role_dist +  # 6
            [veg_ratio, all_veg] +  # 2
            otype_vec,  # 4
            dtype=np.float32
        )  # Total: 23

    # ─────────────────────────────────────────
    # Candidate Item Features
    # ─────────────────────────────────────────
    def _embedding_similarity(self, candidate_id: str, cart_ids: List[str]) -> List[float]:
        """Compute Item2Vec embedding similarity features between candidate and cart."""
        if self.item_embeddings is None:
            return [0.0, 0.0, 0.0, 0.0]

        cand_idx = self.item2idx.get(candidate_id)
        if cand_idx is None:
            return [0.0, 0.0, 0.0, 0.0]

        cand_emb = self.item_embeddings_normed[cand_idx]

        # Cosine similarities with each cart item
        sims = []
        for cart_id in cart_ids:
            cart_idx = self.item2idx.get(cart_id)
            if cart_idx is not None:
                cart_emb = self.item_embeddings_normed[cart_idx]
                sim = float(np.dot(cand_emb, cart_emb))
                sims.append(sim)

        if not sims:
            return [0.0, 0.0, 0.0, 0.0]

        return [
            np.mean(sims),   # avg similarity to cart
            np.max(sims),    # max similarity (closest cart item)
            np.min(sims),    # min similarity (most distant)
            np.std(sims) if len(sims) > 1 else 0.0,  # similarity variance
        ]

    def encode_candidate(self, candidate_item_id: str,
                         cart_item_ids: List[str],
                         cuisine: str, meal_period: str = "dinner",
                         user_segment: str = "mid",
                         user_id: str = None) -> np.ndarray:
        """Encode a candidate add-on item relative to current cart."""
        item = self.item_lookup.get(candidate_item_id, {})
        if not item:
            return np.zeros(22, dtype=np.float32)

        cart_items = [self.item_lookup.get(iid, {}) for iid in cart_item_ids]
        cart_items = [it for it in cart_items if it]

        # Item basic features
        price_norm = item.get("price", 100) / 500.0
        is_veg = float(item.get("is_veg", True))
        popularity = item.get("popularity", 0.5)
        spice = item.get("spice_level", 2) / 5.0
        avg_rating = item.get("avg_rating", 3.5) / 5.0

        # Meal DNA gap score for this candidate
        meal_dna = self.meal_dna_encoder.encode(cart_items, cuisine, meal_period)
        gap_score = self.meal_dna_encoder.get_candidate_gap_score(
            meal_dna, item.get("meal_role", "side")
        )

        # Price coherence
        cart_prices = [it.get("price", 100) for it in cart_items]
        avg_cart_price = np.mean(cart_prices) if cart_prices else 150.0
        price_coherence = 1.0 - min(abs(item.get("price", 100) - avg_cart_price) / max(avg_cart_price, 1), 2.0) / 2.0

        # Price ratio
        cart_total = sum(cart_prices) if cart_prices else 200.0
        price_ratio = min(item.get("price", 100) / max(cart_total, 1), 1.0)

        # Co-occurrence score
        cooc_score = 0.0
        for cart_iid in cart_item_ids:
            cooc_score += self.cooccurrence.get((cart_iid, candidate_item_id), 0)
        cooc_score = min(cooc_score / 100.0, 1.0)

        # Meal role one-hot (6 dims)
        role = item.get("meal_role", "side")
        role_vec = [1.0 if role == r else 0.0 for r in self.meal_dna_encoder.roles]

        # Category match
        cart_roles = set(it.get("meal_role", "") for it in cart_items)
        is_new_role = float(role not in cart_roles)

        # Cuisine match
        item_cuisine = item.get("cuisine", "")
        cuisine_match = float(item_cuisine.lower() == cuisine.lower()) if item_cuisine else 1.0

        # Meal-period × popularity interaction
        period_popularity = popularity
        if meal_period == "breakfast" and role == "drink":
            period_popularity *= 1.5
        elif meal_period in ("dinner", "lunch") and role == "dessert":
            period_popularity *= 1.3
        elif meal_period == "late_night" and role in ("drink", "dessert"):
            period_popularity *= 1.4
        period_popularity = min(period_popularity, 1.0)

        # Item2Vec embedding similarity features (4 dims)
        emb_sims = self._embedding_similarity(candidate_item_id, cart_item_ids)

        return np.array(
            [price_norm, is_veg, popularity, spice, avg_rating,
             gap_score, price_coherence, price_ratio,
             cooc_score, is_new_role, cuisine_match, period_popularity]
            + role_vec
            + emb_sims,
            dtype=np.float32
        )  # Total: 22

    # ─────────────────────────────────────────
    # Full Feature Vector Assembly
    # ─────────────────────────────────────────
    def build_feature_vector(self, user_id: str, cart_item_ids: List[str],
                             candidate_item_id: str, cuisine: str,
                             hour: int, day_of_week: int, meal_period: str,
                             season: str, festival: str, is_weekend: bool,
                             user_row: Dict = None, user_segment: str = "mid",
                             order_type: str = "solo"
                             ) -> np.ndarray:
        """Assemble complete feature vector for model input."""
        temporal = self.encode_temporal(hour, day_of_week, meal_period, season, festival, is_weekend)
        user = self.encode_user(user_id, user_row)
        cart = self.encode_cart(cart_item_ids, cuisine, meal_period, order_type)
        candidate = self.encode_candidate(candidate_item_id, cart_item_ids, cuisine, meal_period, user_segment, user_id)

        return np.concatenate([temporal, user, cart, candidate])

    def build_training_features(self, training_df: pd.DataFrame,
                                users_df: pd.DataFrame,
                                batch_size: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build feature matrices from training data DataFrame.
        Returns (X_features, y_labels, y_c2o_labels).
        """
        user_lookup = users_df.set_index("user_id").to_dict("index")
        features = []
        labels = []
        c2o_labels = []

        total = len(training_df)
        for idx, row in training_df.iterrows():
            cart_ids = json.loads(row["cart_item_ids"])
            user_row = user_lookup.get(row["user_id"], {})

            feat = self.build_feature_vector(
                user_id=row["user_id"],
                cart_item_ids=cart_ids,
                candidate_item_id=row["candidate_item_id"],
                cuisine=row["cuisine"],
                hour=row["hour"],
                day_of_week=row["day_of_week"],
                meal_period=row["meal_period"],
                season=row["season"],
                festival=row["festival"],
                is_weekend=row["is_weekend"],
                user_row=user_row,
                user_segment=user_row.get("segment", "mid"),
                order_type=row.get("order_type", "solo"),
            )
            features.append(feat)
            labels.append(row["label"])
            c2o_labels.append(row.get("order_completed", 1))

            if (idx + 1) % batch_size == 0:
                print(f"  Features: {idx+1}/{total} processed...")

        X = np.vstack(features)
        y = np.array(labels, dtype=np.float32)
        y_c2o = np.array(c2o_labels, dtype=np.float32)

        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Label distribution: {y.mean():.2%} positive")

        return X, y, y_c2o


if __name__ == "__main__":
    # Quick demo
    from data.menu_catalog import MenuCatalog
    catalog = MenuCatalog(seed=42)
    items_df = catalog.get_items_df()

    fe = FeatureEngineer(items_df)

    # Simulated cart: Chicken Biryani (find one)
    biryani_items = items_df[items_df["name"].str.contains("Chicken Biryani")]
    if not biryani_items.empty:
        biryani_id = biryani_items.iloc[0]["item_id"]
        rest_id = biryani_items.iloc[0]["restaurant_id"]
        rest_menu = items_df[items_df["restaurant_id"] == rest_id]

        print(f"Cart: [{biryani_items.iloc[0]['name']}]")
        print(f"Restaurant menu has {len(rest_menu)} items")

        # Encode cart
        cart_vec = fe.encode_cart([biryani_id], "Biryani", "dinner")
        print(f"Cart feature vector ({len(cart_vec)} dims): {cart_vec}")

        # Encode a candidate
        for _, cand in rest_menu.head(5).iterrows():
            if cand["item_id"] != biryani_id:
                cand_vec = fe.encode_candidate(cand["item_id"], [biryani_id], "Biryani", "dinner")
                print(f"  Candidate: {cand['name']:30s} gap={cand_vec[4]:.2f} cooc={cand_vec[6]:.2f}")
