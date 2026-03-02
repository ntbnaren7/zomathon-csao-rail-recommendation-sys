"""
CSAO Synthetic Data Generator — 10K users, 100K+ orders with realistic Indian food delivery patterns.

Generates:
- users.csv: User profiles with segments, preferences, veg/non-veg
- orders.csv: Historical orders with timestamps, cart items, contextual features
- order_items.csv: Individual items within each order
- interactions.csv: User-item interaction matrix for training
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from data.menu_catalog import MenuCatalog, MEAL_TEMPLATES, RESTAURANTS, MEAL_ROLES

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Lucknow"]
CITY_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]  # Population-proportional

USER_SEGMENTS = ["budget", "mid", "premium"]
SEGMENT_WEIGHTS = [0.40, 0.40, 0.20]

MEAL_PERIODS = {
    "breakfast": (7, 10),
    "lunch":     (11, 14),
    "snack":     (15, 17),
    "dinner":    (19, 22),
    "late_night": (22, 26),  # 22:00 - 02:00 (next day)
}
MEAL_PERIOD_WEIGHTS = [0.10, 0.30, 0.10, 0.40, 0.10]

SEASONS = {
    "summer":  [3, 4, 5, 6],
    "monsoon": [7, 8, 9],
    "autumn":  [10, 11],
    "winter":  [12, 1, 2],
}

# Cart-size distribution parameters by ordering context
CART_SIZE_PARAMS = {
    "solo":  {"min": 1, "max": 3, "mode": 2},
    "pair":  {"min": 2, "max": 5, "mode": 3},
    "group": {"min": 4, "max": 8, "mode": 5},
    "party": {"min": 6, "max": 15, "mode": 8},
}
ORDER_TYPE_WEIGHTS = [0.45, 0.25, 0.20, 0.10]


class SyntheticDataGenerator:
    """Generates realistic food delivery order data."""

    def __init__(self, n_users: int = 10000, n_orders: int = 120000, seed: int = 42):
        self.n_users = n_users
        self.n_orders = n_orders
        self.rng = np.random.default_rng(seed)
        self.catalog = MenuCatalog(seed=seed)
        self.items_df = self.catalog.get_items_df()
        self.restaurants_df = self.catalog.get_restaurants_df()

        # Pre-compute restaurant-to-items mapping
        self.rest_items = {}
        for rid in self.restaurants_df["id"]:
            self.rest_items[rid] = self.items_df[self.items_df["restaurant_id"] == rid]

    # ─────────────────────────────────────────
    # User Generation
    # ─────────────────────────────────────────
    def generate_users(self) -> pd.DataFrame:
        """Generate user profiles with realistic distribution."""
        users = []
        for i in range(self.n_users):
            city = self.rng.choice(CITIES, p=CITY_WEIGHTS)
            segment = self.rng.choice(USER_SEGMENTS, p=SEGMENT_WEIGHTS)
            is_veg = self.rng.random() < self._veg_probability(city)

            # Zipf-distributed activity: most users are sparse
            activity_class = self.rng.choice(
                ["sparse", "moderate", "power"],
                p=[0.60, 0.25, 0.15]
            )

            # Cuisine preferences (probability distribution over cuisines)
            cuisine_prefs = self._generate_cuisine_preferences(city, is_veg)

            users.append({
                "user_id": f"U{i+1:05d}",
                "city": city,
                "segment": segment,
                "is_veg": is_veg,
                "activity_class": activity_class,
                "preferred_cuisines": json.dumps(cuisine_prefs),
                "avg_order_value": self._avg_order_for_segment(segment),
                "account_age_days": int(self.rng.integers(30, 730)),
                "has_zomato_pro": self.rng.random() < (0.3 if segment == "premium" else 0.1),
            })
        return pd.DataFrame(users)

    def _veg_probability(self, city: str) -> float:
        """City-specific vegetarian probability (Indian demographics)."""
        return {
            "Mumbai": 0.30, "Delhi": 0.40, "Bangalore": 0.25,
            "Hyderabad": 0.20, "Lucknow": 0.15,
        }.get(city, 0.30)

    def _generate_cuisine_preferences(self, city: str, is_veg: bool) -> dict:
        """Generate user's cuisine preference distribution based on city."""
        base_prefs = {
            "Mumbai":    {"North Indian": 0.15, "South Indian": 0.10, "Chinese": 0.12, "Fast Food": 0.12, "Street Food": 0.12, "Pizza": 0.08, "Biryani": 0.08, "Cafe": 0.06, "Cloud Kitchen": 0.05, "Rolls": 0.04, "Desserts": 0.03, "Healthy": 0.02, "Mughlai": 0.01, "Thali": 0.01, "Regional": 0.01},
            "Delhi":     {"North Indian": 0.22, "Mughlai": 0.10, "Chinese": 0.10, "Street Food": 0.10, "Fast Food": 0.10, "Pizza": 0.07, "Biryani": 0.06, "Cafe": 0.07, "Rolls": 0.05, "South Indian": 0.04, "Desserts": 0.03, "Cloud Kitchen": 0.03, "Healthy": 0.01, "Thali": 0.01, "Regional": 0.01},
            "Bangalore": {"South Indian": 0.20, "North Indian": 0.12, "Chinese": 0.10, "Fast Food": 0.10, "Cafe": 0.10, "Pizza": 0.08, "Biryani": 0.07, "Healthy": 0.06, "Cloud Kitchen": 0.05, "Street Food": 0.04, "Rolls": 0.03, "Desserts": 0.02, "Mughlai": 0.01, "Thali": 0.01, "Regional": 0.01},
            "Hyderabad": {"Biryani": 0.25, "South Indian": 0.12, "North Indian": 0.10, "Chinese": 0.10, "Fast Food": 0.08, "Street Food": 0.07, "Pizza": 0.06, "Mughlai": 0.05, "Cloud Kitchen": 0.05, "Desserts": 0.04, "Cafe": 0.03, "Rolls": 0.02, "Healthy": 0.01, "Thali": 0.01, "Regional": 0.01},
            "Lucknow":   {"Mughlai": 0.22, "North Indian": 0.18, "Biryani": 0.12, "Street Food": 0.10, "Chinese": 0.08, "Fast Food": 0.08, "Cafe": 0.06, "Pizza": 0.04, "Rolls": 0.04, "South Indian": 0.03, "Desserts": 0.02, "Cloud Kitchen": 0.01, "Healthy": 0.01, "Thali": 0.005, "Regional": 0.005},
        }
        prefs = base_prefs.get(city, base_prefs["Mumbai"]).copy()

        # Add per-user noise
        noise = self.rng.dirichlet(np.ones(len(prefs)) * 5)
        keys = list(prefs.keys())
        for j, k in enumerate(keys):
            prefs[k] = prefs[k] * 0.7 + noise[j] * 0.3

        # If veg, reduce biryani/mughlai slightly, boost South Indian/Street Food
        if is_veg:
            prefs["Biryani"] *= 0.5
            prefs["Mughlai"] *= 0.4
            prefs["South Indian"] *= 1.3
            prefs["Street Food"] *= 1.2
            prefs["Thali"] *= 1.5

        # Normalize
        total = sum(prefs.values())
        return {k: round(v / total, 4) for k, v in prefs.items()}

    def _avg_order_for_segment(self, segment: str) -> int:
        """Average order value by user segment."""
        ranges = {"budget": (120, 250), "mid": (250, 500), "premium": (450, 900)}
        lo, hi = ranges[segment]
        return int(self.rng.integers(lo, hi))

    # ─────────────────────────────────────────
    # Order Generation
    # ─────────────────────────────────────────
    def generate_orders(self, users_df: pd.DataFrame) -> tuple:
        """Generate orders with realistic temporal and behavioral patterns."""
        orders = []
        order_items = []
        order_counter = 0

        # Assign order counts per user based on activity class
        user_order_counts = self._assign_order_counts(users_df)

        # Date range: 6 months of data
        start_date = datetime(2025, 9, 1)
        end_date = datetime(2026, 2, 28)
        date_range_days = (end_date - start_date).days

        for _, user in users_df.iterrows():
            n_user_orders = user_order_counts[user["user_id"]]
            user_cuisines = json.loads(user["preferred_cuisines"])

            for _ in range(n_user_orders):
                order_counter += 1
                order_id = f"O{order_counter:06d}"

                # Generate timestamp with meal-period distribution
                timestamp = self._generate_timestamp(start_date, date_range_days)
                meal_period = self._get_meal_period(timestamp.hour)
                season = self._get_season(timestamp.month)

                # Pick restaurant based on user city + cuisine preferences
                restaurant = self._pick_restaurant(user, user_cuisines)
                if restaurant is None:
                    continue

                # Determine order type (solo/pair/group/party)
                order_type = self._pick_order_type(meal_period, timestamp)

                # Generate cart items
                cart_items = self._generate_cart(
                    restaurant, user, meal_period, order_type, season
                )
                if not cart_items:
                    continue

                cart_value = sum(item["price"] for item in cart_items)

                # Determine if this is a festival period
                festival = self._get_festival(timestamp)

                orders.append({
                    "order_id": order_id,
                    "user_id": user["user_id"],
                    "restaurant_id": restaurant["id"],
                    "restaurant_name": restaurant["name"],
                    "cuisine": restaurant["cuisine"],
                    "city": user["city"],
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "hour": timestamp.hour,
                    "day_of_week": timestamp.weekday(),
                    "is_weekend": timestamp.weekday() >= 5,
                    "meal_period": meal_period,
                    "season": season,
                    "festival": festival,
                    "order_type": order_type,
                    "n_items": len(cart_items),
                    "cart_value": cart_value,
                    "order_completed": self.rng.random() < 0.88,  # ~12% abandon
                })

                for seq_idx, item in enumerate(cart_items):
                    order_items.append({
                        "order_id": order_id,
                        "user_id": user["user_id"],
                        "item_id": item["item_id"],
                        "item_name": item["name"],
                        "restaurant_id": restaurant["id"],
                        "cuisine": restaurant["cuisine"],
                        "category": item["category"],
                        "meal_role": item["meal_role"],
                        "is_veg": item["is_veg"],
                        "price": item["price"],
                        "sequence_position": seq_idx,  # Order of addition
                        "is_addon": seq_idx >= 1,  # First item = main, rest = add-ons
                    })

        return pd.DataFrame(orders), pd.DataFrame(order_items)

    def _assign_order_counts(self, users_df: pd.DataFrame) -> dict:
        """Assign number of orders per user (Zipf-like distribution)."""
        counts = {}
        total_target = self.n_orders

        for _, user in users_df.iterrows():
            activity = user["activity_class"]
            if activity == "sparse":
                n = max(1, int(self.rng.exponential(3)))
                n = min(n, 8)
            elif activity == "moderate":
                n = int(self.rng.integers(5, 25))
            else:  # power
                n = int(self.rng.integers(20, 80))
            counts[user["user_id"]] = n

        # Scale to approximate target order count
        current_total = sum(counts.values())
        scale_factor = total_target / current_total
        for uid in counts:
            counts[uid] = max(1, int(counts[uid] * scale_factor))

        return counts

    def _generate_timestamp(self, start_date: datetime, range_days: int) -> datetime:
        """Generate a realistic timestamp with meal-period weighting."""
        day_offset = int(self.rng.integers(0, range_days))
        date = start_date + timedelta(days=day_offset)

        # Pick meal period
        meal_period = self.rng.choice(
            list(MEAL_PERIODS.keys()), p=MEAL_PERIOD_WEIGHTS
        )
        hour_start, hour_end = MEAL_PERIODS[meal_period]

        # Handle late-night wrap-around
        hour = int(self.rng.integers(hour_start, hour_end))
        if hour >= 24:
            hour -= 24
            date += timedelta(days=1)

        minute = int(self.rng.integers(0, 60))
        return date.replace(hour=hour, minute=minute, second=0)

    def _get_meal_period(self, hour: int) -> str:
        """Classify hour into meal period."""
        if 7 <= hour < 11:
            return "breakfast"
        elif 11 <= hour < 15:
            return "lunch"
        elif 15 <= hour < 18:
            return "snack"
        elif 18 <= hour < 22:
            return "dinner"
        else:
            return "late_night"

    def _get_season(self, month: int) -> str:
        """Get season from month."""
        for season, months in SEASONS.items():
            if month in months:
                return season
        return "autumn"

    def _get_festival(self, dt: datetime) -> str:
        """Check if date falls in a festival period (simplified)."""
        month, day = dt.month, dt.day
        # Navratri (Oct 1-9), Diwali (Oct 28-Nov 2), Ramadan (Mar 1-30), Eid (Mar 31)
        if month == 10 and 1 <= day <= 9:
            return "navratri"
        if (month == 10 and day >= 28) or (month == 11 and day <= 2):
            return "diwali"
        if month == 3 and 1 <= day <= 30:
            return "ramadan"
        return "none"

    def _pick_restaurant(self, user: dict, cuisine_prefs: dict) -> dict:
        """Pick a restaurant matching user's city and cuisine preferences."""
        city = user["city"]
        city_restaurants = self.restaurants_df[self.restaurants_df["city"] == city]

        if city_restaurants.empty:
            # Fallback: any restaurant
            city_restaurants = self.restaurants_df

        # Weight by cuisine preference + restaurant rating
        weights = []
        for _, rest in city_restaurants.iterrows():
            cuisine_weight = cuisine_prefs.get(rest["cuisine"], 0.01)
            rating_weight = rest["rating"] / 5.0
            # Segment-price alignment
            segment = user["segment"]
            tier = rest["price_tier"]
            seg_match = 1.5 if segment == tier else 0.8
            weights.append(cuisine_weight * rating_weight * seg_match)

        weights = np.array(weights)
        if weights.sum() == 0:
            weights = np.ones(len(weights))
        weights /= weights.sum()

        idx = self.rng.choice(len(city_restaurants), p=weights)
        return city_restaurants.iloc[idx].to_dict()

    def _pick_order_type(self, meal_period: str, dt: datetime) -> str:
        """Determine order type based on context."""
        is_weekend = dt.weekday() >= 5
        weights = list(ORDER_TYPE_WEIGHTS)  # solo, pair, group, party

        if is_weekend:
            weights[0] *= 0.6   # Less solo
            weights[2] *= 1.5   # More group
            weights[3] *= 1.8   # More party
        if meal_period == "breakfast":
            weights[0] *= 1.5   # More solo breakfast
            weights[3] *= 0.3   # Less party breakfast
        if meal_period == "dinner":
            weights[2] *= 1.3   # More group dinner
        if meal_period == "late_night":
            weights[0] *= 1.4   # More solo late-night

        weights = np.array(weights)
        weights /= weights.sum()
        return self.rng.choice(["solo", "pair", "group", "party"], p=weights)

    def _generate_cart(self, restaurant: dict, user: dict,
                       meal_period: str, order_type: str, season: str) -> list:
        """Generate a realistic cart for this order context."""
        rest_id = restaurant["id"]
        menu = self.rest_items.get(rest_id)
        if menu is None or menu.empty:
            return []

        # Filter veg if user is veg
        available = menu.copy()
        if user["is_veg"]:
            available = available[available["is_veg"] == True]

        if available.empty:
            return []

        # Determine cart size
        params = CART_SIZE_PARAMS[order_type]
        cart_size = int(self.rng.triangular(
            params["min"], params["mode"], params["max"]
        ))
        cart_size = max(1, min(cart_size, len(available)))

        # Build cart with meal-structure awareness
        cart = []
        selected_ids = set()

        # Step 1: Pick the "anchor" item (usually a main course or combo)
        anchor_candidates = available[
            available["category"].isin(["main_course", "combo", "rice"])
        ]
        if anchor_candidates.empty:
            anchor_candidates = available

        anchor_weights = anchor_candidates["popularity"].values
        anchor_weights = anchor_weights / anchor_weights.sum()
        anchor_idx = self.rng.choice(len(anchor_candidates), p=anchor_weights)
        anchor = anchor_candidates.iloc[anchor_idx]
        cart.append(anchor.to_dict())
        selected_ids.add(anchor["item_id"])

        # Step 2: Fill remaining slots with complementary items
        remaining_slots = cart_size - 1
        remaining_items = available[~available["item_id"].isin(selected_ids)]

        if remaining_slots > 0 and not remaining_items.empty:
            # Weight by: popularity + meal-role diversity + season bonus
            filled_roles = {anchor["meal_role"]}
            for _ in range(remaining_slots):
                if remaining_items.empty:
                    break

                weights = []
                for _, item in remaining_items.iterrows():
                    w = item["popularity"]
                    # Boost items that fill a new meal role
                    if item["meal_role"] not in filled_roles:
                        w *= 2.0
                    # Season bonuses
                    if season == "summer" and item["meal_role"] == "drink":
                        w *= 1.8
                    if season == "monsoon" and item["category"] == "beverage" and "chai" in item["name"].lower():
                        w *= 2.0
                    if season == "winter" and item["meal_role"] == "dessert":
                        w *= 1.5
                    # Late-night: boost desserts and drinks
                    if meal_period == "late_night":
                        if item["meal_role"] in ("dessert", "drink"):
                            w *= 1.5
                    # Ensure price coherence with user segment
                    segment = user["segment"]
                    if segment == "budget" and item["price"] > 250:
                        w *= 0.4
                    elif segment == "premium" and item["price"] < 60:
                        w *= 0.6
                    weights.append(max(w, 0.01))

                weights = np.array(weights)
                weights /= weights.sum()

                idx = self.rng.choice(len(remaining_items), p=weights)
                selected = remaining_items.iloc[idx]
                cart.append(selected.to_dict())
                selected_ids.add(selected["item_id"])
                filled_roles.add(selected["meal_role"])
                remaining_items = remaining_items[
                    ~remaining_items["item_id"].isin(selected_ids)
                ]

        return cart

    # ─────────────────────────────────────────
    # Interaction Data for Training
    # ─────────────────────────────────────────
    def generate_training_data(self, orders_df: pd.DataFrame,
                               order_items_df: pd.DataFrame,
                               max_orders: int = 50000) -> pd.DataFrame:
        """
        Generate training examples using optimized dict-based lookups.
        Samples up to max_orders for tractable generation time.
        """
        # Sample orders for training data generation
        unique_orders = orders_df["order_id"].unique()
        if len(unique_orders) > max_orders:
            sampled_orders = set(self.rng.choice(unique_orders, size=max_orders, replace=False))
        else:
            sampled_orders = set(unique_orders)

        # Pre-index: order_id -> order info dict
        orders_indexed = orders_df.set_index("order_id")

        # Pre-index: restaurant_id -> list of item dicts (for fast negative sampling)
        rest_menu_dicts = {}
        for rid, menu_df in self.rest_items.items():
            rest_menu_dicts[rid] = menu_df.to_dict("records")

        training_examples = []
        processed = 0

        for order_id, group in order_items_df.groupby("order_id"):
            if order_id not in sampled_orders:
                continue

            try:
                oi = orders_indexed.loc[order_id]
            except KeyError:
                continue

            items_list = group.sort_values("sequence_position").to_dict("records")
            restaurant_id = oi["restaurant_id"]
            menu_items = rest_menu_dicts.get(restaurant_id, [])
            if not menu_items:
                continue

            all_veg = all(it["is_veg"] for it in items_list)

            # Base context dict (shared across all examples from this order)
            base = {
                "order_id": order_id,
                "user_id": oi["user_id"],
                "restaurant_id": restaurant_id,
                "cuisine": oi["cuisine"],
                "city": oi["city"],
                "meal_period": oi["meal_period"],
                "season": oi["season"],
                "festival": oi["festival"],
                "hour": oi["hour"],
                "day_of_week": oi["day_of_week"],
                "is_weekend": oi["is_weekend"],
                "order_type": oi["order_type"],
                "order_completed": oi["order_completed"],
            }

            for pos in range(1, len(items_list)):
                cart_ids = [items_list[j]["item_id"] for j in range(pos)]
                cart_val = sum(items_list[j]["price"] for j in range(pos))
                added = items_list[pos]

                ex = {
                    **base,
                    "cart_item_ids": json.dumps(cart_ids),
                    "cart_size": pos,
                    "cart_value": cart_val,
                    "candidate_item_id": added["item_id"],
                    "candidate_category": added["category"],
                    "candidate_meal_role": added["meal_role"],
                    "candidate_is_veg": added["is_veg"],
                    "candidate_price": added["price"],
                    "label": 1,
                }
                training_examples.append(ex)

                # Hard negative mining: 1 same-role, 1 popularity-weighted, 1 random
                excluded = set(cart_ids) | {added["item_id"]}
                neg_pool = [m for m in menu_items
                            if m["item_id"] not in excluded
                            and (not all_veg or m["is_veg"])]

                if not neg_pool:
                    continue

                negatives = []

                # Hard neg 1: same meal role as positive (forces model to distinguish within role)
                same_role = [m for m in neg_pool if m["meal_role"] == added["meal_role"]]
                if same_role:
                    negatives.append(same_role[self.rng.integers(len(same_role))])
                elif neg_pool:
                    negatives.append(neg_pool[self.rng.integers(len(neg_pool))])

                # Hard neg 2: popularity-weighted (popular items user skipped are informative)
                remaining = [m for m in neg_pool if m not in negatives]
                if remaining:
                    pop_weights = np.array([m["popularity"] for m in remaining])
                    pop_weights = pop_weights / pop_weights.sum()
                    idx = self.rng.choice(len(remaining), p=pop_weights)
                    negatives.append(remaining[idx])

                # Hard neg 3: random (maintains diversity)
                remaining2 = [m for m in neg_pool if m not in negatives]
                if remaining2:
                    negatives.append(remaining2[self.rng.integers(len(remaining2))])

                for neg in negatives:
                    training_examples.append({
                        **base,
                        "cart_item_ids": json.dumps(cart_ids),
                        "cart_size": pos,
                        "cart_value": cart_val,
                        "candidate_item_id": neg["item_id"],
                        "candidate_category": neg["category"],
                        "candidate_meal_role": neg["meal_role"],
                        "candidate_is_veg": neg["is_veg"],
                        "candidate_price": neg["price"],
                        "label": 0,
                    })

            processed += 1
            if processed % 10000 == 0:
                print(f"  Training data: {processed}/{len(sampled_orders)} orders processed...")

        return pd.DataFrame(training_examples)

    # ─────────────────────────────────────────
    # Save All
    # ─────────────────────────────────────────
    def generate_and_save(self, output_dir: str = "data/generated"):
        """Generate all data and save to CSV."""
        os.makedirs(output_dir, exist_ok=True)

        print("Generating users...")
        users_df = self.generate_users()
        users_df.to_csv(f"{output_dir}/users.csv", index=False)
        print(f"  → {len(users_df)} users")

        print("Generating orders...")
        orders_df, order_items_df = self.generate_orders(users_df)
        orders_df.to_csv(f"{output_dir}/orders.csv", index=False)
        order_items_df.to_csv(f"{output_dir}/order_items.csv", index=False)
        print(f"  → {len(orders_df)} orders, {len(order_items_df)} order-items")

        print("Generating training data...")
        training_df = self.generate_training_data(orders_df, order_items_df)
        training_df.to_csv(f"{output_dir}/training_data.csv", index=False)
        print(f"  → {len(training_df)} training examples "
              f"({training_df['label'].sum()} positive, "
              f"{(training_df['label'] == 0).sum()} negative)")

        # Summary stats
        print("\n=== Dataset Summary ===")
        print(f"Users:    {len(users_df)}")
        print(f"Orders:   {len(orders_df)}")
        print(f"Items:    {len(order_items_df)}")
        print(f"Training: {len(training_df)}")
        print(f"\nOrders per meal period:")
        print(orders_df["meal_period"].value_counts())
        print(f"\nCart size distribution:")
        print(orders_df["n_items"].describe())
        print(f"\nOrder completion rate: {orders_df['order_completed'].mean():.2%}")
        print(f"\nCity distribution:")
        print(orders_df["city"].value_counts())

        return users_df, orders_df, order_items_df, training_df


if __name__ == "__main__":
    generator = SyntheticDataGenerator(n_users=10000, n_orders=120000, seed=42)
    generator.generate_and_save()
