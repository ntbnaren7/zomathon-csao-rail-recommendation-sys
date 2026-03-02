"""
Meal DNA Encoder — Maps cart items into a structured meal completion vector.

The Meal DNA is a 6-dimensional vector representing how "complete" a meal is:
  [protein, carb, side, dessert, drink, accompaniment]

Each dimension ranges 0.0–1.0, where 0 = missing, 1 = fully satisfied.
Cuisine-aware: "complete" means different things for different cuisines.
"""

import numpy as np
from typing import List, Dict, Optional
from data.menu_catalog import MEAL_TEMPLATES as CUISINE_MEAL_TEMPLATES, MEAL_ROLES


class MealDNAEncoder:
    """Encodes cart state into a Meal DNA completion vector."""

    def __init__(self):
        self.roles = MEAL_ROLES  # ["protein", "carb", "side", "dessert", "drink", "accompaniment"]
        self.cuisine_templates = CUISINE_MEAL_TEMPLATES
        # Default template if cuisine or meal period not found
        self.default_template = {r: 0.5 for r in self.roles}

    def encode(self, cart_items: List[Dict], cuisine: str,
               meal_period: str = "dinner") -> Dict[str, float]:
        """
        Compute Meal DNA for the current cart state.

        Args:
            cart_items: List of item dicts with 'meal_role' key
            cuisine: Restaurant cuisine type
            meal_period: Current meal period (breakfast/lunch/dinner/etc.)

        Returns:
            Dict with fill-state for each meal role (0.0 to 1.0)
        """
        # Get ideal template for this cuisine + meal period
        template = self._get_template(cuisine, meal_period)

        # Count items per meal role in the cart
        role_counts = {r: 0 for r in self.roles}
        for item in cart_items:
            role = item.get("meal_role", "side")
            if role in role_counts:
                role_counts[role] += 1

        # Compute fill state relative to template
        meal_dna = {}
        for role in self.roles:
            target = template.get(role, 0.5)
            if target <= 0:
                # This role isn't expected for this cuisine/meal
                meal_dna[role] = 1.0 if role_counts[role] == 0 else 1.0
            else:
                # Fill = count / expected, capped at 1.0
                meal_dna[role] = min(1.0, role_counts[role] / max(target, 0.1))

        return meal_dna

    def get_gap_scores(self, meal_dna: Dict[str, float]) -> Dict[str, float]:
        """
        Compute gap scores — how much each role is "missing."
        Higher gap = more needed. Used to score candidate items.

        Returns:
            Dict mapping each role to its gap score (0.0 = filled, 1.0 = empty)
        """
        return {role: max(0.0, 1.0 - fill) for role, fill in meal_dna.items()}

    def get_completion_score(self, meal_dna: Dict[str, float]) -> float:
        """Overall meal completion percentage (0.0 to 1.0)."""
        return np.mean(list(meal_dna.values()))

    def get_candidate_gap_score(self, meal_dna: Dict[str, float],
                                 candidate_meal_role: str) -> float:
        """
        How much adding a candidate item would fill a gap.
        Returns gap value for the candidate's meal role.
        """
        gaps = self.get_gap_scores(meal_dna)
        return gaps.get(candidate_meal_role, 0.0)

    def encode_as_vector(self, cart_items: List[Dict], cuisine: str,
                         meal_period: str = "dinner") -> np.ndarray:
        """Return Meal DNA as a numpy vector (for model input)."""
        dna = self.encode(cart_items, cuisine, meal_period)
        return np.array([dna[r] for r in self.roles], dtype=np.float32)

    def get_gap_vector(self, cart_items: List[Dict], cuisine: str,
                       meal_period: str = "dinner") -> np.ndarray:
        """Return gap scores as a numpy vector."""
        dna = self.encode(cart_items, cuisine, meal_period)
        gaps = self.get_gap_scores(dna)
        return np.array([gaps[r] for r in self.roles], dtype=np.float32)

    def _get_template(self, cuisine: str, meal_period: str) -> dict:
        """Get ideal meal template for cuisine + meal period."""
        templates = self.cuisine_templates.get(cuisine, {})
        if meal_period in templates:
            return templates[meal_period]
        # Fallback: try "any", then default
        return templates.get("any", self.default_template)

    def get_missing_roles(self, meal_dna: Dict[str, float],
                          threshold: float = 0.3) -> List[str]:
        """Return list of meal roles that are below the fill threshold."""
        return [role for role, fill in meal_dna.items() if fill < threshold]

    def get_explanation(self, meal_dna: Dict[str, float],
                        candidate_meal_role: str) -> str:
        """Generate human-readable explanation for why item is recommended."""
        gaps = self.get_gap_scores(meal_dna)
        gap = gaps.get(candidate_meal_role, 0.0)

        role_labels = {
            "protein": "a main dish",
            "carb": "bread or rice",
            "side": "a side dish",
            "dessert": "something sweet",
            "drink": "a beverage",
            "accompaniment": "an accompaniment",
        }
        label = role_labels.get(candidate_meal_role, candidate_meal_role)

        if gap >= 0.7:
            return f"Completes your meal — you're missing {label}!"
        elif gap >= 0.3:
            return f"Would nicely round out your order"
        else:
            return f"A popular add-on with your items"


if __name__ == "__main__":
    # Demo: Biryani dinner scenario
    encoder = MealDNAEncoder()

    # State 1: Just biryani
    cart_v1 = [{"meal_role": "protein", "name": "Chicken Biryani"}]
    dna_v1 = encoder.encode(cart_v1, "Biryani", "dinner")
    print("=== Cart: [Chicken Biryani] ===")
    for role, fill in dna_v1.items():
        bar = "█" * int(fill * 10) + "░" * (10 - int(fill * 10))
        print(f"  {role:15s} {bar} {fill:.0%}")
    print(f"  Completion: {encoder.get_completion_score(dna_v1):.0%}")
    print(f"  Missing: {encoder.get_missing_roles(dna_v1)}")
    print()

    # State 2: Biryani + Raita + Salan
    cart_v2 = [
        {"meal_role": "protein", "name": "Chicken Biryani"},
        {"meal_role": "accompaniment", "name": "Raita"},
        {"meal_role": "side", "name": "Mirchi Ka Salan"},
    ]
    dna_v2 = encoder.encode(cart_v2, "Biryani", "dinner")
    print("=== Cart: [Chicken Biryani, Raita, Salan] ===")
    for role, fill in dna_v2.items():
        bar = "█" * int(fill * 10) + "░" * (10 - int(fill * 10))
        print(f"  {role:15s} {bar} {fill:.0%}")
    print(f"  Completion: {encoder.get_completion_score(dna_v2):.0%}")
    print(f"  Missing: {encoder.get_missing_roles(dna_v2)}")

    # Check gap scores for candidate items
    print("\n  Gap scores for candidates:")
    for role in ["dessert", "drink", "side", "protein"]:
        gap = encoder.get_candidate_gap_score(dna_v2, role)
        expl = encoder.get_explanation(dna_v2, role)
        print(f"    {role:15s} gap={gap:.2f}  → {expl}")
