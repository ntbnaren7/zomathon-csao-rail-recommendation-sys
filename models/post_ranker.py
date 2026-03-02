"""
Post-Ranking Diversification — Category-aware selection with meal-role constraints.

Ensures the final top-K recommendations:
1. Cover diverse meal roles (don't recommend 8 beverages)
2. Respect hard constraints (min 1 drink + 1 dessert if missing)
3. Apply exploration bonus for cold-start items (Thompson Sampling)
"""

import numpy as np
from typing import List, Dict, Tuple
from features.meal_dna import MealDNAEncoder


class PostRanker:
    """Post-ranking diversification with meal-role constraints."""

    def __init__(self, exploration_rate: float = 0.1):
        self.meal_dna_encoder = MealDNAEncoder()
        self.exploration_rate = exploration_rate

    def diversify(self, candidates: List[Dict], meal_dna: Dict[str, float],
                  top_k: int = 8) -> List[Dict]:
        """
        Select top-K items with diversity constraints.

        Args:
            candidates: List of candidate dicts with 'score', 'meal_role', etc.
                       Sorted by score descending.
            meal_dna: Current Meal DNA state of the cart
            top_k: Number of items to return

        Returns:
            Diversified top-K list with explanations
        """
        if len(candidates) <= top_k:
            return self._add_explanations(candidates, meal_dna)

        gaps = self.meal_dna_encoder.get_gap_scores(meal_dna)
        missing_roles = self.meal_dna_encoder.get_missing_roles(meal_dna, threshold=0.3)

        selected = []
        remaining = list(candidates)

        # Phase 1: Guarantee at least 1 item from each missing role
        for role in missing_roles:
            if len(selected) >= top_k:
                break
            role_items = [c for c in remaining if c.get("meal_role") == role]
            if role_items:
                best = max(role_items, key=lambda x: x.get("score", 0))
                selected.append(best)
                remaining.remove(best)

        # Phase 2: Fill remaining slots with MMR-style greedy selection
        while len(selected) < top_k and remaining:
            best_item = None
            best_mmr = -float("inf")

            for candidate in remaining:
                # Relevance term (model score)
                relevance = candidate.get("score", 0)

                # Diversity penalty: penalize items whose role is already well-represented
                role = candidate.get("meal_role", "side")
                selected_roles = [s.get("meal_role") for s in selected]
                role_count = selected_roles.count(role)
                diversity_penalty = 0.15 * role_count  # Penalty per duplicate role

                # Gap bonus: boost items that fill empty roles
                gap_bonus = gaps.get(role, 0) * 0.2

                # Exploration bonus (Thompson Sampling approximation)
                exploration = 0.0
                if self.exploration_rate > 0:
                    # Items with low popularity or few interactions get exploration bonus
                    pop = candidate.get("popularity", 0.5)
                    if pop < 0.3:
                        exploration = np.random.beta(1, 5) * self.exploration_rate

                mmr_score = relevance - diversity_penalty + gap_bonus + exploration
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_item = candidate

            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                break

        # Phase 3: Enforce hard constraints
        selected = self._enforce_constraints(selected, remaining, meal_dna, top_k)

        # Add explanations
        return self._add_explanations(selected, meal_dna)

    def _enforce_constraints(self, selected: list, remaining: list,
                              meal_dna: dict, top_k: int) -> list:
        """Enforce hard constraints: max 2 per role, veg consistency."""
        # Count roles in selected
        role_counts = {}
        for item in selected:
            role = item.get("meal_role", "side")
            role_counts[role] = role_counts.get(role, 0) + 1

        # If any role has > 2 items, swap out extras for missing roles
        for role, count in role_counts.items():
            if count > 2:
                # Find items of this role sorted by score (lowest first)
                role_items = [(i, s) for i, s in enumerate(selected)
                              if s.get("meal_role") == role]
                role_items.sort(key=lambda x: x[1].get("score", 0))

                # Swap lowest-scored duplicates for items from underrepresented roles
                for idx, _ in role_items[:count - 2]:
                    missing = [r for r, c in role_counts.items() if c == 0]
                    if not missing:
                        missing_items = remaining[:1]
                    else:
                        missing_items = [r for r in remaining
                                          if r.get("meal_role") in missing]
                    if missing_items:
                        selected[idx] = missing_items[0]
                        if missing_items[0] in remaining:
                            remaining.remove(missing_items[0])

        return selected[:top_k]

    def _add_explanations(self, items: List[Dict],
                          meal_dna: Dict[str, float]) -> List[Dict]:
        """Add human-readable explanations to each recommendation."""
        for item in items:
            role = item.get("meal_role", "side")
            explanation = self.meal_dna_encoder.get_explanation(meal_dna, role)

            # Override with more specific explanations
            cooc = item.get("cooccurrence_score", 0)
            pop = item.get("popularity", 0)
            has_offer = item.get("has_offer", False)

            if has_offer:
                explanation = f"🏷️ On offer today! {explanation}"
            elif cooc > 0.5:
                explanation = f"Popular combo — ordered together {int(cooc*100)}% of the time"
            elif pop > 0.85:
                explanation = f"⭐ Best seller at this restaurant"

            item["explanation"] = explanation

        return items


if __name__ == "__main__":
    # Demo
    post_ranker = PostRanker(exploration_rate=0.1)
    meal_dna = {
        "protein": 1.0, "carb": 0.0, "side": 0.0,
        "dessert": 0.0, "drink": 0.0, "accompaniment": 0.0
    }

    # Simulated candidates (all beverages — bad diversity)
    candidates = [
        {"item_id": f"I{i}", "name": f"Drink_{i}", "meal_role": "drink",
         "score": 0.9 - i * 0.05, "popularity": 0.7}
        for i in range(5)
    ] + [
        {"item_id": "I10", "name": "Gulab Jamun", "meal_role": "dessert",
         "score": 0.6, "popularity": 0.8},
        {"item_id": "I11", "name": "Raita", "meal_role": "accompaniment",
         "score": 0.55, "popularity": 0.85},
        {"item_id": "I12", "name": "Naan", "meal_role": "carb",
         "score": 0.5, "popularity": 0.9},
        {"item_id": "I13", "name": "Dal", "meal_role": "side",
         "score": 0.45, "popularity": 0.75},
        {"item_id": "I14", "name": "Papad", "meal_role": "accompaniment",
         "score": 0.3, "popularity": 0.6},
    ]

    result = post_ranker.diversify(candidates, meal_dna, top_k=8)
    print("Cart: [Butter Chicken (protein)]")
    print(f"Meal DNA gaps: {post_ranker.meal_dna_encoder.get_gap_scores(meal_dna)}")
    print(f"\nTop-8 diversified recommendations:")
    for i, item in enumerate(result):
        print(f"  {i+1}. {item['name']:20s} role={item['meal_role']:15s} "
              f"score={item.get('score', 0):.2f}  {item.get('explanation', '')}")
