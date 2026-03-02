"""
A/B Testing Framework Design for CSAO Recommendations.

Defines the experiment structure for validating recommendation quality
in production, with statistical rigor.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ExperimentGroup(Enum):
    CONTROL = "control"       # No add-on recommendations
    BASELINE = "baseline"     # Popularity-based recommendations
    MODEL_V1 = "model_v1"    # Current DCN-V2 reranker
    MODEL_V2 = "model_v2"    # Future improved model


@dataclass
class GuardrailMetric:
    """Metric that must not degrade beyond threshold."""
    name: str
    threshold: float        # Maximum acceptable degradation (%)
    current_value: float


@dataclass
class ABTestConfig:
    """Configuration for a recommendation A/B test."""
    experiment_name: str
    hypothesis: str
    primary_metric: str = "addon_acceptance_rate"

    # Traffic allocation
    control_pct: float = 0.20
    treatment_pct: float = 0.80

    # Statistical parameters
    significance_level: float = 0.05
    power: float = 0.80
    min_effect_size: float = 0.02  # 2% minimum detectable effect

    # Duration
    min_days: int = 7
    max_days: int = 21

    # Guardrails
    guardrails: List[GuardrailMetric] = field(default_factory=lambda: [
        GuardrailMetric("cart_abandonment_rate", 2.0, 12.0),  # Max 2% increase
        GuardrailMetric("avg_session_time", -5.0, 180.0),     # Max 5% decrease
        GuardrailMetric("support_ticket_rate", 1.0, 0.5),     # Max 1% increase
        GuardrailMetric("app_crash_rate", 0.5, 0.1),          # Max 0.5% increase
    ])

    # Metrics to track
    success_metrics: List[str] = field(default_factory=lambda: [
        "addon_acceptance_rate",    # Primary: % of shown recommendations accepted
        "addon_items_per_order",    # Avg add-on items per order
        "aov_lift",                 # Average order value lift
        "cart_to_order_rate",       # Cart completion rate
        "recommendation_ctr",      # Click-through rate on CSAO rail
        "revenue_per_session",      # Revenue per user session
    ])


class ABTestDesign:
    """Designs and validates A/B test parameters."""

    def __init__(self, config: ABTestConfig):
        self.config = config

    def compute_sample_size(self, baseline_rate: float = 0.15,
                             mde: float = 0.02) -> int:
        """
        Compute minimum sample size per variant.
        Uses normal approximation for proportion test.
        """
        from scipy.stats import norm

        alpha = self.config.significance_level
        beta = 1 - self.config.power
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(1 - beta)

        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_avg = (p1 + p2) / 2

        n = (
            (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        ) / (mde ** 2)

        return int(np.ceil(n))

    def estimate_duration(self, daily_orders: int = 50000) -> Dict[str, float]:
        """Estimate experiment duration given daily traffic."""
        sample_size = self.compute_sample_size()
        total_needed = sample_size * 2  # Control + treatment

        days_needed = np.ceil(total_needed / daily_orders)
        days_needed = max(days_needed, self.config.min_days)

        return {
            "sample_size_per_variant": sample_size,
            "total_samples_needed": total_needed,
            "estimated_days": int(days_needed),
            "daily_orders": daily_orders,
        }

    def interleaving_design(self) -> Dict:
        """
        Interleaving test design for rapid preference detection.
        Interleaving merges control + treatment recommendations
        and measures which items users prefer.
        """
        return {
            "method": "Team Draft Interleaving",
            "description": (
                "Control and treatment recommendation lists are interleaved "
                "into a single merged list. User interactions determine "
                "which model's recommendations are preferred."
            ),
            "advantages": [
                "Requires 10-100x fewer samples than standard A/B test",
                "Can detect smaller effects",
                "No traffic splitting needed",
            ],
            "evaluation": {
                "winning_fraction": "Fraction of interactions attributed to treatment model",
                "statistical_test": "Binomial sign test",
                "significance_level": self.config.significance_level,
            },
        }

    def generate_report(self) -> Dict:
        """Generate full experiment design report."""
        duration = self.estimate_duration()
        interleaving = self.interleaving_design()

        return {
            "experiment": {
                "name": self.config.experiment_name,
                "hypothesis": self.config.hypothesis,
                "primary_metric": self.config.primary_metric,
            },
            "traffic_allocation": {
                "control": f"{self.config.control_pct:.0%}",
                "treatment": f"{self.config.treatment_pct:.0%}",
            },
            "statistical_design": {
                "significance_level": self.config.significance_level,
                "power": self.config.power,
                "min_detectable_effect": self.config.min_effect_size,
                **duration,
            },
            "success_metrics": self.config.success_metrics,
            "guardrails": [
                {"metric": g.name, "max_degradation": f"{g.threshold}%",
                 "current_value": g.current_value}
                for g in self.config.guardrails
            ],
            "interleaving_option": interleaving,
            "rollout_strategy": {
                "phase_1": "5% canary (1 day) — monitor guardrails",
                "phase_2": "20% treatment (3 days) — statistical significance check",
                "phase_3": "50/50 split (7 days) — full A/B test",
                "phase_4": "100% rollout if guardrails pass + metrics improve",
                "automated_rollback": "Triggered if any guardrail exceeds threshold",
            },
        }


if __name__ == "__main__":
    import json

    config = ABTestConfig(
        experiment_name="CSAO Rail v1.0 Launch",
        hypothesis=(
            "Adding ML-powered contextual add-on recommendations to the cart page "
            "will increase add-on acceptance rate by ≥2% and AOV by ≥5%, "
            "without increasing cart abandonment."
        ),
    )

    designer = ABTestDesign(config)
    report = designer.generate_report()

    print("=" * 60)
    print("A/B Test Design Report")
    print("=" * 60)
    print(json.dumps(report, indent=2, default=str))
