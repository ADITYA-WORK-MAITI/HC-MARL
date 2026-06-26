"""HC-MARL Baselines package — MAPPO-Lagrangian wrapper + legacy baselines."""
from hcmarl.baselines.safepo_wrapper import SafePOWrapper

# Re-export Phase 2 legacy baselines so existing imports keep working
from hcmarl.baselines._legacy import (
    RandomBaseline, RoundRobinBaseline, GreedyBaseline, GreedySafeBaseline,
    PPOUnconstrainedBaseline, PPOLagrangianBaseline, MAPPONoFilterBaseline,
    MAPPOLagrangianBaseline, FixedScheduleBaseline,
    create_all_baselines,
)
