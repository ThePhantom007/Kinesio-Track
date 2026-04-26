"""
Re-exports the most commonly used query functions so services can import
from ``app.db.queries`` directly.
"""

from app.db.queries.analytics import (  # noqa: F401
    last_n_session_metrics,
    last_n_session_metrics_for_exercise,
    monthly_token_spend,
    pain_trend,
    quality_trend_slope,
    rom_vs_baseline,
)
from app.db.queries.progress import (  # noqa: F401
    get_milestones,
    progress_summary,
    quality_score_series,
    rom_series_all_joints,
    rom_series_by_joint,
    session_frequency,
)