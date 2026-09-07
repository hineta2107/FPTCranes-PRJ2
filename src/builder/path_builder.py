from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    data_raw: Path
    output: Path
    basic: Path
    ml_ready: Path
    comparison: Path
    best: Path
    prediction: Path
    artifacts: Path
    assets: Path
    reports: Path


def build_paths(project_root: Path, raw_path: Path | None = None, output_root: Path | None = None) -> PipelinePaths:
    project_root = project_root.resolve()
    raw = (raw_path or (project_root / "data" / "raw" / "ai_jobs_market_2025_2026.csv")).resolve()
    output = (output_root or (project_root / "outputs")).resolve()
    paths = PipelinePaths(
        root=project_root,
        data_raw=raw,
        output=output,
        basic=output / "01_data_basic_clean",
        ml_ready=output / "02_data_ready_for_machine_learning",
        comparison=output / "03_model_comparison",
        best=output / "04_best_model_and_feature_importance",
        prediction=output / "05_salary_prediction",
        artifacts=project_root / "artifacts",
        assets=project_root / "assets",
        reports=project_root / "reports",
    )
    for p in [
        paths.output,
        paths.basic,
        paths.ml_ready,
        paths.comparison,
        paths.best,
        paths.prediction,
        paths.artifacts,
        paths.assets,
        paths.reports,
    ]:
        p.mkdir(parents=True, exist_ok=True)
    return paths
