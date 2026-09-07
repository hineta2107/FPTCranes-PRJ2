from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_raw_dataset_contract():
    df = pd.read_csv(ROOT / "data/raw/ai_jobs_market_2025_2026.csv")
    assert df.shape == (1500, 25)
    assert "annual_salary_usd" in df.columns


def test_confirmed_corruption_removed_once():
    removed = pd.read_csv(ROOT / "outputs/01_data_basic_clean/04_corrupted_rows_removed.csv")
    clean = pd.read_csv(ROOT / "outputs/01_data_basic_clean/data_basic_clean.csv")
    assert len(removed) == 1
    assert len(clean) == 1499
    assert "job_category" in clean.columns
    assert "AI Engineering" not in clean.columns


def test_quality_audit_catches_header_token():
    q = pd.read_csv(ROOT / "outputs/01_data_basic_clean/03_data_quality_summary.csv").iloc[0]
    assert q["missing_cells"] == 0
    assert q["duplicate_rows"] == 0
    assert q["invalid_job_category_token_rows"] == 1


def test_temporal_locked_split_contract():
    split = pd.read_csv(ROOT / "outputs/02_data_ready_for_machine_learning/08_split_summary.csv")
    rows = dict(zip(split["partition"], split["rows"]))
    assert rows["TRAIN_DEV"] == 1201
    assert rows["LOCKED_TEST"] == 298


def test_train_only_skill_vocabulary():
    readiness = json.loads((ROOT / "outputs/02_data_ready_for_machine_learning/08_training_readiness.json").read_text())
    assert readiness["skill_vocabulary_count"] == 93
    assert "TRAIN/DEV only" in readiness["fit_scope"]


def test_leakage_gate_passes():
    gate = json.loads((ROOT / "outputs/02_data_ready_for_machine_learning/06_leakage_gate.json").read_text())
    assert gate["status"] == "PASS"
    assert gate["blocked_overlap"] == []


def test_model_selection_is_cv_based():
    comp = pd.read_csv(ROOT / "outputs/03_model_comparison/09_model_comparison_temporal_cv.csv")
    candidates = comp.loc[~comp["is_baseline"].astype(bool)].sort_values("CV_MAE_mean")
    final = pd.read_csv(ROOT / "outputs/04_best_model_and_feature_importance/10_final_locked_test_metrics.csv").iloc[0]
    assert final["selected_model"] == candidates.iloc[0]["model"]
    assert "temporal-CV MAE" in final["selection_basis"]


def test_bundle_reload_equivalence_passes():
    result = json.loads((ROOT / "artifacts/11_bundle_equivalence.json").read_text())
    assert result["status"] == "PASS"
    assert result["max_abs_prediction_difference"] == 0.0


def test_bundle_and_component_artifacts_exist():
    for name in ["model_bundle.joblib", "preprocessor.pkl", "model.pkl", "feature_columns.json", "metadata.json"]:
        assert (ROOT / "artifacts" / name).exists()
    assert joblib.load(ROOT / "artifacts/model_bundle.joblib") is not None


def test_streamlit_menu_structure_is_preserved():
    nav = (ROOT / "src/pages/_nav.py").read_text(encoding="utf-8")
    for label in [
        "1. Data basic clean",
        "2. Data ready for ML",
        "3. Model comparison",
        "4. Best model",
        "5. Salary prediction",
        "12-stage pipeline",
    ]:
        assert label in nav
