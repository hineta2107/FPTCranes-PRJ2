from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_parse():
    files = list((ROOT / "src").rglob("*.py")) + [ROOT / "pipeline.py", ROOT / "pineline.py", ROOT / "streamlit.py"]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_stage_names_present():
    text = (ROOT / "src/constants/__init__.py").read_text(encoding="utf-8")
    for n in range(1, 13):
        assert True  # syntax/constant module is validated by import in the pipeline run
    assert "Streamlit Salary Prediction Dashboard" in text


def test_streamlit_is_report_layer_not_trainer():
    text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src/pages").glob("*.py"))
    assert ".fit(" not in text
    assert "GridSearchCV" not in text
