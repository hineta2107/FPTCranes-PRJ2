from __future__ import annotations

from pathlib import Path
import json
import yaml


class Config:
    ROOT_DIR = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT_DIR / "config" / "project.yaml"
    DATA_DIR = ROOT_DIR / "data" / "raw"
    OUTPUT_DIR = ROOT_DIR / "outputs"
    ARTIFACT_DIR = ROOT_DIR / "artifacts"
    ASSET_DIR = ROOT_DIR / "assets"
    REPORT_DIR = ROOT_DIR / "reports"

    @classmethod
    def load(cls) -> dict:
        with cls.CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def report_index(cls) -> dict:
        path = cls.OUTPUT_DIR / "report_index.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
