from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: str | Path = None) -> dict:
    path = Path(path) if path else ROOT / "prompts" / "prompts.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
