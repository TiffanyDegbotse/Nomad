from __future__ import annotations
import os, json, base64, pathlib, uuid
from datetime import datetime, timezone
from typing import Any, Dict
from loguru import logger
from dotenv import load_dotenv

# Absolute project root (no references to other Nomad modules)
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Load environment once
env_path = ROOT / "config" / "keys.env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    logger.warning(f"keys.env not found at {env_path}")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")

def read_yaml(path: pathlib.Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def write_json(path: pathlib.Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def b64_image(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def gen_id(prefix: str = "evt") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def get_openai_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment. Set it in config/keys.env")
    return OpenAI()
