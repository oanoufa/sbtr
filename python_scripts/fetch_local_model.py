import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from python_scripts.src import config

p = argparse.ArgumentParser(description="Download sbtr model files for offline use (needs HF_TOKEN for the gated backbone).")
p.add_argument("--dest", default=None, help="Local directory; default: the Hugging Face cache.")
p.add_argument("--revision", default=config.MODEL_REVISION)
a = p.parse_args()

model_dir = Path(snapshot_download(
    config.MODEL_CONFIG["model_name"], revision=a.revision,
    local_dir=f"{a.dest}/model" if a.dest else None,
))
cfg = json.loads((model_dir / "config.json").read_text())
backbone_dir = snapshot_download(
    cfg["backbone_name"], revision=cfg.get("backbone_revision"),
    allow_patterns=["*.json", "*.py", "*.txt"],
    local_dir=f"{a.dest}/backbone" if a.dest else None,
)
print(f"model:    {model_dir}\nbackbone: {backbone_dir}")