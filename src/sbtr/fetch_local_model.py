import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download, get_token
from huggingface_hub.errors import GatedRepoError
from sbtr import config
import sys


def main(argv=None):

    p = argparse.ArgumentParser(description="Download sbtr model files for offline use (needs HF_TOKEN for the gated backbone).")
    p.add_argument("--dest", default=None, help="Local directory; default: the Hugging Face cache.")
    p.add_argument("--revision", default=config.MODEL_REVISION)
    p.add_argument("--token", default=None,
                help="Hugging Face token (default: HF_TOKEN or the stored login). "
                        "Needed once, for the gated backbone.")
    a = p.parse_args()
    token = a.token or get_token()

    model_dir = Path(snapshot_download(
        config.MODEL_CONFIG["model_name"], revision=a.revision,
        local_dir=f"{a.dest}/model" if a.dest else None,
    ))
    cfg = json.loads((model_dir / "config.json").read_text())
    try:
        backbone_dir = snapshot_download(
            cfg["backbone_name"], revision=cfg.get("backbone_revision"), token=token,
            allow_patterns=["*.json", "*.py", "*.txt"],
            local_dir=f"{a.dest}/backbone" if a.dest else None,
        )

        backbone_code_dir = snapshot_download(
            cfg["backbone_code_name"], revision=cfg.get("backbone_code_revision"), token=token,
            allow_patterns=["*.json", "*.py", "*.txt"],
            local_dir=f"{a.dest}/backbone_code" if a.dest else None,
        )
    except GatedRepoError:
        sys.exit(f"{cfg['backbone_name']} is gated: accept its terms on huggingface.co with the account "
                "that owns the token, then retry with HF_TOKEN set or `hf auth login`.")

    print(                              # ← fix 4
        f"model:         {model_dir}\n"
        f"backbone:      {backbone_dir}\n"
        f"backbone_code: {backbone_code_dir}"
    )

if __name__ == "__main__":
    main()