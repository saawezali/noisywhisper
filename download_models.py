import argparse
import os
import sys

from huggingface_hub import snapshot_download

MODEL_REPOS = {
    "medium": "Systran/faster-whisper-medium",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def parse_models(value):
    if not value:
        return []
    models = []
    for item in value.split(","):
        name = item.strip()
        if name:
            models.append(name)
    return models


def prompt_models():
    print("Available models:")
    for name in MODEL_REPOS:
        print(f"- {name}")
    response = input("Enter comma-separated models (default: large-v3-turbo): ").strip()
    if not response:
        return ["large-v3-turbo"]
    return parse_models(response)


def download_model(name, target_root):
    if name not in MODEL_REPOS:
        raise ValueError(f"Unknown model: {name}")

    repo_id = MODEL_REPOS[name]
    local_dir = os.path.join(target_root, name)

    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {name} to {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Finished {name}")


def main():
    parser = argparse.ArgumentParser(description="Download faster-whisper models.")
    parser.add_argument(
        "--models",
        help="Comma-separated model list (medium, large-v3-turbo, large-v3)",
    )
    args = parser.parse_args()

    models = parse_models(args.models) if args.models else prompt_models()
    if not models:
        print("No models selected.")
        return 1

    root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)

    for name in models:
        try:
            download_model(name, models_dir)
        except Exception as exc:
            print(f"Failed to download {name}: {exc}")
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
