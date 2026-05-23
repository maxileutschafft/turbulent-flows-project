"""Download project datasets from Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "Kokoslocke/NACA_4_Digit_for_ML"
DEFAULT_OUTPUT_DIR = Path("data/raw/NACA_4_Digit_for_ML")


def _parse_patterns(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    patterns: list[str] = []
    for value in values:
        patterns.extend(pattern.strip() for pattern in value.split(",") if pattern.strip())
    return patterns or None


def _write_manifest(
    output_dir: Path,
    repo_id: str,
    revision: str | None,
    local_path: Path,
) -> None:
    files = sorted(
        str(path.relative_to(local_path))
        for path in local_path.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    )
    manifest = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "local_path": str(local_path),
        "files": files,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "download_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def download_dataset(
    repo_id: str = DEFAULT_REPO_ID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
    token: str | None = None,
) -> Path:
    """Download a Hugging Face dataset snapshot into ``output_dir``."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=output_dir,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
            token=token or os.environ.get("HF_TOKEN"),
        )
    )
    _write_manifest(output_dir, repo_id, revision, local_path)
    return local_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset revision, branch, tag, or commit SHA",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional file glob to include. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Optional file glob to exclude. Can be repeated or comma-separated.",
    )
    parser.add_argument("--token", default=None, help="Hugging Face token. Defaults to HF_TOKEN.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    local_path = download_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        revision=args.revision,
        allow_patterns=_parse_patterns(args.allow_pattern),
        ignore_patterns=_parse_patterns(args.ignore_pattern),
        token=args.token,
    )
    print(f"Downloaded {args.repo_id} -> {local_path}")
    print(f"Manifest -> {Path(args.output_dir).resolve() / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
