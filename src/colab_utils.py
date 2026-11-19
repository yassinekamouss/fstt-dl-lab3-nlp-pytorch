"""Google Colab, Google Drive, configuration, and reproducibility utilities."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def is_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the project YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration structure in {path}")
    return data


def setup_environment(
    use_drive: bool = True,
    drive_dir: str = "/content/drive/MyDrive/fstt-dl-lab3-nlp-pytorch",
    repo_root: str | Path | None = None,
) -> dict[str, Path]:
    """Configure workspace paths for Google Colab (with optional Drive) or local runs.

    When running in Colab with ``use_drive=True``, Google Drive is mounted and
    ``data/`` and ``outputs/`` are persisted under ``drive_dir`` so artifacts survive
    Colab runtime resets across the three notebooks.
    """
    resolved_repo = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )

    if is_colab() and use_drive:
        drive_mount_point = Path("/content/drive")
        if not (drive_mount_point / "MyDrive").exists():
            from google.colab import drive  # type: ignore[import-not-found]

            drive.mount(str(drive_mount_point))
        artifact_root = Path(drive_dir)
    else:
        artifact_root = resolved_repo

    paths: dict[str, Path] = {
        "repo_root": resolved_repo,
        "artifact_root": artifact_root,
        "config": resolved_repo / "configs" / "config.yaml",
        "data_raw": artifact_root / "data" / "raw",
        "html_cache": artifact_root / "data" / "raw" / "html_cache",
        "data_processed": artifact_root / "data" / "processed",
        "outputs": artifact_root / "outputs",
        "figures": artifact_root / "outputs" / "figures",
        "reports": artifact_root / "outputs" / "reports",
        "checkpoints": artifact_root / "outputs" / "checkpoints",
    }

    for key in (
        "artifact_root",
        "data_raw",
        "html_cache",
        "data_processed",
        "outputs",
        "figures",
        "reports",
        "checkpoints",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch (when installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> Any:
    """Return the active PyTorch compute device (CUDA GPU if available, else CPU)."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_system_info() -> None:
    """Print runtime diagnostics (Colab status, Python, PyTorch, CUDA GPU, VRAM)."""
    print(f"Environment   : {'Google Colab' if is_colab() else 'Local'}")
    print(f"Python        : {sys.version.split()[0]}")
    try:
        import torch

        print(f"PyTorch       : {torch.__version__}")
        if torch.cuda.is_available():
            gpu_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(gpu_idx)
            vram_gb = props.total_memory / (1024**3)
            print(f"Device        : CUDA ({props.name}, {vram_gb:.2f} GB VRAM)")
        else:
            print("Device        : CPU (GPU not enabled)")
    except ImportError:
        print("PyTorch       : not installed")
