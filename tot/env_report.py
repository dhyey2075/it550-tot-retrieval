"""Environment fingerprint for Colab / local sessions (plan_v2/01 §4.2)."""

from __future__ import annotations

import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _try_import_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001 — report any failure as missing
        return f"not installed ({type(exc).__name__})"


def _gpu_summary() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "CUDA not available"
        n = torch.cuda.device_count()
        parts = []
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            parts.append(f"{i}:{name} ({total_gb:.1f} GB)")
        return "; ".join(parts)
    except Exception as exc:  # noqa: BLE001
        return f"torch unavailable ({type(exc).__name__}: {exc})"


def _ram_summary() -> str:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return (
            f"total={vm.total / (1024**3):.1f} GB, "
            f"available={vm.available / (1024**3):.1f} GB, "
            f"percent={vm.percent}"
        )
    except Exception:
        # Fallback without psutil
        try:
            page_size = None
            with open("/proc/meminfo", encoding="utf-8") as f:
                info = {}
                for line in f:
                    key, val = line.split(":")
                    info[key] = int(val.strip().split()[0])  # kB
            total = info.get("MemTotal", 0) / (1024**2)
            avail = info.get("MemAvailable", 0) / (1024**2)
            return f"total={total:.1f} GB, available={avail:.1f} GB (from /proc/meminfo)"
        except Exception as exc:  # noqa: BLE001
            return f"unavailable ({type(exc).__name__}: {exc})"


def _disk_summary(path: str | Path = ".") -> str:
    usage = shutil.disk_usage(path)
    return (
        f"total={usage.total / (1024**3):.1f} GB, "
        f"used={usage.used / (1024**3):.1f} GB, "
        f"free={usage.free / (1024**3):.1f} GB"
    )


def collect_env_report(libraries: list[str] | None = None) -> str:
    """Return a multi-line environment report string."""
    libs = libraries or [
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "sentence_transformers",
        "faiss",
        "ranx",
        "yaml",
    ]
    lines = [
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"platform: {platform.platform()}",
        f"executable: {sys.executable}",
        f"gpu: {_gpu_summary()}",
        f"ram: {_ram_summary()}",
        f"disk({Path('.').resolve()}): {_disk_summary('.')}",
        "libraries:",
    ]
    for name in libs:
        lines.append(f"  - {name}: {_try_import_version(name)}")
    return "\n".join(lines) + "\n"


def write_env_report(logs_dir: str | Path = "logs", libraries: list[str] | None = None) -> Path:
    """Print and write env report to logs/env_<date>.txt. Returns the path written."""
    report = collect_env_report(libraries=libraries)
    print(report)
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = logs / f"env_{stamp}.txt"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    return out
