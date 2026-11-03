"""Resolve, persist, and inject a linked Ecomodel checkout onto ``sys.path``.

Resolution order for the Ecomodel root:
  1. explicit path (CLI ``--ecomodel-root``)
  2. env ``INSTANCE_ANNOTATOR_ECOMODEL``
  3. config file (see ``config_path``)
  4. interactive prompt on a TTY (first run only)

Config path override: ``INSTANCE_ANNOTATOR_CONFIG``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

ENV_ECOMODEL = "INSTANCE_ANNOTATOR_ECOMODEL"
ENV_CONFIG = "INSTANCE_ANNOTATOR_CONFIG"

_MARKERS = (
    "ecomodel_lite.py",
    "promptable_instance.py",
    "SyntheticPipeline/scripts/benchmark_instance_segmentation.py",
)

_linked: Optional[Path] = None


class EcomodelLinkError(RuntimeError):
    """Missing or invalid Ecomodel link; exit code 2 at the CLI."""


def annotator_repo_root() -> Path:
    """Root of the instance-annotator checkout (parent of the package)."""
    return Path(__file__).resolve().parents[1]


def config_path() -> Path:
    override = os.environ.get(ENV_CONFIG, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "instance-annotator" / "config.json"
    return Path.home() / ".config" / "instance-annotator" / "config.json"


def broken_link_message(path: Optional[Path] = None) -> str:
    where = f" ({path})" if path else ""
    return (
        f"Ecomodel link broken{where}. "
        f"Re-run with --ecomodel-root <Ecomodel> or set {ENV_ECOMODEL}."
    )


def is_ecomodel_root(path: Path) -> bool:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return False
    return all((root / m).is_file() for m in _MARKERS)


def validate_ecomodel_root(path: Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not is_ecomodel_root(root):
        raise EcomodelLinkError(broken_link_message(root))
    return root


def load_config() -> dict:
    p = config_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_ecomodel_root(root: Path) -> Path:
    root = validate_ecomodel_root(root)
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_config()
    data["ecomodel_root"] = str(root)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)
    return root


def synthetic_pipeline(root: Optional[Path] = None) -> Path:
    return (root or ecomodel_root()) / "SyntheticPipeline"


def pytlidar_root(root: Optional[Path] = None) -> Optional[Path]:
    projects = (root or ecomodel_root()).parent
    cand = projects / "PyTLidar"
    return cand if cand.is_dir() else None


def _prompt_ecomodel_root() -> Path:
    no_link = (
        f"No Ecomodel link configured. "
        f"Pass --ecomodel-root <Ecomodel> or set {ENV_ECOMODEL}."
    )
    if not sys.stdin.isatty():
        raise EcomodelLinkError(no_link)
    print("First run: link this annotator to your Ecomodel checkout.")
    print("Example: D:\\Projects\\Ecomodel")
    while True:
        try:
            raw = input("Ecomodel root path: ").strip().strip('"')
        except EOFError as exc:
            raise EcomodelLinkError(no_link) from exc
        if not raw:
            print("Path required.")
            continue
        try:
            return save_ecomodel_root(Path(raw))
        except EcomodelLinkError as exc:
            print(exc)


def resolve_ecomodel_root(
    explicit: Optional[str | Path] = None,
    *,
    persist: bool = True,
    allow_prompt: bool = True,
) -> Path:
    """Return a validated Ecomodel root, optionally writing it to the config file."""
    if explicit:
        root = validate_ecomodel_root(Path(explicit))
        if persist:
            save_ecomodel_root(root)
        return root

    env = os.environ.get(ENV_ECOMODEL, "").strip()
    if env:
        return validate_ecomodel_root(Path(env))

    cfg = load_config().get("ecomodel_root")
    if cfg:
        return validate_ecomodel_root(Path(str(cfg)))

    if allow_prompt:
        return _prompt_ecomodel_root()
    raise EcomodelLinkError(
        f"No Ecomodel link configured. "
        f"Pass --ecomodel-root <Ecomodel> or set {ENV_ECOMODEL}."
    )


def ensure_ecomodel_on_path(
    explicit: Optional[str | Path] = None,
    *,
    persist: bool = True,
    allow_prompt: bool = True,
) -> Path:
    """Validate the link, put Ecomodel on ``sys.path``, and cache the root."""
    global _linked
    root = resolve_ecomodel_root(explicit, persist=persist, allow_prompt=allow_prompt)
    sp = root / "SyntheticPipeline"
    # Annotator package first (this repo), then Ecomodel + SyntheticPipeline for segmenters.
    annot = annotator_repo_root()
    for p in (str(annot), str(root), str(sp)):
        if p not in sys.path:
            sys.path.insert(0, p)
    projects = root.parent
    pyt = projects / "PyTLidar"
    for p in (str(projects), str(pyt)):
        if Path(p).is_dir() and p not in sys.path:
            sys.path.append(p)
    _linked = root
    return root


def ecomodel_root() -> Path:
    if _linked is not None:
        return _linked
    return ensure_ecomodel_on_path(allow_prompt=False, persist=False)


def local_output_dir(*parts: str) -> Path:
    """Writable output under the annotator repo (not under Ecomodel)."""
    out = annotator_repo_root().joinpath("output", *parts)
    out.mkdir(parents=True, exist_ok=True)
    return out
