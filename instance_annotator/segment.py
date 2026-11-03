"""Run benchmark instance methods on an in-memory XYZI cloud."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .ecomodel_link import ecomodel_root, local_output_dir, synthetic_pipeline

METHODS = ("scanline", "treelearn", "treex", "tls2trees", "pointsam",
           "forestformer3d", "segmentanytreev2")


def _paths() -> Tuple[Path, Path, Path]:
    """(ecomodel_root, synthetic_pipeline, pointsam_ft_dir)."""
    root = ecomodel_root()
    sp = synthetic_pipeline(root)
    return root, sp, sp / "pointsam_checkpoints"


def _default_treelearn() -> str:
    root, _, _ = _paths()
    return str(root / "TreeLearn" / "configs" / "pipeline" / "ecomodel.yaml")


def _stock_pointsam() -> str:
    root, _, _ = _paths()
    return str(root / "thirdparty" / "checkpoints" / "point_sam" / "model.safetensors")


def _default_pointsam() -> str:
    return os.environ.get("POINTSAM_CKPT") or _stock_pointsam()


# Mutable default used by ``--pointsam_ckpt`` on the server.
_DEFAULT_POINTSAM = ""


def pointsam_weights() -> list:
    """Selectable Point-SAM checkpoints: [{"path", "label"}], default first."""
    _, _, ft_dir = _paths()
    stock = _stock_pointsam()
    default = _DEFAULT_POINTSAM or _default_pointsam()
    found = []
    if os.path.isfile(stock):
        found.append((stock, "Stock Point-SAM (large)"))
    if ft_dir.is_dir():
        files = sorted(
            ft_dir.glob("*/*.safetensors"),
            key=lambda f: (f.parent.name.startswith("_"), f.parent.name, f.stem != "best", f.stem),
        )
        found += [(str(f), f"Fine-tuned {f.parent.name} / {f.stem}") for f in files]
    if os.path.isfile(default) and all(
        os.path.normcase(p) != os.path.normcase(default) for p, _ in found
    ):
        found.append((default, f"Custom: {Path(default).name}"))
    default_key = os.path.normcase(os.path.abspath(default))
    found.sort(key=lambda x: os.path.normcase(os.path.abspath(x[0])) != default_key)
    return [{"path": p, "label": lab} for p, lab in found]


def unavailable_methods() -> Dict[str, str]:
    """Methods listed in the menu but not runnable. Values are the reason shown in the UI."""
    from ecomodel_forestformer import segmentanytree_v2_reason

    reason = segmentanytree_v2_reason()
    return {"segmentanytreev2": reason} if reason else {}


def resolve_pointsam(path: Optional[str]) -> Optional[str]:
    """Validate a requested checkpoint against pointsam_weights(); None -> default."""
    if not path:
        return None
    want = os.path.normcase(os.path.abspath(path))
    for w in pointsam_weights():
        if os.path.normcase(os.path.abspath(w["path"])) == want:
            return w["path"]
    raise ValueError(f"Unknown Point-SAM checkpoint: {path}")


def default_ckpt_kwargs(
    *,
    treelearn_config: Optional[str] = None,
    pointsam_ckpt: Optional[str] = None,
    treex_stock_tls: bool = True,
    leaf_removal: bool = False,
    results_folder: Optional[str] = None,
) -> Dict:
    return {
        "treelearn_config_path": treelearn_config or _default_treelearn(),
        "treelearn_use_gpu": True,
        "pointsam_ckpt": pointsam_ckpt or _DEFAULT_POINTSAM or _default_pointsam(),
        "pointsam_config": "large",
        "pointsam_use_gpu": True,
        "run_leaf_removal": bool(leaf_removal),
        "results_folder": results_folder or str(local_output_dir("annotator_workdir")),
        "treex_adapt_synthetic": not bool(treex_stock_tls),
    }


SPARSE_COVERAGE = 0.5
DEFAULT_PROPAGATE_RADIUS = 0.2


def propagate_labels(
    xyz: np.ndarray, seed_idx: np.ndarray, seed_labels: np.ndarray, *, radius: float = DEFAULT_PROPAGATE_RADIUS
) -> np.ndarray:
    """Give every point the label of its nearest seed point within `radius` (else -1).

    Seeds with label -1 also count, so ground/non-tree voxels keep nearby points non-tree.
    """
    from scipy.spatial import cKDTree

    full = np.full(len(xyz), -1, dtype=np.int32)
    if len(seed_idx) == 0:
        return full
    tree = cKDTree(xyz[seed_idx])
    dist, nn = tree.query(xyz, k=1, distance_upper_bound=radius, workers=-1)
    hit = np.isfinite(dist)
    full[hit] = seed_labels[nn[hit]]
    full[seed_idx] = seed_labels
    return full


def run_method(
    xyz: np.ndarray,
    intensity01: np.ndarray,
    method: str,
    *,
    leaf_removal: bool = False,
    treex_stock_tls: bool = True,
    treelearn_config: Optional[str] = None,
    pointsam_ckpt: Optional[str] = None,
    results_folder: Optional[str] = None,
    propagate_radius: float = DEFAULT_PROPAGATE_RADIUS,
    merge_fragments: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Run one instance method; return full-length labels (N,) and info dict.

    Points not covered by the segmenter stay -1. Labels are normalized to
    trees >=0, non-tree -1. ``merge_fragments`` auto-merges split trees
    (see merge_fragments.py); review candidates land in info["merge_candidates"].
    """
    method = method.strip().lower()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose from {METHODS}")

    from scripts.benchmark_instance_segmentation import run_segmenter_on_tile

    from .labels import remap_treelearn_labels

    xyz = np.asarray(xyz, dtype=np.float64)
    inten = np.asarray(intensity01, dtype=np.float64).reshape(-1)
    if len(inten) != len(xyz):
        raise ValueError("intensity length mismatch")
    points = np.hstack([xyz, inten.reshape(-1, 1)])

    kwargs = default_ckpt_kwargs(
        treelearn_config=treelearn_config,
        pointsam_ckpt=pointsam_ckpt,
        treex_stock_tls=treex_stock_tls,
        leaf_removal=leaf_removal,
        results_folder=results_folder,
    )
    os.makedirs(kwargs["results_folder"], exist_ok=True)

    work, labels, orig_idx = run_segmenter_on_tile(
        points, segmenter_type=method, **kwargs
    )

    full = np.full(len(xyz), -1, dtype=np.int32)
    info: Dict = {
        "method": method,
        "ok": False,
        "num_pred_points": 0,
        "num_trees": 0,
        "message": "",
    }
    if work is None or labels is None or orig_idx is None:
        info["message"] = "segmentation returned no points"
        return full, info

    lab = np.asarray(labels, dtype=np.int32).reshape(-1)
    oi = np.asarray(orig_idx, dtype=np.int64).reshape(-1)
    if len(lab) != len(oi):
        info["message"] = f"label/index length mismatch {len(lab)} vs {len(oi)}"
        return full, info

    if method == "treelearn":
        lab = remap_treelearn_labels(lab)
    else:
        lab = lab.copy()
        lab[lab < 0] = -1

    # Guard invalid indices
    valid = (oi >= 0) & (oi < len(full))
    oi, lab = oi[valid], lab[valid]
    full[oi] = lab
    info["ok"] = True
    info["num_pred_points"] = int(len(oi))
    if len(np.unique(oi)) < SPARSE_COVERAGE * len(full):
        # Voxelized output (e.g. TreeLearn at 0.1 m) labels one point per voxel;
        # spread each label to the full-resolution points around it.
        full = propagate_labels(xyz, oi, lab, radius=propagate_radius)
        info["propagated"] = True
        info["num_pred_points"] = int((full >= 0).sum())
    if merge_fragments:
        from .merge_fragments import merge_fragments as _merge

        full, rep = _merge(xyz, full)
        info["fragments_merged"] = rep["merged"]
        info["merge_candidates"] = rep["candidates"]
    info["num_trees"] = int(len(np.unique(full[full >= 0])))
    info["message"] = f"trees={info['num_trees']} labeled_pts={info['num_pred_points']}"
    if merge_fragments:
        info["message"] += f" merged_fragments={info['fragments_merged']}"
    return full, info
