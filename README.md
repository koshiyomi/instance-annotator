# TLS Instance Annotator (Web)

Inspect and correct tree **instance** labels on terrestrial / MLS point clouds, then export corrected ground truth in the Ecomodel tile layout.

This repo is a **standalone app**. Segmenters, TreeLearn weights, Point-SAM checkpoints, and `promptable_instance` stay in your **Ecomodel** checkout. On first run you link that path once; it is remembered afterward.

## Install

```powershell
cd D:\Projects\instance-annotator
D:\Projects\PyTLidar\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Or: `pip install -r requirements.txt` then `pip install -e .`.

No Node.js: three.js is vendored under `instance_annotator/web/vendor/`.

## Link Ecomodel (first run)

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
& $py -m instance_annotator --ecomodel-root D:\Projects\Ecomodel --no_browser
```

That writes `%LOCALAPPDATA%\instance-annotator\config.json` (or `~/.config/instance-annotator/config.json`). Later starts reuse it:

```powershell
& $py -m instance_annotator
```

| Override | Meaning |
|----------|---------|
| `--ecomodel-root PATH` | Set / update the link (persisted) |
| `INSTANCE_ANNOTATOR_ECOMODEL` | Env override (not written unless you also pass `--ecomodel-root`) |
| `INSTANCE_ANNOTATOR_CONFIG` | Custom config JSON path (useful for tests) |

If the linked folder is missing or no longer looks like Ecomodel (`ecomodel_lite.py`, `promptable_instance.py`, `SyntheticPipeline/scripts/benchmark_instance_segmentation.py`), startup exits with:

`Ecomodel link broken: <path>. Re-run with --ecomodel-root <Ecomodel> or set INSTANCE_ANNOTATOR_ECOMODEL.`

Segment / ForestFormer / TreeLearn / Point-SAM weights all live under that linked tree (`TreeLearn/`, `thirdparty/checkpoints/`, `SyntheticPipeline/pointsam_checkpoints/`). Annotator workdirs stay local: `instance-annotator/output/`.

## Start

```powershell
cd D:\Projects\instance-annotator
D:\Projects\PyTLidar\.venv\Scripts\python.exe -m instance_annotator
# or after install:
instance-annotator
```

Browser: `http://127.0.0.1:8765`. Useful flags: `--load <tile_prefix>`, `--blank`, `--project <island_project>`, `--max_display`, `--port`, `--no_browser`.

## Workflow

1. **Open** (`O`): tile prefix or path, or browse. Use existing labels or start blank.
2. **Review trees** in the Trees tab; `[` / `]` step, `K` marks reviewed.
3. **Select**: Navigate (`V`), Lasso (`L`), Box (`B`), Brush (`R`); Grow (`G`); height slab / hide non-tree (`H`) / isolate (`I`).
4. **Edit**: New tree (`N`), Assign (`A`), Merge (`M`), Non-tree (`Del`); Undo/Redo (`Ctrl+Z` / `Ctrl+Y`).
5. **Save** (`Ctrl+S`): `{name}_scan.laz`, `{name}_instances.npy`, `{name}_meta.json`, `{name}_preview.ply`.

Press `?` for the full shortcut list.

## Wood / leaf and Segment tabs

Wood/leaf classifiers run in-process (stem-grow / eigen local; RGI / GBSeparation need those packages on the Ecomodel / env path).

The Segment tab runs benchmark methods from the linked Ecomodel stack (`treelearn`, `pointsam`, `treex`, `tls2trees`, `scanline`, `forestformer3d`, …). Point-SAM weight dropdown lists stock + fine-tuned checkpoints under the linked repo.

## Whole-island projects

```powershell
$py = "D:\Projects\PyTLidar\.venv\Scripts\python.exe"
cd D:\Projects\instance-annotator
& $py -m instance_annotator.project build D:\pointclouds\processed2025scans --out D:\pointclouds\island_project
& $py -m instance_annotator --project D:\pointclouds\island_project
```

Island tab: streaming octree, edit regions, segment whole island (buffered tiles), border merge candidates, export. CLI: `python -m instance_annotator.project …` (`build`, `import`, `segment`, `stitch`, `export`, `info`).

## Headless demos

```powershell
& $py -m instance_annotator.cli_demo --ecomodel-root D:\Projects\Ecomodel
& $py -m instance_annotator.cli_demo --skip_treelearn --only manual
```

Artifacts: `output/annotator_demos/`.

## Tests

```powershell
$env:INSTANCE_ANNOTATOR_ECOMODEL = "D:\Projects\Ecomodel"
& $py -m pytest instance_annotator/test_core.py instance_annotator/test_merge_fragments.py -q
```

## Label convention

| Label | Meaning |
|------:|---------|
| `-1` | Non-tree / ground / unlabeled |
| `>=0` | Tree instance ID |

## Tips

- Prefer opening large clouds by disk path, not upload.
- Relative paths resolve under this repo, then the linked `SyntheticPipeline/`, then cwd.
- Only one browser tab should edit at a time.
