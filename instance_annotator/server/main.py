"""FastAPI app: serves the web UI and the annotator API.

  pip install -e .
  python -m instance_annotator --ecomodel-root D:/Projects/Ecomodel
  python -m instance_annotator --load path/to/tile_prefix
  python -m instance_annotator --project D:/pointclouds/island_project
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from instance_annotator.ecomodel_link import (
    EcomodelLinkError,
    ensure_ecomodel_on_path,
    local_output_dir,
)

try:
    ensure_ecomodel_on_path(allow_prompt=False, persist=False)
except EcomodelLinkError:
    # Deferred until main(): first-run prompt / --ecomodel-root.
    pass

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from instance_annotator.server import project_routes  # noqa: E402
from instance_annotator.server.routes import router, session  # noqa: E402

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def create_app() -> FastAPI:
    app = FastAPI(title="TLS Instance Annotator", docs_url="/api/docs")
    app.include_router(router)
    app.include_router(project_routes.router)

    @app.middleware("http")
    async def no_cache(request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("Cache-Control", "no-cache")
        return resp

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=str(WEB_DIR)), name="web")
    return app


app = create_app()


def main(argv=None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="TLS instance annotator (web UI)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--load", default="", help="Tile prefix or cloud to open on start")
    ap.add_argument("--blank", action="store_true", help="Ignore *_instances.npy when loading")
    ap.add_argument("--max_display", type=int, default=None, help="Display LOD point cap")
    ap.add_argument("--no_browser", action="store_true")
    ap.add_argument("--project", default="", help="Island project folder to open on start")
    ap.add_argument(
        "--ecomodel-root",
        default="",
        help="Path to the Ecomodel checkout (saved for later runs)",
    )
    ap.add_argument("--pointsam_ckpt", default="",
                    help="Point-SAM weights for the pointsam method (e.g. pointsam_checkpoints/decoder_ft/best.safetensors)")
    args = ap.parse_args(argv)

    try:
        root = ensure_ecomodel_on_path(
            args.ecomodel_root or None,
            persist=True,
            allow_prompt=True,
        )
    except EcomodelLinkError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    print(f"Ecomodel: {root}")
    local_output_dir("annotator_workdir")
    local_output_dir("annotator_exports")

    if args.pointsam_ckpt:
        from .. import segment

        ckpt = Path(args.pointsam_ckpt).expanduser().resolve()
        if not ckpt.is_file():
            ap.error(f"--pointsam_ckpt not found: {ckpt}")
        segment._DEFAULT_POINTSAM = str(ckpt)
        print(f"Point-SAM weights: {ckpt}")

    if args.project:
        info = project_routes.psession.open(args.project)
        print(f"Project {info['name']}: {len(info['tiles'])} tiles, {info['num_points']:,} pts")
    if args.load:
        print(f"Loading {args.load} ...")
        info = session.load(args.load, "blank" if args.blank else "gt", args.max_display)
        print(f"  {info['num_points']:,} pts, display {info['num_display']:,}, trees={info['num_trees']}")

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Annotator at {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
