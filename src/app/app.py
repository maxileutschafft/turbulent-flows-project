"""FastAPI server for the NACA airfoil surrogate web app (Design view only).

Serves the static single-page design UI (``app.html``) and the prediction
endpoints that back its controls.

The inference device is chosen ONCE at startup (``--device``, or auto-select
cuda > mps > cpu) and used for every request — there is no per-request or
in-UI device switch.

Run with::

    uv run python src/app/app.py                  # binds 127.0.0.1:8001, auto device
    uv run python src/app/app.py --port 9000
    uv run python src/app/app.py --device cpu      # force a specific device

"""

from __future__ import annotations

import argparse
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from utils.naca_geometry import naca4_coords

APP_DIR = Path(__file__).resolve().parent
INDEX_HTML = APP_DIR / "app.html"
STATIC_DIR = APP_DIR / "static"


def _asset_version() -> str:
    """Cache-busting token = newest mtime across the static dir (+ app.html).

    Front-end assets are loaded as ES modules with relative imports, which the
    browser caches by URL. Serving them under a version-stamped path prefix
    (``/static/v/<token>/…``) means any edit changes every module URL at once,
    so a normal reload can never reuse a stale module. The token only changes
    when a file changes, so unchanged reloads still hit the browser cache."""
    paths = list(STATIC_DIR.rglob("*")) + [INDEX_HTML]
    newest = max((p.stat().st_mtime_ns for p in paths if p.is_file()), default=0)
    return str(newest)


ASSET_VERSION = _asset_version()

logger = logging.getLogger(__name__)


class BoundedCache:
    """LRU cache with a hard size cap, drop-in for the dict response cache.

    Supports ``key in cache``, ``cache[key]`` reads, and ``cache[key] = value``
    writes. Reads and writes mark the key most-recently-used; once ``maxsize``
    is exceeded the least-recently-used entry is evicted. This keeps a
    long-running server from accumulating SVG payloads without bound.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self._data: OrderedDict[tuple, dict] = OrderedDict()
        self._maxsize = maxsize

    def __contains__(self, key: tuple) -> bool:
        return key in self._data

    def __getitem__(self, key: tuple) -> dict:
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: tuple, value: dict) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)


# Response-level cache: (naca, round(re), round(aoa,2), view, model) -> full response dict
_RESPONSE_CACHE = BoundedCache()

# Inference device, resolved once in main() before uvicorn starts serving.
# Fixed for the lifetime of the process — there is no per-request override.
DEVICE: str = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all surrogate checkpoints once on server start so no request pays
    a cold-load cost. Models stay CPU-resident for the process lifetime and
    are only moved to the inference device for the duration of a forward pass
    (see ``inference.predict``)."""
    import inference  # noqa: PLC0415 — import inside lifespan intentional

    for name in sorted(inference.VALID_MODELS):
        inference.load_model(name)
    logger.info(
        "All models ready in CPU RAM; will use device=%s for forward passes", DEVICE
    )
    yield


app = FastAPI(title="Airfoil Surrogate", version="0.1.0", lifespan=lifespan)


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that forces revalidation. Browsers cache ES modules
    aggressively; without ``Cache-Control`` a soft refresh reuses a stale module
    (heuristic freshness) and never sees a code change. ``no-cache`` keeps the
    ETag fast-path (304 when unchanged) but guarantees the browser revalidates,
    so edits to the front-end always take effect on reload."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Version-stamped asset mount: app.html references /static/v/<ASSET_VERSION>/…,
# so every module URL changes whenever a front-end file changes — a normal
# reload then fetches the whole graph fresh (no stale ES modules). The bare
# /static mount is kept as a no-cache fallback for any unversioned reference.
app.mount(f"/static/v/{ASSET_VERSION}", _NoCacheStaticFiles(directory=STATIC_DIR), name="static_versioned")
app.mount("/static", _NoCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> Response:
    """Serve the single-page web app with version-stamped asset URLs.

    Rewrites ``/static/…`` references in app.html to the versioned prefix so the
    browser cannot reuse a cached module after a front-end edit. Served
    ``no-cache`` so the (tiny) HTML itself always revalidates and picks up a new
    version token immediately."""
    html = INDEX_HTML.read_text(encoding="utf-8").replace(
        "/static/", f"/static/v/{ASSET_VERSION}/"
    )
    return Response(
        content=html, media_type="text/html", headers={"Cache-Control": "no-cache"}
    )


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    """Liveness probe used to confirm the server is up."""
    return {"status": "ok"}


@app.get("/api/geometry")
def geometry(naca: str = "2412", n: int = 200) -> dict:
    """Return NACA 4-digit airfoil surface coordinates (chord normalised to [0, 1]).

    Upper surface LE->TE in ``(xu, yu)``; lower surface LE->TE in ``(xl, yl)``.
    The frontend rotates these by the angle of attack and renders the silhouette.
    """
    try:
        xu, yu, xl, yl = naca4_coords(naca, n=n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "naca": naca,
        "xu": xu.tolist(),
        "yu": yu.tolist(),
        "xl": xl.tolist(),
        "yl": yl.tolist(),
    }


# Cache STEP file bytes per NACA code — the export depends only on the code
# (fixed sample count / span / chord / units), so repeats are free.
_STEP_CACHE = BoundedCache(maxsize=32)


@app.get("/api/export_step")
def api_export_step(naca: str = "2412") -> Response:
    """Export the NACA airfoil geometry as a solid STEP (.step) CAD file.

    The exported solid is a short spanwise extrusion (10% chord) of the 2D
    airfoil profile in its natural, angle-of-attack-free frame — angle of
    attack is a flow condition, not part of the geometry. Units are metres.
    """
    if len(naca) != 4 or not naca.isdigit():
        raise HTTPException(
            status_code=400,
            detail=f"naca must be a 4-digit numeric code, got {naca!r}",
        )

    if naca in _STEP_CACHE:
        data = _STEP_CACHE[naca]
    else:
        from utils.step_export import naca_airfoil_step_bytes  # noqa: PLC0415

        try:
            data = naca_airfoil_step_bytes(naca)
        except Exception as exc:
            logger.exception("STEP export failed for naca=%s", naca)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _STEP_CACHE[naca] = data

    return Response(
        content=data,
        media_type="application/step",
        headers={"Content-Disposition": f'attachment; filename="naca_{naca}.step"'},
    )


@app.get("/api/predict")
def api_predict(
    naca: str = "2412",
    reynolds: float = 3_000_000.0,
    aoa: float = 0.0,
    view: str = "streamlines",
    model: str = "gno",
) -> dict:
    """Run surrogate inference for the given NACA airfoil and flow conditions.

    Query parameters
    ----------------
    naca     : 4-digit NACA code (e.g. ``2412``)
    reynolds : chord-based Reynolds number
    aoa      : angle of attack in degrees
    view     : one of ``streamlines, vel_mag, u, v, p, k, omega, nut``
    model    : one of ``gno``, ``transolver``, ``domino``

    The inference device is fixed for the process (see ``DEVICE`` / ``--device``)
    and is not a request parameter.

    Returns
    -------
    JSON payload::

        {
            "naca": "2412",
            "reynolds": 3000000.0,
            "aoa": 8.0,
            "view": "vel_mag",
            "model": "gno",
            "device": "cuda",
            "field_svg": "<svg ...>...</svg>",
            "colorbar": {
                "label": "|v| [m/s]",
                "ticks": ["42.3", "31.7", "21.2", "10.6"],
                "vmin": 10.6,
                "vmax": 42.3
            },
            "compute_ms": 1234
        }

    ``colorbar`` is ``null`` for the ``streamlines`` view.
    ``field_svg`` is a transparent SVG string with the flow field rendered in
    screen-coordinate frame (viewBox 0 0 480 260).
    """
    # Validate NACA code (match the /api/geometry convention)
    if len(naca) != 4 or not naca.isdigit():
        raise HTTPException(
            status_code=400,
            detail=f"naca must be a 4-digit numeric code, got {naca!r}",
        )

    import inference  # noqa: PLC0415

    if view not in inference.VALID_VIEWS:
        raise HTTPException(
            status_code=400,
            detail=f"view must be one of {sorted(inference.VALID_VIEWS)}, got {view!r}",
        )

    if model not in inference.VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model must be one of {sorted(inference.VALID_MODELS)}, got {model!r}",
        )

    resolved_device = DEVICE

    cache_key = (naca, round(reynolds), round(aoa, 2), view, model)
    if cache_key in _RESPONSE_CACHE:
        return _RESPONSE_CACHE[cache_key]

    # Lazy-load the model (idempotent)
    inference.load_model(model, resolved_device)

    t0 = time.perf_counter()
    try:
        result = inference.predict(naca, reynolds, aoa, model=model, device=resolved_device)

        if view == "streamlines":
            field_svg = inference.streamline_field_svg(
                result["pos"], result["u"], result["v"], naca, aoa
            )
            colorbar = None
        else:
            label, extractor = inference.FIELD_REGISTRY[view]
            values = extractor(result)
            field_svg, vmin, vmax = inference.scalar_field_svg(
                result["pos"], values, naca, aoa
            )
            colorbar = {
                "label": label,
                "ticks": inference.colorbar_ticks(vmin, vmax),
                "vmin": vmin,
                "vmax": vmax,
            }

    except Exception as exc:
        logger.exception(
            "Inference failed for naca=%s re=%s aoa=%s view=%s model=%s device=%s",
            naca, reynolds, aoa, view, model, resolved_device,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    compute_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "predict naca=%s re=%.0f aoa=%.1f view=%s model=%s device=%s ms=%d",
        naca,
        reynolds,
        aoa,
        view,
        model,
        resolved_device,
        compute_ms,
    )

    response = {
        "naca": naca,
        "reynolds": reynolds,
        "aoa": aoa,
        "view": view,
        "model": model,
        "device": resolved_device,
        "field_svg": field_svg,
        "colorbar": colorbar,
        "coeffs": result.get("coeffs"),
        "compute_ms": compute_ms,
    }
    _RESPONSE_CACHE[cache_key] = response
    return response


def main() -> None:
    global DEVICE

    parser = argparse.ArgumentParser(description="Launch the airfoil surrogate web app.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="bind port (default: 8001)")
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Inference device, fixed for the process (default: auto-select cuda > mps > cpu)",
    )
    args = parser.parse_args()

    import inference  # noqa: PLC0415

    if args.device:
        available = inference.available_devices()
        if args.device not in available:
            parser.error(f"--device {args.device!r} is not available on this machine (available: {available})")
        DEVICE = args.device
    else:
        DEVICE = inference.default_device()
    logger.info("Inference device: %s", DEVICE)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
