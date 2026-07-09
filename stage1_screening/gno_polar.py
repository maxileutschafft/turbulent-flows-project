"""Obtain GNO Cl/Cd for a case. The server/webapp prediction is the source.

Interchangeable sources, all returning ``GnoPolar | None`` for a case:

1. ``from_local``         - run the GNO locally (build_scenario -> infer ->
                            integrate Cl/Cd). Since the GitHub checkpoint matches
                            the deployed server, this equals the webapp output.
2. ``from_csv``           - a precomputed table you export once
                            (columns: naca_code, reynolds, aoa, cl, cd).
3. ``from_server``        - call your deployed inference webapp per case
                            (UChile cluster).
4. ``from_prediction_dir`` - EXPERIMENTAL. Integrate surface pressure from
                            prediction .npz files (pressure-only Cl/Cd).

Pick whichever matches how you get predictions out of the model.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class GnoPolar:
    naca_code: str
    reynolds: float
    aoa: float
    cl: float
    cd: float
    source: str


def _key(code: str, re: float, aoa: float) -> tuple[str, int, float]:
    return (str(code), int(round(float(re))), round(float(aoa), 3))


# --------------------------------------------------------------------------- #
# 1. Local inference (build_scenario -> infer -> integrate)
# --------------------------------------------------------------------------- #


def from_local(
    ckpt: str | Path = "checkpoints/gno_w32_d6_k16_lr0.001_NACA_4_Digit_for_ML/best.pt",
    *,
    device: str | None = None,
) -> Callable[[str, float, float], GnoPolar | None]:
    """Run the GNO locally and integrate Cl/Cd from the predicted pressure field.

    Needs the uv env (torch + torch-geometric). Cl/Cd are pressure-based
    (see stage_common.aero_post); friction drag is not included.
    """
    import sys as _sys
    import pathlib as _pathlib

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if _sys.platform == "win32":
        _pathlib.PosixPath = _pathlib.WindowsPath  # the checkpoint was pickled on Linux
    from surrogate.gno.infer import infer  # lazy: torch only when this source is used
    from surrogate.gno.schema import GNO_TARGET_COLS
    from utils.scenario import build_scenario
    from stage_common.aero_post import cl_cd_from_fields, surface_indices

    p_index = GNO_TARGET_COLS.index("p")
    ckpt = Path(ckpt)

    def lookup(code: str, re: float, aoa: float) -> GnoPolar | None:
        try:
            scenario = build_scenario(code, float(re), float(aoa), device=device)
            result = infer(scenario, ckpt, device=device)
            pred = result["predictions"].cpu().numpy()
            idx = surface_indices(scenario)
            cl, cd = cl_cd_from_fields(
                scenario["x"], scenario["y"], pred[:, p_index], idx, float(re), float(aoa)
            )
            return GnoPolar(code, re, aoa, cl, cd, "local")
        except Exception as exc:  # noqa: BLE001
            print(f"  [local] {code} Re={re:.0f} a={aoa:.1f}: {exc}")
            return None

    return lookup


# --------------------------------------------------------------------------- #
# 2. CSV table
# --------------------------------------------------------------------------- #


def from_csv(path: str | Path) -> Callable[[str, float, float], GnoPolar | None]:
    table: dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            table[_key(row["naca_code"], float(row["reynolds"]), float(row["aoa"]))] = (
                float(row["cl"]), float(row["cd"])
            )

    def lookup(code: str, re: float, aoa: float) -> GnoPolar | None:
        hit = table.get(_key(code, re, aoa))
        if hit is None:
            return None
        return GnoPolar(code, re, aoa, hit[0], hit[1], "csv")

    return lookup


# --------------------------------------------------------------------------- #
# 3. Deployed inference webapp (server)
# --------------------------------------------------------------------------- #


def from_server(
    base_url: str,
    *,
    endpoint: str = "/predict",
    model: str = "GNO",
    build_request: Callable[[str, float, float], dict] | None = None,
    parse_response: Callable[[dict], tuple[float, float]] | None = None,
    timeout: float = 120.0,
) -> Callable[[str, float, float], GnoPolar | None]:
    """Return a lookup that POSTs each case to your inference webapp.

    Adjust ``endpoint`` / ``build_request`` / ``parse_response`` to your API.
    The defaults assume a JSON POST returning cl/cd somewhere in the body.
    """

    def _default_build(code: str, re: float, aoa: float) -> dict:
        return {"naca": code, "reynolds": re, "angle_of_attack": aoa, "model": model}

    def _default_parse(body: dict) -> tuple[float, float]:
        def pick(*names):
            for n in names:
                for k, v in body.items():
                    if k.lower() == n:
                        return float(v)
            raise KeyError(f"none of {names} in response {list(body)}")
        return pick("cl", "c_l", "lift_coefficient"), pick("cd", "c_d", "drag_coefficient")

    build = build_request or _default_build
    parse = parse_response or _default_parse
    url = base_url.rstrip("/") + endpoint

    def lookup(code: str, re: float, aoa: float) -> GnoPolar | None:
        data = json.dumps(build(code, re, aoa)).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            print(f"  [server] {code} Re={re:.0f} a={aoa:.1f}: {exc}")
            return None
        cl, cd = parse(body)
        return GnoPolar(code, re, aoa, cl, cd, "server")

    return lookup


# --------------------------------------------------------------------------- #
# 4. Experimental: pressure integration from prediction .npz files
# --------------------------------------------------------------------------- #


def _surface_indices(d: dict) -> np.ndarray:
    if "is_wall" in d:
        return np.where(np.asarray(d["is_wall"]).astype(bool))[0]
    if "sdf" in d:
        sdf = np.abs(np.asarray(d["sdf"]))
        thr = np.quantile(sdf, 0.02)
        return np.where(sdf <= thr)[0]
    raise KeyError("prediction .npz has neither 'is_wall' nor 'sdf' to locate the surface")


def integrate_pressure_polar(npz_path: str | Path, *, rho: float = 1.0) -> GnoPolar:
    """EXPERIMENTAL pressure-only Cl/Cd from a prediction .npz.

    Friction drag is not included, so Cd is biased low. Prefer your validated
    server-side postprocessing.
    """
    d = dict(np.load(npz_path, allow_pickle=True))
    code = str(d["naca_code"]) if "naca_code" in d else "????"
    re = float(d["reynolds"]) if "reynolds" in d else np.nan
    aoa = float(d["angle_of_attack"]) if "angle_of_attack" in d else np.nan

    idx = _surface_indices(d)
    x = np.asarray(d["x"])[idx].astype(float)
    y = np.asarray(d["y"])[idx].astype(float)
    p = np.asarray(d["p"])[idx].astype(float)

    cx, cy = x.mean(), y.mean()
    order = np.argsort(np.arctan2(y - cy, x - cx))
    x, y, p = x[order], y[order], p[order]

    xn, yn = np.roll(x, -1), np.roll(y, -1)
    dx, dy = xn - x, yn - y
    nx, ny = dy, -dx
    pmid = 0.5 * (p + np.roll(p, -1))
    if np.sum(nx * (0.5 * (x + xn) - cx) + ny * (0.5 * (y + yn) - cy)) < 0:
        nx, ny = -nx, -ny
    fx = -np.sum(pmid * nx)
    fy = -np.sum(pmid * ny)

    nu = 1e-5
    u_mag = re * nu / 1.0
    q = 0.5 * rho * u_mag**2 * 1.0
    a = np.radians(aoa)
    lift = -fx * np.sin(a) + fy * np.cos(a)
    drag = fx * np.cos(a) + fy * np.sin(a)
    return GnoPolar(code, re, aoa, lift / q, drag / q, "npz_pressure")


def from_prediction_dir(dir_path: str | Path) -> Callable[[str, float, float], GnoPolar | None]:
    """Lookup over a directory of prediction .npz files (matched by scalar meta)."""
    polars: dict = {}
    for f in Path(dir_path).glob("*.npz"):
        try:
            gp = integrate_pressure_polar(f)
            polars[_key(gp.naca_code, gp.reynolds, gp.aoa)] = gp
        except Exception as exc:  # noqa: BLE001
            print(f"  [npz] {f.name}: {exc}")

    def lookup(code: str, re: float, aoa: float) -> GnoPolar | None:
        return polars.get(_key(code, re, aoa))

    return lookup
