"""Reference aerodynamic polars (Cl, Cd, Cm) for the Stage-1 comparison.

Two interchangeable backends behind one ``evaluate()`` call:

* ``xfoil``      - real XFOIL via subprocess (the gold-standard viscous panel
                   reference the user asked for). Requires an ``xfoil`` binary
                   on PATH. Faithful geometry: the exact NACA-4 surface is
                   written to a .dat file and LOADed.
* ``neuralfoil`` - a pure-numpy ML surrogate of XFOIL (no binary needed).
                   Instant, differentiable, ships an ``analysis_confidence``
                   score that doubles as a reliability / OOD flag. Great for
                   dense sweeps and for running anywhere.

FIDELITY NOTE: both are viscous panel / integral-boundary-layer methods with a
transition model. Your GNO was trained on RANS k-omega data, which is usually
run *fully turbulent*. For a fairer drag comparison, trip transition near the
leading edge (``xtr_upper=xtr_lower~0.05``) to emulate fully-turbulent RANS.
Use these references for the SHAPE of the Cl-alpha curve, stall onset and
trends/OOD — not for matching absolute Cd to the RANS ground truth.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from naca import naca4_selig, write_dat


@dataclass(frozen=True)
class Polar:
    naca_code: str
    reynolds: float
    aoa: float
    cl: float
    cd: float
    cm: float
    converged: bool
    backend: str
    confidence: float | None = None  # NeuralFoil only


# --------------------------------------------------------------------------- #
# NeuralFoil backend
# --------------------------------------------------------------------------- #


def _neuralfoil_available() -> bool:
    try:
        import neuralfoil  # noqa: F401
        return True
    except Exception:
        return False


def evaluate_neuralfoil(
    code: str,
    reynolds: float,
    aoa: float,
    *,
    n_coords: int = 160,
    n_crit: float = 9.0,
    xtr: float | None = None,
    model_size: str = "large",
) -> Polar:
    import neuralfoil as nf

    coords = naca4_selig(code, n=n_coords)
    kw = dict(coordinates=coords, alpha=float(aoa), Re=float(reynolds), n_crit=n_crit, model_size=model_size)
    if xtr is not None:
        kw["xtr_upper"] = xtr
        kw["xtr_lower"] = xtr
    r = nf.get_aero_from_coordinates(**kw)
    conf = float(np.ravel(r["analysis_confidence"])[0])
    return Polar(
        naca_code=code, reynolds=reynolds, aoa=aoa,
        cl=float(np.ravel(r["CL"])[0]), cd=float(np.ravel(r["CD"])[0]), cm=float(np.ravel(r["CM"])[0]),
        converged=conf > 0.5, backend="neuralfoil", confidence=conf,
    )


# --------------------------------------------------------------------------- #
# XFOIL backend
# --------------------------------------------------------------------------- #


def xfoil_available() -> bool:
    return shutil.which("xfoil") is not None


def evaluate_xfoil(
    code: str,
    reynolds: float,
    aoa: float,
    *,
    n_coords: int = 160,
    n_iter: int = 200,
    xtr: float | None = None,
    timeout: float = 40.0,
    xfoil_bin: str = "xfoil",
) -> Polar:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dat = write_dat(code, tmp / "af.dat", n=n_coords)
        polar = tmp / "polar.txt"

        cmds = ["PLOP", "G F", "", f"LOAD {dat}", "PANE", "OPER", f"VISC {reynolds:.1f}", "MACH 0", f"ITER {n_iter}"]
        if xtr is not None:
            cmds += ["VPAR", f"XTR {xtr} {xtr}", ""]
        cmds += ["PACC", str(polar), "", f"ALFA {aoa:.3f}", "PACC", "", "", "QUIT", ""]
        script = "\n".join(cmds)

        try:
            subprocess.run([xfoil_bin], input=script, text=True, capture_output=True,
                           timeout=timeout, cwd=tmp)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return Polar(code, reynolds, aoa, np.nan, np.nan, np.nan, False, "xfoil")

        cl = cd = cm = np.nan
        converged = False
        if polar.exists():
            for line in polar.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        a = float(parts[0])
                    except ValueError:
                        continue
                    if abs(a - aoa) < 1e-3:
                        cl, cd, cm = float(parts[1]), float(parts[2]), float(parts[4])
                        converged = True
                        break
        return Polar(code, reynolds, aoa, cl, cd, cm, converged, "xfoil")


# --------------------------------------------------------------------------- #
# Unified entry point
# --------------------------------------------------------------------------- #


def resolve_backend(backend: str = "auto") -> str:
    if backend == "auto":
        if xfoil_available():
            return "xfoil"
        if _neuralfoil_available():
            return "neuralfoil"
        raise RuntimeError("Neither an 'xfoil' binary nor the 'neuralfoil' package is available.")
    return backend


def evaluate(code: str, reynolds: float, aoa: float, *, backend: str = "auto", **kw) -> Polar:
    """Evaluate one operating point with the selected (or auto) backend."""
    b = resolve_backend(backend)
    if b == "xfoil":
        return evaluate_xfoil(code, reynolds, aoa, **{k: v for k, v in kw.items()
                                                      if k in ("n_coords", "n_iter", "xtr", "timeout", "xfoil_bin")})
    if b == "neuralfoil":
        return evaluate_neuralfoil(code, reynolds, aoa, **{k: v for k, v in kw.items()
                                                           if k in ("n_coords", "n_crit", "xtr", "model_size")})
    raise ValueError(f"Unknown backend {b!r}")


def sweep(code: str, reynolds: float, alphas, *, backend: str = "auto", **kw) -> list[Polar]:
    """Evaluate an alpha sweep at fixed geometry/Re (used for the Cl-alpha plot)."""
    return [evaluate(code, reynolds, float(a), backend=backend, **kw) for a in alphas]
