"""STEP (ISO 10303 / .step) CAD export for NACA 4-digit airfoil geometry.

Builds a closed planar profile from the analytic surface in
``naca4_coords`` — a smooth B-spline through the upper surface, a smooth
B-spline through the lower surface, and a straight trailing-edge segment
closing the (generally nonzero, open-TE) gap between them — then extrudes it
into a solid prism and writes an AP214 STEP file.

The airfoil is exported in its natural, angle-of-attack-free chord-aligned
frame: angle of attack is a flow condition, not part of the shape.

Requires ``cadquery-ocp-novtk`` (plain OpenCASCADE Python bindings, import
name ``OCP`` — deliberately not the heavier ``cadquery``/``build123d``
wrapper libraries, which pull in unrelated dependencies like VTK/numba/nlopt
that this single export function does not need).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from utils.naca_geometry import naca4_coords

# Default spanwise extrusion depth, as a fraction of chord — turns the 2D
# airfoil profile into a thin solid "wing section" usable directly in CAD
# (a zero-thickness face/wire is not importable as a solid by most CAD and
# 3D-printing tools).
DEFAULT_SPAN_FRACTION = 0.1


def _points_to_harray(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    from OCP.gp import gp_Pnt
    from OCP.TColgp import TColgp_HArray1OfPnt

    n = len(xs)
    arr = TColgp_HArray1OfPnt(1, n)
    for i in range(n):
        arr.SetValue(i + 1, gp_Pnt(float(xs[i]), float(ys[i]), float(zs[i])))
    return arr


def _airfoil_solid(naca_code: str, n: int, span: float, chord: float):
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.gp import gp_Pnt, gp_Vec

    xu, yu, xl, yl = naca4_coords(naca_code, n=n)
    xu, yu, xl, yl = xu * chord, yu * chord, xl * chord, yl * chord
    zeros = np.zeros(n)

    # Upper and lower surfaces each interpolated as their own open B-spline
    # (LE -> TE); interpolating a single closed spline through both surfaces
    # would let the fit round over the sharp (generally nonzero-thickness,
    # open-TE) trailing-edge corner instead of preserving it.
    interp_u = GeomAPI_Interpolate(_points_to_harray(xu, yu, zeros), False, 1e-6)
    interp_u.Perform()
    edge_u = BRepBuilderAPI_MakeEdge(interp_u.Curve()).Edge()

    interp_l = GeomAPI_Interpolate(_points_to_harray(xl, yl, zeros), False, 1e-6)
    interp_l.Perform()
    edge_l = BRepBuilderAPI_MakeEdge(interp_l.Curve()).Edge()

    edge_te = BRepBuilderAPI_MakeEdge(
        gp_Pnt(float(xu[-1]), float(yu[-1]), 0.0),
        gp_Pnt(float(xl[-1]), float(yl[-1]), 0.0),
    ).Edge()

    wire_maker = BRepBuilderAPI_MakeWire()
    wire_maker.Add(edge_u)
    wire_maker.Add(edge_te)
    wire_maker.Add(edge_l)
    if not wire_maker.IsDone():
        raise RuntimeError(f"failed to build a closed airfoil wire for NACA {naca_code}")

    face = BRepBuilderAPI_MakeFace(wire_maker.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, span)).Shape()


def naca_airfoil_step_bytes(
    naca_code: str,
    *,
    n: int = 200,
    span: float | None = None,
    chord: float = 1.0,
) -> bytes:
    """Return an AP214 STEP file (as bytes) for a solid NACA 4-digit airfoil section.

    Parameters
    ----------
    naca_code : 4-digit NACA code, e.g. ``"2412"``.
    n         : surface sample points per side (passed to ``naca4_coords``).
    span      : extrusion depth along Z, same units as ``chord``. Defaults to
                ``DEFAULT_SPAN_FRACTION * chord``.
    chord     : chord length in metres (the STEP file declares metre units).

    Returns
    -------
    bytes — the STEP file contents (ASCII P21 text).
    """
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    if span is None:
        span = DEFAULT_SPAN_FRACTION * chord

    shape = _airfoil_solid(naca_code, n, span, chord)

    if not BRepCheck_Analyzer(shape).IsValid():
        raise RuntimeError(f"generated airfoil solid for NACA {naca_code} is not a valid B-Rep")

    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.unit", "M")
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError(f"STEP transfer failed for NACA {naca_code}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "airfoil.step"
        if writer.Write(str(path)) != IFSelect_RetDone:
            raise RuntimeError(f"STEP write failed for NACA {naca_code}")
        return path.read_bytes()
