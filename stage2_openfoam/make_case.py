"""Generate a simpleFoam (k-omega SST) case for a NACA airfoil at (Re, AoA).

Writes 0/, constant/, system/ into a case directory. BCs match scenario.py:
U_mag = Re*nu/chord, nu=1e-5, chord=1, freestream angled by AoA (airfoil fixed).
Farfield uses the freestream BCs; airfoil is a no-slip wall with wall functions;
frontAndBack are empty (2D). A forceCoeffs function object writes Cl/Cd.

Syntax targets OpenFOAM (ESI / .com) - stable transportProperties +
turbulenceProperties/RASModel. For OpenFOAM.org v11+ rename to
physicalProperties / momentumTransport (see README). The PHYSICS (k-omega SST,
nu, BCs, mesh) is what sets the fidelity, not the dict filenames.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

NU = 1.0e-5
CHORD = 1.0


def _head(cls, obj, loc):
    return (f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n"
            f'    location "{loc}";\n    object {obj};\n}}\n')


def turbulence_inlet(u_mag, intensity=0.008, length=0.08):
    k = 1.5 * (intensity * u_mag) ** 2
    omega = math.sqrt(k) / (0.09 ** 0.25 * length * CHORD)
    nut = k / omega
    return k, omega, nut


def write_case(case: Path, naca: str, reynolds: float, aoa_deg: float,
               end_time: int = 3000, intensity: float = 0.008, dz: float = 0.1):
    u_mag = reynolds * NU / CHORD
    a = math.radians(aoa_deg)
    ux, uy = u_mag * math.cos(a), u_mag * math.sin(a)
    k0, w0, nut0 = turbulence_inlet(u_mag, intensity)
    lift_dir = (-math.sin(a), math.cos(a), 0.0)
    drag_dir = (math.cos(a), math.sin(a), 0.0)

    for d in ("0", "constant", "system"):
        (case / d).mkdir(parents=True, exist_ok=True)

    # ---- 0/ fields ----
    def field(obj, cls, dims, internal, farfield, airfoil):
        return (_head(cls, obj, "0") +
                f"\ndimensions      {dims};\n"
                f"internalField   uniform {internal};\n\n"
                "boundaryField\n{\n"
                f"    farfield\n    {{\n{farfield}    }}\n"
                f"    airfoil\n    {{\n{airfoil}    }}\n"
                "    frontAndBack\n    {\n        type            empty;\n    }\n}\n")

    (case / "0/U").write_text(field(
        "U", "volVectorField", "[0 1 -1 0 0 0 0]", f"({ux:.6g} {uy:.6g} 0)",
        f"        type            freestreamVelocity;\n        freestreamValue uniform ({ux:.6g} {uy:.6g} 0);\n",
        "        type            noSlip;\n"))
    (case / "0/p").write_text(field(
        "p", "volScalarField", "[0 2 -2 0 0 0 0]", "0",
        "        type            freestreamPressure;\n        freestreamValue uniform 0;\n",
        "        type            zeroGradient;\n"))
    (case / "0/k").write_text(field(
        "k", "volScalarField", "[0 2 -2 0 0 0 0]", f"{k0:.6g}",
        f"        type            freestream;\n        freestreamValue uniform {k0:.6g};\n",
        f"        type            kqRWallFunction;\n        value           uniform {k0:.6g};\n"))
    (case / "0/omega").write_text(field(
        "omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"{w0:.6g}",
        f"        type            freestream;\n        freestreamValue uniform {w0:.6g};\n",
        f"        type            omegaWallFunction;\n        value           uniform {w0:.6g};\n"))
    (case / "0/nut").write_text(field(
        "nut", "volScalarField", "[0 2 -1 0 0 0 0]", f"{nut0:.6g}",
        "        type            freestream;\n        freestreamValue uniform 0;\n",
        "        type            nutkWallFunction;\n        value           uniform 0;\n"))

    # ---- constant/ ----
    (case / "constant/transportProperties").write_text(
        _head("dictionary", "transportProperties", "constant") +
        f"\ntransportModel  Newtonian;\nnu              [0 2 -1 0 0 0 0] {NU};\n")
    (case / "constant/turbulenceProperties").write_text(
        _head("dictionary", "turbulenceProperties", "constant") +
        "\nsimulationType  RAS;\nRAS\n{\n    RASModel        kOmegaSST;\n"
        "    turbulence      on;\n    printCoeffs     on;\n}\n")

    # ---- system/ ----
    (case / "system/controlDict").write_text(
        _head("dictionary", "controlDict", "system") + f"""
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   {max(100, end_time // 5)};
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
runTimeModifiable true;

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (airfoil);
        rho             rhoInf;
        rhoInf          1;
        liftDir         ({lift_dir[0]:.6g} {lift_dir[1]:.6g} 0);
        dragDir         ({drag_dir[0]:.6g} {drag_dir[1]:.6g} 0);
        CofR            (0.25 0 0);
        pitchAxis       (0 0 1);
        magUInf         {u_mag:.6g};
        lRef            {CHORD};
        Aref            {CHORD * dz:.6g};
    }}
}}
""")
    (case / "system/fvSchemes").write_text(
        _head("dictionary", "fvSchemes", "system") + """
ddtSchemes      { default steadyState; }
gradSchemes
{
    default         cellLimited Gauss linear 1;
    grad(U)         cellLimited Gauss linear 1;
}
divSchemes
{
    default             none;
    div(phi,U)          bounded Gauss linearUpwindV grad(U);
    div(phi,k)          bounded Gauss upwind;
    div(phi,omega)      bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.33; }
wallDist        { method meshWave; }
""")
    (case / "system/fvSolution").write_text(
        _head("dictionary", "fvSolution", "system") + """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother GaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 2;
    consistent      yes;
    residualControl { p 1e-5; U 1e-6; "(k|omega)" 1e-6; }
}
relaxationFactors
{
    equations { U 0.7; "(k|omega)" 0.5; }
}
""")
    (case / "system/decomposeParDict").write_text(
        _head("dictionary", "decomposeParDict", "system") +
        "\nnumberOfSubdomains 4;\nmethod          scotch;\n")
    print(f"case written: {case}  (U_mag={u_mag:.4g}, k={k0:.4g}, omega={w0:.4g}, nut={nut0:.4g})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--naca", required=True)
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--aoa", type=float, required=True)
    ap.add_argument("--end-time", type=int, default=3000)
    ap.add_argument("--intensity", type=float, default=0.05)
    args = ap.parse_args()
    write_case(Path(args.case), args.naca, args.re, args.aoa, args.end_time, args.intensity)


if __name__ == "__main__":
    main()
