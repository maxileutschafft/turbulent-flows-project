"""After gmshToFoam, set OpenFOAM patch types: airfoil->wall, frontAndBack->empty."""
import re, sys
from pathlib import Path

def set_types(case):
    b = Path(case) / "constant" / "polyMesh" / "boundary"
    txt = b.read_text()
    for name, typ in (("airfoil", "wall"), ("frontAndBack", "empty")):
        txt = re.sub(rf"({name}\s*\{{\s*type\s+)\w+;", rf"\g<1>{typ};", txt, count=1)
    b.write_text(txt)
    print(f"patch types set in {b}")

if __name__ == "__main__":
    set_types(sys.argv[1] if len(sys.argv) > 1 else ".")
