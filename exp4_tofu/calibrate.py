"""Step 1: the tofu's Young's modulus from the literature, each source through its own mechanics.

Brookfield's ball test (NOTES.md) gives the peak force of a 25.4 mm sphere pressed 7 mm at 2 mm/s
into five commercial tofus.  Neo-Hookean, the only one of the engines' materials that reaches
7 mm (axisym_nl.py), has a force proportional to E at a fixed nu, so each tofu's modulus is

  E = E0 F_measured / F_neo-Hookean(7 mm; E0).

The note does not give the block, so its thickness (3, 4, 5 cm), its support (bonded or sliding
on the table) and nu (0.45, 0.49) are varied around the base case, a 4 cm block of the slab's
area, bonded, nu 0.45.  The other sources, converted to E:

  Pattison et al. 2015   shear modulus at 14 Hz (MR elastography), E = 2 mu (1 + nu)
  Dunne et al. 2025      G' at 0.2 to 4.6 rad/s, E = 2 G' (1 + nu)
  Kohyama, Nishinari 1992  silken tofu compressed to fracture: nominal stress 7.1 to 8.4 kPa at
                         strain 0.44 to 0.56; incompressible neo-Hookean in uniaxial compression
                         has nominal stress mu (lambda - lambda^-2), E = 3 mu

  python calibrate.py
"""

import json

import numpy as np

import common as c
import indentation as ind

BROOKFIELD_G = {"Gourmet Soft": 173.0, "Longlife Soft": 183.0, "House Firm": 278.0,
                "Ichiban Firm": 289.5, "Ichiban Regular": 479.0}
G = 9.80665
DEPTH = 7e-3
CASES = {"base: 4 cm, bonded, nu 0.45": dict(height=0.04, bottom="bonded", nu=0.45),
         "3 cm": dict(height=0.03, bottom="bonded", nu=0.45),
         "5 cm": dict(height=0.05, bottom="bonded", nu=0.45),
         "sliding on the table": dict(height=0.04, bottom="slip", nu=0.45),
         "nu 0.49": dict(height=0.04, bottom="bonded", nu=0.49)}
FINE = 2e-4    # m; the force is within 0.2 % of its converged value (axisym_nl.py)


def force_at_7mm(height, bottom, nu):
    path = c.CACHE / f"calibrate_neohookean_{height * 1e3:.0f}mm_{bottom}_{nu:g}.json"
    if path.exists():
        return json.loads(path.read_text())["F"]
    import axisym_nl
    lay = axisym_nl.LayerNL(ind.R_EQUIVALENT, height, c.E0, nu, "neohookean", FINE, 0.015, bottom=bottom)
    for d in np.arange(0.5e-3, DEPTH - 1e-9, 0.5e-3):
        lay.indent(ind.RADIUS, d)
    F = lay.indent(ind.RADIUS, DEPTH)
    c.CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"F": F}))
    return F


def kohyama():
    """The range of E from silken tofu's fracture point in compression."""
    out = []
    for stress in (7.1e3, 8.4e3):
        for strain in (0.44, 0.56):
            lam = 1 - strain
            out.append(3 * stress / (1 / lam**2 - lam))
    return min(out), max(out)


def main():
    forces = {name: force_at_7mm(**kw) for name, kw in CASES.items()}
    base = forces["base: 4 cm, bonded, nu 0.45"]
    print(f"neo-Hookean force at 7 mm, E = {c.E0 / 1e3:.1f} kPa:")
    for name, F in forces.items():
        print(f"  {name:30s} {F:.4f} N  ({F / base - 1:+.1%})")
    print("\nE (kPa) from Brookfield: base case, and the range over the cases")
    for tofu, grams in BROOKFIELD_G.items():
        Es = [c.E0 * grams * 1e-3 * G / F / 1e3 for F in forces.values()]
        print(f"  {tofu:16s} {grams:6.1f} g   E = {c.E0 * grams * 1e-3 * G / base / 1e3:5.1f}   ({min(Es):.1f} to {max(Es):.1f})")
    nu = c.NU
    print("\nE (kPa) from the other sources, nu = 0.45 where it enters:")
    for name, mu in (("Pattison silken soft", 3.88), ("Pattison silken firm", 11.8), ("Pattison silken extra firm", 21.1),
                     ("Dunne firm (G')", 5.67), ("Dunne extra firm (G')", 11.0)):
        print(f"  {name:28s} {2 * mu * (1 + nu):5.1f}")
    lo, hi = kohyama()
    print(f"  {'Kohyama silken (fracture)':28s} {lo / 1e3:5.1f} to {hi / 1e3:.1f}")


if __name__ == "__main__":
    main()
