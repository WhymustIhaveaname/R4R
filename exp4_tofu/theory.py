"""The counterterms of each time scheme for one mode, derived with sympy.

The mode obeys q'' + D q' + w^2 q = 0, with D = beta w^2 for stiffness-proportional damping
(MuJoCo, Genesis's implicit FEM) and D = c for mass-proportional damping (Genesis's explicit
FEM).  A scheme maps (q, v) to (q', v') linearly; its two eigenvalues give the frequency and
the damping it actually produces.  The counterterms are the inputs (w*, D*), expanded to second
order in h, for which these equal the continuum's.  common.time_prediction holds the results.

  uv run python theory.py
"""

import sympy as sp

h, w, b, c = sp.symbols("h omega beta c", positive=True)
W, B = sp.symbols("Omega B", positive=True)   # the inputs: w* and the damping parameter*
a1, a2, c1, c2 = sp.symbols("a1 a2 c1 c2")
x, v, xn, vn = sp.symbols("x v xn vn")


def stiffness_damped(update):
    def eqs(W, B):
        return update(W, B, B * W**2)
    return eqs, b, b * w**2


SCHEMES = {
    "MuJoCo Euler: symplectic Euler, damping explicit": stiffness_damped(
        lambda W, B, D: [sp.Eq(vn, v + h * (-W**2 * x - D * v)), sp.Eq(xn, x + h * vn)]),
    "symplectic Euler, damping implicit": stiffness_damped(
        lambda W, B, D: [sp.Eq(vn, v + h * (-W**2 * x - D * vn)), sp.Eq(xn, x + h * vn)]),
    "MuJoCo discrete, Genesis implicit FEM: backward Euler": stiffness_damped(
        lambda W, B, D: [sp.Eq(vn, v + h * (-W**2 * xn - D * vn)), sp.Eq(xn, x + h * vn)]),
    "Genesis explicit FEM: symplectic Euler, then v <- exp(-c h) v": (
        lambda W, B: [sp.Eq(vn, sp.exp(-B * h) * (v - h * W**2 * x)), sp.Eq(xn, x + h * vn)],
        c, c),
}


def counterterms(eqs, d_true, rate_true):
    sol = sp.solve(eqs(W, B), [xn, vn], dict=True)[0]
    A = sp.Matrix([[sp.diff(sol[xn], x), sp.diff(sol[xn], v)],
                   [sp.diff(sol[vn], x), sp.diff(sol[vn], v)]])
    tr, det = sp.simplify(A.trace()), sp.simplify(A.det())
    # continuum: eigenvalues exp((-s +- i w_d) h), s = rate / 2, w_d = sqrt(w^2 - s^2)
    s = rate_true / 2
    wd = sp.sqrt(w**2 - s**2)
    sub = {W: w * (1 + a1 * h + a2 * h**2), B: d_true + c1 * h + c2 * h**2}
    f1 = sp.expand(sp.series((tr / (2 * sp.sqrt(det))).subs(sub) - sp.cos(wd * h), h, 0, 5).removeO())
    f2 = sp.expand(sp.series((-sp.log(det) / (2 * h)).subs(sub) - s, h, 0, 3).removeO())
    s1 = sp.solve([f1.coeff(h, 3), f2.coeff(h, 1)], [a1, c1], dict=True)[0]
    s2 = sp.solve([f1.coeff(h, 4).subs(s1), f2.coeff(h, 2).subs(s1)], [a2, c2], dict=True)[0]
    e = sp.expand(sp.series((1 + s1[a1] * h + s2[a2] * h**2) ** 2, h, 0, 3).removeO())
    return (sp.factor(e.coeff(h, 1)), sp.factor(e.coeff(h, 2))), (sp.factor(s1[c1]), sp.factor(s2[c2]))


if __name__ == "__main__":
    for name, (eqs, d_true, rate) in SCHEMES.items():
        (e1, e2), (d1, d2) = counterterms(eqs, d_true, rate)
        print(f"{name}\n   E*/E = 1 + ({e1}) h + ({e2}) h^2"
              f"\n   {d_true}* = {d_true} + ({d1}) h + ({d2}) h^2")
