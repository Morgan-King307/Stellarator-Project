#!/usr/bin/env python3
"""
fieldline_tracer.py
===================

A small, general-purpose magnetic field-line tracer, validated against an
EXACT analytic 3D MHD equilibrium (Landreman's iota = 2 family, and the
stellarator-symmetry-breaking generalization of Issan et al.).

WHY THIS FILE EXISTS
--------------------
Almost every stellarator computation starts with "follow a field line":
  * Poincare plots / flux-surface visualization   (EPOS, W7-X)
  * connection lengths to divertor targets        (W7-X island divertor)
  * field-error studies (do the surfaces survive?)(EPOS)
  * geometry for transport / gyrokinetic inputs   (Proll group)
Before trusting a tracer on a real coil field, you want to test it on a
field where the right answer is known in closed form.  The analytic
equilibria are perfect for that: the field lines are *exact* harmonic
oscillators, so we know where every point on every line should be.

WHAT IS IN THE FILE (read top to bottom)
----------------------------------------
  1. AnalyticEquilibrium : the exact B(x,y,z), flux function psi, and the
                           exact field-line map.   (specific to the papers)
  2. trace_field_line, poincare_points : GENERIC tracers.  They accept ANY
                           function B(x) -> 3-vector.  Reuse these for
                           coil fields (Biot-Savart), VMEC/DESC fields, etc.
  3. Diagnostics         : poloidal angle and rotational transform iota.
  4. Validation tests    : numbered checks, each prints PASS/FAIL.
  5. Plots               : figures written to a folder.

HOW TO RUN
----------
    python fieldline_tracer.py                     # eps=0.5, tau0=0 (Landreman)
    python fieldline_tracer.py --tau0 0.5          # symmetry-broken (Issan et al.)
    python fieldline_tracer.py --eps 0.3 --outdir figs

Requires: numpy, scipy, matplotlib.

UNITS
-----
Everything is dimensionless (the papers normalize lengths and B so that
the coefficients are O(1)).  Only the *geometry* matters for tracing.

SOURCES
-------
  [L]  M. Landreman, "Analytic toroidal 3D MHD equilibria and steady Euler
       flows with invariant surfaces", arXiv:2609.26742   (eqs 2.1-2.25)
  [I]  O. Issan et al., "Analytic non-axisymmetric 3D MHD equilibria without
       stellarator symmetry", arXiv:2610.07304            (eqs 4-14)
Equation numbers in comments below refer to these papers.
"""

import argparse
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# 1. THE ANALYTIC EQUILIBRIUM
# =============================================================================
#
# Background (the one-paragraph physics):
#   Static ideal MHD says  (curl B) x B = grad p,   div B = 0.
#   Taking B . (first equation) gives  B . grad p = 0, i.e. pressure is
#   constant along field lines, so field lines live ON surfaces of constant
#   pressure ("flux surfaces").  For confinement these should be nested tori.
#
#   Landreman found closed-form solutions.  The trick: B is a "stretched"
#   version of an axisymmetric Solov'ev tokamak field.  A consequence is that
#   the magnetic tension (B . grad)B equals  -grad Q  with Q QUADRATIC in
#   position, so if you follow a field line with  d x / d(zeta) = B / Omega,
#   each Cartesian coordinate obeys a harmonic-oscillator equation:
#
#        x'' = -x ,    y'' = -y ,    z'' = -4 (z - z_c)           [L eq 2.10, I eq 9]
#
#   (z_c = 0 for the stellarator-symmetric case.)  That is why we can write
#   the field lines down exactly -- and why this is such a nice test case.
#
#   The parameters:
#     eps  in (0,1) : horizontal elongation (eps -> 0 is axisymmetric)
#     tau0 in (-1,1): vertical-asymmetry parameter of Issan et al.
#                     tau0 = 0   -> stellarator-symmetric (Landreman)
#                     tau0 != 0  -> breaks stellarator symmetry


class AnalyticEquilibrium:
    """Exact iota=2 equilibrium of [L] (tau0=0) and [I] (tau0 != 0)."""

    def __init__(self, eps=0.5, tau0=0.0):
        if not (0.0 < eps < 1.0):
            raise ValueError("eps must be in (0,1)")
        if not (-1.0 < tau0 < 1.0):
            raise ValueError("tau0 must be in (-1,1)")
        self.eps = eps
        self.tau0 = tau0
        # Horizontal stretch factors, [I] eq 4 / [L] below eq 2.1.
        self.a = np.sqrt(1.0 + eps)
        self.b = np.sqrt(1.0 - eps)
        # Omega = sqrt(1 - tau0^2); equals 1 when tau0 = 0.
        self.Omega = np.sqrt(1.0 - tau0**2)
        # Phase shift gamma and vertical offset z_c of the field lines
        # ([I] eq 9-10).
        self.gamma = np.arcsin(tau0)
        self.z_c = -tau0 / (2.0 * self.Omega**2)
        # In field-line coordinates (u,v) the magnetic axis sits at
        # (u_a, 0), and flux surfaces are circles about it ([I] sec 2.2).
        self.u_axis = -self.Omega**2 * eps / 2.0
        # Largest allowed boundary flux: surfaces must stay inside the disk
        # u^2+v^2 < 1/4 where the field is smooth ([I] below eq 7).
        self.delta_max = (1.0 - self.Omega**2 * eps) ** 2 / 4.0

    # ---- the magnetic field itself -----------------------------------------
    def B(self, X):
        """Magnetic field at Cartesian point X = (x, y, z).  [I] eq 5.

        s is a "squashed radius squared"; F is the square root that makes the
        horizontal field circulate.  F is real only inside the plasma domain
        (radicand > 0); outside we return NaN so a stray integration fails
        loudly instead of silently producing garbage.
        """
        x, y, z = X
        a, b, t = self.a, self.b, self.tau0
        s = x**2 / a**2 + y**2 / b**2
        radicand = 1.0 - (1.0 - s) ** 2 - 4.0 * z**2 - 4.0 * t * s * z
        if radicand <= 0.0 or s <= 0.0:
            return np.array([np.nan, np.nan, np.nan])
        F = np.sqrt(radicand)
        Bx = ((2.0 * z + t * s) * x - (a / b) * F * y) / s
        By = ((2.0 * z + t * s) * y + (b / a) * F * x) / s
        Bz = 1.0 - s - 2.0 * t * z
        return np.array([Bx, By, Bz])

    # ---- pressure-like scalar H = Q + |B|^2/2 ------------------------------
    def H(self, X):
        """Bernoulli-like function H = Q + |B|^2/2.

        Force balance reads (curl B) x B = -grad H, so p = p_axis - 2 psi/Omega^2
        means grad p = -grad H.  H is therefore CONSTANT on flux surfaces
        (a very useful numerical invariant to monitor along traced lines).
        Q is the scalar whose gradient is minus the magnetic tension, [I] eq 6.
        """
        x, y, z = X
        Q = 0.5 * self.Omega**2 * (x**2 + y**2 + 4.0 * z**2) + 2.0 * self.tau0 * z
        Bv = self.B(X)
        return Q + 0.5 * float(Bv @ Bv)

    # ---- flux function -----------------------------------------------------
    def psi(self, X):
        """Flux label psi, normalized so psi = 0 on the magnetic axis.

        psi = (Omega^2/2) * (H - H_axis).   [I] eq 7 (we fix the additive
        constant numerically from the axis rather than copying the printed
        constant, so a typo can't hide in it).  Test 5 below checks that this
        agrees with the paper's field-line-coordinate form
        psi = (u - u_a)^2 + v^2.
        """
        return 0.5 * self.Omega**2 * (self.H(X) - self._H_axis())

    def _H_axis(self):
        # Cached: evaluate H at one point on the axis.
        if not hasattr(self, "_H_axis_cache"):
            self._H_axis_cache = self.H(self.axis(0.0))
        return self._H_axis_cache

    # ---- exact field-line map (u, v, zeta) -> (x, y, z) --------------------
    def position(self, u, v, zeta):
        """EXACT point on field line (u,v) at toroidal-like parameter zeta.

        [I] eq 10 / [L] eq 2.11.  (u,v) are LABELS of the field line (constants
        along it); zeta runs along it with period 2*pi.  Field lines are
        closed: zeta -> zeta + 2*pi returns to the same point (this is what
        iota = 2 = rational means here).

        u, v may be scalars; zeta may be a scalar or array.
        """
        zeta = np.asarray(zeta, dtype=float)
        q2 = u**2 + v**2
        if q2 >= 0.25:
            raise ValueError("need u^2+v^2 < 1/4 for a valid field line")
        # L solves L^2 + q^2/L^2 = 1, i.e. L = sqrt((1+sqrt(1-4q^2))/2).
        L = np.sqrt((1.0 + np.sqrt(1.0 - 4.0 * q2)) / 2.0)
        x = (self.a / self.Omega) * (
            L * np.cos(zeta) + (u * np.cos(zeta) + v * np.sin(zeta)) / L
        )
        y = (self.b / self.Omega) * (
            L * np.sin(zeta) + (v * np.cos(zeta) - u * np.sin(zeta)) / L
        )
        z = self.z_c + (
            v * np.cos(2 * zeta + self.gamma) - u * np.sin(2 * zeta + self.gamma)
        ) / self.Omega**2
        return np.stack([x, y, z], axis=0)  # shape (3,) or (3, N)

    def uv_from_psi_alpha(self, psi, alpha):
        """Pick a field line on the surface with flux `psi`.

        Surfaces of constant psi are circles of radius sqrt(psi) centred on
        the axis in the (u,v) plane [I sec 2.2]; alpha is the angle around
        that circle.  Together (psi, alpha) label a field line.
        """
        r = np.sqrt(psi)
        return self.u_axis + r * np.cos(alpha), r * np.sin(alpha)

    def axis(self, zeta):
        """Magnetic axis curve: the field line with psi = 0, i.e. (u_a, 0)."""
        return self.position(self.u_axis, 0.0, zeta)


# =============================================================================
# 2. GENERIC FIELD-LINE TRACERS  (reusable for ANY field)
# =============================================================================
#
# A field line is a curve everywhere tangent to B:
#         d x / d t = B(x(t)).
# t is just a curve parameter, NOT arc length (|dx/dt| = |B|).  If you want
# arc length (needed for connection lengths!), set arclength=True, which
# integrates dx/dl = B/|B| instead.  Same geometry, different parametrization.
#
# We use scipy's DOP853 (8th-order Runge-Kutta) with tight tolerances.
# High order matters: Poincare plots and iota need errors that do NOT
# accumulate visibly over hundreds of toroidal turns.


def trace_field_line(B, x0, t_max, arclength=False, rtol=1e-11, atol=1e-13,
                     events=None):
    """Follow a field line from x0 for parameter length t_max.

    Parameters
    ----------
    B        : callable, B(X) -> array(3).  X is array(3).  ANY field works.
    x0       : starting point, array(3).
    t_max    : how far to integrate (curve parameter, or length if arclength).
    arclength: if True, parametrize by arc length l (so t_max is a length).
    events   : optional list of scipy event functions (used for Poincare
               sections and for "hit a target surface" connection-length
               stops -- see poincare_points below).

    Returns
    -------
    scipy OdeResult with dense output: sol.sol(t) gives x(t) for any t.
    """
    if arclength:
        def rhs(t, X):
            Bv = B(X)
            return Bv / np.linalg.norm(Bv)  # unit tangent
    else:
        def rhs(t, X):
            return B(X)

    sol = solve_ivp(rhs, (0.0, t_max), np.asarray(x0, float),
                    method="DOP853", rtol=rtol, atol=atol,
                    dense_output=True, events=events)
    if not sol.success:
        raise RuntimeError(f"field-line integration failed: {sol.message}")
    return sol


def poincare_points(B, x0, phi0, n_crossings, t_max, **kw):
    """Collect the points where a field line crosses the half-plane phi = phi0.

    phi is the usual cylindrical toroidal angle atan2(y, x).  The plane
    phi = phi0 is where  g(X) = -x sin(phi0) + y cos(phi0) = R sin(phi-phi0)
    vanishes.  We ask scipy for upward zero crossings of g (direction=+1),
    which are crossings in the direction of increasing phi; we then discard
    the "other half" of the plane (phi0 + pi), where g also vanishes, by
    requiring R cos(phi - phi0) > 0.

    This assumes the line circulates toroidally in the +phi direction (true
    for tokamaks and stellarators with a dominant toroidal field).

    Returns an (n, 3) array of (x, y, z) crossing points.
    """
    c, s = np.cos(phi0), np.sin(phi0)

    def plane(t, X):          # signed distance-like function; zero on the plane
        return -X[0] * s + X[1] * c
    plane.direction = +1       # only upward (phi-increasing) crossings

    sol = trace_field_line(B, x0, t_max, events=[plane], **kw)
    pts = sol.y_events[0]
    keep = (pts[:, 0] * c + pts[:, 1] * s) > 0.0   # right half-plane only
    return pts[keep][:n_crossings]


# =============================================================================
# 3. DIAGNOSTICS: poloidal angle and rotational transform
# =============================================================================
#
# Rotational transform iota = (poloidal turns) / (toroidal turns) of a field
# line.  To count poloidal turns you need a "poloidal angle" theta measured
# around the magnetic axis.  [L eq 2.23]:
#         theta = arg[ (R - R_axis(phi)) - i (z - Z_axis(phi)) ]
# i.e. the angle of the line's position relative to the axis in the (R,z)
# half-plane at the SAME toroidal angle phi.  The minus sign on z is the
# paper's orientation convention.  iota = total change in theta / total
# change in phi, using UNWRAPPED angles (so 2*pi jumps are counted).


class AxisInterpolator:
    """Axis curve as functions of cylindrical angle phi: R_ax(phi), Z_ax(phi).

    The axis is given parametrically in zeta, but the poloidal angle needs it
    at the same *phi* as the field-line point, so we tabulate and interpolate.
    """

    def __init__(self, axis_func, n=4001):
        zeta = np.linspace(0.0, 2 * np.pi, n)
        xyz = axis_func(zeta)
        phi = np.unwrap(np.arctan2(xyz[1], xyz[0]))
        self.phi0 = phi[0]
        self.period = phi[-1] - phi[0]          # should be 2*pi
        self.phi = phi - self.phi0
        self.R = np.hypot(xyz[0], xyz[1])
        self.Z = xyz[2]

    def at(self, phi):
        """R_ax and Z_ax at (unwrapped) angle(s) phi, periodic in 2*pi."""
        p = np.mod(np.asarray(phi) - self.phi0, self.period)
        return np.interp(p, self.phi, self.R), np.interp(p, self.phi, self.Z)


def rotational_transform(sol, t_end, axis, n=4000):
    """Measure iota along a traced line `sol` (from trace_field_line).

    Samples the dense solution on [0, t_end], builds unwrapped phi and theta,
    and returns (iota, phi_array, theta_array).
    """
    t = np.linspace(0.0, t_end, n)
    X = sol.sol(t)
    phi = np.unwrap(np.arctan2(X[1], X[0]))
    R = np.hypot(X[0], X[1])
    R_ax, Z_ax = axis.at(phi)
    theta = np.unwrap(np.arctan2(-(X[2] - Z_ax), R - R_ax))  # arg[(R-Ra) - i(z-Za)]
    iota = (theta[-1] - theta[0]) / (phi[-1] - phi[0])
    return iota, phi, theta


# =============================================================================
# 4. VALIDATION TESTS
# =============================================================================
#
# Each test returns True/False and prints a one-line verdict with the number
# it measured.  The ladder goes from "is the formula I typed even an
# equilibrium?" up to "does the numerical tracer reproduce the exact lines?".
# If a test ever fails, the order tells you where to look: tests 1-3 check
# the FIELD, tests 4-7 check the TRACER.


def _report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def _fd_jacobian(f, X, h=1e-5):
    """Central-difference Jacobian J[i,j] = d f_i / d X_j for f: R^3 -> R^3."""
    J = np.zeros((3, 3))
    for j in range(3):
        dX = np.zeros(3)
        dX[j] = h
        J[:, j] = (f(X + dX) - f(X - dX)) / (2 * h)
    return J


def _fd_gradient(f, X, h=1e-5):
    """Central-difference gradient of a scalar function f."""
    g = np.zeros(3)
    for j in range(3):
        dX = np.zeros(3)
        dX[j] = h
        g[j] = (f(X + dX) - f(X - dX)) / (2 * h)
    return g


def random_interior_points(eq, delta, n, rng):
    """Random points inside the plasma, sampled via field-line coordinates.

    Choose (psi, alpha, zeta) at random and map through the exact position().
    Using the exact map guarantees the point is inside the valid domain.
    """
    pts = []
    for _ in range(n):
        psi = delta * rng.uniform(0.05, 1.0)
        alpha = rng.uniform(0, 2 * np.pi)
        zeta = rng.uniform(0, 2 * np.pi)
        u, v = eq.uv_from_psi_alpha(psi, alpha)
        pts.append(eq.position(u, v, zeta))
    return np.array(pts)


def test_divergence_free(eq, pts):
    """Test 1: div B = 0.  Trace of the finite-difference Jacobian of B."""
    worst = 0.0
    for X in pts:
        J = _fd_jacobian(eq.B, X)
        scale = np.abs(J).max()
        worst = max(worst, abs(np.trace(J)) / scale)
    return _report("1  div B = 0", worst < 1e-6,
                   f"max |div B| / max|dB/dx| = {worst:.2e}")


def test_force_balance(eq, pts):
    """Test 2: (curl B) x B = -grad H  (equivalently = grad p).

    curl B comes from the antisymmetric part of the Jacobian:
        (curl B)_x = dBz/dy - dBy/dz, etc.
    This is the actual MHD equilibrium equation -- if it holds, the formulas
    in the paper (as transcribed here) really are an equilibrium.
    """
    worst = 0.0
    for X in pts:
        J = _fd_jacobian(eq.B, X)                    # J[i,j] = dB_i/dx_j
        curl = np.array([J[2, 1] - J[1, 2],
                         J[0, 2] - J[2, 0],
                         J[1, 0] - J[0, 1]])
        lorentz = np.cross(curl, eq.B(X))            # (curl B) x B
        gradH = _fd_gradient(eq.H, X)
        resid = np.linalg.norm(lorentz + gradH) / np.linalg.norm(gradH)
        worst = max(worst, resid)
    return _report("2  (curl B) x B = grad p", worst < 1e-5,
                   f"max relative residual = {worst:.2e}")


def test_jacobian_constant(eq, pts_uvz):
    """Test 3: coordinate Jacobian det d(x,y,z)/d(u,v,zeta) = -ab/Omega^3.

    A constant Jacobian means (u,v,zeta) are "volume-preserving" field-line
    coordinates -- that is what lets the paper write the enclosed volume in
    closed form.  Checked by finite differences of the exact map.
    """
    expected = -eq.a * eq.b / eq.Omega**3
    worst = 0.0
    h = 1e-6
    for (u, v, zeta) in pts_uvz:
        def m(p):
            return eq.position(p[0], p[1], p[2])
        p0 = np.array([u, v, zeta])
        J = np.zeros((3, 3))
        for j in range(3):
            d = np.zeros(3)
            d[j] = h
            J[:, j] = (m(p0 + d) - m(p0 - d)) / (2 * h)
        worst = max(worst, abs(np.linalg.det(J) - expected) / abs(expected))
    return _report("3  Jacobian = -ab/Omega^3 (constant)", worst < 1e-6,
                   f"max relative deviation = {worst:.2e}  (expected {expected:.5f})")


def test_exact_map_is_tangent_to_B(eq, pts_uvz):
    """Test 4: the exact map really follows B:  d(position)/d(zeta) = B/Omega...

    Careful with the factor: [I] eq 8 says Omega * dx/dzeta = B.
    This verifies the *map* against the *field* (two independently typed
    formulas), before we involve any numerical integrator.
    """
    worst = 0.0
    h = 1e-6
    for (u, v, zeta) in pts_uvz:
        dx = (eq.position(u, v, zeta + h) - eq.position(u, v, zeta - h)) / (2 * h)
        X = eq.position(u, v, zeta)
        Bv = eq.B(X)
        worst = max(worst, np.linalg.norm(eq.Omega * dx - Bv) / np.linalg.norm(Bv))
    return _report("4  exact map is tangent to B", worst < 1e-6,
                   f"max |Omega dx/dzeta - B| / |B| = {worst:.2e}")


def test_psi_formula(eq, delta, rng):
    """Test 5: psi computed from H equals the paper's (u-u_a)^2 + v^2.

    Two independent routes to the same flux label: one from the field
    in Cartesian coordinates, one from the (u,v) label of the line.
    """
    worst = 0.0
    for _ in range(25):
        psi = delta * rng.uniform(0.05, 1.0)
        alpha, zeta = rng.uniform(0, 2 * np.pi, 2)
        u, v = eq.uv_from_psi_alpha(psi, alpha)
        X = eq.position(u, v, zeta)
        worst = max(worst, abs(eq.psi(X) - psi) / psi)
    return _report("5  psi(H) = (u-u_a)^2 + v^2", worst < 1e-9,
                   f"max relative difference = {worst:.2e}")


def test_tracer_vs_exact(eq, delta, rng, n_lines=8):
    """Test 6: THE core tracer test.  Numerically integrate dx/dt = B and
    compare to the exact closed-form field line at many points.

    The numerical parameter t relates to zeta by  zeta = Omega * t, i.e.
    t = zeta / Omega (because Omega dx/dzeta = B and dx/dt = B), so one
    closed turn (zeta = 2*pi) takes t = 2*pi/Omega.  The error should be tiny
    (~1e-9 or better) over a full toroidal turn; if it's ~1e-3, the
    integrator tolerances (or the field) are wrong.
    """
    worst_pos = 0.0
    worst_closure = 0.0
    worst_psi_drift = 0.0
    zeta_grid = np.linspace(0, 2 * np.pi, 400)
    for _ in range(n_lines):
        psi = delta * rng.uniform(0.1, 1.0)
        alpha = rng.uniform(0, 2 * np.pi)
        u, v = eq.uv_from_psi_alpha(psi, alpha)
        x0 = eq.position(u, v, 0.0)
        sol = trace_field_line(eq.B, x0, 2 * np.pi / eq.Omega)   # one full turn
        num = sol.sol(zeta_grid / eq.Omega)
        exact = eq.position(u, v, zeta_grid)
        worst_pos = max(worst_pos, np.abs(num - exact).max())
        worst_closure = max(worst_closure, np.linalg.norm(sol.y[:, -1] - x0))
        psis = np.array([eq.psi(num[:, k]) for k in range(0, 400, 20)])
        worst_psi_drift = max(worst_psi_drift, np.abs(psis - psi).max() / psi)
    ok = worst_pos < 1e-8 and worst_closure < 1e-8 and worst_psi_drift < 1e-8
    return _report(
        "6  traced line = exact line", ok,
        f"max position error {worst_pos:.1e}; closure after 2pi {worst_closure:.1e}; "
        f"psi drift (rel) {worst_psi_drift:.1e}")


def test_iota(eq, delta, rng):
    """Test 7: measured rotational transform is 2 on every surface.

    Follow lines for several toroidal turns, compute theta(phi), take the
    slope.  [L sec 2.4] proves iota = 2 exactly, for all eps, all psi.
    """
    axis = AxisInterpolator(eq.axis)
    iotas = []
    for frac in (1 / 9, 4 / 9, 1.0):          # the surfaces drawn in the papers
        psi = delta * frac
        for alpha in rng.uniform(0, 2 * np.pi, 3):
            u, v = eq.uv_from_psi_alpha(psi, alpha)
            x0 = eq.position(u, v, 0.0)
            n_turns = 3
            sol = trace_field_line(eq.B, x0, n_turns * 2 * np.pi / eq.Omega)
            iota, _, _ = rotational_transform(sol, n_turns * 2 * np.pi / eq.Omega,
                                              axis)
            iotas.append(iota)
    iotas = np.array(iotas)
    err = np.abs(iotas - 2.0).max()
    return _report("7  iota = 2", err < 1e-3,
                   f"measured iota in [{iotas.min():.6f}, {iotas.max():.6f}]")


def run_validation(eq, delta, seed=0):
    """Run all tests; return True if everything passed."""
    rng = np.random.default_rng(seed)
    pts = random_interior_points(eq, delta, 20, rng)
    # (u, v, zeta) triples for the map tests
    uvz = []
    for _ in range(20):
        psi = delta * rng.uniform(0.05, 1.0)
        u, v = eq.uv_from_psi_alpha(psi, rng.uniform(0, 2 * np.pi))
        uvz.append((u, v, rng.uniform(0, 2 * np.pi)))

    print(f"\nValidation: eps={eq.eps}, tau0={eq.tau0}, "
          f"boundary psi=delta={delta:.5f} (max allowed {eq.delta_max:.5f})\n")
    results = [
        test_divergence_free(eq, pts),
        test_force_balance(eq, pts),
        test_jacobian_constant(eq, uvz),
        test_exact_map_is_tangent_to_B(eq, uvz),
        test_psi_formula(eq, delta, rng),
        test_tracer_vs_exact(eq, delta, rng),
        test_iota(eq, delta, rng),
    ]
    print(f"\n{sum(results)}/{len(results)} tests passed.")
    return all(results)


# =============================================================================
# 5. PLOTS
# =============================================================================


def make_plots(eq, delta, outdir):
    import matplotlib
    matplotlib.use("Agg")                 # no display needed
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    tag = f"eps{eq.eps}_tau{eq.tau0}"
    rng = np.random.default_rng(1)

    # ---- Figure 1: 3D picture: axis, boundary surface, one field line -------
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    # Boundary flux surface from the explicit parametrization [L eq 2.16]:
    # the circle |(u,v)-axis| = sqrt(delta) swept along zeta.
    alphas = np.linspace(0, 2 * np.pi, 40)
    zetas = np.linspace(0, 2 * np.pi, 120)
    A, Z = np.meshgrid(alphas, zetas)
    P = np.zeros((3,) + A.shape)
    for i, al in enumerate(alphas):
        u, v = eq.uv_from_psi_alpha(delta, al)
        P[:, :, i] = eq.position(u, v, zetas)
    ax.plot_surface(P[0], P[1], P[2], alpha=0.25, color="tab:blue", linewidth=0)
    axis_c = eq.axis(zetas)
    ax.plot(*axis_c, color="k", lw=2, label="magnetic axis")
    u, v = eq.uv_from_psi_alpha(delta, 0.7)
    line = eq.position(u, v, zetas)
    ax.plot(*line, color="tab:red", lw=2, label="one (closed) field line")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Boundary surface psi={delta:.4f}  (eps={eq.eps}, tau0={eq.tau0})")
    ax.legend()
    fig.savefig(os.path.join(outdir, f"fig1_3d_{tag}.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: Poincare cross-sections at three toroidal angles ---------
    # Each closed line pierces each phi-plane exactly once per toroidal turn,
    # so we need MANY lines per surface to see the curve.  We TRACE them
    # numerically with the generic tracer (not the exact map) -- the plot is
    # therefore a genuine end-to-end test of poincare_points().
    phis = [0.0, np.pi / 4, np.pi / 2]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax2, phi0 in zip(axes, phis):
        for frac, col in zip((1 / 9, 4 / 9, 1.0), ("tab:green", "tab:orange", "tab:blue")):
            Rs, Zs = [], []
            for alpha in np.linspace(0, 2 * np.pi, 60, endpoint=False):
                u, v = eq.uv_from_psi_alpha(delta * frac, alpha)
                x0 = eq.position(u, v, 0.0)
                pts = poincare_points(eq.B, x0, phi0, 1, 1.05 * 2 * np.pi / eq.Omega)
                if len(pts):
                    Rs.append(np.hypot(pts[0, 0], pts[0, 1]))
                    Zs.append(pts[0, 2])
            ax2.plot(Rs, Zs, ".", ms=4, color=col, label=f"psi = {frac:.2f} delta")
        # mark the axis crossing at this phi
        zeta_grid = np.linspace(0, 2 * np.pi, 20001)
        axc = eq.axis(zeta_grid)
        ph = np.arctan2(axc[1], axc[0])
        k = np.argmin(np.abs(np.angle(np.exp(1j * (ph - phi0)))))
        ax2.plot(np.hypot(axc[0][k], axc[1][k]), axc[2][k], "k+", ms=10, label="axis")
        ax2.set_title(f"phi = {phi0:.3f} rad")
        ax2.set_xlabel("R"); ax2.set_aspect("equal")
    axes[0].set_ylabel("Z"); axes[0].legend(fontsize=8)
    fig.suptitle("Poincare cross-sections from the NUMERICAL tracer "
                 f"(eps={eq.eps}, tau0={eq.tau0})")
    fig.savefig(os.path.join(outdir, f"fig2_poincare_{tag}.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: tracer error vs exact solution, and theta(phi) -----------
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4))
    axis = AxisInterpolator(eq.axis)
    for alpha, col in zip((0.3, 2.0, 4.1), ("tab:red", "tab:green", "tab:blue")):
        u, v = eq.uv_from_psi_alpha(delta, alpha)
        x0 = eq.position(u, v, 0.0)
        n_turns = 5
        T = n_turns * 2 * np.pi / eq.Omega
        sol = trace_field_line(eq.B, x0, T)
        t = np.linspace(0, T, 2000)
        err = np.linalg.norm(sol.sol(t) - eq.position(u, v, t * eq.Omega), axis=0)
        axl.semilogy(t / (2 * np.pi / eq.Omega), err + 1e-17, color=col)
        iota, phi, theta = rotational_transform(sol, T, axis)
        axr.plot(phi / (2 * np.pi), theta / (2 * np.pi), color=col,
                 label=f"alpha={alpha}: iota={iota:.5f}")
    axl.set_xlabel("toroidal turns"); axl.set_ylabel("|x_num - x_exact|")
    axl.set_title("Integrator error (5 turns, rtol=1e-11)")
    axr.set_xlabel("phi / 2pi (toroidal turns)"); axr.set_ylabel("theta / 2pi (poloidal turns)")
    axr.set_title("Poloidal vs toroidal winding: slope = iota")
    axr.legend(fontsize=8)
    fig.savefig(os.path.join(outdir, f"fig3_validation_{tag}.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Figures written to {outdir}/")


# =============================================================================
# 6. MAIN
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("WHY")[0])
    ap.add_argument("--eps", type=float, default=0.5, help="elongation, in (0,1)")
    ap.add_argument("--tau0", type=float, default=0.0,
                    help="symmetry-breaking parameter, in (-1,1); 0 = symmetric")
    ap.add_argument("--delta", type=float, default=1 / 64,
                    help="boundary flux surface psi (default 1/64, as in the papers)")
    ap.add_argument("--outdir", default="fieldline_figures")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    eq = AnalyticEquilibrium(args.eps, args.tau0)
    if not (0 < args.delta < eq.delta_max):
        sys.exit(f"delta must be in (0, {eq.delta_max:.5f}) for these parameters")

    ok = run_validation(eq, args.delta)
    if not args.no_plots:
        make_plots(eq, args.delta, args.outdir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
