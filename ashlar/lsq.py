"""Weighted (and optionally robust) least-squares solvers for tile positions.

Both the within-cycle stitcher (``EdgeAligner``) and the cross-cycle registration
(``LayerAligner``) reduce their alignment measurements to a sparse weighted
least-squares problem of the same form

    minimize over positions p:   sum_r  w_r * || (A p)_r - b_r ||^2

where each constraint (row of the signed incidence matrix ``A``) is either

  * binary  -- a relative measurement ``p_j - p_i = b_r`` (an overlap edge), or
  * unary   -- an absolute measurement ``p_i = b_r`` (a cross-cycle anchor).

This module holds the shared linear algebra so the two callers don't duplicate
it. The gauge (the otherwise-free global translation per connected component) is
fixed either by pinning one node to zero (``pin``), by a small Tikhonov ridge
(``ridge``), or implicitly by the presence of unary anchor rows.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


# Robust IRLS defaults: iteration cap and Huber tuning constant.
ROBUST_ITERS = 10
ROBUST_C = 1.345

# Components larger than this switch from a direct sparse solve to conjugate
# gradient to keep the cost manageable.
_CG_THRESHOLD = 5000


def confidence_weight(error, scheme="inv-encc", eps=1e-9):
    """Map a stored alignment ``error`` to a least-squares weight.

    The cached ``error`` is ``utils.nccw`` = ``-log(NCC)``: non-negative, lower
    meaning more confident. ``inv-encc`` (default) weights by
    ``1 / max(error, eps)``; ``ncc`` recovers the normalized cross-correlation
    ``exp(-error)``; ``uniform`` weights every constraint equally. Inverse
    measurement variance is the principled ideal weight; ENCC and NCC are
    practical proxies for it.
    """
    if scheme == "uniform":
        return 1.0
    elif scheme == "ncc":
        return float(np.exp(-error))
    elif scheme == "inv-encc":
        return 1.0 / max(error, eps)
    raise ValueError("unknown weight scheme %r" % scheme)


def _spsolve(L, b):
    # Direct sparse solve is reliable for typical component sizes; fall back to
    # conjugate gradient for very large components.
    L = scipy.sparse.csc_matrix(L)
    if L.shape[0] > _CG_THRESHOLD:
        return scipy.sparse.linalg.cg(L, b)[0]
    return scipy.sparse.linalg.spsolve(L, b)


def _single_solve(A, b, weights, pin, ridge):
    """One weighted least-squares solve. Returns an ``(n, b.shape[1])`` array."""
    n = A.shape[1]
    ncol = b.shape[1]
    W = scipy.sparse.diags(weights)
    L = (A.T @ W @ A).tocsr()
    rhs = A.T @ (weights[:, None] * b)
    sol = np.zeros((n, ncol))
    if ridge:
        L = L + ridge * scipy.sparse.identity(n, format="csr")
    if ridge or pin is None:
        # Full-rank system (ridge or anchor rows fix the gauge).
        for axis in range(ncol):
            sol[:, axis] = _spsolve(L, rhs[:, axis])
    else:
        # Pin one node to zero, drop its row/column, solve the reduced system.
        keep = np.flatnonzero(np.arange(n) != pin)
        L_red = L[keep][:, keep]
        for axis in range(ncol):
            sol[keep, axis] = _spsolve(L_red, rhs[keep, axis])
        # The pinned node keeps value 0.
    return sol


def _leverages(A, weights, pin, ridge):
    """Per-row leverages ``h_r = w_r * a_r^T M^{-1} a_r`` (hat-matrix diagonal).

    ``M`` is the gauge-fixed normal matrix. Used to studentize residuals so a
    high-leverage (high-confidence) constraint cannot mask its own large
    residual. Computed by solving ``M X = A_red^T`` in column chunks to bound
    memory; only invoked in the robust path.
    """
    n = A.shape[1]
    m = A.shape[0]
    W = scipy.sparse.diags(weights)
    M = (A.T @ W @ A).tocsc()
    if ridge:
        M = (M + ridge * scipy.sparse.identity(n, format="csc")).tocsc()
        keep = np.arange(n)
    elif pin is None:
        keep = np.arange(n)
    else:
        keep = np.flatnonzero(np.arange(n) != pin)
        M = M[keep][:, keep].tocsc()
    A_red = A[:, keep]
    At = A_red.T.tocsc()
    A_red = A_red.tocsr()
    lu = scipy.sparse.linalg.splu(M)
    h = np.empty(m)
    chunk = 512
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        x = lu.solve(At[:, start:stop].toarray())  # M^{-1} a_r per row
        block = A_red[start:stop]
        h[start:stop] = np.asarray(block.multiply(x.T).sum(axis=1)).ravel()
    return np.clip(weights * h, 0.0, 1.0 - 1e-9)


def solve_weighted_lsq(A, b, weights, *, pin=None, ridge=0.0, robust=False,
                       robust_iters=ROBUST_ITERS, robust_c=ROBUST_C):
    """Solve ``min_p sum_r w_r ||(A p)_r - b_r||^2`` for positions ``p``.

    Parameters
    ----------
    A : sparse (m, n)
        Signed incidence matrix; each row is one constraint.
    b : array (m, k)
        Right-hand side targets (k columns solved independently, e.g. x and y).
    weights : array (m,)
        Per-constraint base weights.
    pin : int or None
        If given (and ``ridge`` is 0), fix node ``pin`` to zero to remove the
        gauge freedom. Use for an overlap-only system with no absolute anchors.
    ridge : float
        Tikhonov term ``+ ridge * ||p||^2`` (adds ``ridge * I`` to the normal
        matrix). Fixes the gauge toward zero; keep small.
    robust : bool
        If True, run robust IRLS with a Huber influence function on
        leverage-corrected (studentized) residuals, downweighting constraints
        whose residual is large relative to a median-based scale. Because
        residuals are ~0 on a consistent component, all factors stay 1 and the
        result matches the non-robust solve there.

    Returns
    -------
    array (n, k) of solved positions.
    """
    A = scipy.sparse.csr_matrix(A)
    b = np.asarray(b, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = A.shape[0]

    if not robust:
        return _single_solve(A, b, weights, pin, ridge)

    factor = np.ones(m)
    sol = _single_solve(A, b, weights, pin, ridge)
    for _ in range(robust_iters):
        w = weights * factor
        resid = np.linalg.norm(np.asarray(A @ sol) - b, axis=1)
        leverage = _leverages(A, w, pin, ridge)
        rstud = np.sqrt(w) * resid / np.sqrt(np.maximum(1.0 - leverage, 1e-9))
        scale = 1.4826 * np.median(rstud)
        if scale <= 1e-9:
            break  # residuals consistent; nothing to downweight
        delta = robust_c * scale
        new_factor = np.where(
            rstud <= delta, 1.0, delta / np.maximum(rstud, 1e-9))
        if np.allclose(new_factor, factor, atol=1e-3):
            break
        factor = new_factor
        sol = _single_solve(A, b, weights * factor, pin, ridge)
    return sol
