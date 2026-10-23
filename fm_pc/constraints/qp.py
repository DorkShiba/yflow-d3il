"""Exact Euclidean projection for box constraints on actions and desired-hand positions.

For one plan axis (H steps) the projection is the QP

    min_a  1/2 ||a - a0||^2   s.t.  alo <= a <= ahi,   slo <= L a <= shi,

with L the lower-triangular all-ones matrix (L a = desired-hand offset from
des_0).  Both constraint blocks are boxes, so stacking A = [I; L] gives a single
box-constrained QP solved by ADMM with the splitting z = A a.  Columns (batch x
axis) are independent and solved together.

Pure numpy / float64 on purpose: the problem is tiny (16 x 2), and running it as
thousands of batch-1 GPU kernels is launch-bound and far slower.
rho = 0.3 was chosen by sweeping against an SLSQP reference (max |a - a_ref| ~ 4e-12,
~300 iterations on heavily violating inputs).
"""
import numpy as np


class BoxQP:
    def __init__(self, H, use_a, use_s, rho=0.3, max_iter=5000, tol=1e-11):
        assert use_a or use_s
        self.H, self.use_a, self.use_s = H, use_a, use_s
        self.rho, self.max_iter, self.tol = float(rho), int(max_iter), float(tol)
        self.L = np.tril(np.ones((H, H)))
        rows = ([np.eye(H)] if use_a else []) + ([self.L] if use_s else [])
        self.A = np.vstack(rows)
        self.M = np.linalg.inv(np.eye(H) + self.rho * self.A.T @ self.A)
        self.last = {}

    def solve(self, a0, alo=None, ahi=None, slo=None, shi=None):
        """All arrays (H, N).  Returns a feasible a (H, N)."""
        if self.use_a and not self.use_s:                      # closed form
            self.last = {"iters": 0, "r_prim": 0.0}
            return np.clip(a0, alo, ahi)
        lo = np.vstack(([alo] if self.use_a else []) + [slo])
        hi = np.vstack(([ahi] if self.use_a else []) + [shi])
        A, M, rho = self.A, self.M, self.rho
        a = a0.copy()
        z = np.clip(A @ a, lo, hi)
        u = np.zeros_like(z)
        r = s = np.inf
        for it in range(1, self.max_iter + 1):
            a = M @ (a0 + rho * (A.T @ (z - u)))
            Aa = A @ a
            z_old = z
            z = np.clip(Aa + u, lo, hi)
            u += Aa - z
            r = np.abs(Aa - z).max()
            s = rho * np.abs(A.T @ (z - z_old)).max()
            if r < self.tol and s < self.tol:
                break
        self.last = {"iters": it, "r_prim": float(r), "r_dual": float(s)}
        if not self.use_a:
            zs = z                                             # L a = z exactly
            return np.diff(zs, axis=0, prepend=np.zeros_like(zs[:1]))
        return np.clip(a, alo, ahi)                           # residual on L a is ~tol
