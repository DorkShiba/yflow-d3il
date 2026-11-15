"""User-approved 7-joint analogue of methods 1/2/3/5; never modifies gripper directly."""

import time, math
import numpy as np
import torch
from scipy import sparse
from .qp import solve_qp as _primary_qp
from .rtc import soft_mask


class JointRTCReference:
    """Unexecuted joint deltas of the current H=16, execute=8 plan, per episode."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.actions = None
        self.index = 0

    def install(self, plan):
        self.actions = np.asarray(plan, dtype=np.float64)[:, :7].copy()
        self.index = 0

    def executed(self, invalidated=False):
        self.index += 1
        if invalidated:
            self.clear()

    def suffix(self):
        if self.actions is None:
            return None
        value = self.actions[self.index : self.index + 8]
        return value.copy() if len(value) == 8 else None


def joint_rtc_velocity(p, x, t, suffix, tau, beta):
    """Main RTC VJP/soft-mask rule for seven joint deltas, with gripper held free.

    The network stays float32; constraints and latent bookkeeping stay float64.
    RTC adds no direct gripper correction and never changes the first observation.
    Future observation channels can receive a VJP, as in the main RTC method.
    """
    with torch.enable_grad():
        source = x.detach().to(p.center.dtype).requires_grad_(True)
        v = p.model(source, t)
        free = torch.ones_like(v)
        free[:, 0, 8:] = 0
        v = v * free
        target = torch.zeros_like(source[..., :7])
        target[:, :8] = (
            torch.as_tensor(suffix, dtype=source.dtype, device=p.device) - p.center[:7]
        ) / p.scale[:7]
        mask = torch.as_tensor(soft_mask(16, 8, 0), dtype=source.dtype, device=p.device)
        clean = (source + (1 - tau) * v)[..., :7]
        error = ((target - clean) * mask[None, :, None]).detach()
        gradient = torch.autograd.grad(clean, source, grad_outputs=error)[0]
        gradient = gradient * free
        gradient[:, :, 7] = 0
        strength = (
            beta
            if tau == 0
            else min(beta, (tau * tau + (1 - tau) ** 2) / (tau * (1 - tau)))
        )
        return (v + strength * gradient).detach().to(x.dtype)


def solve_qp(nominal, G, d, settings):
    """Identical minimum-distance QP; strict residual acceptance for both solvers."""
    started = time.perf_counter()
    r = _primary_qp(nominal, G, d, settings)
    if r["status"] not in ["solver_failure", "constraint_violation"]:
        return r
    import clarabel

    nominal, G, d = [np.asarray(v, dtype=np.float64) for v in (nominal, G, d)]
    primary = {k: v for k, v in r.items() if k != "value"}
    opts = clarabel.DefaultSettings()
    opts.verbose = False
    opts.tol_gap_abs = opts.tol_gap_rel = opts.tol_feas = 1e-10
    result = clarabel.DefaultSolver(
        sparse.eye(len(nominal), format="csc"),
        -nominal,
        sparse.csc_matrix(G),
        d,
        [clarabel.NonnegativeConeT(len(d))],
        opts,
    ).solve()
    status = str(result.status)
    r.update(
        primary_solver=primary,
        solver_status="clarabel:" + status,
        iterations=int(result.iterations),
        value=None,
    )
    if (
        status in ["Solved", "AlmostSolved"]
        and result.x is not None
        and np.isfinite(result.x).all()
    ):
        value = np.asarray(result.x)
        res = float(np.maximum(G @ value - d, 0).max(initial=0.0))
        r.update(max_residual=res)
        if res <= settings["feasibility_tolerance_m"]:
            r.update(
                status="ok",
                value=value,
                intervention_norm=float(np.linalg.norm(value - nominal)),
            )
        else:
            r["status"] = "constraint_violation"
    elif "Infeasible" in status:
        r["status"] = "infeasible"
    else:
        r["status"] = "solver_failure"
    r["latency_ms"] = 1000 * (time.perf_counter() - started)
    return r


class JointConstraints:

    def __init__(self, q0, previous, horizon, settings):
        self.q0 = np.asarray(q0, dtype=np.float64)
        self.previous = np.asarray(previous, dtype=np.float64)
        self.H = horizon
        self.settings = settings
        if (
            self.q0.shape != (7,)
            or self.previous.shape != (7,)
            or (not np.isfinite(np.r_[self.q0, self.previous]).all())
        ):
            raise ValueError("invalid joint state")
        self.L = np.kron(np.tril(np.ones((horizon, horizon))), np.eye(7))
        diff = np.eye(horizon)
        diff[np.arange(1, horizon), np.arange(horizon - 1)] = -1
        self.D = np.kron(diff, np.eye(7))
        low = np.tile(np.asarray(settings["joint_low_rad"]) - self.q0, horizon)
        high = np.tile(np.asarray(settings["joint_high_rad"]) - self.q0, horizon)
        a = np.tile(
            np.asarray(settings["joint_acceleration_limit_rad_s2"])
            * settings["dt_s"] ** 2,
            horizon,
        )
        last = np.r_[self.previous, np.zeros(7 * (horizon - 1))]
        self.G = np.vstack([self.L, -self.L, self.D, -self.D])
        self.d = np.r_[high, -low, a + last, a - last]

    def residual(self, actions):
        return float(
            np.maximum(self.G @ np.asarray(actions).reshape(-1) - self.d, 0).max(
                initial=0.0
            )
        )


def joint_lipo(actions, history, bound, settings, G, d):
    """Same minimum-third-difference QP as main LiPo, generalized from 2 to 7 axes."""
    import clarabel

    started = time.perf_counter()
    a = np.asarray(actions, dtype=np.float64)
    hist = np.asarray(history, dtype=np.float64).reshape(-1, 7)[-3:]
    H = len(a)
    n = H * 7
    L = np.kron(np.tril(np.ones((H, H))), np.eye(7))
    D3 = np.diff(np.eye(len(hist) + H), n=3, axis=0)
    Dh = np.kron(D3[:, : len(hist)], np.eye(7))
    Dp = np.kron(D3[:, len(hist) :], np.eye(7))
    unit = 0.001
    M = Dp @ L / unit
    c = Dh @ (hist - hist[-1]).reshape(-1) / unit
    av = a.reshape(-1)
    keep = d - G @ av <= np.abs(G).sum(1) * 2 * bound + 1e-12
    A = np.vstack([L, -L, G[keep]]) / unit
    b = np.r_[np.full(n, bound) + L @ av, np.full(n, bound) - L @ av, d[keep]] / unit
    opts = clarabel.DefaultSettings()
    opts.verbose = False
    opts.tol_gap_abs = opts.tol_gap_rel = opts.tol_feas = 1e-09
    result = clarabel.DefaultSolver(
        sparse.triu(sparse.csc_matrix(2 * M.T @ M), format="csc"),
        2 * M.T @ c,
        sparse.csc_matrix(A),
        b,
        [clarabel.NonnegativeConeT(len(b))],
        opts,
    ).solve()
    status = str(result.status)
    report = dict(
        status="solver_failure",
        solver_status=status,
        correction_max_rad=None,
        max_residual_rad=None,
        latency_ms=0.0,
        value=None,
    )
    if status in ["Solved", "AlmostSolved"] and np.isfinite(result.x).all():
        correction = (L @ (np.asarray(result.x) - av)).reshape(H, 7)
        if np.abs(correction).max() > bound + 1e-08:
            report["status"] = "lipo_bound_violation"
        else:
            correction = np.clip(correction, -bound, bound)
            value = av + np.diff(np.vstack([np.zeros(7), correction]), axis=0).reshape(
                -1
            )
            residual = float(np.maximum(G @ value - d, 0).max(initial=0.0))
            report.update(
                max_residual_rad=residual,
                correction_max_rad=float(np.abs(correction).max()),
            )
            if residual <= settings["feasibility_tolerance_m"]:
                report.update(
                    status="ok",
                    value=value.reshape(H, 7),
                    jerk_cost_before_rad2=float(np.sum((M @ av + c) ** 2) * unit**2),
                    jerk_cost_after_rad2=float(np.sum((M @ value + c) ** 2) * unit**2),
                )
            else:
                report["status"] = "constraint_violation"
    elif "Infeasible" in status:
        report["status"] = "lipo_infeasible"
    report["latency_ms"] = 1000 * (time.perf_counter() - started)
    return report


class JointPotential:
    """7-D train-normalized KDE plus distance to high-density training anchors.
    The original 2-D Cartesian raster is replaced by train-only anchors in 7-D.
    """

    def __init__(self, cache, device):
        with np.load(cache) as z:
            self.support = torch.as_tensor(
                z["support"], dtype=torch.float64, device=device
            )
            self.safe = torch.as_tensor(
                z["safe_anchors"], dtype=torch.float64, device=device
            )
            self.center = torch.as_tensor(
                z["joint_center"], dtype=torch.float64, device=device
            )
            self.scale = torch.as_tensor(
                z["joint_scale"], dtype=torch.float64, device=device
            )
            self.bandwidth = float(z["bandwidth"])
            self.alpha = -1.0

    @torch.no_grad()
    def evaluate(self, raw):
        shape = raw.shape
        points = (raw.reshape(-1, 7).double() - self.center) / self.scale
        phi = []
        grads = []
        for p in points.split(128):
            diff = p[:, None, :] - self.support[None]
            logits = -diff.square().sum(-1) / (2 * self.bandwidth**2)
            logp = (
                torch.logsumexp(logits, -1)
                - math.log(len(self.support))
                - 3.5 * math.log(2 * math.pi * self.bandwidth**2)
            )
            score = -(logits.softmax(-1)[..., None] * diff).sum(1) / self.bandwidth**2
            dist2 = (p[:, None, :] - self.safe[None]).square().sum(-1)
            delta = p - self.safe[dist2.argmin(-1)]
            dist = torch.linalg.vector_norm(delta, dim=-1)
            phi.append(logp + self.alpha * dist)
            grads.append(
                (score + self.alpha * delta / dist.clamp_min(1e-15)[:, None])
                / self.scale
            )
        return (torch.cat(phi).reshape(shape[:-1]), torch.cat(grads).reshape(shape))

    @torch.no_grad()
    def guidance(self, x, center, scale):
        raw = x[:, 1:, 8:15].double() * scale[8:15].double() + center[8:15].double()
        phi, grad = self.evaluate(raw)
        g = torch.zeros_like(x)
        g[:, 1:, 8:15] = (grad * scale[8:15].double()).to(x.dtype)
        assert (
            g[:, :, :8].count_nonzero() == 0
            and g[:, 0, 8:].count_nonzero() == 0
            and (g[:, :, 15:].count_nonzero() == 0)
        )
        return (
            g,
            dict(energy=float(-phi.sum()), gradient_norm=float(g.double().norm())),
        )


class JointPolicy:

    def __init__(self, base, settings, potential, enabled):
        self.base = base
        self.settings = settings
        self.potential = potential
        self.enabled = enabled
        self.last_report = {}

    @torch.no_grad()
    def plan(self, obs, previous, history, noise, rtc_suffix=None):
        p = self.base
        s = self.settings
        started = time.perf_counter()
        r = dict(
            status="ok",
            failure_stage=None,
            qp=[],
            lipo=None,
            pf_calls=0,
            model_calls=0,
            projection_calls=0,
            latency_ms=0.0,
            stages={},
            rtc_enabled=bool(self.enabled and s.get("rtc", {}).get("enabled", False)),
            rtc_guided=False,
            rtc_calls=0,
        )
        self.last_report = r
        use_rtc = r["rtc_enabled"] and rtc_suffix is not None
        beta = s.get("rtc", {}).get("beta", 5.0)
        if use_rtc:
            rtc_suffix = np.asarray(rtc_suffix, dtype=np.float64)
            if (
                rtc_suffix.shape != (8, 7)
                or not np.isfinite(rtc_suffix).all()
                or not np.isfinite(beta)
                or beta < 0
            ):
                r.update(status="invalid_input", failure_stage="rtc")
                return None
            r["rtc_guided"] = beta > 0
        observation = torch.as_tensor(obs, dtype=p.center.dtype, device=p.device)[None]
        cond = {0: (observation - p.center[8:]) / p.scale[8:]}
        x = p.apply_conditioning(noise.clone().to(p.device), cond, 8)
        center = p.center.double() if self.enabled else p.center
        scale = p.scale.double() if self.enabled else p.scale
        if self.enabled:
            x = x.double()
            cond = {0: cond[0].double()}
        anchor = np.asarray(history, dtype=np.float64)[-1]
        con = JointConstraints(anchor, previous, 16, s)
        r["constraint_anchor_rad"] = anchor.tolist()
        r["latent_dtype"] = str(x.dtype)
        times = torch.linspace(0, 1, 11, device=p.device)[:-1]
        dt = times[1] - times[0]
        for k, t in enumerate(times):
            if use_rtc and beta > 0:
                velocity = joint_rtc_velocity(p, x, t, rtc_suffix, k / 10, beta)
                r["rtc_calls"] += 1
            else:
                velocity = p.model(x.to(p.center.dtype), t).to(x.dtype)
            velocity[:, 0, 8:] = 0
            r["model_calls"] += 1
            if self.enabled:
                guidance, pfr = self.potential.guidance(x, center, scale)
                r["pf_calls"] += 1
                velocity = velocity + s["pf_weight"] * guidance
            x = x + dt * velocity
            if self.enabled:
                raw = (x * scale + center)[0]
                gripper_before = raw[:, 7].clone()
                solved = solve_qp(
                    raw[:, :7].double().cpu().numpy().reshape(-1),
                    con.G,
                    con.d,
                    s["solver"],
                )
                value = solved.pop("value")
                r["qp"].append(dict(step=k, **solved))
                r["projection_calls"] += 1
                if value is None:
                    r.update(status=solved["status"], failure_stage="projection")
                    x = None
                    break
                x = x.clone()
                x[0, :, :7] = (
                    torch.as_tensor(
                        value.reshape(16, 7), dtype=x.dtype, device=p.device
                    )
                    - center[:7]
                ) / scale[:7]
                assert torch.equal((x * scale + center)[0, :, 7], gripper_before)
                residual = con.residual(
                    (x * scale + center)[0, :, :7].double().cpu().numpy()
                )
                if residual > s["solver"]["feasibility_tolerance_m"]:
                    r.update(
                        status="constraint_violation",
                        failure_stage="projection_residual",
                    )
                    x = None
                    break
            x = p.apply_conditioning(x, cond, 8).detach()
        if x is not None:
            raw = (x * scale + center)[0].cpu().numpy()
            r["stages"]["predict_joint_actions_rad"] = raw[:, :7].tolist()
            r["stages"]["predict_gripper"] = raw[:, 7].tolist()
            if self.enabled:
                solved = joint_lipo(
                    raw[:, :7], history, s["lipo_bound_rad"], s["solver"], con.G, con.d
                )
                value = solved.pop("value")
                r["lipo"] = solved
                if value is None:
                    r.update(status=solved["status"], failure_stage="lipo")
                    x = None
                else:
                    grip = x[0, :, 7].clone()
                    x = x.clone()
                    x[0, :, :7] = (
                        torch.as_tensor(value, dtype=x.dtype, device=p.device)
                        - center[:7]
                    ) / scale[:7]
                    assert torch.equal(grip, x[0, :, 7])
                    r["stages"]["lipo_joint_actions_rad"] = (
                        (x * scale + center)[0, :, :7].cpu().numpy().tolist()
                    )
        if x is not None:
            result = (x * scale + center)[0].cpu().numpy()
            residual = con.residual(result[:, :7])
            r["final_residual_rad"] = residual
            r["stages"]["final_actions"] = result[:, :8].tolist()
            if not np.isfinite(result).all():
                r.update(status="invalid_policy_output", failure_stage="final")
                result = None
            elif self.enabled and residual > s["solver"]["feasibility_tolerance_m"]:
                r.update(status="constraint_violation", failure_stage="final")
                result = None
        else:
            result = None
        r["latency_ms"] = 1000 * (time.perf_counter() - started)
        return result
