"""Log-variable integrator for the hierarchical background SDE (Eq. 2) and the return SDE.

Contract
--------
``dv_i = -gamma_i (v_i - v_{i-1}) dt + kappa_i v_i dW_i``  with ``beta_i = 2 gamma_i/kappa_i^2``
identical at every level.  Integration is done in ``y_i = ln v_i``::

    dy_i = [ gamma_i (v_{i-1} e^{-y_i} - 1) - kappa_i^2 / 2 ] dt + kappa_i dW_i

so ``v = exp(y) > 0`` *structurally*: no ``abs(v)``, no clipping, no silent truncation.

With the trading clock ``a_chi(t)`` both the drift and the Ito correction pick up ``a`` and
the diffusion picks up ``sqrt(a)``::

    dy_i = [ a gamma_i (v_{i-1} e^{-y_i} - 1) - a kappa_i^2/2 ] dt + sqrt(a) kappa_i dW_i

``a = 0`` freezes the level *exactly* (both terms vanish), which is what makes the
time-change statement of ``docs/theory/novelty_boundary.txt`` sec. 2 hold step by step.

Returns
-------
``ou_mode`` selects the noise amplitude of ``dx = -mu x dt + sqrt(c mu v) dW``:

    paper_literal        c = 1  ->  stationary conditional variance v/2  (Eq. 1 read literally)
    variance_consistent  c = 2  ->  stationary conditional variance v    (matches Eq. 3)

The x-update uses the *exact* OU transition given v frozen over the step,
``x' = e^{-mu dt} x + sqrt(c v (1 - e^{-2 mu dt})/2) Z``.  Its stationary conditional
variance is ``c v / 2`` for every dt, so an OU-variance mismatch cannot be blamed on the
discretisation.

Fault injection
---------------
``integrator="plain_euler_v"`` integrates v directly (no log transform) with an explicit
Euler step.  It exists **only** as the labelled control of W5 and never produces a physical
result.  It does not clamp or take absolute values: negative and non-finite values are
counted, kept, and reported.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import numpy as np

from .. import status
from . import clock as clock_mod

LOG_ABS_MAX = 700.0            # |y| beyond this would overflow exp()

OU_MODES = {"paper_literal": 1.0, "variance_consistent": 2.0}
INTEGRATORS = ("log_euler", "plain_euler_v")
PATTERNS = ("continuous", "ashare")


@dataclass(slots=True)
class HierarchyConfig:
    """All knobs of one simulation.  Names follow ``configs/contracts/interfaces.json``."""

    N: int = 1
    beta: float = 1.0
    v0: float = 1.0
    r: float = 1.0
    gamma_N: float = 1.0
    mu_over_gamma_N: float = 10.0
    dt: float = 0.01
    t_total: float = 50.0
    burn_in: float = 10.0
    ou_mode: str = "variance_consistent"
    chi: float = 1.0
    g_gap: float = 0.0
    session_pattern: str = "continuous"
    n_paths: int = 1000
    n_record: int = 1000
    integrator: str = "log_euler"
    record_all_levels: bool = True
    clock_sub_steps: int = 12
    clock_n_days: int | None = None
    clock_gaps: bool = True

    # ---- derived -------------------------------------------------------------------------
    @property
    def gammas(self) -> tuple:
        """gamma_1 = gamma_N / r^(N-1), ..., gamma_N fixed at ``gamma_N``."""
        return tuple(self.gamma_N / self.r ** (self.N - 1 - k) for k in range(self.N))

    @property
    def kappas(self) -> tuple:
        return tuple(math.sqrt(2.0 * g / self.beta) for g in self.gammas)

    @property
    def mu(self) -> float:
        return self.mu_over_gamma_N * self.gamma_N

    @property
    def c_return(self) -> float:
        return OU_MODES[self.ou_mode]

    def validate(self) -> None:
        if self.N < 1:
            raise ValueError("N must be >= 1")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")
        if self.r <= 0:
            raise ValueError("r must be > 0")
        if self.gamma_N <= 0:
            raise ValueError("gamma_N must be > 0")
        if self.ou_mode not in OU_MODES:
            raise ValueError(f"ou_mode must be one of {sorted(OU_MODES)}")
        if self.integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}")
        if self.session_pattern not in PATTERNS:
            raise ValueError(f"session_pattern must be one of {PATTERNS}")
        if self.integrator == "log_euler" and self.dt <= 0:
            raise ValueError("dt must be > 0")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["gammas"] = list(self.gammas)
        d["kappas"] = list(self.kappas)
        d["mu"] = self.mu
        d["c_return"] = self.c_return
        return d


class _NoiseStream:
    """Block-buffered normal draws.  One RNG, one call per block, views per step."""

    __slots__ = ("_rng", "_buf", "_i", "_blk", "_shape")

    def __init__(self, rng, blk: int, shape: tuple):
        self._rng = rng
        self._blk = int(blk)
        self._shape = tuple(shape)
        self._buf = rng.standard_normal((self._blk,) + self._shape)
        self._i = 0

    def next(self) -> np.ndarray:
        if self._i >= self._blk:
            self._buf = self._rng.standard_normal((self._blk,) + self._shape)
            self._i = 0
        z = self._buf[self._i]
        self._i += 1
        return z


# ---------------------------------------------------------------------------------------
# continuous clock (uniform step grid, step-level recording)
# ---------------------------------------------------------------------------------------


def _integrate_uniform(cfg: HierarchyConfig, rng, dt: float, n_steps: int,
                       burn_step: int, rec_stride: int):
    P, N = int(cfg.n_paths), int(cfg.N)
    v0 = float(cfg.v0)
    log_v0 = math.log(v0)
    gammas, kappas = cfg.gammas, cfg.kappas
    c3 = np.array([k * math.sqrt(dt) for k in kappas])         # kappa sqrt(dt)
    c2 = np.array([0.5 * k * k * dt for k in kappas])          # kappa^2 dt / 2
    c1 = np.array([g * dt for g in gammas])                    # gamma dt
    mu, c_ret = cfg.mu, cfg.c_return

    z = np.zeros((P, N))
    x = np.zeros(P)
    dfrac = np.empty((P, N))
    ed = np.empty((P, N))
    u_old = np.empty((P, N))
    noise = _NoiseStream(rng, 64, (P, N + 1))

    n_rec = max(1, int(math.ceil((n_steps - burn_step) / rec_stride)))
    v_rec = np.empty((P, N, n_rec))
    x_rec = np.empty((P, n_rec))
    t_rec = np.empty(n_rec)

    check_every = max(1, n_steps // 4000)
    neg_every = max(1, n_steps // 400)
    n_check = n_neg_check = n_neg_probe = 0
    max_abs_dy = max_abs_y = 0.0
    n_overflow = n_nonfinite = 0
    i_rec = 0

    for k in range(n_steps):
        Z = noise.next()
        dfrac[:, 0] = log_v0 - z[:, 0]
        if N > 1:
            np.subtract(z[:, :-1], z[:, 1:], out=dfrac[:, 1:])
        np.exp(dfrac, out=ed)
        dz = c1 * ed - (c1 + c2) + c3 * Z[:, :N]
        z += dz

        v_top = np.exp(z[:, -1])
        f = math.exp(-mu * dt)
        sig = math.sqrt(c_ret * 0.5 * (1.0 - f * f))
        x *= f
        x += (sig * np.sqrt(v_top)) * Z[:, N]

        if k % check_every == 0:
            n_check += 1
            m = float(np.max(np.abs(dz)))
            if m > max_abs_dy:
                max_abs_dy = m
            ay = float(np.max(np.abs(z)))
            if ay > max_abs_y:
                max_abs_y = ay
            if not np.isfinite(ay):                                  # pragma: no cover
                n_nonfinite += 1
                z = np.nan_to_num(z, nan=0.0, posinf=LOG_ABS_MAX, neginf=-LOG_ABS_MAX)
            elif ay > LOG_ABS_MAX:
                n_overflow += 1
                np.clip(z, -LOG_ABS_MAX, LOG_ABS_MAX, out=z)

        if k % neg_every == 0:
            n_neg_check += 1
            u_old[:, 0] = v0
            if N > 1:
                np.exp(z[:, :-1], out=u_old[:, 1:])
            v_old = np.exp(z)
            v_eul = v_old * (1.0 - c1 + c3 * Z[:, :N]) + c1 * u_old
            n_neg_probe += int(np.count_nonzero(v_eul <= 0.0))

        if k >= burn_step and ((k - burn_step) % rec_stride == 0) and i_rec < n_rec:
            np.exp(z, out=v_rec[:, :, i_rec])
            x_rec[:, i_rec] = x
            t_rec[i_rec] = k * dt
            i_rec += 1

    neg_rate = n_neg_probe / max(1, n_neg_check * P * N)
    return dict(v_rec=v_rec, x_rec=x_rec, t_rec=t_rec, i_rec=i_rec,
                n_overflow=n_overflow, n_nonfinite=n_nonfinite,
                max_abs_dy=max_abs_dy, max_abs_y=max_abs_y,
                n_neg_probe=n_neg_probe, n_neg_check=n_neg_check,
                neg_rate=neg_rate, n_neg_est=neg_rate * n_steps * P * N,
                n_neg_actual=0, check_every=check_every, neg_every=neg_every)


# ---------------------------------------------------------------------------------------
# trading clock (irregular step grid, per-bar recording)
# ---------------------------------------------------------------------------------------


def _integrate_clock(cfg: HierarchyConfig, rng, clk: clock_mod.TradingClock):
    P, N = int(cfg.n_paths), int(cfg.N)
    v0 = float(cfg.v0)
    log_v0 = math.log(v0)
    gammas, kappas = cfg.gammas, cfg.kappas
    dt = clk.dt
    c3 = np.array([k * math.sqrt(dt) for k in kappas])
    c2 = np.array([0.5 * k * k * dt for k in kappas])
    c1 = np.array([g * dt for g in gammas])
    mu, c_ret = cfg.mu, cfg.c_return

    n_steps, n_bars = clk.n_steps, clk.n_bars
    a_arr = clk.a
    bar_of_step = clk.bar_of_step
    is_bar_end = clk.is_bar_end
    sqrt_cache = {a: math.sqrt(a) for a in np.unique(a_arr).tolist()}
    drift_cache = {a: (a * c1, a * (c1 + c2), sqrt_cache[a] * c3) for a in sqrt_cache}

    z = np.zeros((P, N))
    x = np.zeros(P)
    dfrac = np.empty((P, N))
    ed = np.empty((P, N))
    u_old = np.empty((P, N))
    noise = _NoiseStream(rng, 64, (P, N + 1))

    bar_dx = np.zeros((P, n_bars))
    bar_vN = np.zeros((P, n_bars))
    acc = np.zeros(P)

    check_every = max(1, n_steps // 4000)
    neg_every = max(1, n_steps // 400)
    n_check = n_neg_check = n_neg_probe = 0
    max_abs_dy = max_abs_y = 0.0
    n_overflow = n_nonfinite = 0

    for k in range(n_steps):
        a = float(a_arr[k])
        Z = noise.next()
        dfrac[:, 0] = log_v0 - z[:, 0]
        if N > 1:
            np.subtract(z[:, :-1], z[:, 1:], out=dfrac[:, 1:])
        np.exp(dfrac, out=ed)
        ac1, ac1c2, sac3 = drift_cache[a]
        dz = ac1 * ed - ac1c2 + sac3 * Z[:, :N]
        z += dz

        if a > 0.0:
            v_top = np.exp(z[:, -1])
            f = math.exp(-a * mu * dt)
            sig = math.sqrt(c_ret * 0.5 * (1.0 - f * f))
            dx = (f - 1.0) * x + (sig * np.sqrt(v_top)) * Z[:, N]
            x += dx
            b = bar_of_step[k]
            if b >= 0:
                acc += dx
                if is_bar_end[k]:
                    bar_dx[:, b] = acc
                    bar_vN[:, b] = v_top
                    acc[:] = 0.0

        if k % check_every == 0:
            n_check += 1
            m = float(np.max(np.abs(dz)))
            if m > max_abs_dy:
                max_abs_dy = m
            ay = float(np.max(np.abs(z)))
            if ay > max_abs_y:
                max_abs_y = ay
            if not np.isfinite(ay):                                  # pragma: no cover
                n_nonfinite += 1
                z = np.nan_to_num(z, nan=0.0, posinf=LOG_ABS_MAX, neginf=-LOG_ABS_MAX)
            elif ay > LOG_ABS_MAX:
                n_overflow += 1
                np.clip(z, -LOG_ABS_MAX, LOG_ABS_MAX, out=z)

        if k % neg_every == 0:
            n_neg_check += 1
            u_old[:, 0] = v0
            if N > 1:
                np.exp(z[:, :-1], out=u_old[:, 1:])
            v_old = np.exp(z)
            v_eul = v_old * (1.0 - a * c1 + math.sqrt(a) * c3 * Z[:, :N]) + a * c1 * u_old
            n_neg_probe += int(np.count_nonzero(v_eul <= 0.0))

    neg_rate = n_neg_probe / max(1, n_neg_check * P * N)
    burn_bar = int(np.searchsorted(clk.bar_day, clk.burn_days, side="left"))
    return dict(bar_dx=bar_dx, bar_vN=bar_vN, burn_bar=burn_bar,
                n_overflow=n_overflow, n_nonfinite=n_nonfinite,
                max_abs_dy=max_abs_dy, max_abs_y=max_abs_y,
                n_neg_probe=n_neg_probe, n_neg_check=n_neg_check,
                neg_rate=neg_rate, n_neg_est=neg_rate * n_steps * P * N,
                n_neg_actual=0, check_every=check_every, neg_every=neg_every)


# ---------------------------------------------------------------------------------------
# labelled fault injection: explicit Euler on v, no log transform (W5 control only)
# ---------------------------------------------------------------------------------------


def _integrate_plain_euler_v(cfg: HierarchyConfig, rng, dt: float, n_steps: int,
                             burn_step: int, rec_stride: int):
    P, N = int(cfg.n_paths), int(cfg.N)
    v0 = float(cfg.v0)
    gammas, kappas = cfg.gammas, cfg.kappas
    sq = np.array([k * math.sqrt(dt) for k in kappas])
    c1 = np.array([g * dt for g in gammas])
    noise = _NoiseStream(rng, 64, (P, N))

    v = np.full((P, N), v0)
    u_old = np.empty((P, N))
    n_rec = max(1, int(math.ceil((n_steps - burn_step) / rec_stride)))
    v_rec = np.empty((P, N, n_rec))
    n_nonpos = n_nonpos_steps = n_nonfinite = 0
    worst = 0.0
    i_rec = 0

    for k in range(n_steps):
        Z = noise.next()
        u_old[:, 0] = v0
        if N > 1:
            u_old[:, 1:] = v[:, :-1]
        v = v + (-c1 * (v - u_old)) + sq * v * Z
        bad = int(np.count_nonzero(v <= 0.0))
        if bad:
            n_nonpos += bad
            n_nonpos_steps += 1
            w = float(np.nanmin(v))
            if w < worst:
                worst = w
        if not np.all(np.isfinite(v)):
            n_nonfinite += 1
        if k >= burn_step and ((k - burn_step) % rec_stride == 0) and i_rec < n_rec:
            v_rec[:, :, i_rec] = v
            i_rec += 1

    return dict(v_rec=v_rec, i_rec=i_rec, n_nonpos=n_nonpos,
                n_nonpos_steps=n_nonpos_steps, n_nonfinite=n_nonfinite,
                worst_negative_value=worst, n_neg_actual=n_nonpos,
                n_overflow=0, max_abs_dy=float("nan"), max_abs_y=float("inf"),
                n_neg_probe=n_nonpos, n_neg_check=n_steps,
                neg_rate=n_nonpos / max(1, n_steps * P * N),
                n_neg_est=float(n_nonpos))


# ---------------------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------------------


def simulate_hierarchy(cfg: HierarchyConfig, seed: int) -> dict:
    """Run one simulation.  ``seed`` is the only source of randomness (reproducible)."""
    cfg.validate()
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))

    clk = None
    if cfg.session_pattern == "continuous":
        dt = float(cfg.dt)
        burn_step = int(round(cfg.burn_in / dt))
        n_steps = int(round((cfg.burn_in + cfg.t_total) / dt))
        n_post = n_steps - burn_step
        rec_stride = max(1, int(round(n_post / max(1, cfg.n_record))))
        if cfg.integrator == "log_euler":
            raw = _integrate_uniform(cfg, rng, dt, n_steps, burn_step, rec_stride)
        else:
            raw = _integrate_plain_euler_v(cfg, rng, dt, n_steps, burn_step, rec_stride)
        t_rec = raw.get("t_rec")
        rec_burn = burn_step
    else:
        if cfg.integrator != "log_euler":
            raise NotImplementedError("plain_euler_v is only wired for the continuous "
                                      "pattern; the W5 control runs there")
        clk = clock_mod.build_clock(cfg.g_gap, cfg.chi, sub_steps=cfg.clock_sub_steps,
                                    n_days=cfg.clock_n_days, gaps_on=cfg.clock_gaps)
        raw = _integrate_clock(cfg, rng, clk)
        dt = clk.dt
        t_rec = None
        rec_burn = clk.burn_step
        rec_stride = 1

    wall = time.perf_counter() - t0
    gammas, kappas = cfg.gammas, cfg.kappas

    if clk is None:
        v_rec = raw["v_rec"][:, :, :raw["i_rec"]]
        x_rec = raw.get("x_rec")
        if x_rec is not None:
            x_rec = x_rec[:, :raw["i_rec"]]
        n_samples = int(v_rec.shape[0] * v_rec.shape[2])
        n_steps_total = int(round((cfg.burn_in + cfg.t_total) / cfg.dt))
    else:
        v_rec = raw["bar_vN"]
        x_rec = raw["bar_dx"]
        n_samples = int(x_rec.size)
        n_steps_total = int(clk.n_steps)

    diag = {
        "N": cfg.N, "beta": cfg.beta, "r": cfg.r, "gamma_N": cfg.gamma_N,
        "gammas": list(gammas), "kappas": list(kappas), "mu": cfg.mu,
        "ou_mode": cfg.ou_mode, "c_return": cfg.c_return,
        "dt": dt, "n_steps": n_steps_total, "burn_step": rec_burn,
        "n_paths": int(cfg.n_paths), "n_samples": n_samples,
        "n_overflow": int(raw.get("n_overflow", 0)),
        "n_nonfinite": int(raw.get("n_nonfinite", 0)),
        "n_negative": int(raw.get("n_neg_actual", 0)),
        "n_negative_counterfactual_est": float(raw.get("n_neg_est", 0.0)),
        "neg_counterfactual_rate": float(raw.get("neg_rate", 0.0)),
        "max_abs_dy": float(raw.get("max_abs_dy", 0.0)),
        "max_abs_y": float(raw.get("max_abs_y", 0.0)),
        "steps_per_fastest_relaxation": float(dt * gammas[-1]),
        "effective_steps_per_slowest_relaxation": float(dt * gammas[0]),
        "wall_time_s": wall,
        "session_pattern": cfg.session_pattern,
        "chi": cfg.chi, "g_gap": cfg.g_gap,
        "record_kind": "step" if clk is None else "bar",
        "record_stride_steps": int(rec_stride),
        "safety_check_every": int(raw.get("check_every", 1)),
        "neg_probe_every": int(raw.get("neg_every", 1)),
        "integrator": cfg.integrator,
    }
    if clk is not None:
        diag["clock"] = clk.summary()
        diag["burn_bar"] = int(raw["burn_bar"])
    if "n_nonpos" in raw:
        diag["fault_injection"] = {
            "n_nonpos": int(raw["n_nonpos"]),
            "n_steps_with_nonpos": int(raw["n_nonpos_steps"]),
            "worst_negative_value": float(raw["worst_negative_value"]),
        }

    latent = {"vN": v_rec if v_rec.ndim == 2 else v_rec[:, -1, :]}
    if cfg.record_all_levels and v_rec.ndim == 3:
        latent["v"] = v_rec
    calendar = None
    if clk is not None:
        calendar = {"bar_day": clk.bar_day, "bar_slot": clk.bar_slot,
                    "bar_gap_before": clk.bar_gap_before,
                    "burn_bar": int(raw["burn_bar"]),
                    "gaps_on": bool(clk.gaps_on), "chi": clk.chi, "g": clk.g}
    return {
        "latent_paths": latent,
        "returns": x_rec,
        "t": t_rec,
        "calendar": calendar,
        "diagnostics": diag,
        "status": status.OK,
        "seed": int(seed),
        "config": cfg.as_dict(),
    }


__all__ = ["HierarchyConfig", "simulate_hierarchy", "INTEGRATORS", "OU_MODES", "PATTERNS"]
