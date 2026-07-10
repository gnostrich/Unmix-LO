"""
mz_kernel.py — the self-expanding Operator-Valued Mori–Zwanzig memory kernel (the aggregator state).

The kernel maintains a minimal state-space realization (A,B,C) of the shared task dynamics from the
aggregated Markov parameters. Its order self-expands / prunes by the Hankel-rank / atomicity criterion
against the second-FDT noise floor:

  SELF-EXPAND: append a state dimension while the next (closure-residual) Hankel singular value
               clears the second-FDT noise floor.
  PRUNE      : balanced-truncation — drop the top state while its Hankel singular value is below the floor.

Second-FDT noise floor: the Hankel singular-value level attributable to fluctuation (the residual/worker-
disagreement covariance), estimated by Monte-Carlo of a same-structure white-noise Hankel. No free knob —
it is set by the measured noise, per the fluctuation–dissipation link between the MZ noise and memory kernel.

Reuses the block-Hankel closure primitive from the archived gate2_mzkernel lineage; Ho-Kalman from resolvent.py.
"""
import numpy as np
from resolvent import block_hankel, hankel_svals, ho_kalman, markov_from_system


class MZKernel:
    def __init__(self, p=3, L=18, floor_mc=24, floor_q=99.5, seed=0):
        self.p = p
        self.L = L
        self.floor_mc = floor_mc          # Monte-Carlo draws for the FDT floor
        self.floor_q = floor_q            # quantile of white-noise Hankel top-SV -> the floor
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.A = self.B = self.C = None
        self.h_hat = None
        self.expansions = 0
        self.prunes = 0
        self.last_sv = None
        self.last_floor = None
        self._floor_cache = {}

    # --------------------------------------------------- second-FDT noise floor
    def fdt_floor(self, sigma):
        """Top singular value a same-structure white-noise block-Hankel of per-entry std `sigma` would
        produce (quantile over MC draws). Modes above this are dissipative memory; below is fluctuation."""
        if sigma <= 0:
            return 0.0
        key = round(float(sigma), 9)
        if key not in self._floor_cache:
            tops = []
            K = 2 * self.L + 2
            for _ in range(self.floor_mc):
                hn = self.rng.normal(scale=sigma, size=(K, self.p, self.p))
                tops.append(np.linalg.svd(block_hankel(hn, self.L), compute_uv=False)[0])
            self._floor_cache[key] = float(np.percentile(tops, self.floor_q))
        return self._floor_cache[key]

    # --------------------------------------------------- what the kernel currently explains
    def predicted_markov(self, K):
        if self.n == 0:
            return np.zeros((K, self.p, self.p))
        return markov_from_system(self.A, self.B, self.C, K)

    def closure_residual(self):
        """Relative MZ closure residual: ||h_hat - kernel_reproduction|| / ||h_hat||."""
        if self.h_hat is None:
            return 1.0
        K = self.h_hat.shape[0]
        pred = self.predicted_markov(K)
        return float(np.linalg.norm(pred - self.h_hat) / (np.linalg.norm(self.h_hat) + 1e-12))

    # --------------------------------------------------- the self-expansion / prune mechanism
    def step(self, h_hat, sigma):
        """Ingest aggregated Markov estimate h_hat (K,p,p) with fluctuation std sigma; self-expand/prune."""
        self.h_hat = h_hat
        sv, L = hankel_svals(h_hat, self.L)
        floor = self.fdt_floor(sigma)
        self.last_sv, self.last_floor = sv, floor

        # SELF-EXPAND: while the next (residual) Hankel singular value clears the FDT floor, append a state dim
        while self.n < len(sv) and sv[self.n] > floor:
            self.n += 1
            self.expansions += 1
        # BALANCED-TRUNCATION PRUNE: drop the top state while its Hankel singular value is below the floor
        while self.n > 0 and sv[self.n - 1] <= floor:
            self.n -= 1
            self.prunes += 1

        if self.n > 0:
            self.A, self.B, self.C, _ = ho_kalman(h_hat, self.n, self.L)
        else:
            self.A = self.B = self.C = None
        return self.n

    # --------------------------------------------------- cost (dimension-independent of K)
    def memory_cost(self):
        """Kernel memory footprint: A (n^2) + B (n*p) + C (p*n). Depends ONLY on order n, not on K."""
        return int(self.n * self.n + 2 * self.n * self.p)

    def spectral_gap(self):
        """sv[n-1]/sv[n]: clean gap (>>1) = atomic support; ~1 = no clean termination (continuous)."""
        sv = self.last_sv
        if sv is None or self.n == 0 or self.n >= len(sv):
            return float("inf") if self.n > 0 else 0.0
        return float(sv[self.n - 1] / (sv[self.n] + 1e-12))
