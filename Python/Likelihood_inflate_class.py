"""

Author: Lorenzo Baldazzi
Affiliation: Università degli Studi di Roma Tor Vergata

==================
Cobaya Likelihood for Planck 2018 Plik lite with TT cut at ℓ = 650.

Implements the Gaussian likelihood:
    -2 ln L = Δ^T · C_cut^{-1} · Δ
where
    Δ_b = C_b^obs - C_b^th
    C_b^obs  : observed bandpower from cl_cmb_plik_v22.dat  (units μK²)
    C_b^th   : theoretical bandpower computed via W @ C_ell^th
    C_cut    : covariance matrix with TT ℓ>650 variances inflated by 10^{10}
               (file c_matrix_plik_v22_TT650cut.dat, produced by apply_TT650_cut.py)

The theoretical bandpower is:
    C_b^th = Σ_ell  W[b, ell] · C_ell^th(μK²)

where W is the binning matrix constructed from blmin/blmax/bweight
following the procedure described in the _build_window_matrix() comments.

Data vector structure (613 bandpowers):
    indices   0 .. 214   →  TT  (ℓ 32–2492)   215 bins
    indices 215 .. 413   →  TE  (ℓ 32–1988)   199 bins
    indices 414 .. 612   →  EE  (ℓ 32–1988)   199 bins


"""

from __future__ import annotations

import os
import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError


# ─────────────────────────────────────────────────────────────────────────────
class PlikLiteTT650(Likelihood):
    """
    Planck 2018 Plik lite – Gaussian likelihood with TT cut at ℓ > 650.

    Attributes configurable via YAML (with default values):
        data_folder  : folder containing data files
        cl_file      : observed bandpowers
        cov_file     : covariance matrix (already cut)
        blmin_file   : minimum ℓ for each fine-bin
        blmax_file   : maximum ℓ for each fine-bin
        bweight_file : binning weights for each fine-bin
        lmax         : maximum ℓ required from theory code
    """

    # ── YAML-configurable attributes ────────────────────────────────────────
    data_folder:  str = "."
    cl_file:      str = "cl_cmb_plik_v22.dat"
    cov_file:     str = "c_matrix_plik_v22_TT650cut.dat"
    blmin_file:   str = "blmin.dat"
    blmax_file:   str = "blmax.dat"
    bweight_file: str = "bweight.dat"
    lmax:         int = 2600          # maximum ℓ required from theory

    # ── Initialization ───────────────────────────────────────────────────────
    def initialize(self) -> None:
        """Reads files and precomputes fixed quantities (W, C^{-1})."""

        def path(fname: str) -> str:
            return os.path.join(self.data_folder, fname)

        # ── 1. Observed bandpowers ──────────────────────────────────────────
        cl_raw        = np.loadtxt(path(self.cl_file))
        self.ells_obs = cl_raw[:, 0]          # central ℓ of each bandpower
        self.cl_obs   = cl_raw[:, 1]          # observed C_ell  [μK²]
        self.n_bins   = len(self.cl_obs)      # 613

        # Identifies TT/TE/EE blocks from negative jumps in ℓ sequence
        jumps       = np.where(np.diff(self.ells_obs) < 0)[0]
        self.n_TT   = jumps[0] + 1             # 215
        self.n_TE   = jumps[1] - jumps[0]      # 199
        self.n_EE   = self.n_bins - jumps[1] - 1  # 199

        if self.n_TT + self.n_TE + self.n_EE != self.n_bins:
            raise LoggedError(
                self.log,
                "Inconsistent TT/TE/EE structure: %d+%d+%d ≠ %d",
                self.n_TT, self.n_TE, self.n_EE, self.n_bins,
            )

        self.log.info(
            "Bandpowers: %d TT + %d TE + %d EE = %d total",
            self.n_TT, self.n_TE, self.n_EE, self.n_bins,
        )

        # ── 2. Covariance matrix (Fortran unformatted format) ───────────────
        # Structure: [int32 nbytes] [float64 lower triangle 613×613]
        #            [int32 nbytes]
        raw       = np.fromfile(path(self.cov_file), dtype=np.uint8)
        cov_lower = (
            np.frombuffer(raw[4:-4].tobytes(), dtype=np.float64)
            .reshape(self.n_bins, self.n_bins)
            .copy()
        )
        # Symmetrization of lower triangle
        cov = cov_lower + cov_lower.T - np.diag(cov_lower.diagonal())

        # Precomputes inverse (used at each likelihood evaluation)
        self.inv_cov = np.linalg.inv(cov)
        self.log.debug("Covariance matrix inverted (%dx%d).", *cov.shape)

        # ── 3. Binning matrix W ──────────────────────────────────────────────
        blmin        = np.loadtxt(path(self.blmin_file)).astype(int)
        blmax        = np.loadtxt(path(self.blmax_file)).astype(int)
        bweight_flat = np.loadtxt(path(self.bweight_file))

        self.W = self._build_window_matrix(blmin, blmax, bweight_flat)
        self.log.info("Binning matrix W built: shape %s.", self.W.shape)

    # ── Building the binning matrix ─────────────────────────────────────────
    def _build_window_matrix(
        self,
        blmin: np.ndarray,
        blmax: np.ndarray,
        bweight_flat: np.ndarray,
    ) -> np.ndarray:
        """
        Builds the W matrix of shape (n_bins, lmax+1) such that:

            C_b^th = Σ_ell  W[b, ell] · C_ell^th

        where W[b, ell] is the weight of ell in bandpower b, derived from
        blmin/blmax/bweight.

        Strategy
        ---------
        The 645 "fine-bins" in blmin/blmax do not correspond 1-to-1 to the 613
        bandpowers of cl_cmb: wider bins (width 9) contain more observed ℓ.
        For each bandpower b we define its effective range [ℓ_min_b, ℓ_max_b]
        as the range of ℓ "belonging" to that bandpower (halfway between
        consecutive bandpowers), then for each ℓ in that range we use the weight
        from its fine-bin. Finally we normalize each row so that weights sum to 1.

        This way:
            C_b^th ≈ C_ell(ℓ_b)   for a smooth C_ell,
        which is exactly the expected behavior.
        """

        # ── a. Lookup: for each ℓ → fine-bin and raw weight ────────────────
        lmax_w              = int(blmax.max())
        ell_to_finebin      = np.full(lmax_w + 1, -1, dtype=int)
        ell_weight_lookup   = np.zeros(lmax_w + 1)

        idx_w = 0
        for bf in range(len(blmin)):
            n_ell_bf = blmax[bf] - blmin[bf] + 1
            w_bf     = bweight_flat[idx_w : idx_w + n_ell_bf]
            idx_w   += n_ell_bf
            for k, ell in enumerate(range(blmin[bf], blmax[bf] + 1)):
                if ell <= lmax_w:
                    ell_to_finebin[ell]    = bf
                    ell_weight_lookup[ell] = w_bf[k]

        # ── b. Effective range [ℓ_min_b, ℓ_max_b] for each bandpower ─────────
        def eff_range(ells: np.ndarray, b: int) -> tuple[int, int]:
            """ℓ boundary as halfway between consecutive bandpowers."""
            n = len(ells)
            if b == 0:
                lo = int(ells[0] - (ells[1] - ells[0]) / 2)
            else:
                lo = int((ells[b - 1] + ells[b]) / 2) + 1
            if b == n - 1:
                hi = int(ells[-1] + (ells[-1] - ells[-2]) / 2)
            else:
                hi = int((ells[b] + ells[b + 1]) / 2)
            return max(2, lo), min(hi, lmax_w)

        # ── c. Building W ───────────────────────────────────────────────────
        W = np.zeros((self.n_bins, self.lmax + 1))

        tt_ells = self.ells_obs[: self.n_TT]
        te_ells = self.ells_obs[self.n_TT : self.n_TT + self.n_TE]
        ee_ells = self.ells_obs[self.n_TT + self.n_TE :]

        spectra = [
            (tt_ells, 0),                        # TT: offset=0
            (te_ells, self.n_TT),                # TE: offset=n_TT
            (ee_ells, self.n_TT + self.n_TE),    # EE: offset=n_TT+n_TE
        ]

        for ells_spec, offset in spectra:
            for b_local in range(len(ells_spec)):
                b_global = offset + b_local
                lo, hi   = eff_range(ells_spec, b_local)

                # Collects weights from lookup for ℓ ∈ [lo, hi]
                for ell in range(lo, min(hi, self.lmax) + 1):
                    if ell <= lmax_w and ell_to_finebin[ell] >= 0:
                        W[b_global, ell] = ell_weight_lookup[ell]

                # Renormalize (sum of weights = 1 within each bandpower)
                row_sum = W[b_global, :].sum()
                if row_sum > 0:
                    W[b_global, :] /= row_sum
                else:
                    self.log.warning(
                        "Bandpower b=%d (ells_spec[%d]=%.0f): all weights null.",
                        b_global, b_local, ells_spec[b_local],
                    )

        return W

    # ── Requirements from theory code ────────────────────────────────────────
    def get_requirements(self) -> dict:
        """
        Requests from Cobaya the C_ell (not D_ell) for TT, TE, EE up to lmax.

        Note: ell_factor=False in get_Cl() → returns C_ell [μK²].
        """
        return {
            "Cl": {
                "tt": self.lmax,
                "te": self.lmax,
                "ee": self.lmax,
            }
        }

    # ── Computation of log-likelihood ────────────────────────────────────────
    def logp(self, **params_values) -> float:
        """
        Returns  -0.5 · χ²  where
            χ² = Δ^T · C_cut^{-1} · Δ
            Δ_b = C_b^obs – C_b^th
            C_b^th = W[b, :] @ C_ell^th

        C_ell^th is provided by Cobaya in μK² (ell_factor=False).
        TT bandpowers with ℓ > 650 have inflated variance in C_cut,
        so their contribution to χ² is negligible (~10^{-14}).
        """

        # ── Get the theoretical C_ell in μK² ──────────────────────────────
        Cls = self.provider.get_Cl(ell_factor=False, units="muK2")

        def _pad(spec: str) -> np.ndarray:
            """Returns C_ell[spec] padded to length lmax+1."""
            arr = Cls.get(spec, np.zeros(self.lmax + 1))
            if len(arr) < self.lmax + 1:
                arr = np.pad(arr, (0, self.lmax + 1 - len(arr)))
            return arr[: self.lmax + 1]

        Cl_TT = _pad("tt")
        Cl_TE = _pad("te")
        Cl_EE = _pad("ee")

        # ── Compute theoretical bandpowers: C_b^th = W @ C_ell ──────────────
        sl_TT = slice(0, self.n_TT)
        sl_TE = slice(self.n_TT, self.n_TT + self.n_TE)
        sl_EE = slice(self.n_TT + self.n_TE, self.n_bins)

        cl_th = np.empty(self.n_bins)
        cl_th[sl_TT] = self.W[sl_TT] @ Cl_TT
        cl_th[sl_TE] = self.W[sl_TE] @ Cl_TE
        cl_th[sl_EE] = self.W[sl_EE] @ Cl_EE

        # ── Gaussian likelihood ──────────────────────────────────────────────
        delta = self.cl_obs - cl_th
        chi2  = delta @ self.inv_cov @ delta

        return -0.5 * float(chi2)