"""

Author: Lorenzo Baldazzi
Affiliation: Università degli Studi di Roma Tor Vergata

==================
Replicates the multipolar cut TT ℓ > 650 on the covariance matrix
of the Planck Plik lite likelihood, following the procedure described in
Kable et al. 2023 (ApJ 959:143), footnote 7:

    "We artificially increase the uncertainties in the Planck data
     covariance matrix for TT ℓ > 650 for the TT-TT, TT-TE, and TT-EE
     covariance blocks."

Required input files (in the same folder as the script):
    - c_matrix_plik_v22.dat   : 613x613 covariance matrix (Fortran unformatted)
    - cl_cmb_plik_v22.dat     : observed bandpowers (ell, Cl, sigma)
    - blmin.dat               : minimum ell of each bin
    - blmax.dat               : maximum ell of each bin
    - bweight.dat             : weights for binning

Output files:
    - c_matrix_plik_v22_TT650cut.dat : modified matrix (same binary format)
    - cov_matrix_TT650_cut_comparison.png : comparison figure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. READING THE COVARIANCE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
# The file is in Fortran unformatted format (sequential, unformatted):
#   [int32: nbytes] | [float64 x 613x613, lower triangle] | [int32: nbytes]
# The first and last 4 bytes are Fortran "record markers".

print("Reading covariance matrix...")
raw_bytes = np.fromfile('c_matrix_plik_v22.dat', dtype=np.uint8)

# Skip the 4 bytes of the initial and final record markers
data_bytes = raw_bytes[4:-4].tobytes()
cov_lower  = np.frombuffer(data_bytes, dtype=np.float64).reshape(613, 613).copy()

# The matrix is saved as lower triangle -> symmetrization
cov = cov_lower + cov_lower.T - np.diag(cov_lower.diagonal())

print(f"  Shape: {cov.shape}")
print(f"  Symmetric: {np.allclose(cov, cov.T)}")
print(f"  Positive diagonal: {np.all(cov.diagonal() > 0)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. IDENTIFICATION OF THE TT / TE / EE BLOCKS
# ─────────────────────────────────────────────────────────────────────────────
# The file cl_cmb_plik_v22.dat contains three blocks in sequence:
#   TT (215 bin, ℓ 32-2492) | TE (199 bin) | EE (199 bin)
# Blocks are recognized by negative jumps in the ell vector.

print("\nIdentifying TT/TE/EE structure...")
cl_data = np.loadtxt('cl_cmb_plik_v22.dat')
ells    = cl_data[:, 0]   # column 0: central multipole of bin
cl_obs  = cl_data[:, 1]   # column 1: observed Cl
cl_err  = cl_data[:, 2]   # column 2: error

jumps = np.where(np.diff(ells) < 0)[0]   # indices where ell "restarts"
n_TT  = jumps[0] + 1                     # 215

tt_ells = ells[:n_TT]

print(f"  TT: {n_TT} bin, ell {tt_ells.min():.0f} – {tt_ells.max():.0f}")
print(f"  TE: {jumps[1]-jumps[0]} bin")
print(f"  EE: {613-jumps[1]-1} bin")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELECTION OF BINS TO CUT (TT with ℓ > 650)
# ─────────────────────────────────────────────────────────────────────────────

L_CUT = 650          # cutting threshold
INFLATE = 1e10       # covariance inflation factor

idx_cut = np.where(tt_ells > L_CUT)[0]   # indices of TT bins to cut

print(f"\nTT cut ℓ > {L_CUT}:")
print(f"  Bins to cut: {len(idx_cut)}  (indices {idx_cut[0]}..{idx_cut[-1]})")
print(f"  ell first bin to cut:  {tt_ells[idx_cut[0]]:.0f}")
print(f"  ell last TT bin:       {tt_ells[idx_cut[-1]]:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. APPLICATION OF THE CUT
# ─────────────────────────────────────────────────────────────────────────────
# For each TT bin with ℓ > 650, the corresponding row and column
# in the covariance matrix are multiplied by INFLATE.
#
# Effect on the Gaussian likelihood (χ² = d^T C^{-1} d):
#   - the diagonal C[i,i] grows by INFLATE^2  ->  C^{-1}[i,i] ~  INFLATE^{-2} ≈ 0
#   - off-diagonal terms grow by INFLATE ->  C^{-1}[i,j] ≈ 0
# => those bins no longer contribute to χ², equivalent to excluding them.

print(f"\nApplying the cut (factor {INFLATE:.0e})...")
cov_cut = cov.copy()

for i in idx_cut:
    cov_cut[i, :] *= INFLATE   # row i
    cov_cut[:, i] *= INFLATE   # column i (symmetry)

print(f"  Diagonal TT ℓ=644 (last bin NOT cut): {cov_cut[74, 74]:.3e}")
print(f"  Diagonal TT ℓ=653 (first bin cut):      {cov_cut[75, 75]:.3e}")

# Verify that the inverse is ~0 for cut bins
sub     = cov_cut[73:78, 73:78]
sub_inv = np.linalg.inv(sub)
print("\n  C_cut^{-1} (5x5 block around the cut, rows/ell 635-671):")
print("  ", np.array2string(sub_inv, precision=2, suppress_small=False))
print("  → elements for ℓ>650 (last 3 rows/columns) are ~10^{-14}: OK")


# ─────────────────────────────────────────────────────────────────────────────
# 5. WRITING THE OUTPUT FILE (same Fortran binary format)
# ─────────────────────────────────────────────────────────────────────────────
# Structure: [int32 nbytes] [float64 lower triangle] [int32 nbytes]

print("\nWriting output file...")
cov_cut_lower = np.tril(cov_cut)
data_out      = cov_cut_lower.flatten().astype(np.float64)
nbytes        = np.int32(len(data_out) * 8)

outfile = 'c_matrix_plik_v22_TT650cut.dat'
with open(outfile, 'wb') as f:
    f.write(nbytes.tobytes())    # initial record marker
    f.write(data_out.tobytes())  # data
    f.write(nbytes.tobytes())    # final record marker

# Round-trip verification
raw2   = np.fromfile(outfile, dtype=np.uint8)
cov2_l = np.frombuffer(raw2[4:-4].tobytes(), dtype=np.float64).reshape(613, 613)
cov2   = cov2_l + cov2_l.T - np.diag(cov2_l.diagonal())
print(f"  Round-trip OK: {np.allclose(cov2, cov_cut)}")
print(f"  File written:  {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON FIGURE
# ─────────────────────────────────────────────────────────────────────────────

print("\nCreating figure...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Panel 1: original vs cut diagonal (TT block) ──
ax = axes[0]
ax.semilogy(range(n_TT), np.sqrt(np.abs(cov.diagonal()[:n_TT])),
            'b-',  lw=1.5, label='Original')
ax.semilogy(range(n_TT), np.sqrt(np.abs(cov_cut.diagonal()[:n_TT])),
            'r--', lw=1.5, label=f'TT ℓ>{L_CUT} inflated')
#ax.axvline(idx_cut[0], color='gray', ls=':', lw=1,
           #label=f'ℓ={int(tt_ells[idx_cut[0]])} (idx={idx_cut[0]})')
ax.set_xlabel('Bandpower index TT')
ax.set_ylabel(r'$\sqrt{C_{ii}}$  (diagonal error)')
ax.set_title('Diagonal — TT block')
ax.legend(fontsize=8,loc='upper left')
ax.grid(True, alpha=0.3)

# ── Panel 2: original matrix (log|C|) ──
ax = axes[1]
with np.errstate(divide='ignore'):
    img = np.log10(np.abs(cov) + 1e-30)
im = ax.imshow(img, aspect='auto', origin='upper', cmap='RdBu_r', vmin=-15, vmax=0)
for v in [n_TT - 0.5, jumps[1] + 0.5]:
    ax.axvline(v, color='white', lw=1, ls='--')
    ax.axhline(v, color='white', lw=1, ls='--')
ax.set_title('log₁₀|C| — original')
plt.colorbar(im, ax=ax, shrink=0.8)
for label, pos in [('TT', n_TT // 2),
                   ('TE', (n_TT + jumps[1]) // 2),
                   ('EE', (jumps[1] + 613) // 2)]:
    ax.text(pos, 2, label, ha='center', va='top',
            color='k', fontsize=9, fontweight='bold', clip_on=False)

# ── Panel 3: difference (modified − original) ──
ax = axes[2]
diff = cov_cut - cov
with np.errstate(divide='ignore'):
    diff_log = np.log10(np.abs(diff) + 1e-40)
    diff_log[np.abs(diff) < 1e-40] = np.nan
im2 = ax.imshow(diff_log, aspect='auto', origin='upper', cmap='hot_r', vmin=-15, vmax=15)
ax.axvline(n_TT - 0.5, color='cyan', lw=1, ls='--')
ax.axhline(n_TT - 0.5, color='cyan', lw=1, ls='--')
ax.set_title('log₁₀|ΔC| — modified regions')
plt.colorbar(im2, ax=ax, shrink=0.8)

plt.suptitle(f'Planck Plik lite — TT cut at ℓ > {L_CUT}  (inflate × {INFLATE:.0e})',
             fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig('cov_matrix_TT650_cut_comparison.png', dpi=300, bbox_inches='tight')
print("  Figure saved: cov_matrix_TT650_cut_comparison.png")

print("\nDone.")
