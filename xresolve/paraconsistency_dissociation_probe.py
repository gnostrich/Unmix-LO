import numpy as np
rng = np.random.default_rng(1)
# FIXED probe: collapse must be IRREVERSIBLY lossy, and the disambiguating evidence must ONLY be
# interpretable THROUGH the held hypotheses (not readable on its own). Then does holding win?

N, D = 6000, 5
z = rng.normal(size=(N,D))
branch = rng.integers(0,2,size=N)
# obs loses the sign (aliasing): two hypotheses h+ = +|.|, h- = -|.|
obs = np.abs(z) + 0.1*rng.normal(size=(N,D))

ntr=N//2
def acc(p,t): return np.mean((p>0.5)==(t>0.5))

# magnitude is recoverable; sign/branch is NOT recoverable from obs alone
Wm,*_=np.linalg.lstsq(np.abs(obs[:ntr]), np.abs(z[:ntr]), rcond=None)
mag = np.abs(obs)@Wm
hplus, hminus = mag, -mag

# DISAMBIGUATING EVIDENCE that is ONLY interpretable through the hypotheses:
# a measurement m = <true_signed_state, r> for a known random probe r. Given m and r, you can check WHICH
# hypothesis (h+ or h-) is consistent (dot with r close to m). But m ALONE (without the held hypotheses)
# is uninterpretable -- it's just a scalar. This is the key: the evidence disambiguates ONLY relative to
# the held superposition. A collapse model that discarded one hypothesis CANNOT use m to recover it.
r_probe = rng.normal(size=(N,D))
true_signed = np.where(branch[:,None]==1, mag, -mag)
m = (true_signed * r_probe).sum(1) + 0.3*rng.normal(size=N)   # scalar evidence

# downstream target needs the correct branch
y = (true_signed.sum(1) > 0).astype(float)

# ---- COLLAPSE: picks ONE hypothesis at obs time (before m arrives), say hplus, IRREVERSIBLY. ----
# later it gets m and r_probe but has already committed to hplus -> can only check hplus against m.
# it cannot evaluate hminus (discarded). Its branch estimate: is hplus consistent with m? if not, it's stuck.
collapse_consistency = np.abs((hplus*r_probe).sum(1) - m)   # how consistent its ONE kept hyp is with m
# collapse predicts branch=1 (hplus) if consistent, else it's WRONG (can't switch to hminus it discarded)
collapse_branch_guess = (collapse_consistency < 1.0).astype(float)  # crude: consistent->trust hplus
Xc = np.column_stack([collapse_branch_guess[:ntr], np.ones(ntr)])
Wc,*_=np.linalg.lstsq(Xc, y[:ntr], rcond=None)
B_collapse = acc(np.column_stack([collapse_branch_guess[ntr:], np.ones(N-ntr)])@Wc, y[ntr:])

# ---- HOLD: keeps BOTH hplus, hminus. When m arrives, SELECTS whichever is consistent with m. ----
cons_plus  = np.abs((hplus *r_probe).sum(1) - m)
cons_minus = np.abs((hminus*r_probe).sum(1) - m)
selected_branch = (cons_plus < cons_minus).astype(float)   # pick the hypothesis consistent with evidence
Xh = np.column_stack([selected_branch[:ntr], np.ones(ntr)])
Wh,*_=np.linalg.lstsq(Xh, y[:ntr], rcond=None)
B_hold = acc(np.column_stack([selected_branch[ntr:], np.ones(N-ntr)])@Wh, y[ntr:])

# reconstruction task A (collapse should tie/beat): both emit MAP hplus -> same
A_collapse = acc((hplus[ntr:].sum(1)>0).astype(float), (z[ntr:].sum(1)>0).astype(float))  # ~chance, sign lost
A_hold     = A_collapse

print("FIXED double-dissociation (collapse irreversible; evidence interpretable ONLY via held hypotheses):\n")
print(f"  TASK A reconstruction (sign lost, both ~chance): collapse {A_collapse:.3f}, hold {A_hold:.3f} (tie, ok)")
print(f"  TASK B ambiguity-dependent (evidence needs held hyps):")
print(f"    collapse (discarded a hyp) acc = {B_collapse:.3f}")
print(f"    hold (kept both, selects)  acc = {B_hold:.3f}")
print(f"    -> hold beats collapse? {'YES' if B_hold>B_collapse+0.03 else 'no'}  (+{B_hold-B_collapse:.3f})")
print()
print("  This task is HONEST iff the evidence m is uninterpretable without the held hypotheses (it's a bare")
print("  scalar dotted with a probe) -- so collapse, having discarded a hypothesis, structurally cannot use it.")
print("  If hold wins HERE, paraconsistency has real content on irreversibly-lossy ambiguity tasks.")
