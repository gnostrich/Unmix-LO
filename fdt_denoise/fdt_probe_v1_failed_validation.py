import numpy as np
rng = np.random.default_rng(0)
# LOAD-BEARING HONESTY TEST: does the model-disagreement dynamics satisfy a second-FDT-type relation?
# If YES -> the MZ noise floor is PRINCIPLED (native denoising, your suspicion is real).
# If NO  -> the "noise floor" is just a threshold we chose (denoising is heuristic, not native).
#
# Second FDT (Mori-Zwanzig): the MEMORY KERNEL K(t) and the NOISE autocorrelation <xi(t)xi(0)> are LINKED:
#   K(t) proportional to <xi(t) xi(0)>  (the memory is set by the noise's own autocorrelation).
# Test: build a disagreement time-series, extract memory kernel and noise term via MZ projection, check the link.

# --- Build a system WITH a genuine generator (should satisfy FDT) vs WITHOUT (should not) ---
T, D = 6000, 4

def mz_extract(x):
    # x: (T,D) trajectory of the "resolved" observable (the disagreement/swirl signal)
    # Projection: resolved = x_t; fit x_{t+1} = M x_t + residual(noise). Memory via lagged autocorrelation.
    M,*_ = np.linalg.lstsq(x[:-1], x[1:], rcond=None)
    noise = x[1:] - x[:-1]@M                      # the orthogonal fluctuation = MZ noise term xi
    # memory kernel proxy: autocorrelation of the resolved dynamics at lag tau
    def autocorr(series, tau):
        a = series[:-tau] if tau>0 else series
        b = series[tau:] if tau>0 else series
        return np.mean(np.sum(a*b,axis=1))
    lags = range(1,15)
    K = np.array([autocorr(x, t) for t in lags])          # memory-side
    Xi = np.array([np.mean(np.sum(noise[:-t]*noise[t:],axis=1)) if t< len(noise) else 0 for t in lags])  # noise-side
    return K, Xi, noise

def fdt_ratio(K, Xi):
    # second FDT: K(t) ∝ Xi(t). Test by correlation of the two decay profiles (shape match), not scale.
    K1, X1 = K/ (np.abs(K[0])+1e-9), Xi/(np.abs(Xi[0])+1e-9)
    return np.corrcoef(K1, X1)[0,1]

# CASE A: genuine linear generator + thermal noise (a real MZ system -> should satisfy FDT)
A = 0.85*np.linalg.qr(rng.normal(size=(D,D)))[0]
xA = np.zeros((T,D)); 
for t in range(1,T): xA[t] = xA[t-1]@A.T + 0.3*rng.normal(size=D)   # OU-like: generator + noise, FDT holds
KA, XiA, _ = mz_extract(xA); rA = fdt_ratio(KA, XiA)

# CASE B: arbitrary NON-generator disagreement (two unrelated processes' difference -> FDT should FAIL)
p1 = np.cumsum(rng.normal(size=(T,D)),axis=0)*0.02      # random walk 1
p2 = np.cumsum(rng.normal(size=(T,D)),axis=0)*0.02      # random walk 2 (independent)
xB = np.tanh(p1) - np.tanh(p2)                           # arbitrary disagreement, no shared generator
KB, XiB, _ = mz_extract(xB); rB = fdt_ratio(KB, XiB)

# CASE C: STRUCTURED disagreement -- two models viewing a SHARED latent with different gauges (the realistic case)
z = np.zeros((T,D))
G = 0.9*np.linalg.qr(rng.normal(size=(D,D)))[0]
for t in range(1,T): z[t] = z[t-1]@G.T + 0.2*rng.normal(size=D)   # shared latent HAS a generator
R1 = np.linalg.qr(rng.normal(size=(D,D)))[0]; R2 = np.linalg.qr(rng.normal(size=(D,D)))[0]
xC = z@R1 - z@R2                                          # disagreement of two gauges of a generated latent
KC, XiC, _ = mz_extract(xC); rC = fdt_ratio(KC, XiC)

print("SECOND-FDT test: is memory-kernel shape correlated with noise-autocorrelation shape? (=FDT holds)\n")
print(f"  CASE A (genuine generator + noise, FDT should HOLD)     : FDT-corr = {rA:+.3f}")
print(f"  CASE B (arbitrary non-generator disagreement, should FAIL): FDT-corr = {rB:+.3f}")
print(f"  CASE C (disagreement of 2 gauges of a GENERATED latent)  : FDT-corr = {rC:+.3f}   <- the realistic case")
print()
print("  READING:")
print("  - If C ~ A (high corr): model-disagreement of a shared GENERATED world satisfies FDT")
print("    -> the MZ noise floor is PRINCIPLED -> Baur denoising is NATIVE (your suspicion holds).")
print("  - If C ~ B (low corr): disagreement does NOT satisfy FDT -> noise floor is a chosen threshold,")
print("    denoising is heuristic not native. Honest downgrade.")
print("  Key insight this probes: FDT holds iff the disagreement inherits a GENERATOR from the shared world.")
