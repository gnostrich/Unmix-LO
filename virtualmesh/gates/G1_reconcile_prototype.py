import numpy as np
rng = np.random.default_rng(0)
# ---------------------------------------------------------------------------
# QUESTION: does recurrent SETTLING across models coupled through a shared frame
# beat one-shot output POOLING (the router/ensemble baseline), specifically when
# the knowledge needed for a query is SPLIT across models (each holds a piece)?
#
# Setup: a shared latent "world" of facts. Each small model observes only a SUBSET
# of the world's dimensions (its expertise), in its OWN rotated coordinates (gauge).
# A query needs several dimensions that no single model fully covers.
#   - POOLING: each model decodes its guess of the answer; average the guesses.
#   - SETTLING: models are coupled through a shared frame; each iteratively updates
#     a shared state using ONLY the dims it knows, reading others' contributions via
#     the frame, until the joint state settles (deep-equilibrium style).
# If knowledge is genuinely split, settling should reconstruct the full answer where
# pooling cannot (pooling averages partial/ignorant guesses).
# ---------------------------------------------------------------------------
D = 30            # world dimensions (facts)
M = 8             # small models
cover = 6         # each model "knows" this many of the D dims
N = 300           # queries

# each model knows a subset of dims, and represents them in its own random basis (gauge)
know = [rng.choice(D, size=cover, replace=False) for _ in range(M)]
R = [np.linalg.qr(rng.normal(size=(D,D)))[0] for _ in range(M)]   # each model's private rotation

# SHARED FRAME: assume we've aligned each model to common coords via anchors.
# (We grant the anchoring — that's the maintained frame; the test is settling-vs-pooling GIVEN a frame.)
# Each model can read/write ONLY its known dims of the shared-frame state; unknown dims it must leave to others.

def model_readout(m, x_true):
    # model m observes only its known dims (in shared-frame coords), with noise; unknown dims -> 0 (ignorance)
    obs = np.zeros(D)
    obs[know[m]] = x_true[know[m]] + rng.normal(0, 0.05, size=cover)
    mask = np.zeros(D); mask[know[m]] = 1.0
    return obs, mask

def pooling_answer(x_true):
    # each model outputs its best full-vector guess: known dims filled, unknown dims = its PRIOR (0/mean).
    guesses = []
    for m in range(M):
        obs, mask = model_readout(m, x_true)
        g = obs.copy()          # unknown dims stay 0 (model has no knowledge there)
        guesses.append(g)
    return np.mean(guesses, axis=0)   # one-shot pool

def settling_answer(x_true, iters=50):
    # shared state; each model iteratively writes its known dims, we reconcile by
    # confidence-weighted consensus on each dim (models that KNOW a dim dominate it).
    x = np.zeros(D)
    conf = np.zeros((M, D))
    for m in range(M):
        conf[m, know[m]] = 1.0
    csum = conf.sum(0) + 1e-9
    for _ in range(iters):
        contrib = np.zeros(D)
        for m in range(M):
            obs, mask = model_readout(m, x_true)     # model re-reads (its known dims; noise resampled)
            # a model contributes its known dims; for coupling, it also nudges toward current consensus on unknown dims
            local = obs*mask + x*(1-mask)
            contrib += conf[m]*local
        x_new = contrib / csum
        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new; break
        x = x_new
    return x

# queries: random world states; error = how well the recovered answer matches full truth on ALL dims
err_pool, err_settle, err_best = [], [], []
for _ in range(N):
    x_true = rng.normal(size=D)
    p = pooling_answer(x_true); s = settling_answer(x_true)
    err_pool.append(np.linalg.norm(p - x_true)/np.linalg.norm(x_true))
    err_settle.append(np.linalg.norm(s - x_true)/np.linalg.norm(x_true))
    # best single model: the one covering most, filling unknown with 0
    bm = 0; obs,_ = model_readout(bm, x_true); g=obs.copy()
    err_best.append(np.linalg.norm(g - x_true)/np.linalg.norm(x_true))

print("RECONCILIATION vs POOLING gate (knowledge split across models)")
print(f"  D={D} facts, {M} models, each knows {cover}/{D} dims; union covers {len(set().union(*[set(k) for k in know]))}/{D}")
print(f"  best single model  rel-error = {np.mean(err_best):.3f}")
print(f"  one-shot POOLING   rel-error = {np.mean(err_pool):.3f}")
print(f"  recurrent SETTLING rel-error = {np.mean(err_settle):.3f}")
print(f"  settling improvement over pooling = {np.mean(err_pool)/np.mean(err_settle):.2f}x")
