import numpy as np
rng = np.random.default_rng(0)
# PRECONDITION TEST (not mechanism-confirmation): does one model's frame resolve ANOTHER model's genuine
# ambiguity, or do their ambiguities COINCIDE (Platonic convergence -> nothing to resolve)?
# This is the fork that says whether "cross-model resolves paraconsistency" is real or toy-bound.
#
# Setup that can say NO: two models are two different lossy typings of a shared latent. Model A ALIASES
# certain latent pairs (maps them to the same rep). Question: does model B DISTINGUISH the pairs A aliases,
# or does B alias the SAME pairs (coinciding ambiguity -> re-framing has nothing to resolve)?
# We test THREE regimes to make the instrument honest:
#   (I)  INDEPENDENT typings  -> B should resolve A's aliasing (the effect exists)
#   (II) COINCIDING typings   -> B aliases the same pairs (the null: Platonic convergence, no resolution)
#   (III) real-ish CORRELATED -> the honest in-between; measures how much resolution survives shared structure

Nlat, D = 4000, 10
z = rng.normal(size=(Nlat, D))          # shared latent

def typing(seed, alias_dirs, alias_strength=1.0):
    # a lossy typing: projects out (aliases) certain latent directions -> reps identical along them
    r = np.random.default_rng(seed)
    G = np.linalg.qr(r.normal(size=(D,D)))[0]
    proj = np.eye(D)
    for d in alias_dirs:                  # collapse these directions (aliasing = can't see them)
        proj = proj - alias_strength*np.outer(G[:,d], G[:,d])
    return z @ proj + 0.05*r.normal(size=(Nlat,D)), G

def resolves(repA, repB):
    # Does B distinguish pairs that A aliases? Measure: for pairs A maps CLOSE (aliased), is B FAR (resolved)?
    # sample pairs, find A-aliased ones (A-distance tiny), check B-distance on those.
    i,j = rng.integers(0,Nlat,20000), rng.integers(0,Nlat,20000)
    dA = np.linalg.norm(repA[i]-repA[j],axis=1)
    dB = np.linalg.norm(repB[i]-repB[j],axis=1)
    aliased = dA < np.percentile(dA,5)    # the 5% A sees as most identical
    # resolution score = how distinguishable B finds A's aliased pairs, relative to B's overall scale
    return dB[aliased].mean()/dB.mean()   # >1 => B resolves them (finds them MORE distinct than average)

# (I) INDEPENDENT: A and B alias DIFFERENT directions
repA_I,_ = typing(1, alias_dirs=[0,1,2])
repB_I,_ = typing(2, alias_dirs=[5,6,7])            # different dirs -> should resolve
# (II) COINCIDING: A and B alias the SAME directions (Platonic convergence)
repA_II,_ = typing(1, alias_dirs=[0,1,2])
repB_II,_ = typing(3, alias_dirs=[0,1,2])           # same dirs -> should NOT resolve
# (III) CORRELATED: partial overlap
repA_III,_ = typing(1, alias_dirs=[0,1,2])
repB_III,_ = typing(4, alias_dirs=[1,2,3])          # overlap on 1,2; differ on 0 vs 3

print("PRECONDITION: does model B's typing RESOLVE model A's genuine aliasing? (score>1 = resolves)\n")
print(f"  (I)  INDEPENDENT typings  : B-resolves-A score = {resolves(repA_I,repB_I):.3f}   (expect >1: resolves)")
print(f"  (II) COINCIDING typings   : B-resolves-A score = {resolves(repA_II,repB_II):.3f}   (expect ~1: NULL, no resolution)")
print(f"  (III) CORRELATED (partial): B-resolves-A score = {resolves(repA_III,repB_III):.3f}   (in-between)")
print()
print("  READING: if independent >> coinciding, the instrument works AND cross-model resolution is real")
print("  WHEN typings differ. The real-model question then becomes: are real models' ambiguities INDEPENDENT")
print("  (resolution real) or COINCIDING (Platonic convergence -> the conviction was toy-bound)?")
print("  This toy CANNOT answer the real-model question -- it validates the MEASURE and shows the fork is real.")
