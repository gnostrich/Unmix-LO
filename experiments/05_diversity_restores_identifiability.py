# experiment 05 — diversity restores identifiability (genre-calibrated). See experiments/README.md
import numpy as np
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
rng=np.random.default_rng(0)

# 8 latent "skills" (shared operators the world composes), each a direction in D-dim gradient-like space
skills=["syntax","scoping","numeric","symbolic","coref","narrative","pragmatics","retrieval"]
K=len(skills); D=32
S,_=np.linalg.qr(rng.normal(0,1,(D,K))); S=S[:,:K]
idx={s:i for i,s in enumerate(skills)}

# realistic training GENRES: which skills are active, and which PAIR is locked (co-occurs, correlated) IN THAT GENRE
genres={
 "code":       (["syntax","scoping","numeric"],      ("syntax","scoping")),
 "math":       (["numeric","symbolic","syntax"],     ("numeric","symbolic")),
 "prose":      (["coref","narrative","syntax"],       ("coref","narrative")),
 "dialogue":   (["coref","pragmatics","retrieval"],   ("coref","pragmatics")),
 "scientific": (["symbolic","retrieval","numeric"],   ("symbolic","retrieval")),
 "web":        (["retrieval","narrative","pragmatics"],("retrieval","narrative")),
}
# note: e.g. 'coref' is locked-with-narrative in prose but locked-with-pragmatics in dialogue
#       -> pooling prose+dialogue should DECORRELATE coref and let ICA separate it.

def gen_samples(genre, M=500):
    active,(a,b)=genres[genre]; X=[]
    for _ in range(M):
        loads={sk:rng.laplace() for sk in active}
        loads[b]=loads[a]                      # LOCK: correlated loadings within this genre
        v=sum(loads[sk]*S[:,idx[sk]] for sk in active)+rng.normal(0,0.02,D)
        X.append(v)
    return np.array(X)

def recovery(genre_list):
    X=np.vstack([gen_samples(g) for g in genre_list])
    ica=FastICA(n_components=K, whiten='unit-variance', max_iter=2000, random_state=0)
    try: ica.fit(X)
    except Exception: return np.nan
    A=ica.mixing_; A=A/(np.linalg.norm(A,axis=0,keepdims=True)+1e-12)
    C=np.abs(S.T@A); r,c=linear_sum_assignment(-C)
    return C[r,c].mean()          # mean best-match cosine of each true skill to a recovered component (1=perfect)

order=["code","math","prose","dialogue","scientific","web"]
print("Individual-skill recovery vs. number of training genres pooled (1.0 = every skill cleanly separated):\n")
for n in range(1,len(order)+1):
    reps=[recovery(list(rng.permutation(order))[:n]) for _ in range(6)]   # avg over which n genres
    print(f"  {n} genre(s):  recovery = {np.nanmean(reps):.3f}")

print("\nControl — one skill pair locked in ALL genres (truly fused, should NEVER separate):")
for g in genres: 
    a,b=genres[g][1]; 
# force 'numeric'&'symbolic' locked everywhere
fused=dict(genres)
for g in fused:
    act=fused[g][0]
    fused[g]=(list(set(act)|{"numeric","symbolic"}),("numeric","symbolic"))
def gen_fused(genre,M=500):
    active,(a,b)=fused[genre]; X=[]
    for _ in range(M):
        loads={sk:rng.laplace() for sk in active}; loads[b]=loads[a]
        v=sum(loads[sk]*S[:,idx[sk]] for sk in active)+rng.normal(0,0.02,D); X.append(v)
    return np.array(X)
Xf=np.vstack([gen_fused(g) for g in order])
ica=FastICA(n_components=K,whiten='unit-variance',max_iter=2000,random_state=0); ica.fit(Xf)
A=ica.mixing_; A=A/(np.linalg.norm(A,axis=0,keepdims=True)+1e-12)
C=np.abs(S.T@A)
ns=C[idx["numeric"]].max(); sy=C[idx["symbolic"]].max()
print(f"  all 6 genres pooled, but numeric&symbolic fused everywhere:")
print(f"  numeric recovery={ns:.3f}  symbolic recovery={sy:.3f}  (low => correctly NOT separated when truly fused)")
