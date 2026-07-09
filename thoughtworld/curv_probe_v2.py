import numpy as np
rng = np.random.default_rng(0)
# CORRECTED instrument check: eff-rank AND held-out predictivity distinguishes atomic from noise.
D, Nstates = 24, 4000
E = np.linalg.qr(rng.normal(size=(D,D)))[0]
states = rng.normal(size=(Nstates, D)); engine_next = states @ E.T

def dev_of(kind, rank=3, noise=0.3):
    if kind=="atomic":
        U=np.linalg.qr(rng.normal(size=(D,rank)))[0]; W=rng.normal(size=(D,rank))*0.5
        return (states@W)@U.T
    if kind=="noise":  return rng.normal(0,noise,size=(Nstates,D))
    if kind=="mixed":
        U=np.linalg.qr(rng.normal(size=(D,rank)))[0]; W=rng.normal(size=(D,rank))*0.5
        return (states@W)@U.T + rng.normal(0,noise,size=(Nstates,D))
    if kind=="random_fragment": return rng.normal(0,1.0,size=(Nstates,D))  # control

def characterize(dev):
    n=len(dev)//2
    A,*_=np.linalg.lstsq(states[:n], dev[:n], rcond=None)     # fit on half
    sv=np.linalg.svd(A,compute_uv=False)
    eff_rank=(sv.sum()**2)/(sv**2).sum()
    pred=states[n:]@A                                          # test on held-out half
    ss_res=((dev[n:]-pred)**2).sum(); ss_tot=((dev[n:]-dev[n:].mean(0))**2).sum()
    r2=1-ss_res/ss_tot
    F=0.5*(A-A.T); sym=0.5*(A+A.T)
    directed_frac=np.linalg.norm(F)/(np.linalg.norm(A)+1e-9)
    return eff_rank, r2, directed_frac

print(f"{'case':<16}{'eff-rank':>9}{'heldout-R2':>12}{'directed-frac':>14}  verdict")
for kind in ["atomic","noise","mixed","random_fragment"]:
    er,r2,df=characterize(dev_of(kind))
    atomic = (er < D*0.4) and (r2 >= 0.3)
    print(f"{kind:<16}{er:>9.2f}{r2:>12.2f}{df:>14.2f}  {'ATOMIC' if atomic else 'NOISE'}")
print(f"\n  D={D}. ATOMIC needs eff-rank<{D*0.4:.0f} AND held-out R2>=0.3.")
print("  Expect: atomic->ATOMIC, noise->NOISE, mixed->ATOMIC, random_fragment->NOISE (control passes).")
