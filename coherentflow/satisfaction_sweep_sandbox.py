import numpy as np
D, T = 24, 600; ntr = T//2
def r2(p,t):
    d=((t-t.mean(0))**2).sum(); return 1-((t-p)**2).sum()/d if d>0 else 0.0
def acc(p,t): return float(np.mean((p>0.5)==(t>0.5)))
def iface(z, seed, extra=None, noise=0.1):
    r=np.random.default_rng(seed); R=np.linalg.qr(r.normal(size=(D,D)))[0]
    v=z@R+noise*r.normal(size=(T,D))
    if extra is not None: v=v+extra
    A,*_=np.linalg.lstsq(v[:ntr],z[:ntr],rcond=None); return v@A
def is_structured(d, z):
    dc=d-d.mean(0); U,S,Vt=np.linalg.svd(dc[:ntr],full_matrices=False)
    eff=max(1,int(round((S.sum()**2)/(S**2).sum()))); P=Vt[:eff].T@Vt[:eff]
    cap=((dc[ntr:]@P)**2).sum()/((dc[ntr:]**2).sum()+1e-9)
    A,*_=np.linalg.lstsq(z[:ntr],d[:ntr],rcond=None); ho=r2(z[ntr:]@A,d[ntr:])
    return (cap>1.5*eff/D) and (ho>0.3), P
def settle(ifaces, z, n=20):
    state=np.mean(ifaces,axis=0); circ=np.zeros_like(state); res=[]; held=0
    for it in range(n):
        prev=state.copy(); c=np.zeros_like(state); held=0
        for f in ifaces:
            d=f-state; s,P=is_structured(d,z)
            if s: c=c+0.5*(d@P); held+=1
        circ=c; state=prev+0.5*((np.mean(ifaces,axis=0)+circ)-prev)
        res.append(np.linalg.norm(state-prev))
    return state, circ, res, held

# ===== FIX T5: valid contraction test (tail-slope, clean input, state NOT init at stitch) =====
print("T5-FIXED: contraction via tail-slope (last-third residuals decreasing), clean structured input")
for seed in [5,15,25]:
    z=np.random.default_rng(seed).normal(size=(T,D))
    br=np.random.default_rng(seed+1).integers(0,2,T); ext=np.zeros((T,D)); ext[:,0]=(br*2-1)*3
    ifaces=[iface(z,seed*10,extra=ext),iface(z,seed*10+1),iface(z,seed*10+2)]
    # init state AWAY from stitch so there's a real transient to contract
    state=np.random.default_rng(seed).normal(size=(T,D))*0.5+np.mean(ifaces,axis=0)
    circ=np.zeros_like(state); res=[]
    for it in range(30):
        prev=state.copy(); c=np.zeros_like(state)
        for f in ifaces:
            d=f-state; s,P=is_structured(d,z)
            if s: c=c+0.5*(d@P)
        state=prev+0.5*((np.mean(ifaces,axis=0)+c)-prev); res.append(np.linalg.norm(state-prev))
    tail=res[len(res)*2//3:]; contracts = tail[-1] < tail[0] and tail[-1] < res[0]
    print(f"  seed {seed}: res {res[0]:.3f} -> {res[-1]:.4f}, tail {tail[0]:.4f}->{tail[-1]:.4f}  contracts={contracts}")

# ===== QUANTIFY T3: detection sensitivity vs injection strength, across seeds =====
print("\nT3-SWEEP: detection rate + read-payoff vs injection strength (30 seeds each)")
print(f"  {'strength':>8} | {'detect%':>8} | {'mean_held':>9} | {'combined-consensus payoff (when held)':>10}")
for strength in [1.0, 2.0, 3.0, 4.0, 6.0]:
    detects=0; payoffs=[]; helds=[]
    for seed in range(30):
        z=np.random.default_rng(1000+seed).normal(size=(T,D))
        br=np.random.default_rng(2000+seed).integers(0,2,T); ext=np.zeros((T,D)); ext[:,0]=(br*2-1)*strength
        ifaces=[iface(z,3000+seed,extra=ext),iface(z,4000+seed),iface(z,5000+seed)]
        st,circ,res,held=settle(ifaces,z)
        helds.append(held)
        if held>=1:
            detects+=1
            Wc,*_=np.linalg.lstsq(np.column_stack([st[:ntr],np.ones(ntr)]),br[:ntr].astype(float),rcond=None)
            cons=acc(np.column_stack([st[ntr:],np.ones(T-ntr)])@Wc,br[ntr:])
            Wh,*_=np.linalg.lstsq(np.column_stack([circ[:ntr],np.ones(ntr)]),br[:ntr].astype(float),rcond=None)
            comb=acc(np.column_stack([circ[ntr:],np.ones(T-ntr)])@Wh,br[ntr:])
            payoffs.append(comb-cons)
    pay = np.mean(payoffs) if payoffs else 0.0
    print(f"  {strength:>8.1f} | {100*detects/30:>7.0f}% | {np.mean(helds):>9.2f} | {pay:>+10.3f}")

# ===== FALSE-POSITIVE rate: how often does it hold on PURE COHERENT input (should be ~0%) =====
print("\nFALSE-POSITIVE check: hold-rate on coherent input across 30 seeds (must be ~0%)")
fp=0
for seed in range(30):
    z=np.random.default_rng(6000+seed).normal(size=(T,D))
    ifaces=[iface(z,7000+seed),iface(z,8000+seed),iface(z,9000+seed)]
    st,circ,res,held=settle(ifaces,z)
    if held>=1: fp+=1
print(f"  false-positive hold rate = {100*fp/30:.0f}%  ({'clean' if fp==0 else 'LEAKY'})")
