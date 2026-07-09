import numpy as np
rng_global = np.random.default_rng(0)
D, T = 24, 600; ntr = T//2
def r2(p,t):
    d=((t-t.mean(0))**2).sum(); return 1-((t-p)**2).sum()/d if d>0 else 0.0
def acc(p,t): return float(np.mean((p>0.5)==(t>0.5)))
def make_world(seed=0):
    r=np.random.default_rng(seed); return r.normal(size=(T,D))
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

R={}
# T1 coverage-union beats best-single (bankable use-case)
z=make_world(1)
def partial(seed,dims):
    r=np.random.default_rng(seed); m=np.zeros(D); m[list(dims)]=1; Rm=np.linalg.qr(r.normal(size=(D,D)))[0]
    v=(z*m)@Rm+0.1*r.normal(size=(T,D)); A,*_=np.linalg.lstsq(v[:ntr],z[:ntr],rcond=None); return v@A
mods=[partial(1,range(0,8)),partial(2,range(6,14)),partial(3,range(12,20)),partial(4,range(18,24))]
singles=[r2(m[ntr:],z[ntr:]) for m in mods]; union=r2(np.mean(mods,axis=0)[ntr:],z[ntr:])
R['T1 coverage-union > best-single']=(union>max(singles)+0.02, f"union={union:.3f} best={max(singles):.3f}")

# T2 coherent input -> honest no-op
z=make_world(2); coh=[iface(z,10),iface(z,11),iface(z,12)]
st,circ,res,held=settle(coh,z)
R['T2 coherent -> honest no-op']=(held==0 and np.linalg.norm(circ)<0.5, f"held={held} circ={np.linalg.norm(circ):.4f}")

# T3 structured decoherence -> combined read beats consensus-collapse
z=make_world(3); branch=rng_global.integers(0,2,T); ext=np.zeros((T,D)); ext[:,0]=(branch*2-1)*3
sm=[iface(z,20,extra=ext),iface(z,21),iface(z,22)]
st,circ,res,held=settle(sm,z)
Wc,*_=np.linalg.lstsq(np.column_stack([st[:ntr],np.ones(ntr)]),branch[:ntr].astype(float),rcond=None)
cons=acc(np.column_stack([st[ntr:],np.ones(T-ntr)])@Wc,branch[ntr:])
Wh,*_=np.linalg.lstsq(np.column_stack([circ[:ntr],np.ones(ntr)]),branch[:ntr].astype(float),rcond=None)
comb=acc(np.column_stack([circ[ntr:],np.ones(T-ntr)])@Wh,branch[ntr:])
R['T3 structured read > consensus']=(held>=1 and comb>cons+0.15, f"held={held} consensus={cons:.3f} combined={comb:.3f}")

# T4 noise decoherence -> rejected, no circulation (no G1)
z=make_world(4); noisy=iface(z,30); noisy=noisy+2.0*np.random.default_rng(99).normal(size=(T,D))
nm=[noisy,iface(z,31),iface(z,32)]
st,circ,res,held=settle(nm,z)
R['T4 noise -> rejected (no G1)']=(held==0 and np.linalg.norm(circ)<0.5, f"held={held} circ={np.linalg.norm(circ):.4f}")

# T5 settling stable (contracts, never amplifies) across all above
z=make_world(5); mm=[iface(z,40,extra=np.tile((rng_global.integers(0,2,T)*2-1)[:,None]*2.0,(1,D))*np.eye(D)[0]),iface(z,41)]
st,circ,res,held=settle(mm,z)
R['T5 settling contracts (stable)']=(res[-1]<res[0], f"res {res[0]:.3f}->{res[-1]:.3f}")

# T6 SPECIFICITY: does it circulate ONLY the structured dim, not spray into others? (not cheating)
z=make_world(6); branch=rng_global.integers(0,2,T); ext=np.zeros((T,D)); ext[:,0]=(branch*2-1)*3
sm=[iface(z,50,extra=ext),iface(z,51),iface(z,52)]
st,circ,res,held=settle(sm,z)
# circulated energy should concentrate; measure fraction of circ energy in its top-1 direction
U,S,Vt=np.linalg.svd(circ-circ.mean(0),full_matrices=False)
conc=(S[0]**2)/((S**2).sum()+1e-9)
R['T6 circulation concentrated (not spray)']=(conc>0.4, f"top-dir energy frac={conc:.3f}")

# T7 FALSIFICATION: give it PURE noise as "structure" -> must NOT hold it (guard against false positives)
z=make_world(7); fake=iface(z,60)+1.5*np.random.default_rng(7).normal(size=(T,D))
fm=[fake,iface(z,61),iface(z,62)]
st,circ,res,held=settle(fm,z)
R['T7 pure-noise NOT held (falsification)']=(held==0, f"held={held} (must be 0)")

print("="*70); print("INTERNAL SATISFACTION BATTERY"); print("="*70)
allp=True
for k,(ok,detail) in R.items():
    allp = allp and ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"          {detail}")
print("="*70)
print(f"ALL PASS: {allp}")
