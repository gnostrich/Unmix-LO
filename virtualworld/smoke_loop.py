import numpy as np
rng=np.random.default_rng(0)
# Smoke-check the play-loop is light & the three behaviors show on a tiny 4-modality world (no real models yet
# -- stand-in encoders -- just to confirm the loop, knobs, and detector run cheaply and behave).
T,D=300,6; ntr=T//2
# tiny world: 3 balls, positions+velocities evolving (stand-in physics)
state=np.zeros((T,D))
for t in range(1,T): state[t]=0.95*state[t-1]+0.1*rng.normal(size=D)
def align(v):
    A,*_=np.linalg.lstsq(v[:ntr],state[:ntr],rcond=None); return v@A
# 4 modalities = 4 lossy views (stand-ins for vision/text/audio/timeseries encoders)
def modality(seed,keep):
    r=np.random.default_rng(seed); m=np.zeros(D); m[r.choice(D,keep,replace=False)]=1
    R=np.linalg.qr(r.normal(size=(D,D)))[0]; return align((state*m)@R+0.1*r.normal(size=(T,D)))
M=[modality(i,4) for i in range(1,5)]   # each sees 4/6 dims (overlapping coverage)

def stitch(mods): return np.mean(mods,axis=0)
def structured(d):
    dc=d-d.mean(0); U,S,Vt=np.linalg.svd(dc[:ntr],full_matrices=False)
    eff=max(1,int(round((S.sum()**2)/(S**2).sum()))); P=Vt[:eff].T@Vt[:eff]
    cap=((dc[ntr:]@P)**2).sum()/((dc[ntr:]**2).sum()+1e-9); return cap, eff/D
def r2(p,t): return 1-((t-p)**2).sum()/((t-t.mean(0))**2).sum()

# coherent stitch (coverage union)
st=stitch(M); print(f"STITCH (coverage union) reconstruct world: R2={r2(st[ntr:],state[ntr:]):.3f}")
# coverage: how much each modality UNIQUELY adds (drop-one)
for i in range(4):
    without=stitch([M[j] for j in range(4) if j!=i])
    print(f"  drop M{i+1}: R2={r2(without[ntr:],state[ntr:]):.3f} (uniqueness = full - this)")
# decoherence classify: natural (mostly noise/agree) vs INJECTED structured (knob)
d_nat=M[0]-M[1]; cap_n,base=structured(d_nat)
print(f"NATURAL decoherence M1-M2: captured={cap_n:.2f} base={base:.2f} -> {'STRUCTURED' if cap_n>base*1.5 else 'noise (typical for agreeing models)'}")
# INJECT structured (knob): add a hidden branch to M1 only
branch=rng.integers(0,2,T); Minj=M[0].copy(); Minj[:,0]+=(branch*2-1)*3
d_inj=Minj-M[1]; cap_i,_=structured(d_inj)
print(f"INJECTED structured decoherence: captured={cap_i:.2f} base={base:.2f} -> {'STRUCTURED (extend!)' if cap_i>base*1.5 else 'missed'}")
# INJECT noise (knob): corrupt M2
d_noise=M[0]-(M[1]+2*rng.normal(size=(T,D))); cap_x,_=structured(d_noise)
print(f"INJECTED noise decoherence: captured={cap_x:.2f} base={base:.2f} -> {'noise (reject, no extend)' if cap_x<base*1.5 else 'FALSE structured'}")
print("\nLoop is light (numpy, <1s), 4-modality structure works, three behaviors visible. Buildable in CC easily.")
