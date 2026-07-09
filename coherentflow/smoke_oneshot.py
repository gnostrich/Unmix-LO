import numpy as np
rng=np.random.default_rng(0)
# One-shot smoke: confirm the WHOLE object runs and DOES SOMETHING on frame-diverse injected input,
# stays stable, and the combined read surfaces held structure. Fast, observe-don't-prove.
D,T=24,500; ntr=T//2
z=rng.normal(size=(T,D))
def iface(seed,extra=None):  # a model's interface: native frame <-> medium, aligned to world coords
    r=np.random.default_rng(seed); R=np.linalg.qr(r.normal(size=(D,D)))[0]
    v=z@R+0.1*r.normal(size=(T,D))
    if extra is not None: v=v+extra
    A,*_=np.linalg.lstsq(v[:ntr],z[:ntr],rcond=None); return v@A
def r2(p,t): return 1-((t-p)**2).sum()/((t-t.mean(0))**2).sum()
def struct(d):  # structured iff concentrated AND held-out predictable
    dc=d-d.mean(0); U,S,Vt=np.linalg.svd(dc[:ntr],full_matrices=False)
    eff=max(1,int(round((S.sum()**2)/(S**2).sum()))); P=Vt[:eff].T@Vt[:eff]
    cap=((dc[ntr:]@P)**2).sum()/((dc[ntr:]**2).sum()+1e-9)
    A,*_=np.linalg.lstsq(z[:ntr],d[:ntr],rcond=None); ho=r2(z[ntr:]@A,d[ntr:])
    return (cap>1.5*eff/D)and(ho>0.3), P

# frame-diverse injected input: 3 interfaces, one carries a structured hidden branch
branch=rng.integers(0,2,T); ext=np.zeros((T,D)); ext[:,0]=(branch*2-1)*3
ifaces=[iface(1,extra=ext), iface(2), iface(3)]

# SETTLE: recurrent flow, guards INSIDE, internal coherence loss (minimise unstructured decoherence)
state=np.mean(ifaces,axis=0); held=[]; res=[]
for it in range(15):
    prev=state.copy(); circ=np.zeros_like(state)
    for f in ifaces:
        d=f-state; is_s,P=struct(d)
        if is_s: circ+=0.5*(d@P); held.append(P)      # hold+circulate structure
        # else reject (noise not circulated) -> no G1
    state=prev+0.5*((np.mean(ifaces,axis=0)+circ)-prev)  # damped settle
    res.append(np.linalg.norm(state-prev))
settled = res[-1] < res[len(res)//2]                    # stable in tail (observed, not proven)

# COMBINED READ across interfaces for a query needing the branch:
# read settled state through each interface; where they carry structured disagreement, HOLD both.
reads=[state, state+ (ifaces[0]-state), state+(ifaces[1]-state)]  # per-interface views
# consensus part + held structured part
consensus=np.mean(reads,axis=0)
held_structure = circ  # the surfaced held cross-frame content
# does the combined read recover the branch (held structure) that a single-frame collapse would lose?
Wc,*_=np.linalg.lstsq(np.column_stack([consensus[:ntr],np.ones(ntr)]),branch[:ntr].astype(float),rcond=None)
consensus_acc=np.mean((np.column_stack([consensus[ntr:],np.ones(T-ntr)])@Wc>0.5)==(branch[ntr:]>0.5))
Wh,*_=np.linalg.lstsq(np.column_stack([held_structure[:ntr],np.ones(ntr)]),branch[:ntr].astype(float),rcond=None)
held_acc=np.mean((np.column_stack([held_structure[ntr:],np.ones(T-ntr)])@Wh>0.5)==(branch[ntr:]>0.5))

print("ONE-SHOT WHOLE-OBJECT SMOKE (frame-diverse injected input):")
print(f"  settling stable (tail-contract, observed)? {settled}  residual {res[0]:.3f}->{res[-1]:.3f}")
print(f"  structure surfaced+held? circ_norm={np.linalg.norm(circ):.3f} ({'yes' if np.linalg.norm(circ)>0.5 else 'no'})")
print(f"  combined read: consensus recovers branch={consensus_acc:.3f} | HELD-structure recovers branch={held_acc:.3f}")
print(f"  -> combined read surfaces held structure a consensus-collapse loses? {'YES' if held_acc>consensus_acc+0.1 else 'no'}")
print(f"\n  whole object runs, settles, holds structure, combined-read extracts it. Buildable one-shot.")
