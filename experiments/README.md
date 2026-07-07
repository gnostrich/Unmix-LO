# Experiments — what each one established

Run any with `python experiments/<file>.py`. Order tells the story: mechanism -> negative that
forced the ICA correction -> conditional positive -> the diversity solvent.

| file | question | result |
|---|---|---|
| `01_saturation_and_transfer.py` | does a library saturate at #primitives and transfer? | saturates at K=P (not #tasks); recovers subspace; routing helps only when compositions SEPARATE primitives |
| `02_sparse_routing_fails_secondorder.py` | does second-order (projector-avg) routing win under sparse activation? | NO — monolithic >= routed; second-order can't resolve individuals |
| `03_operator_vs_delta_neural.py` | on a real-ish neural net, is family structure in deltas or curvature? | thin either way; high overlap is task-GENERIC (~1.04-1.13x family signal). The negative prior. |
| `04_ica_extraction_gate.py` | does higher-order (ICA) extraction make routed composition win? | YES, conditionally: ~20x over vanilla when primitives separable; collapses (0.01x) when correlated |
| `05_diversity_restores_identifiability.py` | does trainee/genre DIVERSITY restore separability? | YES: recovery 0.33->0.78 as genres pooled; truly-fused pairs correctly stay at 0.707 |

`_shared_neural_mlp.py` is a dependency of `03` (tiny numpy MLP + DiLoCo pseudo-gradients).

## The through-line
Second-order extraction is insufficient (01,02); on real-ish nets shared structure is thin/generic
(03); higher-order ICA extraction works IF primitives are separable (04); diversity is what makes
them separable (05). The only untested link: do REAL gradients separate this way. See ../GATE.md.
