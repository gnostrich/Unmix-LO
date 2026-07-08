# What "cost" means (measure all three, report separately)
1. Per-query inference cost: FLOPs / specialist-calls / latency to answer one query.
2. Coordination/memory cost: size of the routing memory / kernel that must be maintained, vs N.
3. Upfront alignment cost: anchors + alignment compute per specialist (the maintained artifact) — one-time.
The G2 claim is specifically about (1) and (2) staying flat as N grows. (3) is the honest upfront price.
