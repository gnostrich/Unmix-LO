# BASELINES — must be FAIR (same specialists, same tasks, same accuracy target)
- best_single.py       : strongest individual specialist per query (floor).
- static_pooling.py    : BioVERSE-style — align all N specialists, pool features once, decode. Cost ~ O(N)/query.
- orchestration.py     : Het-MedAgent-style — LLM calls each specialist as a tool, stitches outputs. Cost ~ O(N) calls + LLM overhead.
The BIOMESH layer's whole claim is cost-vs-N FLATTER than these at EQUAL accuracy. If it can't match
their accuracy, it does not win — cheaper-but-worse is not the claim.
