"""
registry/registry.py — append-only run ledger + preflight (§9). Constants are frozen (§10); preflight
refuses to run if a frozen constant was moved. Every gate/event appends a record.
"""
import os, json, hashlib

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "REGISTRY.jsonl")

# Frozen by pre-registration (§10). Moving any of these invalidates a result.
FROZEN = {"lambda_grow": 0.05, "gamma": 1.0, "eta": 1.0, "W": 50, "L": 12,
          "k_max": 6, "T_max": 20, "m0": 4, "probes": 16}


def preflight(config):
    """Refuse runs that move a frozen constant."""
    bad = {k: config[k] for k in FROZEN if k in config and config[k] != FROZEN[k]}
    if bad:
        raise RuntimeError(f"preflight: frozen constant(s) moved {bad} (§10) — result would be dead")
    return True


def _stamp(record, when):
    record = dict(record)
    record["ts"] = when
    record["hash"] = hashlib.sha1(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]
    return record


def append(record, when="unstamped"):
    """Append a JSON record (caller supplies a timestamp string; time is not read here for determinism)."""
    rec = _stamp(record, when)
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def read_all():
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]
