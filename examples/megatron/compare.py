"""Join real (Megatron) + simulated (SysSim) per-case results and compute %error.

Each case writes two files in the results dir: ``<model>_<case>.real.json`` and
``<model>_<case>.sim.json``, each with ``step_time_ms`` and ``peak_memory_gb``.
``main()`` joins them by ``<model>_<case>`` into ``summary.json`` (a list of rows).
"""
import argparse
import glob
import json
import os


def pct_error(sim, real):
    """Relative error ``|sim - real| / real``; ``None`` if either is missing or real==0."""
    if sim is None or real is None or real == 0:
        return None
    return abs(sim - real) / real


def join_case(model, case, real, sim, tol=0.10):
    """Build one comparison row.

    Two regimes:
    * If either side OOMs, the test is whether SysSim predicts OOM correctly (``real_oom ==
      sim_oom``); peak magnitudes aren't comparable (real is measured at the OOM point, sim is
      the would-be peak), so no %error gate.
    * Otherwise both ran: ``pass`` requires step time AND peak memory within ``tol``.
    """
    real_oom = bool(real.get("oom"))
    sim_oom = bool(sim.get("oom"))
    step_time_pct = pct_error(sim.get("step_time_ms"), real.get("step_time_ms"))
    memory_pct = pct_error(sim.get("peak_memory_gb"), real.get("peak_memory_gb"))
    if real_oom or sim_oom:
        passed = real_oom == sim_oom
    else:
        available = [p for p in (step_time_pct, memory_pct) if p is not None]
        passed = bool(available) and all(p <= tol for p in available)
    return {
        "model": model,
        "case": case,
        "real": real,
        "sim": sim,
        "real_oom": real_oom,
        "sim_oom": sim_oom,
        "step_time_pct": step_time_pct,
        "memory_pct": memory_pct,
        "pass": passed,
    }


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="docs/megatron_gh200_validation/results")
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for real_path in sorted(glob.glob(os.path.join(args.results, "*.real.json"))):
        key = os.path.basename(real_path)[: -len(".real.json")]
        sim_path = os.path.join(args.results, key + ".sim.json")
        if not os.path.exists(sim_path):
            continue
        model, _, case = key.partition("_")  # model slugs use '-'; '_' separates the case
        rows.append(join_case(model, case, _load(real_path), _load(sim_path), tol=args.tol))

    out = args.out or os.path.join(args.results, "summary.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
