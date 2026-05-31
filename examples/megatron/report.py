"""Render the HTML validation report (per-case tables + embedded plots) from compare.py rows.

Plots are matplotlib PNGs encoded inline as base64 ``data:`` URIs so ``report.html`` is a
single self-contained file. ``main()`` reads ``results/summary.json`` and writes ``report.html``.
"""
import argparse
import base64
import io
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _fmt_pct(p):
    return "n/a" if p is None else f"{p * 100:.1f}%"


def _plot_model(model, rows):
    """Grouped bar chart (sim vs real for step time + peak memory across cases) → base64 PNG."""
    cases = [r["case"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in (
        (axes[0], "step_time_ms", "Step time (ms)"),
        (axes[1], "peak_memory_gb", "Peak memory (GB)"),
    ):
        real = [(r["real"].get(key) or 0) for r in rows]
        sim = [(r["sim"].get(key) or 0) for r in rows]
        x = range(len(cases))
        ax.bar([i - 0.2 for i in x], real, width=0.4, label="real")
        ax.bar([i + 0.2 for i in x], sim, width=0.4, label="sim")
        ax.set_xticks(list(x))
        ax.set_xticklabels(cases)
        ax.set_title(f"{model} — {title}")
        ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def render_html(rows, tol=0.10):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    n_pass = sum(1 for r in rows if r["pass"])
    parts = [
        "<html><head><meta charset='utf-8'><title>Megatron vs SysSim — GH200 validation</title>",
        "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:4px 8px}.pass{color:green;font-weight:bold}"
        ".fail{color:red;font-weight:bold}</style></head><body>",
        "<h1>Megatron vs SysSim — GH200 validation</h1>",
        f"<p>Target: within {tol * 100:.0f}% on step time <b>and</b> peak memory. "
        f"Passing: {n_pass}/{len(rows)} cases.</p>",
        "<table><tr><th>Model</th><th>Case</th><th>Step real/sim (ms)</th><th>Step %err</th>"
        "<th>Mem real/sim (GB)</th><th>Mem %err</th><th>Result</th></tr>",
    ]
    for r in rows:
        verdict = "PASS" if r["pass"] else "FAIL"
        cls = "pass" if r["pass"] else "fail"
        parts.append(
            f"<tr><td>{r['model']}</td><td>{r['case']}</td>"
            f"<td>{r['real'].get('step_time_ms')} / {r['sim'].get('step_time_ms')}</td>"
            f"<td>{_fmt_pct(r['step_time_pct'])}</td>"
            f"<td>{r['real'].get('peak_memory_gb')} / {r['sim'].get('peak_memory_gb')}</td>"
            f"<td>{_fmt_pct(r['memory_pct'])}</td>"
            f"<td class='{cls}'>{verdict}</td></tr>"
        )
    parts.append("</table>")

    for model, mrows in by_model.items():
        png = _plot_model(model, mrows)
        parts.append(f"<h2>{model}</h2><img alt='{model} sim vs real' "
                     f"src='data:image/png;base64,{png}'/>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="docs/megatron_gh200_validation/results/summary.json")
    ap.add_argument("--out", default="docs/megatron_gh200_validation/report.html")
    ap.add_argument("--tol", type=float, default=0.10)
    args = ap.parse_args()
    with open(args.summary) as f:
        rows = json.load(f)
    html = render_html(rows, tol=args.tol)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
