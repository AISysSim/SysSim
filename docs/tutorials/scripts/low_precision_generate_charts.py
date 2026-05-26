"""Generate Tutorial 4 comparison charts from generated JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = REPO_ROOT / "docs/tasks/results"
ASSET_DIR = REPO_ROOT / "docs/tasks/assets"
ACTUAL_JSON = RESULT_DIR / "low_precision_actual_h100.json"
ROOFLINE_JSON = RESULT_DIR / "low_precision_roofline_h100.json"
PROFILE_JSON = RESULT_DIR / "low_precision_profile_model_h100.json"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.relative_to(REPO_ROOT)}. Run the earlier tutorial scripts first.")
    return json.loads(path.read_text())


def collect_wall_time_rows() -> list[tuple[str, float]]:
    actual = read_json(ACTUAL_JSON)
    roofline = read_json(ROOFLINE_JSON)
    profile = read_json(PROFILE_JSON)

    rows: list[tuple[str, float]] = []
    for row in actual.get("results", []):
        if row.get("wall_time_ms") is not None:
            rows.append((row["label"], row["wall_time_ms"]))
    for row in roofline.get("predictions", []):
        rows.append((row["label"], row["wall_time_ms"]))
    for row in profile.get("predictions", []):
        rows.append((row["label"], row["wall_time_ms"]))
    return rows


def collect_memory_rows() -> list[tuple[str, float]]:
    actual = read_json(ACTUAL_JSON)
    roofline = read_json(ROOFLINE_JSON)

    rows: list[tuple[str, float]] = []
    for row in actual.get("results", []):
        if row.get("peak_memory_gb") is not None:
            rows.append((row["label"], row["peak_memory_gb"]))
    for row in roofline.get("predictions", []):
        rows.append((row["label"], row["memory"]["total_model_state_gb"]))
    return rows


def bar_chart(rows: list[tuple[str, float]], ylabel: str, title: str, output_path: Path) -> None:
    labels = [name.replace("_", "\n") for name, _ in rows]
    values = [value for _, value in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1"][: len(rows)]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    wall_time_png = ASSET_DIR / "low_precision_h100_wall_time.png"
    memory_png = ASSET_DIR / "low_precision_h100_memory.png"

    bar_chart(
        collect_wall_time_rows(),
        "Wall time per forward + backward step (ms)",
        "Qwen3.5-9B low-precision wall time on 1 x H100",
        wall_time_png,
    )
    bar_chart(
        collect_memory_rows(),
        "Memory (GB)",
        "Measured peak memory and SysSim model-state memory",
        memory_png,
    )
    print(f"Wrote {wall_time_png.relative_to(REPO_ROOT)}")
    print(f"Wrote {memory_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
