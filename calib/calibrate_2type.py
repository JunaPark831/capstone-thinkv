"""Two-type (|T|=2) fallback calibration on layer 4.

This script is the post-hoc exploratory fallback described in
docs/e3_branch_a_2type_fallback.md (§6). The preregistered three-mode
branch-A criterion in docs/e3_branch_a_criteria.md was NOT met; this
analysis does not re-open or relax that criterion.

Fixed analysis settings (doc §6):
  - candidate layer: 4 (0-indexed)
  - KDE bandwidth: Scott base factor x1 (same base as the preregistered check)
  - evaluation interval: [0, 1]
  - evaluation grid: 512 points
  - density normalization: divide each KDE by its own max
  - peak detection: scipy.signal.find_peaks, prominence 0.05
  - qualifying prompts: those with exactly 2 peaks at Scott x1 on layer 4
  - per-problem threshold theta_p: KDE minimum between the two peaks
  - final threshold theta: median of the per-problem thresholds

Fixed input (doc §2): the first 8 files of the merged Qwen-7B run,
0000.pt .. 0007.pt. Files 0008.pt+ are excluded from this fixed input.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


CALIB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CALIB_DIR.parent

DEFAULT_INPUT_DIR = CALIB_DIR / "sparsity" / "qwen7b_s1k_p11_max35000"
DEFAULT_OUTPUT_JSON = CALIB_DIR / "calib_2type.json"
DEFAULT_ARTIFACT_DIR = CALIB_DIR / "two_type_artifacts"

# Fixed input indices for the P=8 snapshot (doc §2).
SOURCE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]

# Fixed analysis settings (doc §6). Kept identical to criteria_checker.py.
LAYER = 4
GRID = np.linspace(0.0, 1.0, 512)
BANDWIDTH_SCALE = 1.0  # Scott base factor x1
PROMINENCE = 0.05

# collect_sparsity.py saved model_name=None in this run, so the model is
# recorded here from the documented collection condition (doc §2), not read
# from the .pt files.
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-type fallback calibration on layer 4 (doc §6)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with the fixed 7B/s1K .pt files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"Figure output directory (default: {DEFAULT_ARTIFACT_DIR})",
    )
    return parser.parse_args()


def load_layer_values(input_dir: Path) -> list[dict[str, Any]]:
    """Load layer-4 sparsity for the fixed indices 0000.pt .. 0007.pt."""
    records: list[dict[str, Any]] = []
    for idx in SOURCE_INDICES:
        path = input_dir / f"{idx:04d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing fixed input file: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)

        for key in ("idx", "sparsity", "token_ids"):
            if key not in saved:
                raise RuntimeError(f"{path.name} is missing key: {key}")
        if int(saved["idx"]) != idx:
            raise RuntimeError(f"{path.name} has idx={saved['idx']}, expected {idx}")

        sparsity = saved["sparsity"]
        if not isinstance(sparsity, torch.Tensor) or sparsity.ndim != 2:
            raise RuntimeError(f"{path.name} has invalid sparsity tensor")
        if sparsity.shape[0] <= LAYER:
            raise RuntimeError(
                f"{path.name} has {sparsity.shape[0]} layers, cannot index layer {LAYER}"
            )

        values = sparsity[LAYER].float().numpy().astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{path.name} layer {LAYER} contains NaN or Inf")
        if values.min() < 0.0 or values.max() > 1.0:
            raise RuntimeError(f"{path.name} layer {LAYER} outside [0, 1]")
        if values.size < 2:
            raise RuntimeError(f"{path.name} layer {LAYER} has too few samples for KDE")
        if np.var(values) == 0.0:
            raise RuntimeError(f"{path.name} layer {LAYER} has zero variance")

        records.append(
            {
                "idx": idx,
                "path": path,
                "steps": int(values.size),
                "values": values,
            }
        )
    return records


def normalized_kde(values: np.ndarray) -> np.ndarray:
    """KDE on GRID, Scott base factor x BANDWIDTH_SCALE, max-normalized.

    Matches criteria_checker.normalized_kde: compute the Scott factor from
    the data first, then multiply by the scale.
    """
    scott_factor = float(gaussian_kde(values, bw_method="scott").factor)
    kde = gaussian_kde(values, bw_method=scott_factor * BANDWIDTH_SCALE)
    density = np.asarray(kde(GRID), dtype=np.float64)
    maximum = float(density.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise RuntimeError("KDE produced an invalid density")
    return density / maximum


def analyze(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute per-prompt density, peaks, and (if 2 peaks) the valley theta_p."""
    per_prompt: list[dict[str, Any]] = []
    for record in records:
        density = normalized_kde(record["values"])
        peaks, _ = find_peaks(density, prominence=PROMINENCE)
        entry: dict[str, Any] = {
            "idx": record["idx"],
            "steps": record["steps"],
            "density": density,
            "peaks": peaks.tolist(),
            "peak_count": int(peaks.size),
            "qualifies": peaks.size == 2,
            "theta_p": None,
            "valley_idx": None,
            "peak_positions": [float(GRID[p]) for p in peaks.tolist()],
        }
        if peaks.size == 2:
            left, right = int(peaks[0]), int(peaks[1])
            between = density[left : right + 1]
            valley_idx = left + int(np.argmin(between))
            entry["valley_idx"] = valley_idx
            entry["theta_p"] = float(GRID[valley_idx])
        per_prompt.append(entry)
    return per_prompt


def kde_view_xlim(per_prompt: list[dict[str, Any]]) -> tuple[float, float]:
    """x-range where the layer-4 KDE mass actually lives.

    All the structure (peaks and valleys) sits near sparsity ~0.9-1.0, so the
    full [0, 1] view wastes most of the width. Zoom to the smallest window that
    still contains every curve's visible density, padded a little on the left.
    """
    threshold = 0.01  # ignore the near-zero tail
    left = 1.0
    for entry in per_prompt:
        density = entry["density"]
        above = np.nonzero(density >= threshold)[0]
        if above.size:
            left = min(left, float(GRID[above[0]]))
    left = max(0.0, left - 0.02)
    # Round down to a tidy gridline so the axis reads cleanly.
    left = float(np.floor(left * 20) / 20)
    return left, 1.0


def _draw_kde_curve(
    axis: "plt.Axes",
    entry: dict[str, Any],
    color: Any,
    *,
    emphasize: bool,
    label: str | None,
) -> None:
    """Draw one prompt's KDE curve with its peak/valley markers."""
    qual = entry["qualifies"]
    axis.plot(
        GRID,
        entry["density"],
        color=color,
        linewidth=1.8 if emphasize else (1.6 if qual else 0.9),
        alpha=0.95 if emphasize else (0.9 if qual else 0.4),
        label=label,
    )
    for p in entry["peaks"]:
        axis.plot(GRID[p], entry["density"][p], "o", color=color, markersize=4)
    if qual:
        vx = GRID[entry["valley_idx"]]
        vy = entry["density"][entry["valley_idx"]]
        axis.plot(vx, vy, "v", color=color, markersize=9)
        axis.axvline(vx, color=color, linestyle=":", linewidth=0.9, alpha=0.6)


def plot_kde_valleys(per_prompt: list[dict[str, Any]], artifact_dir: Path) -> list[Path]:
    """Render the layer-4 KDE/valley views.

    Produces two files:
      - layer04_kde_valleys.png: one combined overlay of every prompt
      - layer04_kde_valleys_grid.png: one small-multiple subplot per prompt

    Both zoom the x-axis to the region where the density actually lives so the
    two peaks and the valley between them are legible instead of being crushed
    against the right edge.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    xlim = kde_view_xlim(per_prompt)

    thetas = [e["theta_p"] for e in per_prompt if e["qualifies"]]
    theta = float(np.median(thetas)) if thetas else None

    # --- Combined overlay ---------------------------------------------------
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for i, entry in enumerate(per_prompt):
        label = f"P{entry['idx']} ({entry['peak_count']} peaks)" + (
            f", θ_p={entry['theta_p']:.3f}" if entry["qualifies"] else ""
        )
        _draw_kde_curve(
            axis, entry, colors[i % len(colors)], emphasize=False, label=label
        )
    if theta is not None:
        axis.axvline(
            theta,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=f"θ (median) = {theta:.3f}",
        )
    axis.set_xlim(*xlim)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Layer-4 sparsity")
    axis.set_ylabel("Normalized KDE")
    axis.set_title(
        f"Two-type fallback — layer {LAYER} KDE & valleys "
        f"(Scott x{BANDWIDTH_SCALE:g}, prominence={PROMINENCE:g}, P={len(per_prompt)})"
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="upper left")
    combined_path = artifact_dir / "layer04_kde_valleys.png"
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)

    # --- Per-example small multiples ----------------------------------------
    count = len(per_prompt)
    ncols = 2 if count <= 4 else 3
    nrows = (count + ncols - 1) // ncols
    grid_fig, grid_axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.0 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = grid_axes.ravel()
    for i, entry in enumerate(per_prompt):
        ax = flat_axes[i]
        color = colors[i % len(colors)]
        _draw_kde_curve(ax, entry, color, emphasize=True, label=None)
        if theta is not None:
            ax.axvline(theta, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)
        title = f"P{entry['idx']} — {entry['peak_count']} peaks"
        if entry["qualifies"]:
            title += f" (θ_p={entry['theta_p']:.3f})"
        ax.set_title(title, fontsize=10)
        if i % ncols == 0:
            ax.set_ylabel("Normalized KDE")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Layer-4 sparsity")

    for j in range(count, nrows * ncols):
        flat_axes[j].axis("off")

    grid_fig.suptitle(
        f"Two-type fallback — layer {LAYER} KDE per prompt "
        f"(Scott x{BANDWIDTH_SCALE:g}, prominence={PROMINENCE:g})",
        fontsize=12,
    )
    grid_path = artifact_dir / "layer04_kde_valleys_grid.png"
    grid_fig.savefig(grid_path, dpi=180)
    plt.close(grid_fig)

    return [combined_path, grid_path]


def plot_theta_timeline(
    record_values: dict[int, np.ndarray],
    per_prompt: list[dict[str, Any]],
    theta: float,
    artifact_dir: Path,
) -> list[Path]:
    """Per-qualifying-prompt layer-4 sparsity timeline with theta and MA(128)."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    window = 128
    for entry in per_prompt:
        if not entry["qualifies"]:
            continue
        idx = entry["idx"]
        values = record_values[idx]
        steps = np.arange(values.size)

        fig, axis = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
        axis.plot(steps, values, color="#4c78a8", linewidth=0.4, alpha=0.6, label="sparsity")
        if values.size >= window:
            kernel = np.ones(window) / window
            ma = np.convolve(values, kernel, mode="valid")
            ma_x = np.arange(window - 1, values.size)
            axis.plot(ma_x, ma, color="#f58518", linewidth=1.2, label=f"MA({window})")
        axis.axhline(
            entry["theta_p"],
            color="#54a24b",
            linestyle=":",
            linewidth=1.4,
            label=f"θ_p={entry['theta_p']:.3f}",
        )
        axis.axhline(theta, color="black", linestyle="--", linewidth=1.2, label=f"θ={theta:.3f}")

        # Real sparsity values sit almost entirely in ~0.8-1.0. A linear y-axis
        # wastes the lower 80% of the panel. Apply a power transform y->y^k so
        # the bottom is compressed and the top is stretched, keeping the full
        # [0, 1] range visible but giving most height to the high band.
        power = 4.0
        axis.set_yscale(
            "function",
            functions=(
                lambda y, p=power: np.power(np.clip(y, 0.0, 1.0), p),
                lambda y, p=power: np.power(np.clip(y, 0.0, 1.0), 1.0 / p),
            ),
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        axis.set_xlabel("Decode step")
        axis.set_ylabel("Layer-4 sparsity")
        axis.set_title(f"P{idx}: layer-{LAYER} sparsity over decode steps")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="upper right", ncol=4)

        path = artifact_dir / f"layer04_theta_timeline_{idx:04d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_json = args.output_json.resolve()
    artifact_dir = args.artifact_dir.resolve()

    records = load_layer_values(input_dir)
    record_values = {r["idx"]: r["values"] for r in records}
    per_prompt = analyze(records)

    qualifying = [e for e in per_prompt if e["qualifies"]]
    qualifying_indices = [e["idx"] for e in qualifying]
    theta_per_problem = [e["theta_p"] for e in qualifying]

    if not theta_per_problem:
        raise RuntimeError(
            "No qualifying 2-peak prompts on layer 4; cannot compute theta."
        )

    final_theta = float(np.median(theta_per_problem))

    kde_images = plot_kde_valleys(per_prompt, artifact_dir)
    kde_combined, kde_grid = kde_images[0], kde_images[1]
    timeline_images = plot_theta_timeline(
        record_values, per_prompt, final_theta, artifact_dir
    )

    result = {
        "schema_version": 1,
        "method": "branch_a_two_type_fallback",
        "analysis_status": "post_hoc_exploratory",
        "analysis_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "P": len(SOURCE_INDICES),
        "source_dir": relative_path(input_dir),
        "source_indices": SOURCE_INDICES,
        "num_thought_types": 2,
        "layer": LAYER,
        "bandwidth": "scott_x1",
        "grid_points": int(GRID.size),
        "grid_min": 0.0,
        "grid_max": 1.0,
        "density_normalization": "divide_each_curve_by_its_max",
        "prominence": PROMINENCE,
        "peak_count_by_prompt": {
            str(e["idx"]): e["peak_count"] for e in per_prompt
        },
        "peak_positions_by_prompt": {
            str(e["idx"]): e["peak_positions"] for e in per_prompt
        },
        "qualifying_prompt_indices": qualifying_indices,
        "theta_per_problem": theta_per_problem,
        "theta": final_theta,
        "original_three_type_criterion_passed": False,
        "artifacts": {
            "layer04_kde_valleys": relative_path(kde_combined),
            "layer04_kde_valleys_grid": relative_path(kde_grid),
            "layer04_theta_timelines": [relative_path(p) for p in timeline_images],
        },
    }

    atomic_write_text(
        output_json,
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )

    print(f"Qualifying prompts (2 peaks): {qualifying_indices}")
    print(f"theta_per_problem: {theta_per_problem}")
    print(f"final theta (median): {final_theta:.6f}")
    print(f"JSON: {output_json}")
    print(f"KDE figure (combined): {kde_combined}")
    print(f"KDE figure (grid): {kde_grid}")
    for p in timeline_images:
        print(f"Timeline figure: {p}")


if __name__ == "__main__":
    main()
