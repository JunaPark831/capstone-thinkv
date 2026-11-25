"""Check the preregistered E3 branch-A criterion on collected sparsity data.

The decision rule is fixed in docs/criteria/e3_branch_a_criteria.md. This script does
not tune thresholds from the observed result. It writes a machine-readable
record, a Markdown summary, three bandwidth-specific KDE figures, and one
cross-bandwidth fraction figure.
"""

from __future__ import annotations

import argparse
import hashlib
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
from tqdm.auto import tqdm


CALIB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CALIB_DIR.parent
DEFAULT_INPUT_DIR = (
    CALIB_DIR / "sparsity" / "aime_llama8b_p10_max35000"
)
DEFAULT_OUTPUT_DIR = (
    CALIB_DIR / "criteria_artifacts" / "aime_llama8b_p10_max35000"
)
CRITERIA_SOURCE = PROJECT_DIR / "docs" / "criteria" / "e3_branch_a_criteria.md"

GRID = np.linspace(0.0, 1.0, 512)
BANDWIDTH_SCALES = (0.5, 1.0, 2.0)
PROMINENCE = 0.05
MIN_THREE_MODE_FRACTION = 0.5
MIN_ROBUST_LAYERS = 4

SCALE_KEYS = {0.5: "scott_x0.5", 1.0: "scott_x1", 2.0: "scott_x2"}
SCALE_FILE_KEYS = {0.5: "scott_x0_5", 1.0: "scott_x1", 2.0: "scott_x2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered E3 branch-A criterion."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing P=10 .pt files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for records and figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--expected-p",
        type=int,
        default=10,
        help="Required number of prompt records (default: 10)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(input_dir: Path, expected_p: int) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("*.pt"))
    expected_names = [f"{idx:04d}.pt" for idx in range(expected_p)]
    actual_names = [path.name for path in paths]
    if actual_names != expected_names:
        raise RuntimeError(
            f"Expected exactly {expected_names}, but found {actual_names} in {input_dir}"
        )

    records: list[dict[str, Any]] = []
    required_keys = {"idx", "problem", "sparsity", "token_ids", "input_len"}
    layer_count: int | None = None

    for expected_idx, path in enumerate(paths):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        missing = required_keys - set(saved)
        if missing:
            raise RuntimeError(f"{path.name} is missing keys: {sorted(missing)}")
        if saved["idx"] != expected_idx:
            raise RuntimeError(
                f"{path.name} has idx={saved['idx']}, expected {expected_idx}"
            )

        sparsity = saved["sparsity"]
        token_ids = saved["token_ids"]
        if not isinstance(sparsity, torch.Tensor) or sparsity.ndim != 2:
            raise RuntimeError(f"{path.name} has invalid sparsity tensor")

        current_layer_count = int(sparsity.shape[0])
        if current_layer_count < 1:
            raise RuntimeError(f"{path.name} has no sparsity layers")
        if layer_count is None:
            layer_count = current_layer_count
        elif current_layer_count != layer_count:
            raise RuntimeError(
                f"{path.name} has {current_layer_count} layers, "
                f"but previous files have {layer_count}"
            )

        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise RuntimeError(f"{path.name} has invalid token_ids tensor")
        if sparsity.shape[1] != token_ids.numel() - 1:
            raise RuntimeError(
                f"{path.name}: sparsity steps={sparsity.shape[1]}, "
                f"generated tokens={token_ids.numel()}"
            )
        if sparsity.shape[1] < 2:
            raise RuntimeError(f"{path.name} has too few sparsity samples for KDE")

        values = sparsity.float().numpy().astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{path.name} contains NaN or Inf")
        if values.min() < 0.0 or values.max() > 1.0:
            raise RuntimeError(f"{path.name} has sparsity outside [0, 1]")
        zero_variance_layers = np.flatnonzero(np.var(values, axis=1) == 0.0)
        if zero_variance_layers.size:
            raise RuntimeError(
                f"{path.name} has zero-variance layers: "
                f"{zero_variance_layers.tolist()}"
            )

        records.append(
            {
                "idx": expected_idx,
                "path": path,
                "sha256": sha256_file(path),
                "problem": saved["problem"],
                "input_len": int(saved["input_len"]),
                "generated_tokens": int(token_ids.numel()),
                "sparsity": values,
            }
        )

    return records


def normalized_kde(
    values: np.ndarray, scott_factor: float, scale: float
) -> np.ndarray:
    kde = gaussian_kde(values, bw_method=scott_factor * scale)
    density = np.asarray(kde(GRID), dtype=np.float64)
    maximum = float(density.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise RuntimeError("KDE produced an invalid density")
    return density / maximum


def analyze(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise RuntimeError("No sparsity records to analyze")

    prompt_count = len(records)
    layer_count = int(records[0]["sparsity"].shape[0])
    scale_count = len(BANDWIDTH_SCALES)
    curves = np.empty(
        (scale_count, prompt_count, layer_count, GRID.size), dtype=np.float32
    )
    peak_counts = np.empty(
        (scale_count, prompt_count, layer_count), dtype=np.int16
    )

    total = prompt_count * layer_count
    with tqdm(total=total, desc="Computing KDE", unit="prompt-layer") as progress:
        for prompt_idx, record in enumerate(records):
            current_layer_count = int(record["sparsity"].shape[0])
            if current_layer_count != layer_count:
                raise RuntimeError(
                    f"Record {record['idx']} has {current_layer_count} layers, "
                    f"expected {layer_count}"
                )
            for layer_idx in range(layer_count):
                values = record["sparsity"][layer_idx]
                scott_factor = float(
                    gaussian_kde(values, bw_method="scott").factor
                )
                for scale_idx, scale in enumerate(BANDWIDTH_SCALES):
                    density = normalized_kde(values, scott_factor, scale)
                    peaks, _ = find_peaks(density, prominence=PROMINENCE)
                    curves[scale_idx, prompt_idx, layer_idx] = density
                    peak_counts[scale_idx, prompt_idx, layer_idx] = len(peaks)
                progress.update(1)

    fractions = (peak_counts == 3).mean(axis=1)
    return curves, peak_counts, fractions


def plot_kde_panels(
    output_dir: Path,
    curves: np.ndarray,
    fractions: np.ndarray,
) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    colors = plt.get_cmap("tab10").colors
    layer_count = int(curves.shape[2])
    ncols = min(8, layer_count)
    nrows = (layer_count + ncols - 1) // ncols

    for scale_idx, scale in enumerate(BANDWIDTH_SCALES):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(2.35 * ncols, 2.55 * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
            squeeze=False,
        )
        flat_axes = axes.flat
        for layer_idx in range(layer_count):
            axis = flat_axes[layer_idx]
            for prompt_idx in range(curves.shape[1]):
                axis.plot(
                    GRID,
                    curves[scale_idx, prompt_idx, layer_idx],
                    color=colors[prompt_idx % len(colors)],
                    linewidth=0.65,
                    alpha=0.55,
                )
            fraction = fractions[scale_idx, layer_idx]
            title_color = (
                "#138a36"
                if fraction >= MIN_THREE_MODE_FRACTION
                else "black"
            )
            axis.set_title(
                f"L{layer_idx:02d} | 3-mode {fraction:.0%}",
                fontsize=9,
                color=title_color,
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.05)
            axis.grid(alpha=0.15, linewidth=0.5)
            if layer_idx >= (nrows - 1) * ncols:
                axis.set_xlabel("Sparsity", fontsize=8)
            if layer_idx % ncols == 0:
                axis.set_ylabel("Normalized KDE", fontsize=8)

        for axis in flat_axes[layer_count:]:
            axis.set_visible(False)

        fig.suptitle(
            f"E3 layer KDEs — Scott bandwidth ×{scale:g} "
            f"(prominence={PROMINENCE:g}, P={curves.shape[1]}, "
            f"layers={layer_count})",
            fontsize=14,
        )
        path = output_dir / f"e3_kde_{SCALE_FILE_KEYS[scale]}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        image_paths[SCALE_KEYS[scale]] = path

    return image_paths


def plot_fraction_summary(
    output_dir: Path,
    fractions: np.ndarray,
    robust_layers: list[int],
    passed: bool,
) -> Path:
    layer_count = int(fractions.shape[1])
    layers = np.arange(layer_count)
    width = 0.25
    fig_width = max(12.0, 0.5 * layer_count)
    fig, axis = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)

    for layer in robust_layers:
        axis.axvspan(layer - 0.48, layer + 0.48, color="#b7e4c7", alpha=0.45)

    colors = ("#4c78a8", "#f58518", "#54a24b")
    for scale_idx, (scale, color) in enumerate(zip(BANDWIDTH_SCALES, colors)):
        offset = (scale_idx - 1) * width
        axis.bar(
            layers + offset,
            fractions[scale_idx],
            width=width,
            label=f"Scott ×{scale:g}",
            color=color,
        )

    axis.axhline(
        MIN_THREE_MODE_FRACTION,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold {MIN_THREE_MODE_FRACTION:.0%}",
    )
    axis.set_xticks(layers)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Fraction of prompts with exactly 3 modes")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, loc="upper right")
    result_label = "PASS" if passed else "FAIL"
    axis.set_title(
        f"E3 branch-A criterion: {result_label} — "
        f"{len(robust_layers)} robust layers (required ≥{MIN_ROBUST_LAYERS})"
    )

    path = output_dir / "e3_three_mode_fractions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def build_result(
    records: list[dict[str, Any]],
    peak_counts: np.ndarray,
    fractions: np.ndarray,
    image_paths: dict[str, Path],
    summary_image: Path,
) -> dict[str, Any]:
    scale_results: dict[str, Any] = {}
    qualifying_by_scale = fractions >= MIN_THREE_MODE_FRACTION
    layer_count = int(fractions.shape[1])

    for scale_idx, scale in enumerate(BANDWIDTH_SCALES):
        scale_results[SCALE_KEYS[scale]] = {
            "scale": scale,
            "three_mode_fraction_by_layer": fractions[scale_idx].tolist(),
            "three_mode_count_by_layer": (peak_counts[scale_idx] == 3)
            .sum(axis=0)
            .astype(int)
            .tolist(),
            "peak_count_by_prompt_and_layer": peak_counts[scale_idx]
            .astype(int)
            .tolist(),
            "qualifying_layers": np.flatnonzero(
                qualifying_by_scale[scale_idx]
            ).tolist(),
            "kde_image": relative_path(image_paths[SCALE_KEYS[scale]]),
        }

    robust_layers = np.flatnonzero(qualifying_by_scale.all(axis=0)).tolist()
    passed = len(robust_layers) >= MIN_ROBUST_LAYERS

    return {
        "schema_version": 2,
        "analysis_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "criteria_source": relative_path(CRITERIA_SOURCE),
        "criteria": {
            "grid_min": 0.0,
            "grid_max": 1.0,
            "grid_points": int(GRID.size),
            "bandwidth_base": "scott_factor",
            "bandwidth_scales": list(BANDWIDTH_SCALES),
            "density_normalization": "divide_each_curve_by_its_max",
            "peak_method": "scipy.signal.find_peaks",
            "peak_prominence": PROMINENCE,
            "required_peak_count": 3,
            "min_three_mode_fraction_per_scale": MIN_THREE_MODE_FRACTION,
            "min_robust_layers": MIN_ROBUST_LAYERS,
            "robust_layer_rule": (
                "same layer meets the fraction threshold at all bandwidth scales"
            ),
        },
        "inputs": {
            "P": len(records),
            "layer_count": layer_count,
            "files": [
                {
                    "idx": record["idx"],
                    "path": relative_path(record["path"]),
                    "sha256": record["sha256"],
                    "input_len": record["input_len"],
                    "generated_tokens": record["generated_tokens"],
                    "sparsity_steps": int(record["sparsity"].shape[1]),
                }
                for record in records
            ],
        },
        "results_by_scale": scale_results,
        "result": {
            "robust_layers": robust_layers,
            "robust_layer_count": len(robust_layers),
            "branch_a_passed": passed,
            "decision": "PASS" if passed else "FAIL",
            "summary_image": relative_path(summary_image),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    decision = result["result"]["decision"]
    robust_layers = result["result"]["robust_layers"]
    p = result["inputs"]["P"]
    layer_count = result["inputs"]["layer_count"]
    lines = [
        "# E3 분기점 A 판정 결과",
        "",
        f"- 분석 시각: {result['analysis_time']}",
        f"- 사전 기준: `{result['criteria_source']}`",
        f"- 입력: P={p}, 레이어 {layer_count}개",
        f"- 판정: **{decision}**",
        f"- 세 bandwidth 공통 통과 레이어: `{robust_layers}` "
        f"({len(robust_layers)}개, 필요 {MIN_ROBUST_LAYERS}개 이상)",
        "",
        "## 레이어별 3봉 비율",
        "",
        "| Layer | Scott ×0.5 | Scott ×1 | Scott ×2 | Robust |",
        "|---:|---:|---:|---:|:---:|",
    ]

    scale_results = result["results_by_scale"]
    fraction_lists = [
        scale_results[SCALE_KEYS[scale]]["three_mode_fraction_by_layer"]
        for scale in BANDWIDTH_SCALES
    ]
    robust_set = set(robust_layers)
    for layer in range(layer_count):
        cells = [f"{values[layer]:.0%}" for values in fraction_lists]
        lines.append(
            f"| {layer} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{'✓' if layer in robust_set else ''} |"
        )

    lines.extend(["", "## 근거 이미지", ""])
    for scale in BANDWIDTH_SCALES:
        scale_result = scale_results[SCALE_KEYS[scale]]
        lines.append(f"- Scott ×{scale:g}: `{scale_result['kde_image']}`")
    lines.append(f"- 비교 요약: `{result['result']['summary_image']}`")
    lines.extend(
        [
            "",
            "## 판정 규칙",
            "",
            "각 bandwidth에서 정확히 3봉인 프롬프트 비율이 50% 이상인 "
            "동일 레이어가 세 bandwidth 모두에서 4개 이상이면 통과한다.",
            "",
            "이 문서는 checker가 관측 결과를 기준에 대입해 자동 생성했다. "
            "기준 원문은 수정하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_dir, args.expected_p)
    curves, peak_counts, fractions = analyze(records)
    image_paths = plot_kde_panels(output_dir, curves, fractions)

    qualifying = fractions >= MIN_THREE_MODE_FRACTION
    robust_layers = np.flatnonzero(qualifying.all(axis=0)).tolist()
    passed = len(robust_layers) >= MIN_ROBUST_LAYERS
    summary_image = plot_fraction_summary(
        output_dir, fractions, robust_layers, passed
    )

    result = build_result(
        records, peak_counts, fractions, image_paths, summary_image
    )
    json_path = output_dir / "criteria_result.json"
    markdown_path = output_dir / "criteria_result.md"
    atomic_write_text(
        json_path,
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic_write_text(markdown_path, render_markdown(result))

    print(f"Decision: {result['result']['decision']}")
    print(f"Robust layers: {result['result']['robust_layers']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    for path in image_paths.values():
        print(f"Image: {path}")
    print(f"Image: {summary_image}")


if __name__ == "__main__":
    main()
