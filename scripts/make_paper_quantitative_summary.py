from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis import load_jsonl
from src.utils.cec_style import CEC_BLUE, CEC_GREEN, CEC_ORANGE, CEC_RED
from src.utils.journal_figures import apply_journal_style, save_figure as save_journal_figure, add_panel_labels
from src.utils.stats import mean_t_ci95


OUTPUT_DEFAULT = Path("outputs/runs/paper_figures_full/quantitative")
MODEL_ORDER = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "microsoft/Phi-4-mini-instruct",
]
MODEL_LABELS = {
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Ministral-3-8B-Instruct-2512": "Ministral-3-8B",
    "mistralai/Ministral-3-14B-Instruct-2512": "Ministral-3-14B",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
}
TASK_STYLE = {
    "consensus": {"metric": "final_dispersion", "ylabel": "MAD", "color": CEC_BLUE},
    "diffusion": {"metric": "roughness_final", "ylabel": "roughness", "color": CEC_GREEN},
}
RUN_GROUPS = {
    "consensus": {
        "with_pi": [
            "20260109_102923_consensus_4x4_4llms",
            "20260116_110026_consensus_ministral14B",
            "20260120_003407_consensus",
        ],
        "no_pi": [
            "20260117_174520_consensus_5seeds",
            "20260124_010246_consensus_add5nopi",
        ],
        "t30": ["20260125_111339_consensus_T=30"],
        "grid6": ["20260204_082317_consensus_6x6"],
        "grid8": ["20260206_103341_consensus_8x8"],
    },
    "diffusion": {
        "with_pi": [
            "20260115_073051_diffusion_4x4_4llms",
            "20260116_110026_diffusion_ministral14B",
            "20260120_003407_diffusion",
        ],
        "no_pi": [
            "20260117_174520_diffusion_5seeds",
            "20260124_010246_diffusion_add5nopi",
        ],
        "t30": ["20260125_111339_diffusionT=30"],
        "grid6": ["20260317_diffusion_6x6_diffusion"],
        "grid8": ["20260317_diffusion_8x8_diffusion"],
    },
}

apply_journal_style(font_size=10, title_size=11, label_size=10, tick_size=9, legend_size=9)


def _prompt_mode(record: dict) -> str:
    mode = record.get("prompt_mode")
    if isinstance(mode, str) and mode:
        return mode
    return "with_pi"


def _load_run(outputs_root: Path, run_id: str) -> dict:
    run_dir = outputs_root / run_id
    trials = load_jsonl(run_dir / "trial_summary.jsonl")
    steps = load_jsonl(run_dir / "steps.jsonl")
    manifest_path = run_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    for record in trials:
        record.setdefault("run_id", run_id)
    for record in steps:
        record.setdefault("run_id", run_id)
    return {"run_id": run_id, "trials": trials, "steps": steps, "manifest": manifest}


def _mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    return mean_t_ci95(values)


def _cohens_d(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    pooled_var = ((x.size - 1) * sx * sx + (y.size - 1) * sy * sy) / max(x.size + y.size - 2, 1)
    if pooled_var <= 0.0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled_var))


def _fmt_num(value: float, digits: int = 3) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "nan"
    return f"{value:.{digits}f}"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_journal_figure(fig, path, pad_inches=0.05)


def _collect_trials(payloads: list[dict], model_id: str | None = None) -> list[dict]:
    trials = []
    for payload in payloads:
        for record in payload["trials"]:
            if model_id is not None and record.get("model_id") != model_id:
                continue
            trials.append(record)
    return trials


def _collect_steps(payloads: list[dict], model_id: str | None = None) -> list[dict]:
    steps = []
    for payload in payloads:
        for record in payload["steps"]:
            if model_id is not None and record.get("model_id") != model_id:
                continue
            steps.append(record)
    return steps


def _filter_trials(
    trials: list[dict],
    *,
    grid_size: int | None = None,
    steps: int | None = None,
    prompt_mode: str | None = None,
) -> list[dict]:
    filtered = []
    for record in trials:
        if grid_size is not None and int(record.get("grid_size", grid_size)) != grid_size:
            continue
        if steps is not None and int(record.get("steps", steps)) != steps:
            continue
        mode = _prompt_mode(record)
        if prompt_mode == "with_pi" and mode != "with_pi":
            continue
        if prompt_mode == "no_pi" and mode != "no_pi":
            continue
        filtered.append(record)
    return filtered


def _filter_steps_by_trial(steps: list[dict], trials: list[dict]) -> list[dict]:
    keys = {(record.get("run_id"), int(record.get("trial_id", -1))) for record in trials}
    return [
        record
        for record in steps
        if (record.get("run_id"), int(record.get("trial_id", -1))) in keys
    ]


def _group_values_by_mu(trials: list[dict], metric_key: str) -> dict[float, list[float]]:
    grouped = defaultdict(list)
    for record in trials:
        metric = record.get(metric_key)
        mu = record.get("mu")
        if mu is None or not isinstance(metric, (int, float)):
            continue
        grouped[float(mu)].append(float(metric))
    return dict(grouped)


def _best_mu_summary(trials: list[dict], metric_key: str) -> dict:
    grouped = _group_values_by_mu(trials, metric_key)
    if not grouped:
        return {
            "mu": "",
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "ci95": math.nan,
            "values": [],
        }
    best_mu = min(grouped, key=lambda mu: float(np.mean(grouped[mu])))
    mean, std, ci = _mean_ci(grouped[best_mu])
    return {
        "mu": best_mu,
        "n": len(grouped[best_mu]),
        "mean": mean,
        "std": std,
        "ci95": ci,
        "values": list(grouped[best_mu]),
    }


def _summary_at_mu(trials: list[dict], metric_key: str, mu: float) -> dict:
    grouped = _group_values_by_mu(trials, metric_key)
    values = grouped.get(float(mu), [])
    mean, std, ci = _mean_ci(values)
    return {
        "mu": mu,
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95": ci,
        "values": list(values),
    }


def _trial_runtime_hours(trials: list[dict]) -> float:
    vals = [float(record.get("duration_s", 0.0)) for record in trials if isinstance(record.get("duration_s"), (int, float))]
    return float(sum(vals) / 3600.0)


def _mean_step_change_by_mu(step_records: list[dict], trials: list[dict]) -> dict[float, tuple[float, float]]:
    trial_keys = {(record.get("run_id"), int(record.get("trial_id", -1))) for record in trials}
    by_trial_step = defaultdict(dict)
    for record in step_records:
        key = (record.get("run_id"), int(record.get("trial_id", -1)))
        if key not in trial_keys:
            continue
        by_trial_step[(key, int(record["step"]))][int(record["agent_id"])] = record

    by_trial = defaultdict(list)
    trial_mu = {
        (record.get("run_id"), int(record.get("trial_id", -1))): float(record.get("mu", 0.0))
        for record in trials
        if record.get("mu") is not None
    }
    for (trial_key, step), items in by_trial_step.items():
        if step == 0:
            continue
        prev = by_trial_step.get((trial_key, step - 1), {})
        deltas = []
        for agent_id, record in items.items():
            prev_record = prev.get(agent_id)
            if prev_record is None:
                continue
            try:
                deltas.append(abs(int(record["value"]) - int(prev_record["value"])))
            except Exception:
                continue
        if deltas:
            by_trial[trial_key].append(float(np.mean(deltas)))

    grouped = defaultdict(list)
    for trial_key, per_step in by_trial.items():
        mu = trial_mu.get(trial_key)
        if mu is None:
            continue
        grouped[mu].append(float(np.mean(per_step)))

    summary = {}
    for mu, values in grouped.items():
        mean, _std, ci = _mean_ci(values)
        summary[mu] = (mean, ci)
    return dict(summary)


def _build_qwen_condition_rows(payloads_by_task: dict[str, dict[str, list[dict]]]) -> list[dict]:
    rows = []
    for task, groups in payloads_by_task.items():
        metric_key = TASK_STYLE[task]["metric"]
        main_trials = _filter_trials(_collect_trials(groups["with_pi"], "Qwen/Qwen3-8B"), grid_size=4, steps=10, prompt_mode="with_pi")
        no_pi_trials = _filter_trials(_collect_trials(groups["no_pi"], "Qwen/Qwen3-8B"), grid_size=4, steps=10, prompt_mode="no_pi")
        t30_trials = _filter_trials(_collect_trials(groups["t30"], "Qwen/Qwen3-8B"), grid_size=4, steps=30, prompt_mode="with_pi")
        grid6_trials = _filter_trials(_collect_trials(groups["grid6"], "Qwen/Qwen3-8B"), grid_size=6, steps=10, prompt_mode="with_pi")
        grid8_trials = _filter_trials(_collect_trials(groups["grid8"], "Qwen/Qwen3-8B"), grid_size=8, steps=10, prompt_mode="with_pi")

        conditions = [
            ("main_4x4_t10_with_pi", main_trials, True),
            ("no_pi", no_pi_trials, False),
            ("t30", t30_trials, True),
            ("grid6", grid6_trials, True),
            ("grid8", grid8_trials, True),
        ]
        for condition_name, trials, has_sweep in conditions:
            if not trials:
                continue
            summary = _best_mu_summary(trials, metric_key) if has_sweep else _summary_at_mu(trials, metric_key, -5.0)
            rows.append(
                {
                    "task": task,
                    "condition": condition_name,
                    "metric": metric_key,
                    "mu": _fmt_num(float(summary["mu"]), 2) if summary["mu"] != "" else "",
                    "mean": _fmt_num(summary["mean"]),
                    "ci95": _fmt_num(summary["ci95"]),
                    "std": _fmt_num(summary["std"]),
                    "n": summary["n"],
                    "runtime_h": _fmt_num(_trial_runtime_hours(trials), 2),
                }
            )
    return rows


def _build_regime_rows(payloads_by_task: dict[str, dict[str, list[dict]]]) -> list[dict]:
    rows = []
    for task, groups in payloads_by_task.items():
        metric_key = TASK_STYLE[task]["metric"]
        trials = _filter_trials(_collect_trials(groups["with_pi"], "Qwen/Qwen3-8B"), grid_size=4, steps=10, prompt_mode="with_pi")
        if not trials:
            continue
        best = _best_mu_summary(trials, metric_key)
        low = _summary_at_mu(trials, metric_key, -5.0)
        high = _summary_at_mu(trials, metric_key, 5.0)
        rows.append(
            {
                "task": task,
                "metric": metric_key,
                "low_mu": "-5.00",
                "low_mean": _fmt_num(low["mean"]),
                "low_ci95": _fmt_num(low["ci95"]),
                "best_mu": _fmt_num(float(best["mu"]), 2),
                "best_mean": _fmt_num(best["mean"]),
                "best_ci95": _fmt_num(best["ci95"]),
                "high_mu": "5.00",
                "high_mean": _fmt_num(high["mean"]),
                "high_ci95": _fmt_num(high["ci95"]),
                "best_vs_low_pct": _fmt_num((low["mean"] - best["mean"]) / max(low["mean"], 1e-9) * 100.0, 1),
                "best_vs_high_pct": _fmt_num((high["mean"] - best["mean"]) / max(high["mean"], 1e-9) * 100.0, 1),
            }
        )
    return rows


def _build_model_rows(payloads_by_task: dict[str, dict[str, list[dict]]]) -> list[dict]:
    rows = []
    for task, groups in payloads_by_task.items():
        metric_key = TASK_STYLE[task]["metric"]
        for model_id in MODEL_ORDER:
            trials = _filter_trials(_collect_trials(groups["with_pi"], model_id), grid_size=4, steps=10, prompt_mode="with_pi")
            if not trials:
                continue
            best = _best_mu_summary(trials, metric_key)
            rows.append(
                {
                    "task": task,
                    "model": MODEL_LABELS.get(model_id, model_id),
                    "metric": metric_key,
                    "best_mu": _fmt_num(float(best["mu"]), 2),
                    "best_mean": _fmt_num(best["mean"]),
                    "best_ci95": _fmt_num(best["ci95"]),
                    "n": best["n"],
                }
            )
    return rows


def _build_ablation_rows(payloads_by_task: dict[str, dict[str, list[dict]]]) -> list[dict]:
    rows = []
    for task, groups in payloads_by_task.items():
        metric_key = TASK_STYLE[task]["metric"]
        for model_id in MODEL_ORDER:
            with_trials = _filter_trials(_collect_trials(groups["with_pi"], model_id), grid_size=4, steps=10, prompt_mode="with_pi")
            no_pi_trials = _filter_trials(_collect_trials(groups["no_pi"], model_id), grid_size=4, steps=10, prompt_mode="no_pi")
            if not with_trials or not no_pi_trials:
                continue
            best = _best_mu_summary(with_trials, metric_key)
            no_pi = _summary_at_mu(no_pi_trials, metric_key, -5.0)
            rows.append(
                {
                    "task": task,
                    "model": MODEL_LABELS.get(model_id, model_id),
                    "metric": metric_key,
                    "best_mu_with_pi": _fmt_num(float(best["mu"]), 2),
                    "with_pi_mean": _fmt_num(best["mean"]),
                    "with_pi_ci95": _fmt_num(best["ci95"]),
                    "no_pi_mean": _fmt_num(no_pi["mean"]),
                    "no_pi_ci95": _fmt_num(no_pi["ci95"]),
                    "abs_improvement": _fmt_num(no_pi["mean"] - best["mean"]),
                    "rel_improvement_pct": _fmt_num((no_pi["mean"] - best["mean"]) / max(no_pi["mean"], 1e-9) * 100.0, 1),
                    "cohens_d": _fmt_num(_cohens_d(no_pi["values"], best["values"]), 3),
                }
            )
    return rows


def _plot_regime_bars(payloads_by_task: dict[str, dict[str, list[dict]]], out_dir: Path) -> None:
    for task, groups in payloads_by_task.items():
        metric_key = TASK_STYLE[task]["metric"]
        ylabel = TASK_STYLE[task]["ylabel"]
        color = TASK_STYLE[task]["color"]
        trials = _filter_trials(_collect_trials(groups["with_pi"], "Qwen/Qwen3-8B"), grid_size=4, steps=10, prompt_mode="with_pi")
        if not trials:
            continue
        best = _best_mu_summary(trials, metric_key)
        low = _summary_at_mu(trials, metric_key, -5.0)
        high = _summary_at_mu(trials, metric_key, 5.0)
        labels = ["low μ", f"best μ={_fmt_num(float(best['mu']), 2)}", "high μ"]
        means = [low["mean"], best["mean"], high["mean"]]
        cis = [low["ci95"], best["ci95"], high["ci95"]]
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=cis, color=[color, CEC_BLUE, CEC_RED], capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{task}: low / best / high μ")
        _save(fig, out_dir / f"fig_regime_bars_{task}.png")


def _plot_model_best(payloads_by_task: dict[str, dict[str, list[dict]]], out_dir: Path) -> None:
    for task, groups in payloads_by_task.items():
        metric_key = TASK_STYLE[task]["metric"]
        ylabel = TASK_STYLE[task]["ylabel"]
        labels = []
        means = []
        cis = []
        for model_id in MODEL_ORDER:
            trials = _filter_trials(_collect_trials(groups["with_pi"], model_id), grid_size=4, steps=10, prompt_mode="with_pi")
            if not trials:
                continue
            best = _best_mu_summary(trials, metric_key)
            labels.append(MODEL_LABELS.get(model_id, model_id))
            means.append(best["mean"])
            cis.append(best["ci95"])
        if not labels:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=cis, color=CEC_BLUE, capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{task}: best achieved performance by model")
        _save(fig, out_dir / f"fig_model_best_{task}.png")


def _plot_compute_scaling(payloads_by_task: dict[str, dict[str, list[dict]]], out_dir: Path) -> None:
    condition_labels = ["4x4/T10", "4x4/T30", "6x6/T10", "8x8/T10"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    for ax, task in zip(axes, ["consensus", "diffusion"]):
        groups = payloads_by_task[task]
        main_trials = _filter_trials(_collect_trials(groups["with_pi"], "Qwen/Qwen3-8B"), grid_size=4, steps=10, prompt_mode="with_pi")
        t30_trials = _filter_trials(_collect_trials(groups["t30"], "Qwen/Qwen3-8B"), grid_size=4, steps=30, prompt_mode="with_pi")
        g6_trials = _filter_trials(_collect_trials(groups["grid6"], "Qwen/Qwen3-8B"), grid_size=6, steps=10, prompt_mode="with_pi")
        g8_trials = _filter_trials(_collect_trials(groups["grid8"], "Qwen/Qwen3-8B"), grid_size=8, steps=10, prompt_mode="with_pi")
        values = [
            _trial_runtime_hours(main_trials),
            _trial_runtime_hours(t30_trials),
            _trial_runtime_hours(g6_trials),
            _trial_runtime_hours(g8_trials),
        ]
        ax.bar(np.arange(len(values)), values, color=TASK_STYLE[task]["color"])
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(condition_labels, rotation=20, ha="right")
        ax.set_ylabel("runtime [h]")
        ax.set_title(task)
    add_panel_labels(axes)
    _save(fig, out_dir / "fig_compute_scaling.png")


def _plot_masc_vs_error(payloads_by_task: dict[str, dict[str, list[dict]]], out_dir: Path) -> None:
    for task, groups in payloads_by_task.items():
        metric_key = TASK_STYLE[task]["metric"]
        ylabel = TASK_STYLE[task]["ylabel"]
        trials = _filter_trials(_collect_trials(groups["with_pi"], "Qwen/Qwen3-8B"), grid_size=4, steps=10, prompt_mode="with_pi")
        steps = _filter_steps_by_trial(_collect_steps(groups["with_pi"], "Qwen/Qwen3-8B"), trials)
        if not trials or not steps:
            continue
        grouped = _group_values_by_mu(trials, metric_key)
        masc = _mean_step_change_by_mu(steps, trials)
        mus = sorted(set(grouped.keys()) & set(masc.keys()))
        if not mus:
            continue
        x = [masc[mu][0] for mu in mus]
        y = [float(np.mean(grouped[mu])) for mu in mus]
        fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
        ax.scatter(x, y, s=42, color=TASK_STYLE[task]["color"])
        for mu, x_val, y_val in zip(mus, x, y):
            ax.text(x_val, y_val, _fmt_num(mu, 2), fontsize=8, ha="left", va="bottom")
        ax.set_xlabel("MASC (mean step change)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{task}: responsiveness vs performance")
        _save(fig, out_dir / f"fig_masc_vs_error_{task}.png")


def _write_readme(out_dir: Path) -> None:
    lines = [
        "# Quantitative Summary",
        "",
        "- table_qwen_conditions: Qwen main/no_pi/T30/grid conditions with mean, 95% CI, runtime.",
        "- table_qwen_regimes: low/best/high μ comparison for the main 4x4/T10/with_pi setting.",
        "- table_model_best: best achieved metric and best μ for each model.",
        "- table_pi_ablation: with_pi vs no_pi numerical comparison with effect size.",
        "- fig_regime_bars_*: low/best/high μ bars with 95% CI.",
        "- fig_model_best_*: best achieved metric by model with 95% CI.",
        "- fig_masc_vs_error_*: diagnostic scatter showing responsiveness vs task performance.",
        "- fig_compute_scaling: runtime comparison across T and grid size.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate quantitative summaries for the paper runs.")
    parser.add_argument("--outputs-root", default="outputs/runs")
    parser.add_argument("--out-dir", default=str(OUTPUT_DEFAULT))
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads_by_task: dict[str, dict[str, list[dict]]] = {}
    for task, groups in RUN_GROUPS.items():
        payloads_by_task[task] = {}
        for group_name, run_ids in groups.items():
            payloads_by_task[task][group_name] = [_load_run(outputs_root, run_id) for run_id in run_ids]

    qwen_rows = _build_qwen_condition_rows(payloads_by_task)
    regime_rows = _build_regime_rows(payloads_by_task)
    model_rows = _build_model_rows(payloads_by_task)
    ablation_rows = _build_ablation_rows(payloads_by_task)

    _write_csv(out_dir / "table_qwen_conditions.csv", qwen_rows, list(qwen_rows[0].keys()))
    _write_markdown_table(out_dir / "table_qwen_conditions.md", qwen_rows, list(qwen_rows[0].keys()))

    _write_csv(out_dir / "table_qwen_regimes.csv", regime_rows, list(regime_rows[0].keys()))
    _write_markdown_table(out_dir / "table_qwen_regimes.md", regime_rows, list(regime_rows[0].keys()))

    _write_csv(out_dir / "table_model_best.csv", model_rows, list(model_rows[0].keys()))
    _write_markdown_table(out_dir / "table_model_best.md", model_rows, list(model_rows[0].keys()))

    _write_csv(out_dir / "table_pi_ablation.csv", ablation_rows, list(ablation_rows[0].keys()))
    _write_markdown_table(out_dir / "table_pi_ablation.md", ablation_rows, list(ablation_rows[0].keys()))

    _plot_regime_bars(payloads_by_task, out_dir)
    _plot_model_best(payloads_by_task, out_dir)
    _plot_compute_scaling(payloads_by_task, out_dir)
    _plot_masc_vs_error(payloads_by_task, out_dir)
    _write_readme(out_dir)

    print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
