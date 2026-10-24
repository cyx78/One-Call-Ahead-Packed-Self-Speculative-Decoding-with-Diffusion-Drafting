"""Recompute paper measurements and prompt-bootstrap intervals offline."""

from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
from common import output_path

METHODS = ["Linear", "Quadratic", "Packed", "Native"]
DISPLAY_NAMES = {m: f"{m}-SS" for m in METHODS[:3]} | {"Native": "Native"}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load():
    specs = json.loads((ROOT / "configs/experiment_matrix.json").read_text())
    by_group = defaultdict(list)
    with gzip.open(ROOT / "reference/measurements.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            by_group[row["experiment"]].append(row)
    if set(by_group) != {s["id"] for s in specs}:
        raise ValueError("Experiment inventory mismatch")
    for s in specs:
        rows = by_group[s["id"]]
        methods = ["Native"] if s["order"] == "Native" else METHODS[:3]
        expected = {
            (i, rep, m)
            for i in range(s["n"])
            for rep in range(s["repeats"])
            for m in methods
        }
        actual = {(r["index"], r["replicate"], r["method"]) for r in rows}
        if len(actual) != len(rows) or actual != expected:
            raise ValueError(f"Coverage mismatch: {s['id']}")
        for r in rows:
            if any(
                type(r[k]) is not int for k in ("tokens", "nfe", "index", "replicate")
            ):
                raise ValueError("Counts and indices must be integers")
            if (
                not 0 < r["tokens"] <= s["cap"]
                or not math.isfinite(r["wall_s"])
                or r["wall_s"] <= 0
                or r["nfe"] <= 0
            ):
                raise ValueError("Invalid measurement")
            if r["correct"] is not None and type(r["correct"]) is not bool:
                raise ValueError("Correctness must be boolean or null")
            if r["model_s"] is not None and (
                not math.isfinite(r["model_s"]) or r["model_s"] < 0
            ):
                raise ValueError("Invalid model duration")
            if s["task"] == "writingbench" and r["correct"] is not None:
                raise ValueError("WritingBench must have no score")
            if s["family"] == "setdlm" and r["nfe"] != r["decode_nfe"] + 1:
                raise ValueError("SetDLM setup call missing")
    return specs, by_group


def summarize(spec, rows):
    result = []
    for method in METHODS:
        rs = [r for r in rows if r["method"] == method]
        if not rs:
            continue
        n = len(rs)
        wall = sum(r["wall_s"] for r in rs)
        tokens = sum(r["tokens"] for r in rs)
        nfe = sum(r["nfe"] for r in rs)
        scored = all(r["correct"] is not None for r in rs)
        model = (
            sum(r["model_s"] for r in rs)
            if all(r["model_s"] is not None for r in rs)
            else None
        )
        result.append(
            dict(
                experiment=spec["id"],
                scale=spec["scale"],
                task=spec["task"],
                temperature=spec["temperature"],
                order=spec["order"],
                method=method,
                prompts=spec["n"],
                repeats=spec["repeats"],
                samples=n,
                cap=spec["cap"],
                mean_wall_s=wall / n,
                mean_tokens=tokens / n,
                mean_total_nfe=nfe / n,
                mean_decode_nfe=sum(r["decode_nfe"] for r in rs) / n
                if "decode_nfe" in rs[0]
                else None,
                tokens_per_s=tokens / wall,
                tokens_per_total_nfe=tokens / nfe,
                seconds_per_total_nfe=wall / nfe,
                score_pct=100 * sum(r["correct"] for r in rs) / n if scored else None,
                mean_model_s=model / n if model is not None else None,
                mean_residual_s=(wall - model) / n if model is not None else None,
                residual_boundary="setup plus other work"
                if spec["family"] == "setdlm"
                else "other decoder work",
                cap_hits=sum(r["stopped_on_eos"] is False for r in rs)
                if all(r["stopped_on_eos"] is not None for r in rs)
                else None,
            )
        )
    return result


def seed_for(spec):
    if spec["family"] in ("setdlm", "lora"):
        return 20260909 + int(hashlib.sha256(spec["id"].encode()).hexdigest()[:8], 16)
    if spec["temperature"] == 1 and spec["task"] in ("humaneval", "mbpp"):
        return 20260916
    return 20260911


def comparisons(spec, rows, draws):
    import numpy as np

    if spec["order"] == "Native":
        return []
    n = spec["n"]
    repeats = spec["repeats"]
    scored = rows[0]["correct"] is not None
    # Aggregate every repeat within its prompt before drawing prompt clusters.
    arrays = {m: np.zeros((n, 4), dtype=np.float64) for m in METHODS[:3]}
    for r in rows:
        arrays[r["method"]][r["index"]] += [
            r["tokens"],
            r["wall_s"],
            r["nfe"],
            r["correct"] if scored else 0,
        ]
    boot_sums = None
    if draws:
        rng = np.random.default_rng(seed_for(spec))
        data = np.stack([arrays[m] for m in METHODS[:3]], axis=1)
        chunks = []
        for offset in range(0, draws, 500):
            pick = rng.integers(0, n, size=(min(500, draws - offset), n))
            chunks.append(data[pick].sum(1))
        boot_sums = np.concatenate(chunks)
    pairs = []
    for a, b in [
        ("Quadratic", "Linear"),
        ("Packed", "Linear"),
        ("Packed", "Quadratic"),
        ("Linear", "Quadratic"),
    ]:
        av, bv = arrays[a], arrays[b]

        def values(aa, bb):
            return np.stack(
                [
                    (aa[..., 0] / aa[..., 1]) / (bb[..., 0] / bb[..., 1]),
                    (aa[..., 0] / aa[..., 2]) / (bb[..., 0] / bb[..., 2]),
                    (aa[..., 1] / aa[..., 2]) / (bb[..., 1] / bb[..., 2]),
                    (aa[..., 1] - bb[..., 1]) / (n * repeats),
                    100 * (aa[..., 3] - bb[..., 3]) / (n * repeats),
                ],
                axis=-1,
            )

        point = values(av.sum(0), bv.sum(0))
        cis = None
        if draws:
            boot = values(
                boot_sums[:, METHODS.index(a)], boot_sums[:, METHODS.index(b)]
            )
            cis = np.quantile(boot, [0.025, 0.975], axis=0)
        row = dict(
            experiment=spec["id"],
            numerator=a,
            denominator=b,
            bootstrap_unit="prompt_cluster_all_repeats",
            bootstrap_draws=draws,
            bootstrap_seed=seed_for(spec),
        )
        for j, key in enumerate(
            [
                "throughput_ratio",
                "tokens_per_call_ratio",
                "time_per_call_ratio",
                "wall_delta_s",
                "score_delta_pp",
            ]
        ):
            row[key] = float(point[j]) if j != 4 or scored else None
            if cis is not None:
                row[key + "_low"] = float(cis[0, j]) if j != 4 or scored else None
                row[key + "_high"] = float(cis[1, j]) if j != 4 or scored else None
        pairs.append(row)
    return pairs


def appendix(out):
    width = json.loads((ROOT / "reference/width_probe.json").read_text())
    validate_width(width)
    write_csv(
        out / "table03_width.csv",
        [
            dict(
                method=r["method"],
                B=r["width"],
                prompts=r["samples"],
                mean_wall_s=r["mean_walltime_s"],
                mean_total_nfe=r["mean_nfe"],
                mean_tokens=r["mean_generated_tokens"],
                score_pct=100 * r["mean_correct"],
                max_rows=r["max_rows"],
            )
            for r in width
        ],
    )
    timing = json.loads((ROOT / "reference/lora_timing.json").read_text())
    base, lora = timing["base"], timing["lora"]
    rows = []
    for key, method, selected, evaluated in [
        ("linear_w32", "Linear", 32, 32),
        ("quadratic_w8", "Quadratic", 64, 72),
        ("packed_w32", "Packed", 32, 64),
    ]:
        a = base[key]["mean_decode_iteration_model_call_walltime_s"] * 1000
        measured = lora[key]
        recurrent = sum(
            value
            for role, value in measured["model_call_walltime_s_by_role"].items()
            if not role.startswith("prompt_prefill")
        )
        b = (
            1000
            * recurrent
            / (measured["mean_decode_iterations"] * measured["samples"])
        )
        rows.append(
            dict(
                method=method,
                adapter_selected_rows=selected,
                adapter_evaluated_rows=evaluated,
                base_ms_per_iteration=a,
                lora_ms_per_iteration=b,
                increase_pct=100 * (b / a - 1),
            )
        )
    write_csv(out / "table06_lora_iteration_cost.csv", rows)


def validate_width(cells):
    methods = {
        "strict_ltr_self_speculative",
        "strict_ltr_quadratic_self_speculative",
        "strict_ltr_packed_self_speculative",
    }
    expected = {(width, method) for width in (8, 16, 32) for method in methods}
    keys = [(cell["width"], cell["method"]) for cell in cells]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Width inventory mismatch")
    for cell in cells:
        rows = cell["records"]
        if (
            cell["samples"] != 10
            or len(rows) != 10
            or sorted(r["index"] for r in rows) != list(range(10))
        ):
            raise ValueError("Width prompt coverage mismatch")
        for row in rows:
            if row["method"] != cell["method"] or type(row["correct"]) is not bool:
                raise ValueError("Invalid width record")
            if (
                not math.isfinite(row["walltime_s"])
                or row["walltime_s"] <= 0
                or not 0 < row["generated_tokens"] <= 512
                or row["nfe"] <= 0
            ):
                raise ValueError("Invalid width measurement")
        for field, metric in [
            ("mean_walltime_s", "walltime_s"),
            ("mean_nfe", "nfe"),
            ("mean_generated_tokens", "generated_tokens"),
            ("mean_correct", "correct"),
        ]:
            if not math.isclose(
                cell[field],
                sum(row[metric] for row in rows) / 10,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Width aggregate mismatch")


def figures(out, summaries, pairs):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for temp in [0, 1]:
        tasks = ["mbpp", "humaneval", "gsm8k", "math500", "writingbench"]
        ids = [
            f"{'greedy' if temp == 0 else 'rejection'}_{s}_{t}"
            for s in ["3B", "8B"]
            for t in tasks
        ]
        if temp == 0:
            ids += ["setdlm_16_LTR", "setdlm_32_LTR"]
        fig, axes = plt.subplots(1, 4, figsize=(13, 7), sharey=True)
        keys = [
            "throughput_ratio",
            "tokens_per_call_ratio",
            "time_per_call_ratio",
            "score_delta_pp",
        ]
        titles = [
            "Throughput ÷ Linear-SS",
            "Tokens / total NFE ÷ Linear-SS",
            "Time / total NFE ÷ Linear-SS",
            "Accuracy − Linear-SS (pp)",
        ]
        for ax, key, title in zip(axes, keys, titles):
            ax.axvline(0 if key == "score_delta_pp" else 1, color="#888", lw=0.8)
            for method, offset, color in [
                ("Quadratic", -0.12, "#c27830"),
                ("Packed", 0.12, "#247f75"),
            ]:
                for i, gid in enumerate(ids):
                    r = next(
                        (
                            p
                            for p in pairs
                            if p["experiment"] == gid
                            and p["numerator"] == method
                            and p["denominator"] == "Linear"
                        ),
                        None,
                    )
                    if r is None or r[key] is None:
                        continue
                    ax.plot(
                        r[key],
                        i + offset,
                        "o",
                        color=color,
                        ms=4,
                        label=DISPLAY_NAMES[method] if i == 0 else None,
                    )
                    if key + "_low" in r:
                        ax.plot(
                            [r[key + "_low"], r[key + "_high"]],
                            [i + offset] * 2,
                            color=color,
                            lw=1,
                        )
            ax.set_title(title, fontsize=10)
            ax.grid(axis="x", alpha=0.2)
        setdlm_labels = {
            "setdlm_16_LTR": "SetDLM W=8 (smax16), LTR",
            "setdlm_32_LTR": "SetDLM W=16 (smax32), LTR",
        }
        axes[0].set_yticks(
            range(len(ids)),
            [
                setdlm_labels.get(
                    i,
                    "Nemotron "
                    + i.replace("greedy_", "").replace("rejection_", "").replace("_", " "),
                )
                for i in ids
            ],
        )
        axes[0].invert_yaxis()
        axes[0].legend(loc="best")
        fig.suptitle(
            f"Temperature {temp}: speculative schedules from recorded measurements"
        )
        fig.tight_layout()
        fig.savefig(out / f"figure{3 if temp == 0 else 4:02d}_metrics.pdf")
        fig.savefig(out / f"figure{3 if temp == 0 else 4:02d}_metrics.png", dpi=150)
        plt.close(fig)
    # Figure 5: normalize each math task by its own Linear time, then average tasks equally.
    components = []
    for temp in [0, 1]:
        for scale in ["3B", "8B"]:
            for method in METHODS[:3]:
                model = []
                other = []
                for task in ["gsm8k", "math500"]:
                    candidates = [
                        r
                        for r in summaries
                        if r["temperature"] == temp
                        and r["scale"] == scale
                        and r["task"] == task
                        and r["experiment"].startswith(("greedy_", "rejection_"))
                    ]
                    ref = next(r for r in candidates if r["method"] == "Linear")
                    r = next(r for r in candidates if r["method"] == method)
                    model.append(r["mean_model_s"] / ref["mean_wall_s"])
                    other.append(r["mean_residual_s"] / ref["mean_wall_s"])
                components.append(
                    dict(
                        temperature=temp,
                        scale=scale,
                        method=method,
                        model=sum(model) / 2,
                        other=sum(other) / 2,
                    )
                )
    write_csv(out / "figure05_timing_components.csv", components)
    components = [r for r in components if r["temperature"] == 0]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = list(range(len(components)))
    ax.bar(x, [r["model"] for r in components], label="Model execution time")
    ax.bar(
        x,
        [r["other"] for r in components],
        bottom=[r["model"] for r in components],
        label="Non-model decoder time",
    )
    ax.set_xticks(
        x,
        [f"{r['scale']}\n{DISPLAY_NAMES[r['method']]}" for r in components],
        fontsize=8,
    )
    ax.set_ylabel("Time / task-specific Linear-SS wall time")
    ax.set_title("Temperature 0: GSM8K + MATH-500, cap 1,024")
    ax.axhline(1, color="#888", linestyle="--", linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure05_timing.pdf")
    plt.close(fig)


def printed_rows():
    return json.loads((ROOT / "reference/expected_metrics.json").read_text())


def check_metrics(output):
    """Compare rebuilt metrics with archived table values."""
    with (output / "all_measurements.csv").open() as handle:
        actual = {(r["experiment"], r["method"]): r for r in csv.DictReader(handle)}
    checked = 0
    failures = []
    for expected in printed_rows():
        r = actual[expected["experiment"], expected["method"]]
        for key, value in expected.items():
            if key in ("table", "experiment", "method"):
                continue
            decimals = len(value.split(".")[1])
            got = f"{float(r[key]):.{decimals}f}"
            checked += 1
            if got != value:
                failures.append(
                    dict(
                        experiment=r["experiment"],
                        method=r["method"],
                        metric=key,
                        expected=value,
                        recomputed=got,
                    )
                )
    report = dict(
        passed=not failures, printed_values_checked=checked, failures=failures
    )
    (output / "point_estimate_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/rebuilt")
    p.add_argument("--bootstrap-draws", type=int, default=20000)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--check-only", action="store_true", help="Check previously rebuilt tables")
    args = p.parse_args()
    if args.bootstrap_draws < 0:
        p.error("bootstrap-draws must be nonnegative")
    args.output = output_path(args.output)
    if args.check_only:
        check_metrics(args.output)
        return
    specs, groups = load()
    summaries = []
    pairs = []
    for s in specs:
        summaries.extend(summarize(s, groups[s["id"]]))
        pairs.extend(comparisons(s, groups[s["id"]], args.bootstrap_draws))
        print(f"Rebuilt {s['id']}", flush=True)
    write_csv(args.output / "all_measurements.csv", summaries)
    write_csv(args.output / "paired_comparisons.csv", pairs)
    write_csv(
        args.output / "table07_greedy.csv",
        [
            r
            for r in summaries
            if r["experiment"].startswith("greedy_")
        ],
    )
    write_csv(
        args.output / "table08_sampling.csv",
        [r for r in summaries if r["temperature"] == 1],
    )
    write_csv(
        args.output / "table04_setdlm.csv",
        [r for r in summaries if r["experiment"].startswith("setdlm_")],
    )
    write_csv(
        args.output / "table05_lora.csv",
        [r for r in summaries if r["experiment"].startswith("lora_")],
    )
    appendix(args.output)
    check_metrics(args.output)
    if not args.skip_figures:
        figures(args.output, summaries, pairs)
    (args.output / "validation.json").write_text(
        json.dumps(
            dict(
                coverage_verified=True,
                groups=len(specs),
                cells=len(summaries),
                responses=sum(map(len, groups.values())),
                bootstrap_draws=args.bootstrap_draws,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
