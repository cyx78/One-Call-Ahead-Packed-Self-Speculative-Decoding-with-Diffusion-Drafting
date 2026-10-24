"""Run greedy or rejection-sampled Nemotron benchmarks."""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import (
    append_row,
    bind_protocol,
    check_context,
    check_coverage,
    file_sha,
    output_path,
    rows_at,
    run_lock,
    source_hashes,
    validate_measurement,
    write_json,
    write_rows,
)

COUNTS = dict(humaneval=164, mbpp=378, gsm8k=1319, math500=500, writingbench=555)
REVISIONS = {
    "3B": "b9d4dfe782378d663699cf0f8a272e10f5956a25",
    "8B": "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc",
}
VARIANTS = [
    dict(
        id="linear",
        method="strict_ltr_self_speculative",
        block_length=32,
        max_uncached_rows_per_forward=32,
    ),
    dict(
        id="quadratic",
        method="strict_ltr_quadratic_self_speculative",
        block_length=8,
        max_uncached_rows_per_forward=72,
    ),
    dict(
        id="packed",
        method="strict_ltr_packed_self_speculative",
        block_length=32,
        max_uncached_rows_per_forward=64,
    ),
]
NAMES = dict(zip([v["id"] for v in VARIANTS], ["Linear", "Quadratic", "Packed"]))


def now():
    return datetime.now(timezone.utc).isoformat()


def cap_for(task, suite="main"):
    return 1024 if task in ("gsm8k", "math500") and suite == "main" else 512


def repeats_for(task, policy):
    return {"humaneval": 20, "mbpp": 7}.get(task, 1) if policy == "rejection" else 1


def sample_seed(scale, task, policy, replicate, index, suite="main"):
    if policy == "rejection" and task in ("humaneval", "mbpp"):
        return (
            202609160000
            + list(REVISIONS).index(scale) * 1000000
            + ["humaneval", "mbpp"].index(task) * 100000
            + replicate * 1000
            + index
        )
    if task == "writingbench":
        return (20260917 if policy == "greedy" else 20260909) + index
    if policy == "rejection":
        return 20260910 + index
    if suite == "width":
        return 20260822 + index
    if suite == "lora-base":
        return 20260823 + index
    return (20260827 if scale == "3B" else 20260826) + index


def load_task(task):
    if task != "writingbench":
        from nemotron_ltr.data import prepared_rows

        data = prepared_rows(task)
    else:
        import urllib.request

        spec = json.loads((ROOT / "configs/writingbench_dataset.json").read_text())
        path = ROOT / ".cache/writingbench_all.jsonl"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(spec["url"], timeout=120) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != spec["sha256"]:
                raise ValueError("WritingBench checksum mismatch")
            path.write_bytes(content)
        if file_sha(path) != spec["sha256"]:
            raise ValueError("WritingBench checksum mismatch")
        data = [r for r in rows_at(path) if r["lang"] == "en"]
    if len(data) != COUNTS[task]:
        raise ValueError(f"Unexpected {task} dataset size")
    return data


def configuration(scale, suite):
    config = dict(
        model=dict(
            id=f"nvidia/Nemotron-Labs-Diffusion-{scale}",
            revision=REVISIONS[scale],
            cache_dir=str(ROOT / ".cache/models"),
        ),
        generation=dict(proposal_only_lora=suite == "lora"),
    )
    if suite == "lora":
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                config["model"]["id"],
                revision=REVISIONS[scale],
                cache_dir=config["model"]["cache_dir"],
                allow_patterns=["linear_spec_lora/*"],
            )
        )
        adapter = snapshot / "linear_spec_lora"
        expected = {
            "adapter_config.json": "2e9a72d47d4ae4f70ab624cca17ef41161ddd6ad0e72204817a3fcdca7297b34",
            "adapter_model.safetensors": "5893f4aec2f43bd0ea065776a269f5f7b70c52db52be3270686349a78e824908",
        }
        for filename, digest in expected.items():
            if file_sha(adapter / filename) != digest:
                raise ValueError(f"Adapter checksum mismatch: {filename}")
        config["model"]["adapter_path"] = str(adapter)
    return config


def stage(args, scale, task):
    import torch
    from nemotron_ltr.loading import _load_model
    from nemotron_ltr.data import messages_for_row
    from nemotron_ltr.scoring import (
        _prompt_ids,
        _score_output,
        _row_metadata,
        score_output,
    )
    from nemotron_ltr.decoders import (
        strict_ltr_self_speculative,
        strict_ltr_quadratic_self_speculative,
        strict_ltr_packed_self_speculative,
    )

    METHODS = {
        f.__name__: f
        for f in (
            strict_ltr_self_speculative,
            strict_ltr_quadratic_self_speculative,
            strict_ltr_packed_self_speculative,
        )
    }
    from nemotron_ltr.instrumentation import _aggregate_call_times, compact
    from common import sample_timeout
    from nemotron_ltr import instrumentation as instrumented
    from nemotron_ltr import sampling

    data = load_task(task)
    n = min(args.num_samples or len(data), len(data))
    repeats = args.repeats or repeats_for(task, args.policy)
    cap = cap_for(task, args.suite)
    out = args.output / scale / task
    out.mkdir(parents=True, exist_ok=True)
    config = configuration(scale, args.suite)
    sources = source_hashes()
    protocol = dict(
        policy=args.policy,
        scale=scale,
        task=task,
        suite=args.suite,
        prompts=n,
        repeats=repeats,
        cap=cap,
        model_revision=REVISIONS[scale],
        dtype="bfloat16",
        variants=VARIANTS,
        source_sha256=sources,
        config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        model_call_timing=not (task == "writingbench" and args.policy == "rejection"),
    )
    bind_protocol(out, protocol)
    raw = out / "raw.jsonl"
    rows = rows_at(raw, recover_tail=True)
    done, expected = check_coverage(rows, n, repeats, [v["id"] for v in VARIANTS])
    for row in rows:
        validate_measurement(row, cap)
    if done == expected:
        return
    model, tokenizer, device = _load_model(config)
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    if dtypes != ["torch.bfloat16"] and args.suite != "lora":
        raise RuntimeError(f"Unexpected model precision: {dtypes}")
    cohort = f"fresh-{args.policy}-{scale}-{task}-{os.getpid()}"
    write_json(
        out / f"{cohort}.json",
        dict(
            started_utc=now(),
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            parameter_dtypes=dtypes,
        ),
    )
    eos = {tokenizer.eos_token_id}
    if task == "writingbench" and args.policy == "rejection":
        raw_eos = model.config.eos_token_id
        eos.update(raw_eos if isinstance(raw_eos, list) else [raw_eos])
        eos.discard(None)

    def generate(rep, i, v, limit):
        if task == "writingbench":
            ids = tokenizer.apply_chat_template(
                [dict(role="user", content=data[i]["query"])],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=False,
            ).to(device)
            prompt_sha = hashlib.sha256(
                json.dumps(ids[0].tolist()).encode()
            ).hexdigest()
        else:
            ids, prompt_sha = _prompt_ids(
                tokenizer, messages_for_row(task, data[i]), device
            )
        check_context(model, ids.shape[1], limit, v["block_length"])
        seed = sample_seed(scale, task, args.policy, rep, i, args.suite)
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with sample_timeout(300), torch.inference_mode():
            if args.policy == "greedy":
                result = METHODS[v["method"]](
                    model,
                    ids,
                    max_new_tokens=limit,
                    block_length=v["block_length"],
                    threshold=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                    max_uncached_rows_per_forward=v["max_uncached_rows_per_forward"],
                    record_model_call_walltimes=True,
                    use_draft_lora=args.suite == "lora",
                )
            else:
                result = (sampling if task == "writingbench" else instrumented).decode(
                    model, ids, eos, NAMES[v["id"]].lower(), v["block_length"], limit
                )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if args.policy == "greedy":
            output = result.output_ids[0, ids.shape[1] :].detach().cpu().tolist()
            extra = dict(
                nfe=result.nfe,
                stopped_on_eos=result.stopped_on_eos,
                model_call_work_walltime_s=sum(result.model_call_walltimes_s),
                prefill_walltime_s=result.prefill_walltime_s,
                max_uncached_rows=result.max_uncached_rows,
                accepted_lengths=result.accepted_lengths,
                stable_prefix_lengths=result.stable_prefix_lengths,
                iteration_records=result.iteration_records,
                model_call_walltime_s_by_role=_aggregate_call_times(
                    result.model_call_roles, result.model_call_walltimes_s
                ),
                model_call_count_by_role=dict(Counter(result.model_call_roles)),
            )
        else:
            output = result.pop("output_ids")
            extra = result
        if not 0 < len(output) <= limit:
            raise RuntimeError("Invalid output length")
        if (output[-1] in eos) != bool(extra["stopped_on_eos"]):
            raise RuntimeError("EOS accounting mismatch")
        if not extra["stopped_on_eos"] and len(output) != limit:
            raise RuntimeError("Incomplete response")
        text = tokenizer.decode(
            output,
            skip_special_tokens=not (
                task == "writingbench" and args.policy == "rejection"
            ),
        )
        row = dict(
            extra,
            index=i,
            replicate=rep,
            variant=v["id"],
            method=NAMES[v["id"]],
            scale=scale,
            task=task,
            policy=args.policy,
            seed=seed,
            walltime_s=elapsed,
            generated_tokens=len(output),
            output_token_ids=output,
            output_text=text,
            output_token_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest(),
            prompt_tokens=ids.shape[1],
            prompt_sha256=prompt_sha,
            effective_max_new_tokens=limit,
            measured_max_new_tokens=limit,
            cohort_id=cohort,
            origin="fresh_reproduction",
            completed_utc=now(),
        )
        if task == "writingbench":
            row.update(
                correct=None,
                score_status="not_evaluated_open_ended_generation",
                dataset_index=data[i]["index"],
            )
        else:
            # Select the configured answer extraction rule.
            row.update(
                score_output(scale, task, text, data[i])
                if args.suite in ("main", "lora-base", "width")
                else _score_output(task, text, data[i])
            )
            row.update(_row_metadata(task, data[i]))
        if "model_call_work_walltime_s" in row:
            row["other_work_walltime_s"] = max(
                0.0, elapsed - row["model_call_work_walltime_s"]
            )
        return compact(row)

    warmup = [generate(0, 0, v, 32) for v in VARIANTS]
    write_json(
        out / f"{cohort}_warmup.json",
        dict(
            included=False,
            rows=[
                {k: r[k] for k in ("variant", "nfe", "walltime_s", "generated_tokens")}
                for r in warmup
            ],
        ),
    )
    orders = list(itertools.permutations(VARIANTS))
    for rep in range(repeats):
        for i in range(n):
            for v in orders[(rep * COUNTS[task] + i) % 6]:
                if (rep, i, v["id"]) in done:
                    continue
                if shutil.disk_usage(out).free < 512 * 1024**2:
                    raise RuntimeError(
                        "Less than 512 MiB free disk; saved rows are resumable"
                    )
                row = generate(rep, i, v, cap)
                validate_measurement(row, cap)
                append_row(raw, row)
                done.add((rep, i, v["id"]))
                write_json(
                    out / "status.json",
                    dict(
                        status="running",
                        completed=len(done),
                        expected=len(expected),
                        updated_utc=now(),
                    ),
                )
            print(
                f"{scale}/{task} repeat {rep + 1}: {len(done)}/{len(expected)}",
                flush=True,
            )
    write_json(
        out / "status.json",
        dict(status="generation_complete", completed=len(done), expected=len(expected)),
    )


def grade_and_merge(args, scale, task):
    out = args.output / scale / task
    rows = rows_at(out / "raw.jsonl")
    data = load_task(task)
    n = min(args.num_samples or len(data), len(data))
    repeats = args.repeats or repeats_for(task, args.policy)
    check_coverage(rows, n, repeats, [v["id"] for v in VARIANTS], complete=True)
    for row in rows:
        validate_measurement(row, cap_for(task, args.suite))
    if task == "humaneval":
        from nemotron_ltr import code_eval as grader

        grader._DATASETS["humaneval"] = data
        for row in rows:
            result = grader._score_code_row(task, row)
            if result["outcome"] == "sandbox_error" or any(
                a.get("outcome") == "sandbox_error" for a in result["attempts"]
            ):
                raise RuntimeError("Code sandbox failed; no score recorded")
            row.update(
                correct=result["passed"],
                code_eval=result,
                score_status="scored_humaneval_sandbox_v1",
            )
    elif task == "mbpp":
        from nemotron_ltr.code_eval import _mbpp_candidate
        from nemotron_ltr.evalplus_mbpp_base import evaluate_mbpp_base

        for rep in sorted({r["replicate"] for r in rows}):
            for v in VARIANTS:
                rs = sorted(
                    (
                        r
                        for r in rows
                        if r["replicate"] == rep and r["variant"] == v["id"]
                    ),
                    key=lambda r: r["index"],
                )
                if len(rs) != 378:
                    for r in rs:
                        r.update(
                            correct=None,
                            score_status="partial_generation_not_full_set_scored",
                        )
                    continue
                samples = [
                    dict(
                        task_id=f"Mbpp/{int(data[r['index']]['task_id'])}",
                        solution=_mbpp_candidate(r)[0],
                    )
                    for r in rs
                ]
                result = evaluate_mbpp_base(
                    samples,
                    dataset_version="v0.2.0",
                    noextreme=True,
                    parallel_workers=4,
                )
                if result["dataset_md5"] != "85d8d7a406c0686e80353e676189bfee":
                    raise ValueError("MBPP release checksum mismatch")
                for r, s in zip(rs, samples):
                    r.update(
                        correct=result["eval"][s["task_id"]][0]["base_status"]
                        == "pass",
                        score_status="scored_evalplus_0.3.1_mbpp_base",
                    )
    write_rows(
        out / "combined_raw.jsonl",
        sorted(rows, key=lambda row: (row["replicate"], row["index"], row["variant"])),
    )
    summary = []
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v["id"]]
        n = len(rs)
        wall = sum(r["walltime_s"] for r in rs)
        tokens = sum(r["generated_tokens"] for r in rs)
        nfe = sum(r["nfe"] for r in rs)
        summary.append(
            dict(
                method=NAMES[v["id"]],
                samples=n,
                mean_wall_s=wall / n,
                mean_tokens=tokens / n,
                mean_nfe=nfe / n,
                tokens_per_s=tokens / wall,
                tokens_per_nfe=tokens / nfe,
                score_pct=100 * sum(r["correct"] for r in rs) / n
                if all(isinstance(r.get("correct"), bool) for r in rs)
                else None,
            )
        )
    write_json(out / "summary.json", summary)
    scored = task == "writingbench" or all(r.get("correct") is not None for r in rows)
    write_json(
        out / "status.json",
        dict(
            status="complete" if scored else "generation_complete_grading_incomplete",
            responses=len(rows),
            updated_utc=now(),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["greedy", "rejection"], default="greedy")
    p.add_argument(
        "--scales", nargs="+", choices=list(REVISIONS), default=list(REVISIONS)
    )
    p.add_argument("--tasks", nargs="+", choices=list(COUNTS), default=list(COUNTS))
    p.add_argument(
        "--suite", choices=["main", "lora", "lora-base", "width"], default="main"
    )
    p.add_argument("--num-samples", type=int)
    p.add_argument(
        "--repeats",
        type=int,
        help="Samples per prompt; T=1 defaults to 20 HumanEval and 7 MBPP",
    )
    p.add_argument("--output", type=Path)
    widths = p.add_mutually_exclusive_group()
    widths.add_argument("--width", type=int, choices=[8, 16, 32], help="Run one block size")
    widths.add_argument("--widths", nargs="+", type=int, choices=[8, 16, 32],
                        help="Block-size ablation; defaults to 8, 16, and 32")
    p.add_argument("--execute", action="store_true")
    operation = p.add_mutually_exclusive_group()
    operation.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    operation.add_argument("--grade", nargs=2, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.num_samples is not None and args.num_samples < 1:
        p.error("num-samples must be positive")
    if args.repeats is not None and args.repeats < 1:
        p.error("repeats must be positive")
    if args.suite in ("lora", "lora-base", "width"):
        if args.policy != "greedy":
            p.error("This suite requires greedy decoding")
        args.scales = ["8B"]
        args.tasks = ["gsm8k"]
    if args.suite == "width":
        args.num_samples = args.num_samples or 10
        if args.num_samples > 10:
            p.error("Width probe has ten prompts")
        if args.width is None:
            if args.worker or args.grade:
                p.error("Width workers require a single --width")
            selected_widths = list(dict.fromkeys(args.widths or [8, 16, 32]))
            output = output_path(args.output or ROOT / "outputs/width_probe")
            print(json.dumps(dict(
                suite="width", model="Nemotron 8B", widths=selected_widths,
                prompts=args.num_samples, cap=512,
                repeats=args.repeats or 1,
                methods=["Linear", "Quadratic", "Packed"],
                output=str(output), execute=args.execute,
            ), indent=2))
            if args.execute:
                for width in selected_widths:
                    command = [
                        sys.executable, str(Path(__file__).resolve()),
                        "--suite", "width", "--width", str(width),
                        "--num-samples", str(args.num_samples),
                        "--output", str(output / f"B{width}"), "--execute",
                    ]
                    if args.repeats is not None:
                        command += ["--repeats", str(args.repeats)]
                    subprocess.run(command, check=True)
            return
        for v in VARIANTS:
            v["block_length"] = args.width
            v["max_uncached_rows_per_forward"] = (
                args.width * (args.width + 1)
                if v["id"] == "quadratic"
                else args.width * (2 if v["id"] == "packed" else 1)
            )
    elif args.widths is not None:
        p.error("--widths requires --suite width")
    args.width = args.width or 8
    default_output = (
        ROOT / f"outputs/width_probe/B{args.width}"
        if args.suite == "width"
        else ROOT / f"outputs/nemotron_{args.suite}_{args.policy}"
    )
    args.output = output_path(args.output or default_output)
    if args.worker:
        if not args.execute:
            p.error("Worker requires --execute")
        if args.worker[0] not in args.scales or args.worker[1] not in args.tasks:
            p.error("Invalid worker selection")
        with run_lock(args.output / args.worker[0] / args.worker[1]):
            stage(args, *args.worker)
        return
    if args.grade:
        if not args.execute:
            p.error("Grading requires --execute")
        if args.grade[0] not in args.scales or args.grade[1] not in args.tasks:
            p.error("Invalid grading selection")
        with run_lock(args.output / args.grade[0] / args.grade[1]):
            grade_and_merge(args, *args.grade)
        return
    cells = [
        dict(
            scale=s,
            task=t,
            cap=cap_for(t, args.suite),
            prompts=min(args.num_samples or COUNTS[t], COUNTS[t]),
            repeats=args.repeats or repeats_for(t, args.policy),
        )
        for s in args.scales
        for t in args.tasks
    ]
    print(
        json.dumps(
            dict(
                cells=cells,
                policy=args.policy,
                suite=args.suite,
                dtype="bfloat16",
                variants=VARIANTS,
                output=str(args.output),
                execute=args.execute,
            ),
            indent=2,
        )
    )
    if not args.execute:
        return
    if any(cell["task"] in ("humaneval", "mbpp") for cell in cells):
        from grading_sandbox import preflight

        preflight()
    # Load each scale/task in a separate process; grading follows model release.
    for cell in cells:
        s, t = cell["scale"], cell["task"]
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--execute",
            "--policy",
            args.policy,
            "--suite",
            args.suite,
            "--output",
            str(args.output),
        ]
        if args.num_samples:
            command += ["--num-samples", str(args.num_samples)]
        if args.repeats:
            command += ["--repeats", str(args.repeats)]
        command += ["--width", str(args.width)]
        subprocess.run(command + ["--worker", s, t], check=True)
        env = dict(os.environ)
        if t == "mbpp":
            from evalplus.data import get_mbpp_plus
            from evalplus.data.mbpp import _ready_mbpp_plus_path

            # Download the canonical oracle before running candidate code in isolation.
            previous = os.environ.pop("MBPP_OVERRIDE_PATH", None)
            try:
                get_mbpp_plus(version="v0.2.0", noextreme=True)
                canonical = Path(
                    _ready_mbpp_plus_path(version="v0.2.0", noextreme=True)
                ).resolve()
            finally:
                if previous is not None:
                    os.environ["MBPP_OVERRIDE_PATH"] = previous
            from grading_sandbox import command as sandbox_command

            grading_env = dict(
                MBPP_OVERRIDE_PATH=str(canonical),
                HF_HUB_OFFLINE="1",
                HF_DATASETS_OFFLINE="1",
            )
            sandbox = sandbox_command(
                command + ["--grade", s, t],
                readonly=[ROOT, canonical],
                writable=[args.output / s / t],
                environment=grading_env,
            )
            subprocess.run(sandbox, check=True)
        else:
            subprocess.run(command + ["--grade", s, t], env=env, check=True)


if __name__ == "__main__":
    main()
