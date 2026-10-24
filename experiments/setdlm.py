"""Benchmark FP32 SetDLM on GSM8K with LTR and confidence ordering."""

from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import (
    append_row,
    bind_protocol,
    check_coverage,
    output_path,
    rows_at,
    run_lock,
    source_hashes,
    validate_measurement,
    write_json,
)


def variants_for(suite):
    if suite == "native":
        return [{"id": "native", "method": "Native", "order": "Native"}]
    return [
        {"id": f"{order}_{method.lower()}", "method": method, "order": order}
        for order in ("LTR", "Confidence")
        for method in ("Linear", "Quadratic", "Packed")
    ]


def run(args, horizon):
    import torch
    from setdlm.loading import load_model
    from setdlm import benchmark
    from setdlm.decoders import linear_decode, packed_or_quadratic_decode
    from nemotron_ltr.data import prepared_rows

    out = args.output / f"{args.suite}_smax{horizon}"
    variants = variants_for(args.suite)
    cap, window = 1024, horizon // 2
    row_cap = 128 if horizon == 16 else 289
    with run_lock(out):
        protocol = dict(
            suite=args.suite,
            horizon=horizon,
            window=window,
            dtype="float32",
            prompts=args.num_samples,
            cap=cap,
            seed=1234,
            variants=variants,
            source_sha256=source_hashes(),
            native_confidence_threshold=0.9,
        )
        bind_protocol(out, protocol)
        raw = out / "raw.jsonl"
        rows = rows_at(raw, recover_tail=True)
        done, expected = check_coverage(
            rows, args.num_samples, 1, [v["id"] for v in variants]
        )
        for row in rows:
            validate_measurement(row, cap)
        if done != expected:
            data = prepared_rows("gsm8k")
            model, tokenizer, device = load_model(horizon, ROOT / ".cache/models")
            dtypes = {
                parameter.dtype
                for parameter in model.parameters()
                if parameter.is_floating_point()
            }
            if dtypes != {torch.float32}:
                raise RuntimeError(f"Expected FP32 parameters, got {dtypes}")
            stop_sequences = {
                tuple(tokenizer.encode(text, add_special_tokens=False))
                for text in ("<|im_end|>", "<|eot_id|>")
            }
            if tokenizer.eos_token_id is not None:
                stop_sequences.add((tokenizer.eos_token_id,))
            stop_sequences.discard(())
            if not stop_sequences:
                raise ValueError("Tokenizer has no termination sequences")
            stop_sequences = tuple(
                sorted(stop_sequences, key=lambda item: (len(item), item))
            )

            def generate(index, variant, limit):
                benchmark.seed_everything(1234 + index)
                prompt = benchmark._gsm_prompt(tokenizer, str(data[index]["question"]))
                ids = benchmark._encode(tokenizer, model, prompt, target_length=limit)
                if ids.shape[1] == 0 or ids.shape[1] >= model.config.length:
                    raise ValueError(
                        "Prompt must be nonempty and shorter than the context window"
                    )
                common = dict(
                    target_length=limit,
                    window_size=window,
                    max_uncached_rows_per_forward=row_cap,
                    stop_sequences=stop_sequences,
                )
                with torch.inference_mode():
                    if variant["method"] == "Native":
                        tokens, metrics = benchmark._native_decode(
                            model, tokenizer, ids, seed=1234 + index, **common
                        )
                        wall, nfe = metrics["latency_s"], metrics["total_forward_calls"]
                        acceptance = metrics["generated_tokens_per_step"]
                        model_s = None
                        stopped = metrics["termination_position"] is not None
                        max_rows = metrics["max_uncached_rows"]
                    else:
                        order = "ltr" if variant["order"] == "LTR" else "confidence"
                        decoder = (
                            linear_decode
                            if variant["method"] == "Linear"
                            else packed_or_quadratic_decode
                        )
                        kwargs = (
                            {}
                            if variant["method"] == "Linear"
                            else {"execution_mode": variant["method"].lower()}
                        )
                        result = decoder(
                            model,
                            ids,
                            order_policy=order,
                            record_model_call_walltimes=True,
                            **common,
                            **kwargs,
                        )
                        if not result.converged:
                            raise RuntimeError("Decoder did not complete the response")
                        tokens = result.target_ids
                        wall, nfe = result.latency_s, result.total_forward_calls
                        model_s = sum(result.model_call_walltimes_s)
                        acceptance = result.generated_tokens_per_step
                        stopped = result.termination_position is not None
                        max_rows = result.max_uncached_rows
                if bool((tokens == model.mask_token_id).any()):
                    raise RuntimeError("Output contains unresolved mask tokens")
                if sum(acceptance) != tokens.numel():
                    raise RuntimeError("Acceptance counts disagree with output length")
                if not stopped and tokens.numel() != limit:
                    raise RuntimeError("Output ended before EOS or the token cap")
                raw_text, text, prediction = benchmark._decode_completion(
                    tokenizer, tokens
                )
                gold = benchmark._normalized_answer(
                    benchmark._gold_answer(str(data[index]["answer"]))
                )
                row = dict(
                    index=index,
                    replicate=0,
                    variant=variant["id"],
                    method=variant["method"],
                    order=variant["order"],
                    horizon=horizon,
                    seed=1234 + index,
                    generated_tokens=tokens.numel(),
                    walltime_s=wall,
                    nfe=nfe,
                    decode_nfe=nfe - 1,
                    stopped_on_eos=stopped,
                    max_uncached_rows=max_rows,
                    model_call_work_walltime_s=model_s,
                    output_token_ids=tokens[0].tolist(),
                    output_text=text,
                    raw_output_text=raw_text,
                    prediction=prediction,
                    gold_answer=gold,
                    correct=prediction == gold and gold is not None,
                    emitted_tokens_per_round_histogram=dict(Counter(acceptance)),
                )
                validate_measurement(row, limit)
                return row

            warmup = [generate(0, variant, 16) for variant in variants]
            write_json(
                out / "warmup.json",
                [
                    {"variant": row["variant"], "walltime_s": row["walltime_s"]}
                    for row in warmup
                ],
            )
            for index in range(args.num_samples):
                shift = index % len(variants)
                for variant in variants[shift:] + variants[:shift]:
                    key = (0, index, variant["id"])
                    if key in done:
                        continue
                    row = generate(index, variant, cap)
                    append_row(raw, row)
                    rows.append(row)
                    done.add(key)
                write_json(
                    out / "status.json",
                    dict(status="running", completed=len(done), expected=len(expected)),
                )
                print(
                    f"smax{horizon}: {len(done)}/{len(expected)} responses", flush=True
                )
        check_coverage(
            rows, args.num_samples, 1, [v["id"] for v in variants], complete=True
        )
        summaries = []
        for variant in variants:
            selected = [row for row in rows if row["variant"] == variant["id"]]
            n = len(selected)
            wall = sum(row["walltime_s"] for row in selected)
            tokens = sum(row["generated_tokens"] for row in selected)
            nfe = sum(row["nfe"] for row in selected)
            summaries.append(
                dict(
                    **variant,
                    samples=n,
                    mean_wall_s=wall / n,
                    mean_tokens=tokens / n,
                    mean_nfe=nfe / n,
                    tokens_per_s=tokens / wall,
                    tokens_per_nfe=tokens / nfe,
                    score_pct=100 * sum(row["correct"] for row in selected) / n,
                )
            )
        write_json(out / "summary.json", summaries)
        write_json(
            out / "status.json",
            dict(status="complete", completed=len(rows), expected=len(expected)),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons", nargs="+", type=int, choices=[16, 32], default=[16, 32]
    )
    parser.add_argument("--suite", choices=["matrix", "native"], default="matrix")
    parser.add_argument("--num-samples", type=int, default=1319)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/setdlm")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", type=int, choices=[16, 32], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.num_samples <= 1319:
        parser.error("num-samples must be between 1 and 1319")
    args.output = output_path(args.output)
    if args.worker:
        if not args.execute:
            parser.error("Worker requires --execute")
        run(args, args.worker)
        return
    print(
        json.dumps(
            dict(
                suite=args.suite,
                horizons=args.horizons,
                prompts=args.num_samples,
                dtype="float32",
                windows={str(h): h // 2 for h in args.horizons},
                variants=variants_for(args.suite),
                max_new_tokens=1024,
                output=str(args.output),
                execute=args.execute,
            ),
            indent=2,
        )
    )
    if args.execute:
        for horizon in dict.fromkeys(args.horizons):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--execute",
                    "--worker",
                    str(horizon),
                    "--suite",
                    args.suite,
                    "--num-samples",
                    str(args.num_samples),
                    "--output",
                    str(args.output),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
