# Experiments and reproducibility

Paper locations follow the September 25, 2026 ICLR manuscript. LaTeX labels
identify the same results if section or float numbering changes again.

| Paper location | LaTeX label | Experiment | Entry point / reference group |
|---|---|---|---|
| Section 4.1, Fig. 3, Tables 7, 9, 11 | `fig:efficiency-greedy`, `tab:nemotron`, `tab:score-comparisons`, `tab:paired-ci` | Greedy Nemotron 3B/8B, five tasks, three schedules | `experiments/nemotron.py --policy greedy`; `greedy_*` |
| Section 4.2, Fig. 4, Tables 8, 9, 12 | `fig:efficiency-sampling`, `tab:rejection`, `tab:score-comparisons`, `tab:sampling-ci` | Temperature-1 Nemotron, five tasks, three schedules | `experiments/nemotron.py --policy rejection`; `rejection_*` |
| Section 4.1, Appendix F.3, Fig. 3, Tables 4, 10 | `app:setdlm-conf`, `tab:setdlm-order`, `tab:setdlm-paired-ci` | SetDLM smax16/W=8 and smax32/W=16, LTR/confidence-ordered verification | `experiments/setdlm.py --suite matrix`; `setdlm_*_LTR`, `setdlm_*_Confidence` |
| Appendix E.6, Fig. 3, Table 4; accuracy comparisons in F.7 | `app:native-baseline`, `tab:setdlm-order`, `app:accuracy-records` | Native SetDLM confidence-threshold decoding at 0.9 | `experiments/setdlm.py --suite native`; `setdlm_*_native` |
| Appendix F.2, Table 3 | `app:width`, `tab:width-probe` | Block-size ablation of greedy Nemotron 8B, B=8,16,32 | `experiments/nemotron.py --suite width`; `reference/width_probe.json` |
| Appendix F.4, Tables 5–6 | `app:lora-details`, `tab:lora`, `tab:lora-iteration` | Proposal-only LoRA and base iteration cost | `experiments/nemotron.py --suite lora` and `--suite lora-base`; `lora_8B_gsm8k`, `reference/lora_timing.json` |
| Appendix F.5, Fig. 5 | `app:time-breakdown`, `fig:nemotron-breakdown` | Greedy mathematics model execution / non-model decoder time | `analyze.py`; greedy GSM8K and MATH-500 groups |
| Appendix F.1, Table 2 | `app:rows`, `tab:structures` | Logical queries and evaluated uncached rows per iteration | Decoder mask builders and row counters |
| Appendix E.1 | `app:experimental-details` | Model/checkpoint setup | `configs/models.json` records the four pinned revisions |

`configs/experiment_matrix.json` lists the prompt count, repeats, temperature,
ordering, and output cap for every recorded group. The 27 groups contain 77
method cells. The block-size probe adds nine cells, each with ten responses.
The two SetDLM checkpoints have training horizons 16/32 and active windows
8/16; this is not a same-weights window-size ablation. `Native` has no
verification order: its separate group does not define an LTR or confidence
variant.

## Exported tables and figures

| Output in `outputs/rebuilt/` | Current paper counterpart |
|---|---|
| `table03_width.csv` | Table 3: block-size probe |
| `table04_setdlm.csv` | Table 4: LTR, confidence-ordered, and Native SetDLM measurements |
| `table05_lora.csv` | Table 5: proposal-only LoRA |
| `table06_lora_iteration_cost.csv` | Table 6: model execution time per iteration |
| `table07_greedy.csv` | Table 7: greedy Nemotron measurements |
| `table08_sampling.csv` | Table 8: temperature-1 Nemotron measurements |
| `paired_comparisons.csv` | Speculative-schedule comparisons underlying Tables 9–12 |
| `figure03_metrics.pdf`, `figure04_metrics.pdf` | Figures 3–4: speculative-schedule efficiency and accuracy comparisons |
| `figure05_timing.pdf` | Figure 5: greedy mathematics timing decomposition |
| `figure05_timing_components.csv` | Figure 5 components, plus supplementary temperature-1 components |

The point-estimate check retains 252 archived rounded values for 66 cells
in Tables 4, 7, and 8. It does not yet check every column or the latest
manuscript's increased throughput precision. `Native` absolute values are
exported, but the analysis does not yet join Native and LTR prompt records
to reconstruct the Native ratios, accuracy differences, and paired intervals
shown in the latest Figure 3 and Appendix F.7. Figure 3's generated plot
therefore covers only the speculative schedules.

## Generation and measurement settings

### Runtime requirements

Use Python 3.12 on Linux, including WSL on Windows. Shared runner utilities
use Linux file locking (`fcntl`), including during analysis. The two model
families use separate environments because their dependency versions differ.
Generation requires an NVIDIA GPU with CUDA; the recorded runs used one
A100 80 GB GPU at batch size one. Model and dataset downloads use the pinned
revisions and are stored in `.cache/`.

HumanEval and MBPP grading require bubblewrap (`bwrap`) and a Linux system
that permits unprivileged user namespaces. These requirements apply to code
grading; saved-result analysis and mathematics scoring do not use bubblewrap.

### Models

| Checkpoint | Revision | Layers | Dtype |
|---|---|---:|---|
| `nvidia/Nemotron-Labs-Diffusion-3B` | `b9d4dfe782378d663699cf0f8a272e10f5956a25` | 26 | BF16 |
| `nvidia/Nemotron-Labs-Diffusion-8B` | `16c67f0560b912e93e0cabb6e0c4f5c3086d95fc` | 34 | BF16 |
| `kuleshov-group/setdlm-gsm8k-smax16` | `555271c1ef014ecd78a5c11d1e8849a6fc721388` | 28 | FP32 |
| `kuleshov-group/setdlm-gsm8k-smax32` | `fe334b030407c145e76fd3f64472fc0fc1235a86` | 28 | FP32 |

Nemotron uses root-inclusive B=32 for Linear-SS/Packed-SS and B=8 for
Quadratic-SS; the speculative width is W=B−1. SetDLM uses active windows
W=8 and W=16 for the smax16 and smax32 training horizons, respectively.
The windows use separately adapted checkpoints, not the same model weights.
Both SetDLM orderings use zero ordering-prior strength. Native SetDLM uses
confidence threshold 0.9 and the checkpoint's first-hitting-time cache ordering.
It commits confident parallel predictions without speculative verification,
so Native has no verification order and is a separate decoding policy.

### Tasks and generation

| Task | Prompts | Output cap | Responses per prompt at T=1 |
|---|---:|---:|---:|
| HumanEval | 164 | 512 | 20 |
| MBPP | 378 | 512 | 7 |
| GSM8K | 1,319 | 1,024 | 1 |
| MATH-500 | 500 | 1,024 | 1 |
| English WritingBench | 555 | 512 | 1 |

Greedy runs use one response per prompt. Both SetDLM checkpoints use the full
GSM8K test split and a 1,024-token canvas.

The block-size probe, proposal-only LoRA run, and base iteration-cost runner
use a total cap of 512 newly generated tokens, shared by thinking and answer
tokens; prompt tokens are excluded. Generation ends at EOS or that total cap.
All 3,957 archived LoRA responses (1,319 per schedule) record both measured
and effective caps of 512. There is no separate thinking cap; thinking and
answer generation share the same total output budget.

Nemotron prompt text and dataset revisions are in
`src/nemotron_ltr/data.py`; SetDLM's prompt is in
`src/setdlm/benchmark.py`. WritingBench selects every record
with `lang == 'en'` in source order and disables thinking in the chat template.
Tokenizer EOS is used for code, mathematics, and greedy WritingBench. Sampled
WritingBench uses the union of model and tokenizer EOS IDs.

Rejection sampling uses FP32 categorical probabilities, temperature 1,
top-p 1, acceptance probability min(1,p/q), and a normalized positive
(p−q) correction. Full draft acceptance adds one sample from the target
distribution. Packed and Quadratic retain the proposal distributions aligned
with the next speculative tokens.

### Scores

HumanEval uses extracted Python candidates and base tests in an isolated
subprocess. MBPP uses EvalPlus 0.3.1 with its 378-task v0.2.0 NoExtreme release
and original base tests. Code scores average per-response correctness across
all prompts and repeats.

GSM8K and MATH-500 use boxed-answer extraction and numeric/exact comparison.
The Nemotron 8B GSM8K scorer uses a flat-box regex; the 3B and LoRA scorers
parse balanced braces. SetDLM uses its boxed/answer-marker extractor.
WritingBench correctness is null; throughput is measured without a quality score.

### Timing and aggregation

Wall time covers prompt-cache/bootstrap work and decoding, with CUDA
synchronization at timing boundaries. Model loading, tokenization, text
decoding, serialization, and grading occur outside that interval. Model execution
timing excludes the vocabulary head and sampling operations. The remaining
inclusive decoder time is labeled non-model decoder time.

NFE is the total number of model forward calls, including prompt processing
and setup. Throughput is sum(tokens)/sum(wall seconds). Tokens/NFE is
sum(tokens)/sum(total NFE), and Time/NFE is sum(wall seconds)/sum(total NFE).
Time/NFE includes decoder overhead outside the model forward pass.
SetDLM records decode calls and its one setup call separately, but the latest
paper's Table 4 and all per-NFE comparisons use total NFE, including setup.
`mean_decode_nfe` is an auxiliary exported diagnostic, not the paper's NFE.
The comparison CSV keys `tokens_per_call_ratio` and `time_per_call_ratio`
also refer to total NFE.

The recorded greedy mathematics groups use an effective 1,024-token cap:
EOS-complete 512-token responses are retained, and prompts truncated by any
schedule are rerun for all three schedules at 1,024 tokens. Each record stores
its measured and effective caps. Fresh generation runs use 1,024 tokens directly.

Paired intervals use 20,000 prompt bootstrap draws. All repeated responses to
a prompt remain in the same resampled cluster. Bootstrap seeds are included
in the output CSV. The current reconstruction aggregates Native SetDLM groups
independently and does not yet produce the paired Native-versus-LTR comparisons
in the latest paper. Those comparisons require resampling the same prompt
indices in both groups.
Figure 5 uses greedy decoding only: it normalizes each mathematics task by
its Linear-SS wall time, then averages GSM8K and MATH-500 with equal weight.

## Validation

The simplified layout was validated on October 11, 2026:

- **160 CPU tests passed** across the separate Nemotron and SetDLM environments.
  These cover decoder outputs, rejection sampling, cache behavior, LoRA row
  selection, scoring, checkpoint loading, output recovery, and the consolidated
  block-size and analysis commands. Small model tests exercise the actual
  SetDLM backbone and native generation.
- Analysis validates all **89,719** main response records and **90** block-size
  comparison records. Before/after reconstruction with 200 bootstrap draws
  produced identical results in all 11 CSV/JSON exports and regenerated the
  three PDF figures. Normal analysis still defaults to 20,000 bootstrap draws.
- All **252** archived rounded table values match. This does not cover every
  column or the Native paired comparisons in the latest manuscript.
- Saved numerical measurements and upstream license texts were preserved.
  Tests and analysis were run from a separate clean copy of the repository.

Full-size GPU generation was not rerun. Code-grading isolation was checked
through subprocess construction and failure handling; an end-to-end
bubblewrap grading run was not executed in the validation environment.

Run the test commands in the README to repeat the included checks. `analyze.py`
validates measurement coverage and automatically checks the rebuilt table values.
