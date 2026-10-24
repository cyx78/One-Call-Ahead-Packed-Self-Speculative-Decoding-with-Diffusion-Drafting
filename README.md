# One Call Ahead: Packed Self-Speculative Decoding with Diffusion Drafting

Code and recorded measurements for the paper **One Call Ahead: Packed Self-Speculative Decoding with Diffusion Drafting**.
The three self-speculative schedules are **Linear-SS**, **Quadratic-SS**, and
**Packed-SS**, evaluated with Nemotron-Labs-Diffusion and SetDLM.
In configuration files, recorded data, and CSVs, their keys remain `Linear`,
`Quadratic`, and `Packed` (the paper also abbreviates them L, Q, and P).
`Native` denotes SetDLM's released non-speculative diffusion decoder.

| Experiments | Models | Tasks |
|---|---|---|
| Greedy and temperature-1 rejection sampling, including a greedy block-size ablation | Nemotron 3B, 8B | HumanEval, MBPP, GSM8K, MATH-500, English WritingBench |
| Greedy with LTR and confidence-ordered verification; native diffusion decoding | SetDLM smax16 (W=8), smax32 (W=16) | GSM8K |
| Proposal-only LoRA | Nemotron 8B | GSM8K |

## Setup

Run the commands below from the repository root in a **Linux terminal**
(Ubuntu/WSL on Windows), with **Python 3.12** installed.

Choose the setup for what you want to do. Each `.venv-*` folder is a Python
virtual environment: it holds the packages for one workflow. You only need
to create the environments you will use. The command paths below select
the appropriate environment directly.

### Rebuild saved results — no GPU needed

This installs the tools for calculating tables and drawing figures from the
included measurements. It does not download or run the language models.

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r configs/requirements-analysis.txt
```

Then follow [Reconstruct the recorded results](#reconstruct-the-recorded-results).

### Run Nemotron experiments — NVIDIA GPU required

Use this environment for the main Nemotron experiments, the block-size
ablation, and proposal-only LoRA.

```bash
python3.12 -m venv .venv-nemotron
.venv-nemotron/bin/python -m pip install -r configs/requirements-nemotron.txt
```

### Run SetDLM experiments — NVIDIA GPU required

Nemotron and SetDLM require different versions of some Python packages, so
SetDLM has its own environment. Create it if you want to run SetDLM.

```bash
python3.12 -m venv .venv-setdlm
.venv-setdlm/bin/python -m pip install -r configs/requirements-setdlm.txt
```

Generation requires a working CUDA setup. The runners download the required
models and datasets automatically. See the
[experiment settings](EXPERIMENTS.md#generation-and-measurement-settings)
for the recorded hardware and detailed settings.

### Additional setup for HumanEval and MBPP

These tasks grade generated Python code. On Ubuntu/WSL, install `bubblewrap`
to run that code in an isolated process:

```bash
sudo apt-get install bubblewrap
```

This extra setup is needed for code grading, not for rebuilding saved results
or running the mathematics tasks. System requirements for grading are listed
in [EXPERIMENTS.md](EXPERIMENTS.md#runtime-requirements).

## Reconstruct the recorded results

```bash
.venv-analysis/bin/python analyze.py
```

The analysis writes summary tables, paired confidence intervals, and plots
for Figures 3–5 to `outputs/rebuilt/`. The same command checks the rebuilt values
with the archived results at their original rounding precision. Add
`--bootstrap-draws 0 --skip-figures` to `analyze.py` for a quick
table-only check. Use `--check-only` to check previously rebuilt tables.

The Figure 3 reconstruction currently includes the speculative schedules only;
it does not yet reproduce the Native SetDLM ratios and paired intervals added
to the latest paper. Native absolute measurements are included in the CSVs.
See [EXPERIMENTS.md](EXPERIMENTS.md) for current paper labels and export coverage.

## Run experiments

Every runner prints its configuration without loading a model by default.
Add `--execute` to run generation and grading.

### Nemotron

Greedy decoding and temperature-1 rejection sampling:

```bash
.venv-nemotron/bin/python experiments/nemotron.py --policy greedy --execute
.venv-nemotron/bin/python experiments/nemotron.py --policy rejection --execute
```

The block-size probe is a small ablation of the greedy Nemotron experiment:
it compares B=8, 16, and 32 on the first ten GSM8K questions using Nemotron 8B.
Use `--width 16` to run just one block size.

```bash
.venv-nemotron/bin/python experiments/nemotron.py --suite width --execute
```

### SetDLM

Speculative decoding under LTR and confidence orders, and the native baseline:

```bash
.venv-setdlm/bin/python experiments/setdlm.py --suite matrix --execute
.venv-setdlm/bin/python experiments/setdlm.py --suite native --execute
```

### Proposal-only LoRA

The adapter experiment and its base-model timing comparison:

```bash
.venv-nemotron/bin/python experiments/nemotron.py --suite lora --execute
.venv-nemotron/bin/python experiments/nemotron.py --suite lora-base --execute
```

### Small runs and resuming

For a small generation run, select `--scales 3B --tasks gsm8k --num-samples 2`
for Nemotron or `--horizons 16 --num-samples 2` for SetDLM. Rejection sampling
defaults to 20 responses per HumanEval prompt and 7 per MBPP prompt; use
`--repeats 1` to shorten a trial. MBPP grading requires the full 378-task set;
smaller runs save their generated responses with an ungraded status.

Choose a separate `--output` directory for each configuration. Rerunning the
same command resumes from completed responses. Grading follows generation.
After changing the code or configuration, use a new output directory; saved
runs record the source version and reject incompatible resumes.

## Repository layout

```text
analyze.py          Rebuild and check the recorded tables and figures
experiments/        Run Nemotron or SetDLM experiments
src/                Decoder and model code
  nemotron_ltr/     Nemotron schedules, sampling, and scoring
  setdlm/           SetDLM schedules and benchmark
    model/          Supporting SetDLM model code
configs/            Experiment settings, model revisions, and requirements
reference/          Saved numerical results used by analyze.py
tests/              Correctness and regression tests
licenses/           Upstream attribution and license notices
```

To change the decoding algorithms, start with `src/nemotron_ltr/decoders.py`
or `src/setdlm/decoders.py`. To run experiments, use the commands above.

## Tests

```bash
.venv-analysis/bin/python -m pytest tests/test_package.py tests/test_runs.py
.venv-nemotron/bin/python -m pytest tests/nemotron
.venv-setdlm/bin/python -m pytest tests/setdlm
```

[EXPERIMENTS.md](EXPERIMENTS.md) contains the paper's figure/table mapping,
generation settings, measurement definitions, and validation details.
[licenses/README.md](licenses/README.md) provides upstream attribution.
