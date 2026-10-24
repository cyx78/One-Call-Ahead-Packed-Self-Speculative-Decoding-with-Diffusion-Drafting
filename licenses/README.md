# Attribution

SetDLM model code and Qwen integration derive from
[kuleshov-group/setdlms](https://github.com/kuleshov-group/setdlms).
Its MIT notice is retained in [licenses/implementation_LICENSE](implementation_LICENSE).

Nemotron decoding, runtime attention, and model interfaces build on
[NVlabs/Nemotron-Labs-Diffusion](https://github.com/NVlabs/Nemotron-Labs-Diffusion).
The upstream source-code and model-weight license notice is retained in
[licenses/nvlabs_reference_LICENSE](nvlabs_reference_LICENSE).
The local decoder and evaluation files implement the schedules and measurement
interfaces used here. Model checkpoints are downloaded from their pinned
repositories and are governed by their respective model licenses.

The Qwen attention implementation retains its source copyright and Apache 2.0
notice. EvalPlus is an installed dependency; its MBPP correctness checker is
called through `src/nemotron_ltr/evalplus_mbpp_base.py`.

Dataset sources and revisions are declared in the task loader and WritingBench
manifest. `reference/` contains derived numerical measurements and table values.
