# Tutorial: build a MoE operator graph

This tutorial shows the entrypoints added by the MoE support PR.
It follows the same path covered by the MoE tests, but uses standalone
snippets that are easier to copy into an experiment.

## What this PR adds

The main entrypoint is `build_moe_operator_graph(...)` in `syssim.moe`.
It builds a regular `OperatorGraph` with explicit MoE stage nodes:

```text
moe_router -> moe_dispatch -> moe_expert -> moe_combine
```

When `expert_parallel_size > 1`, the graph also inserts collective
all-to-all stages around expert execution:

```text
moe_dispatch -> collective -> moe_expert -> collective -> moe_combine
```

For Hugging Face models, use the wrappers in
`syssim.integrations.huggingface`:

- `trace_hf_moe_model_for_training(...)`
- `trace_hf_moe_model_for_inference(...)`

The unit-test reference for the manual graph path is
`tests/test_moe_graph.py`. The Hugging Face config extraction reference is
`tests/test_moe_hf_spec.py`.

## Build a graph from a SysSim MoE spec

Use `MoEModelSpec` for model structure and `MoERuntimeConfig` for the
runtime shape. This path does not require Transformers.

```python
import torch

from syssim import (
    HardwareInfo,
    MoEModelSpec,
    MoERuntimeConfig,
    SimulatorConfig,
    build_moe_operator_graph,
)

hw = HardwareInfo(
    peak_tflops_mm=989.0,
    peak_tflops_math=989.0,
    peak_memory_bandwidth_gbps=3350.0,
)
config = SimulatorConfig(hw_info=hw)

spec = MoEModelSpec(
    num_layers=2,
    hidden_size=256,
    intermediate_size=128,
    num_experts=4,
    top_k=2,
    vocab_size=1000,
    name="tiny_moe",
)
runtime = MoERuntimeConfig(
    batch_size=1,
    seq_len=8,
    dtype=torch.bfloat16,
)

graph = build_moe_operator_graph(spec, runtime, config)
print(graph.summary())
```

Expected stage counts for this example:

```text
moe_router: 2
moe_dispatch: 2
moe_expert: 2
moe_combine: 2
```

## Add expert-parallel communication

Set `expert_parallel_size` above 1 to insert two collective all-to-all
nodes per sparse layer. Without an explicit network topology and LogGP
model, SysSim uses a memory-roofline estimate for those collective nodes.

```python
ep_runtime = MoERuntimeConfig(
    batch_size=1,
    seq_len=8,
    dtype=torch.bfloat16,
    expert_parallel_size=2,
)

ep_graph = build_moe_operator_graph(spec, ep_runtime, config)
print(ep_graph.summary())
```

For the two-layer example above, the graph contains four `collective`
nodes: dispatch and combine all-to-all for each sparse layer.

## Model non-uniform routing

The default routing model is deterministic uniform routing because fake
tensor tracing cannot know data-dependent top-k choices. To model an
imbalanced batch, pass `tokens_per_expert`. Its length must match
`num_experts`, and its sum must equal `batch_size * seq_len * top_k`.

```python
imbalanced_runtime = MoERuntimeConfig(
    batch_size=1,
    seq_len=8,
    dtype=torch.bfloat16,
    tokens_per_expert=(16, 0, 0, 0),
)

imbalanced_graph = build_moe_operator_graph(
    spec,
    imbalanced_runtime,
    config,
)
```

This increases the `moe_expert` stage estimate relative to uniform
routing because the graph models the busiest expert load.

## Build a graph from a Hugging Face MoE config

For Qwen3-style Hugging Face models, the wrapper extracts these required
config fields:

- `num_hidden_layers`
- `hidden_size`
- `moe_intermediate_size`
- `num_experts`
- `num_experts_per_tok`

It also reads optional `vocab_size`, `decoder_sparse_step`, and
`first_sparse_layer`.

```python
import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from syssim import (
    HardwareInfo,
    MoERuntimeConfig,
    SimulatorConfig,
    trace_hf_moe_model_for_training,
)

model_config = Qwen3MoeConfig(
    num_hidden_layers=2,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    vocab_size=1000,
    num_experts=4,
    num_experts_per_tok=2,
)

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(
        model_config,
        torch_dtype=torch.bfloat16,
    )

inputs = {"input_ids": torch.randint(0, model_config.vocab_size, (1, 8))}
runtime = MoERuntimeConfig(batch_size=1, seq_len=8, dtype=torch.bfloat16)
sim_config = SimulatorConfig(
    hw_info=HardwareInfo(989.0, 989.0, 3350.0),
)

graph = trace_hf_moe_model_for_training(
    model,
    inputs,
    sim_config,
    runtime=runtime,
)
print(graph.summary())
```

## Run the included example

The Qwen3 MoE example is the executable entrypoint for the PR:

```bash
python examples/huggingface/train_qwen3_moe_single.py --batch-size 1 --seq-len 32
python examples/huggingface/train_qwen3_moe_single.py --batch-size 1 --seq-len 32 --expert-parallel-size 2
```

The first command reports only MoE stage nodes. The second command also
reports collective all-to-all nodes.

## Validate the tutorial path

Run the tests that cover the tutorial concepts:

```bash
python -m pytest tests/test_moe_graph.py tests/test_moe_hf_spec.py -q
python -m pytest tests/test_moe_tracing.py -q
```
