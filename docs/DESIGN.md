# SysSim — Design and Mechanism

## 1. What SysSim is

SysSim predicts the **step time** and **per-GPU peak memory** of distributed LLM training on hardware you do not physically have, **without running real computation**.

Its founding bet is a statement about the physics of deep-learning kernels:

> The cost of a deep-learning computation is determined by its **structure** — the operator types, the tensor shapes, the dtypes, and the dependency/overlap structure — together with the **target hardware's constants**. It is *not* determined by the numeric values in the tensors.

A matmul of a given shape in bf16 performs the same number of floating-point operations and moves the same number of bytes whether its operands hold trained weights or uninitialized garbage. An all-reduce of a given size occupies the fabric for the same duration regardless of what is being summed. So the cost-relevant information about a training step is exactly the information a **fake tensor** carries — shape, dtype, and device label — and *nothing* that requires real data or a launched kernel.

That bet buys a clean separation into three concerns, each independent of the others:

1. **STRUCTURE.** Run the real training stack *once*, on a single ordinary GPU, with fake tensors, and intercept every operation PyTorch dispatches. The result is an operator DAG: every GEMM, attention op, elementwise op, collective, and cross-stream synchronization, each annotated with exact tensor metadata and an edge to whatever must finish before it. This DAG is the hardware-independent fingerprint of the step.
2. **COST.** Price each operator analytically from its shape/dtype and the target accelerator's published peaks — a multi-resource roofline. Optionally, a learned per-device residual tightens the roofline toward measured reality.
3. **TIMING.** Replay the DAG through a discrete-event simulator that models stream overlap, network contention, and pipeline bubbles, and recover the wall-clock critical path. The step time is the makespan; the exposed communication is measured by re-running with collectives stripped.

Memory is read off the **same trace** by a separate clean pass.

Because structure is captured once and is hardware-independent, the *same* trace can be re-priced for GH200, for a fictional future accelerator, for TP=2/DP=4 or TP=8/PP=4 — with no re-tracing. That re-usability is the entire point.

```
                       ┌──────────────────────────────────────────────────┐
   model YAML  ─┐      │  STRUCTURE (once, on any single GPU, fake tensors) │
 (architecture) │      │                                                    │
                ├────► │  fake process group @ real world size              │
 parallelism /  │      │  → real Megatron builds the per-rank sharded model │
 training kwargs┘      │  → TorchDispatchMode intercepts every aten op      │
                       │  → operator DAG (shapes, dtypes, streams, sync)    │
                       └───────────────────────┬───────────────┬───────────┘
                                                │ (immutable)   │
                          ┌─────────────────────▼──┐         ┌──▼──────────────┐
   hardware YAML ───────► │  COST (per op, analytic)│         │  MEMORY pass    │
 (peaks, topology)        │  roofline = max(tensor, │         │  (1 microbatch, │
                          │   fma, sfu, mem, launch) │         │   real DDP +    │
                          │  [+ learned residual]    │         │   dist-Adam,    │
                          └─────────────┬───────────┘         │   MemTracker)   │
                                        │ ms/op                └──┬──────────────┘
                          ┌─────────────▼────────────────────┐   │ per-bucket bytes
                          │  TIMING (discrete-event replay)   │   │
   inject: DP grad sync,  │  per-stream FIFO queues +         │   │  × 1F1B in-flight
   analytic optimizer,    │  cross-stream sync edges          │   ▼
   PP P2P edges           │  collectives timed by flow sim    │  per-GPU peak,
                          │  (max-min fair over links)        │  OOM check
                          └─────────────┬─────────────────────┘
                                        ▼
                          step time, fwd/bwd/opt, exposed comm, MFU/HFU
```

Throughout this document one **worked example** is advanced stage by stage as a high-level walkthrough — no arithmetic, no measurements. It is a single transformer (GPT) decoder block sized like the `qwen3-8b` configuration in the repository (GQA attention, a SwiGLU FFN), in bf16, parallelized **TP=2** (tensor) and **DP=4** (data) across **8 GH200 GPUs** arranged as two 4-GPU nodes. Each step says what the engine *does* to the block and what *qualitatively* falls out — which operators are compute- versus memory-bound, which collectives ride fast intra-node links versus the slow inter-node fabric, and what drives the memory peak — all following from the mechanisms below, not from profiled numbers.

---

## 2. How a number is produced — end to end

Before the deep dives, here is the whole pipeline in one breath, using the worked example.

We start with two inputs that are kept rigorously separate. The **model YAML** carries *only* architecture — layers, hidden, heads, query-groups, ffn, vocab, seq. The **hardware YAML** carries *only* device peaks, per-GPU memory, and a topology block. Parallelism (`tp/dp/cp/pp`) and training knobs (micro/global batch, dtype, recompute, distributed-optimizer) are Python kwargs, never YAML. This split is what makes the captured structure reusable: architecture is intrinsic to the model, a parallelization is a per-run decision about how to map that fixed architecture onto a machine.

From `tp·dp·cp·pp` we derive `world_size = 8`. We start a process, install a **fake process group** that reports `world_size = 8` and `rank = r`, and let the **real Megatron** code build a `GPTModel`. Because the framework genuinely believes it is rank `r` of an 8-rank world, it builds **column/row-parallel** linears whose weights are already divided by `tp=2`, a vocab-parallel embedding sliced by `tp`, and it wires the TP all-reduces into the forward. We then run the real forward+backward on **fake CUDA tensors** and intercept every dispatched aten op. Out comes a DAG of a few dozen operator nodes for the block — the four projection GEMMs, the SwiGLU gate/up/down GEMMs, the explicit attention sequence (a QKᵀ score `baddbmm`, a `_softmax` with `masked_fill`, and a scores·V `bmm`), RMSNorm/RoPE/SwiGLU/residual math ops, plus the captured TP collectives — with view/reshape ops emitting *no* node.

Each node is then **priced** analytically. A representative GEMM (the FFN gate+up projection, sharded by TP) is large and dense, so the tensor-core term dominates and it comes out compute-bound. The TP all-reduces captured in the trace, and the **DP gradient sync injected afterward**, are timed by a flow-level network simulator over the real topology.

Finally the DAG is replayed through a **discrete-event simulator**. Compute lives on one CUDA stream, async collectives on another; the simulator advances a clock to the next finishing op, so communication **overlaps** compute exactly to the degree the dependency structure allows. The step time is the makespan. Memory comes from a separate one-microbatch pass that measures the live byte set under the real optimizer.

The rest of this document opens each of these stages: the idea, the mechanism, why it works, and the worked example advanced one step.

---

## 3. Tracing and the fake-tensor trick

### The idea

The structure layer captures *what the step is made of* — the thing the rest of the engine turns into time and memory. Rather than profile the model on the target accelerator (which you may not have, and which would conflate the kernel's intrinsic cost with that specific GPU's quirks), we run the model once on **any** GPU with **fake tensors** — objects that carry only shape, dtype, and device label, never real bytes and never a launched kernel — and observe every operation PyTorch dispatches.

### The mechanism

1. **Swap the model to fakes.** Every parameter and buffer is replaced in place by a fake tensor with the same shape/dtype/stride but allocated on a *fake CUDA device*; the inputs are converted the same way. A restore log records every swap so the real model is put back verbatim afterward — tracing is non-destructive.

2. **Run the real training stack on the fakes.** The tracer invokes Megatron's actual `forward_backward_func` with the fake model, so the *same code path* real training would take issues the *same sequence of ops*. Nothing is hand-modeled; the structure is observed, not assumed.

3. **Intercept at the dispatcher.** A `TorchDispatchMode` sees every aten op. Pure-metadata queries (sizes, strides, dtype) pass straight through. **View ops** (transpose/slice/reshape) are executed to propagate the fake storage alias but emit **no node** — they launch no kernel, cost nothing, and only rename existing bytes. **Tensor-creation ops** emit a **zero-time node** (they reserve memory but move no data). Everything else becomes a real operator node.

4. **Classify and shape-tag each real op.** The op type tells the cost layer which roofline ceiling applies and tells the timing replay which hardware resource the work consumes — so each dispatched op is mapped to one type: `c10d` ops → `COLLECTIVE`; `mm/addmm/bmm/matmul/linear` → `GEMM` with M/N/K read straight off the (already TP-sharded) operand shapes; SDPA ops, when present, → `ATTN`; cross-device copies → `MEMORY`; everything else (RMSNorm, softmax, masked_fill, RoPE, SwiGLU, residual adds) → `MATH`. The node stores input and output tensor metadata, so the cost layer later has exact shapes and dtypes without re-running anything.

5. **Tag the training phase.** A module tracker runs alongside, exposing an `is_bw` flag; each op is stamped *forward* or *backward* at capture time. There is no separate backward trace — autograd's own backward ops flow through the same dispatch interception, just stamped backward.

6. **Record causality, not just order.** Each op's structural predecessor is the previous op on its **own CUDA stream** (a per-stream FIFO chain). Cross-stream dependencies are captured *explicitly* — never inferred from data — from three sync sources: patched CUDA Event record/wait, the collective-capture sync logic, and async-collective handle `.wait()` calls. This preserves true concurrency: compute on stream 0 and a collective on stream 1 are independent in the DAG and can overlap during replay.

7. **Neutralize real collectives.** During tracing `torch.distributed` collectives and `send/recv` are patched to no-ops (real NCCL would crash on data-less fake tensors). But the no-op is not silent: it **records** a `COLLECTIVE` node on a dedicated comm stream (stream 1) carrying the byte count (`numel × element_size`) and the participating rank list, and for synchronous collectives it injects a `STREAM_SYNC` back on the caller stream so subsequent compute correctly waits — exactly mirroring NCCL's post-collective sync. Async collectives return a mock handle whose `.wait()` emits that sync later, where the program actually waits.

The finished DAG (plus model/parallelism/training provenance) becomes an immutable `Trace`. A `simulate_on(hardware)` method works on a *copy*, injects the hardware-dependent pieces, and replays the graph — so one trace re-costs for many hardware targets.

### Why it works

The structure-determines-cost premise is *physically exact* for the dominant ops: a GEMM's FLOP count is `2·M·N·K` and its HBM traffic is the sum of operand and result bytes — both functions purely of shape and dtype; a collective's wire time is `bytes / effective-bandwidth`. None of these depend on numeric contents, so a valueless fake tensor of the right shape/dtype carries *all* the cost-relevant information.

The fakes must still claim to live on **CUDA**. PyTorch's dispatcher selects the kernel variant — and therefore the GPU-shaped decompositions — based on device *type*. A fake CUDA device makes the recorded operator set and shapes match what would really run on a GPU, while no kernel executes. (The tracer hard-fails on a CPU-only torch build precisely for this reason.)

TP sharding is captured for free and correctly because the tracer runs Megatron's *real* tensor-parallel modules: a column- or row-parallel linear already presents its **local post-shard** shape at dispatch. There is no separate sharding model that could drift from reality, because there is no separate model.

Cross-stream causality is recorded from explicit synchronization primitives rather than guessed from producer/consumer tensor reuse. This avoids both *false serialization* (which would hide real overlap and over-estimate step time) and *false parallelism* (which would hide contention). The DAG therefore encodes exactly the concurrency the real runtime has.

A separate **memory pass** runs one microbatch under a tracker with the real DDP + distributed-Adam optimizer on the fake model, stepping the optimizer once *before* the backward so master weights and Adam m/v are resident at the backward peak. The two passes are deliberately distinct: the cleanest memory peak is a single-microbatch snapshot, while timing needs the full multi-op runtime graph; conflating them would corrupt both.

### Worked example — step 1

Tracing one `qwen3-8b` decoder block under TP=2 yields a few dozen real operator nodes per microbatch: the projection GEMMs, the explicit attention sequence (the QK score `baddbmm`, the `_softmax` with its `masked_fill`, and the scores·V context `bmm`), the two SwiGLU FFN GEMMs, the RMSNorm/RoPE/SwiGLU/residual `MATH` ops, and the TP collectives — while every view/reshape op emits no node. Because the layer uses Megatron's explicit (unfused) attention path, the attention score matrix is materialized to HBM as its own tensor, which is what makes the softmax a real, separately-priced node in §5 rather than something hidden inside a fused kernel.

The one effect to hold onto: TP=2 means every projection GEMM is recorded *already sharded* — its model-parallel dimension halved — because the real tensor-parallel module presents its local post-shard shape at dispatch. We do no shape arithmetic; the sharded shapes simply arrive.

---

## 4. The operator-DAG representation and how overlap is encoded

### The idea

An LLM training step is **not** a serial list of kernels whose times you add up. It is a partially-ordered set of operations, some of which run *concurrently* on different hardware engines — the SM array doing matmuls, the copy engines moving bytes, the NVLink/NIC fabric carrying collectives. **Step time is the length of the longest dependency chain through that concurrency**, not the sum of the pieces. The DAG is the representation that makes both the partial order and the concurrency explicit and machine-readable.

### The mechanism

Every operator becomes one node tagged with its **operator type** — `GEMM`, `ATTN`, `MATH` (work on the SMs), `COLLECTIVE` (traffic on the fabric), `MEMORY`, or the two synchronization kinds `BARRIER` and `STREAM_SYNC`. The type tells the cost layer which roofline ceiling to apply and tells the timing replay which resource the work consumes.

Each node records the live CUDA stream it was issued on as `stream_id`, read straight from `torch.cuda.current_stream()` at dispatch time. A *stream* is an in-order hardware execution lane — the queue of work the GPU runs front-to-back. Compute is captured on stream 0; observed/injected collectives go on a dedicated comm stream (stream 1). Two different `stream_id`s mean two lanes the hardware can advance simultaneously.

Ordering lives in **one `predecessors` list per node**, deliberately overloaded to hold two kinds of edge:

- **Same-stream FIFO edge.** When a node is created, the tracer looks up the last op on its stream and adds it as a predecessor, then records itself as the new last op on that stream. This threads each stream into a linear chain — the GPU runs that lane in issue order, no more, no less.
- **Cross-stream causality edge** — the *only* thing that couples lanes. When a stream records a CUDA event, the engine remembers which op on which stream that event stands for; when another stream waits on that event, a `STREAM_SYNC` node is emitted on the waiting stream with predecessors `(a)` the previous op on the waiting stream and `(b)` the op the event captured on the source stream. That `STREAM_SYNC` becomes the new head of the waiting lane, so everything after it transitively depends on the source-stream work — overlap stops exactly where the program asked it to.

Collectives get the same treatment automatically. A captured collective is a `COLLECTIVE` node on stream 1 whose predecessors are the current heads of both the caller's compute stream and the comm stream. For a **synchronous** collective the engine drops a `STREAM_SYNC` back onto the caller's compute stream depending on that collective; for an **async** collective no such sync is inserted at launch — the later `.wait()` emits the `STREAM_SYNC`, and in between, compute on stream 0 is free to overlap the comm on stream 1 because no edge joins them. This is precisely how gradient-reduction overlap with backward compute is represented.

`BARRIER` is the heavyweight ordering primitive: honored in the replay only when *every other stream has fully drained*, modeling a global wait. The DAG validates itself — reference integrity (no dangling predecessor) and cycle detection by DFS coloring — guaranteeing a true DAG with a topological order. The topological sort is cached and dropped on any mutation.

### Why it works

The representation is **faithful to the hardware's actual concurrency model**: real GPUs execute each stream in issue order and run independent streams concurrently, synchronizing only through events/barriers. The IR mirrors this one-to-one — one FIFO chain per stream, cross-stream edges only where a real CUDA event/barrier existed. So the set of legal execution orderings the DAG admits is exactly the set the hardware admits: no spurious serialization, no illegal overlap.

Overlap is therefore the **default** — it emerges from the *absence* of an edge — and serialization is the thing that must be *earned* by a real sync. Critical-path timing is the correct objective on concurrent hardware (the longest weighted chain of dependent ops), and the DAG is exactly what makes it computable. Summing op times would over-count (serializing work the hardware overlaps); taking a single max would under-count.

One overloaded `predecessors` list — rather than separate "data" and "sync" edge sets — is the simplest structure that carries all timing-relevant information, because the replay only ever asks one question of an edge: *"is the source completed, and when did it finish?"* Both kinds of edge answer that identically (gate eligibility, push the start time). The `stream_id` already tells you which kind an edge is when you need to know (export draws same-stream edges solid, cross-stream dashed).

### Worked example — step 2

For the block, tracing yields the few-dozen nodes above. With TP=2 the row-parallel halves of the attention output projection and the FFN down projection must be summed, so Megatron emits an **all-reduce after each** — these appear as `COLLECTIVE` nodes on stream 1, independent of the compute nodes on stream 0 until their sync. The DP gradient reduction (injected later, §8) is a separate `COLLECTIVE` hung off the last traced op on stream 1, with no edge joining it to the in-flight backward GEMMs until its `.wait()` — which is what lets the replay overlap it.

---

## 5. The roofline cost physics

### The idea

Given one operator's structure (op type, input/output shapes and dtypes) and the target accelerator's published peak rates, the cost layer returns how many nanoseconds that operator should take — analytically, no kernel run. It rests on one physical truth: a GPU kernel is a pipeline of distinct hardware resources — tensor-core matrix units, FP32 vector/FMA lanes, the special-function unit (SFU) for transcendentals like `exp`, and the HBM subsystem — and the kernel **cannot finish faster than whichever resource it saturates**. So the runtime is the **maximum** over the time each resource alone would need, floored by a fixed kernel-launch overhead.

This generalizes the textbook single-knee roofline into a **multi-resource bound**, which is what makes it correct for the non-GEMM operators (softmax, layernorm, gelu) that a single tensor-core ceiling would badly mis-predict.

### The mechanism

For each operator the cost layer receives the function packet, the argument tensors, the output tensor(s), and an op-type tag. View ops and pure allocation ops short-circuit to zero — they launch no kernel and move no data.

- **Compute demand (`tensor_ns`).** Derive the FLOP count purely from shapes: a matmul of `[M,K]·[K,N]` is exactly `2·M·N·K`; attention decomposes into its two batched matmuls (QKᵀ and scores·V) summed. Divide by the tensor-core peak (in FLOP/s).
  - *Size-aware peak:* a kernel only reaches the headline tensor-core peak when it is large enough to amortize launch and fill the systolic array. If **all** of M, N, K are ≥ 512, use the full tensor peak; otherwise use a conservative (launch-dominated) peak. This is a two-tier knee on operator *size*, not just arithmetic intensity. Note the conservative peak **defaults to the full tensor peak** unless a separate `peak_tflops_mm_conservative` is supplied — and the GH200 YAML does not supply one — so for the worked-example hardware the size-knee collapses to a single tier: every GEMM, large or small, is priced at 1979 TFLOP/s. The mechanism exists but is dormant on GH200.
- **Memory demand (`mem_ns`).** Sum the bytes of every input read plus every output written (true storage `nbytes`, rounded to the allocator's block), divided by HBM bandwidth in bytes/s. Bytes are dtype-aware (bf16 = 2 B/element). Two corrections keep the count honest: a `beta=0` `baddbmm` does not read its accumulator, and `copy_` overwrites (not reads) its destination.
- **Vector / transcendental demands (`fma_ns`, `sfu_ns`).** An instruction-mix table returns, per op, how many non-MMA FP32 ops and how many transcendental ops it performs. Softmax, for instance, is tagged ~one `exp` per element (transcendental) plus a comparable count of max/sub/div (FP32 vector). These counts are divided by the math peak and the SFU peak (defaulting to ¼ of the math peak, matching NVIDIA's special-function throughput). For a pure GEMM both are zero — all its work is MMA, already in `tensor_ns`.

Combine: `roofline_ns = max(tensor_ns, fma_ns, sfu_ns, mem_ns, launch_ns)`, converted ns → ms for the public estimate. The tracer and DES never know how this number was computed — the only estimation touchpoint is a single `estimate_runtime(...)` call.

### Why it works

Each per-resource term is a **true lower bound**: a kernel physically cannot push N FLOPs through a unit faster than its peak FLOP/s, nor move B bytes faster than peak bandwidth. The max across resources is therefore the tightest bound the kernel's structure alone implies — exactly what a well-fused, saturating kernel achieves.

The numerators are *exact*, not estimated: FLOP and byte counts are exact functions of shape and dtype, which fake-tensor tracing preserves perfectly. Only the achievable *fraction* of peak is modeled, and for large GEMMs that fraction is near 1 — which is why the bound is trustworthy where most FLOPs live.

The compute/memory crossover — the **ridge point**, peak FLOP/s divided by peak bandwidth — is implicit in the max: an op whose arithmetic intensity sits above the ridge is compute-bound, below it memory-bound. The crossover is a property of the device's two peaks, so it shifts automatically when you re-target other hardware.

Adding the FP32-vector and SFU demands repairs the classic single-ceiling failure: for memory-/transcendental-bound ops the tensor demand is ~0, so a single-ceiling model would predict near-zero time. The multi-demand max instead reports the real binding resource, so these ops are not silently free.

Hardware-agnosticism is structural: nothing in the formula references a specific device, only the four scalar peaks. Substituting another accelerator's peaks re-prices every operator with no code change.

### Worked example — step 3

The representative FFN gate+up GEMM is large and dense: its arithmetic intensity sits far above the ridge point, so the tensor-core term dominates the max and the operator is firmly **compute-bound**, priced at the full tensor-core peak.

The attention softmax is the opposite case, and shows why the multi-resource max matters. It does no matrix-multiply, so its tensor term is zero — but it streams the full materialized score matrix through HBM, so its memory term binds and the operator is **memory-bound**. A single tensor-core ceiling would have priced this softmax at essentially zero; the memory demand is what correctly accounts for it. Because the traced layer uses the explicit attention path, those scores really are written to and read back from HBM, so this memory-bound node genuinely exists in the DAG.

---

## 6. The learned per-device residual correction

### The idea

The roofline is an idealized lower bound: it assumes the operator runs at the device's published peak and that the only fixed cost is a launch. Real kernels never hit peak — they pay launch latency, imperfect occupancy and tiling, **wave quantization** (the M-mod-128 / N-mod-64 effects), library heuristics, and memory-access inefficiency. Rather than model each mechanism from first principles, an optional **calibrated tree** learns the single aggregate gap between the idealized roofline and what the target device actually achieves, as a function of the operator's observable structure.

Crucially it does not regress latency directly. It predicts a **multiplicative correction to the physical anchor**, so the physics carries the prediction and the model learns only the residual.

### The mechanism

A rule-based router maps each dispatched op to one of five families from the op identity plus the operator-type tag: `GEMM`, `ATTENTION`, `NORMALIZATION`, `ELEMENTWISE`, `REDUCTION`. No learning is involved in routing.

The analytical roofline is computed first, with the family's *measured* launch floor injected as `t_launch_ns`. The binding bound (the max over the pipeline demands, or the launch floor if larger) is the **anchor**. If the anchor is non-positive (view/creation), the estimate is 0. If no trained booster exists for that family, the residual is exactly 0 and the prediction collapses to the bare roofline. (In the shipped GH200 bundle only `gemm`, `elementwise`, and `reduction` are calibrated; attention and normalization fall back to pure roofline.)

If a booster exists, a feature row is built by the **same** `featurize()` used at training time: universal columns derived from the roofline itself (log of each pipeline demand, log anchor, arithmetic intensity, total bytes, dtype, bits/element) plus family-specific columns (for GEMM: M/N/K, the batched dim, their logs, and the modulo-alignment flags mod 8/16/64/128 that capture tensor-core tiling quantization). Categoricals are converted to the integer codes pinned in the manifest, the row is ordered by the manifest's feature-column list, and one regularized tree per family predicts a scalar **log-residual**.

The final op time is:

```
time = max(anchor_ns · exp(residual), anchor_ns)   →  ms
```

The `max(...)` is an out-of-distribution **rail**: a learned correction can only ever pull a prediction *upward* (slower than ideal), never below the physical roofline floor. The whole estimate is wrapped so any failure (missing column, bad model) silently falls back to the bare roofline — a calibration model can never crash a simulation.

### Why it works

The multiplicative/log-residual framing matches the physics of inefficiency: slowdowns compose multiplicatively (occupancy, tiling, and library overheads stack as factors, not additive terms), and latency spans many orders of magnitude across op sizes. Regressing `log(time)` directly would force the tree to relearn the FLOP-count scaling the roofline already computes exactly. Targeting `y = log(measured) − log(anchor)` gives the tree a near-zero-mean, low-variance signal, spending all its capacity on the genuinely hard-to-model gap.

The anchor is a true lower bound, so clamping to it is physically sound — the worst case for an unseen shape is *reverting to the roofline* instead of producing a confidently-wrong sub-peak number. This is what makes calibration safe to ship as a default.

Per-family routing gives each model a homogeneous feature space and failure mode: a GEMM's residual is dominated by tile/wave quantization (the mod-alignment features), an elementwise op's by launch overhead and coalescing, a reduction's by the reduced-axis size vs. fan-out. Mixing them would blur distinct mechanisms; splitting also lets families with no data fall straight back to roofline.

**Feature parity** is the precondition for the residual to mean what it was fit to mean: `featurize()` and the feature-column list are a single shared module imported by *both* the estimator and the offline calibrator, and a pinned schema version is asserted at load. If the inference row differed even slightly from the training row (column order, a missing modulo flag, a different dtype encoding), the correction would be applied to the wrong point in feature space and become noise. Sharing the code makes train/serve skew structurally impossible.

Empirically, on GH200 the GEMM family reaches ~6% median absolute percentage error over held-out ops, elementwise ~6%, reduction ~4.6%, with small mean-signed-log error (≈ +0.10 for GEMM) — exactly the regime where a thin correction on a physics anchor is the right tool.

### Worked example — step 4

For the representative GEMM, its roofline value becomes the anchor and the GEMM tree is handed that op's feature row — including the tile-alignment flags, which here are the favorable case (its dimensions are clean multiples of the tensor-core tile). The tree predicts a small *upward* correction, so the calibrated time lands modestly above the idealized roofline — capturing real-kernel inefficiency — while the clamp guarantees it can never fall below the physical floor. These lightly-corrected per-op times are what the DES schedules.

---

## 7. The flow-level network simulator

### The idea

Collective time has **no closed form** on real hardware. A ring all-reduce's many simultaneous point-to-point sends share physical links, and the time any one send takes depends on what *else* is on the wire at that instant. The standard "algorithmic bandwidth" formula (`bytes × 2(P−1)/P ÷ busBW`) silently assumes a contention-free, perfectly-balanced ring on homogeneous links; it cannot represent two collectives colliding, an incast saturating one NIC, or a ring that must cross a slow inter-node link while other hops run on fast NVLink.

SysSim instead **decomposes** every collective into its real directed point-to-point sends, turns each send into a **flow**, lays those flows over an explicit link graph (the topology), and runs a discrete-event simulator that, at every event, recomputes each flow's rate by **max-min fair water-filling** over the links it shares. The collective's wall-clock time is the **makespan** — the finish time of the last flow — which *emerges* from contention rather than being assumed away.

### The mechanism

1. **Decompose.** Each collective expands into a list of directed sends, each a tiny `Op(src, dst, size, deps, tag)`. A ring all-reduce over P ranks → `2(P−1)` rounds; in each round every rank sends one chunk (`total/P`) to its ring-neighbor. Reduce-scatter and all-gather are the two halves (`P−1` rounds each). A broadcast is a binary tree. Data causality is inlined as `deps` (a round's send out of rank `r` depends on the previous round's send that wrote *into* `r`). The explicit "serialize sends leaving rank r" dependency is deliberately **dropped** — per-rank uplink contention is left for the fair solver to discover, not hard-coded.

2. **Place.** Each `(src,dst)` resolves to an ordered list of physical `Link`s via the topology's route table: routed one differing coordinate-dimension at a time, concatenating that dimension's link segments. Intra-node hops → a single NVLink edge; inter-node hops → a send-link into the switch plus a recv-link out.

3. **Index the flow–link incidence.** Collect every unique `Link`, convert each `capacity_GBps` to bytes/sec, and record per `Op` the integer indices of the links on its path plus the summed path latency (which delays when the flow joins the active set).

4. **Event loop.** State is three sets — `pending` (deps unmet), `ready` (deps met, waiting out path latency), `active` (draining bytes). At each event: promote `ready → active` when due; build a boolean flow×link incidence matrix for the active set; call the max-min fair solver for every active flow's instantaneous rate; compute `dt` = the soonest of {an active flow draining to zero, a ready flow's latency expiring}; advance time by `dt`, subtract `rate·dt` bytes, retire flows that hit zero (recording finish time), and promote newly dep-satisfied pending ops. **Rates are re-solved from scratch on every event** because the active flow set just changed.

5. **Water-fill** (the heart). The solver allocates bandwidth so no flow can be sped up without slowing an equal-or-slower flow. It iterates: (1) per link, tentative fair share = `remaining_capacity / (#unassigned flows on it)`; (2) the bottleneck link is the argmin of those shares; (3) freeze every unassigned flow crossing it at that rate; (4) subtract those flows' committed demand from the remaining capacity of *every* link on their paths; repeat until all flows are assigned. Each iteration removes ≥1 link, so it terminates in ≤ L iterations (typically ≤5 for DL workloads).

6. **Makespan.** After the loop drains, the collective time = `max(finish_time)`, in seconds → ms. The runner calls this for each TP/SP collective captured during tracing and for the injected DP gradient sync, then feeds those millisecond costs back as edge weights into the operator-graph DES.

Per-link capacity is **derived from datasheet aggregates by node degree**, so one advertised number stays physically consistent across topologies: `per-link = per-node aggregate ÷ degree` (fully-connected → `size−1` peer links, ring → 2 neighbors, switch → 1 uplink). Links are modeled **full-duplex** (each direction an independent directed `Link`), matching real NICs/NVLink, so a ring's a→b and b→a traffic do not falsely contend. Path latency is a one-time delay before a flow joins the active set, capturing the small-message latency floor without distorting steady-state sharing.

### Why it works

Max-min fairness is the correct contention model for a lossless, backpressured fabric. NVLink/NVSwitch and credit-flow-controlled Slingshot converge, at steady state, to the rate vector where each flow is bottlenecked by some link and shares it equally with the other flows pinned there — exactly the max-min fair allocation. Solving the fixed point per event reproduces what the real congestion-controlled fabric settles into, with no packet simulation. The water-filling iteration is provably that vector (standard progressive-filling proof): the globally smallest per-link share identifies the true bottleneck, freezing those flows is optimal, and recursing on residual capacity correctly re-allocates the rest.

Per-rank contention needs no special case: each send is an independent flow, and sharing each physical link among exactly the flows on it means a rank firing several concurrent sends through one uplink automatically has that uplink split among them — derived from the link graph, so it stays correct under any topology. When a DP ring spans nodes, its chunks must traverse the 25 GB/s Cassini segment, whose share is far smaller than the NVLink segments on the same ring; water-filling pins the whole ring's progress to that slow link, so the makespan reflects the genuine inter-node bottleneck rather than an NVLink-optimistic average.

The same machinery reproduces measured GH200 behavior — the calibrated network profile records an intra-node all-reduce busBW around 330 GB/s and a ~60 µs intra-node latency floor — and the GH200 Megatron calibration landed 14–16 of 16 collective-inclusive step times within 10%.

### Worked example — step 5

Two collectives matter for the block, and their placement on the topology is the whole point. With Megatron's tensor-parallel-fastest rank order, the **DP=4 group spans both nodes**, so the data-parallel ring is forced across the slow inter-node Slingshot fabric — exactly the case a contention-free formula would misprice. The **TP=2 all-reduces**, by contrast, stay inside a node on fast NVLink.

The flow simulator decomposes each collective into its ring sends, lays them over the link graph, and water-fills. Every round of the DP collective is pinned to the slow inter-node link, so that link sets the makespan — the simulator surfaces the genuine inter-node bottleneck instead of an NVLink-optimistic average. The TP all-reduce, riding NVLink, is comparatively cheap. The network subsystem's job ends at producing each collective's makespan; the DES (§9) decides how much of it actually lands on the critical path.

Under the distributed optimizer the injected DP sync is a single bf16 all-gather of the TP-sharded parameters; plain DDP would instead inject a larger gradient all-reduce. In a real run gradients are bucketed across all blocks and overlapped with backward compute — the per-block view here just makes the mechanism concrete. The example uses the datasheet topology constants from the hardware YAML, which the training path actually reads, not the measured-and-inverted `data/gh200/network.json` that §11 builds.

---

## 8. How parallelism reshapes and composes the computation

### The idea

Distributed training never runs "the model" on any one device — each rank runs a sharded, replicated, or sliced fragment plus the communication that glues fragments together. SysSim's bet is that you can reproduce one rank's *exact* fragment cheaply by initializing a **fake process group at the real world size**, letting the real Megatron parallelism machinery build the model exactly as it would in a real job, and tracing it. Because the framework genuinely believes it is rank `r` of a W-rank world, every shard split, every collective insertion, and every sequence/layer partition happens for real — **the shapes that come out are already the per-rank shapes.**

### The mechanism — five orthogonal dimensions

`world_size = tp·dp·cp·pp`; `num_nodes = ceil(world_size / gpus_per_node)`; if `>1`, an inter-node topology is mandatory. Megatron's canonical rank order is **tensor-parallel-fastest**, then DP, then CP, then PP outermost — load-bearing for routing collectives onto the right links.

For each pipeline stage a process is started; it installs a fake distributed backend so `get_world_size()/get_rank()/new_group()` return the requested values with zero inter-process negotiation. Then real `initialize_model_parallel(tp, pp, vpp, cp)` runs and a real `GPTModel` is built on fake tensors. What each dimension does:

- **TP** shards the attention and FFN weight matrices and **inserts** one all-reduce after attention-out and one after the FFN down-projection (forward). These land in the trace naturally with their real byte counts. The traced GEMMs already have the sharded N or K dimension — **captured for free**.
- **SP** (sequence parallel) additionally splits the sequence dimension across the TP group, swaps the TP all-reduces for reduce-scatter + all-gather pairs, and divides LayerNorm/dropout activation memory by `tp` — also captured.
- **CP** (context parallel) shards the sequence dimension across a separate group so each rank holds `s/cp` tokens, shrinking attention activations.
- **DP** replicates the whole model. A single traced rank looks identical regardless of `dp`, so **DP changes nothing in the trace**; its only effect is a gradient sync added afterward.
- **PP** splits the layer stack across stages — each stage is a *different* model fragment, so each is traced in its own process (MPMD).

**Real collectives are patched to no-ops during tracing** (recorded as `COLLECTIVE` nodes with `estimated_time_ms = 0`, name + byte count + group_ranks); timing is filled in later by the network simulator.

**Inject what a single rank cannot see — DP sync.** After tracing, for `pp==1`, the runner anchors a DP gradient collective on a separate stream to the **last traced op** (the tail of the backward, taken by insertion order — `next(reversed(graph.operators))` — not a phase-tag search). The DP group's global ranks are `[0, tp, 2·tp, …]` — **strided by `tp`** because of the tp-fastest layout — so when DP spans nodes the collective is routed over the inter-node fabric, not mistakenly over NVLink. Plain DDP injects an all-reduce of the bf16 gradient; the distributed optimizer (ZeRO-1) injects instead a bf16 all-gather of only the TP-sharded parameters (replicated embeddings/norms are not DP-gathered).

**Inject the optimizer step analytically.** Real Megatron fuses mixed-precision Adam into one `multi_tensor_apply` kernel that is purely bandwidth-bound and has no FakeTensor implementation (tracing it would explode into ~100× too many per-parameter ops touching ~100× the real traffic). So it is added as a single `MATH` node with `time = bytes_moved / peak_HBM`. The resident fp32 master + Adam m + v is 12 B/param (three fp32 copies); the fused update *reads and writes* all three, so that is **24 B/param** of HBM traffic (`12 · param_bytes` with `param_bytes = 2 B` for bf16), plus a 4 B/param fp32 grad read and a 2 B/param bf16 param write. ZeRO-1 divides the whole `bytes_moved` by `dp`.

**Compose pipeline stages with timed P2P edges.** With `pp>1`, each stage emits P2P send/recv nodes carrying their peer global rank and a tag. Composition renames every node per-stage, re-namespaces stream IDs as `pp_rank*1000 + local`, pairs each send with the matching recv on the peer stage, and adds a cross-stage predecessor edge (recv waits on send). Each P2P edge is timed by simulating a single point-to-point `Op` of that byte size over the topology. The schedule is **1F1B** only.

### Why it works

**Per-rank shapes are exactly right by construction:** the trace runs the real framework's real sharding code under a real world size, so a column-parallel GEMM is genuinely `[.., N/tp]` and a vocab-parallel embedding is genuinely sliced. No manual shape arithmetic to get wrong.

**DP correctly changes nothing in the trace:** data parallelism replicates the model bit-for-bit, so every rank's operator graph is identical. Tracing once and reusing for any `dp` is *exact*; the only `dp`-dependent effect is the gradient sync, added analytically.

**Injecting DP sync and the optimizer is more faithful than tracing them.** The gradient sync is genuinely invisible to a single traced rank (the peers don't exist), so it *must* be added. The fused Adam kernel is bandwidth-bound by physics (it streams master/m/v/grad/param through HBM), so `bytes_moved/bandwidth` is the correct model; tracing its FakeTensor decomposition would mis-count it by ~100×.

**DP rank striding routes the collective over the right links.** Because replicas of a given shard sit at global ranks `[0, tp, 2tp, …]`, using this stride (not `[0,1,2,…]`) ensures that when DP spans nodes the modeled collective traverses the inter-node fabric — matching where the real traffic goes.

**1F1B memory scaling is derived, not guessed** (see §10), and the pipeline bubble emerges from the dependency structure in the DES rather than a hand-coded bubble formula.

### Worked example — step 6

Under the fake process group at `world_size = 8` with `tp = 2`, the framework builds the block and the tracer captures it already sharded — TP halves each projection GEMM's model-parallel dimension, so it does roughly half the work it would unsharded. The TP all-reduces are captured in the trace; the DP=4 all-gather is injected on the comm stream after the last traced op, routed via the tp-strided rank list across the inter-node fabric the DP group spans; and the fused-Adam update is injected as one bandwidth-bound `MATH` node, its traffic sharded across the DP group under ZeRO-1. The fully-assembled graph hands off to the DES.

---

## 9. The discrete-event timing replay, and MFU/HFU

### The idea

This is where a single step time finally emerges. Every node already carries a duration in ms (from the estimator), a phase tag, and a `stream_id`. The runtime stage does **not** recompute any costs — its job is purely **temporal scheduling**: replay the annotated DAG respecting two constraints simultaneously — within one stream, ops run strictly in issue order (a CUDA stream is FIFO); across streams, an op cannot start until every cross-stream dependency it points at has finished. The instant the last node finishes is the step time. **Overlap and pipeline bubbles are emergent, not modeled.**

### The mechanism

Group every op by `stream_id` into ordered queues; only the head of each queue is a candidate, and the head advances only when that op completes. Maintain per-stream head pointers, an `active` map (op → finish time), a `completed` map, and a clock `t_now = 0`.

Each iteration:
- Scan the head op of every stream. An op is **eligible** when (a) all named predecessors are in `completed`, and (b) if it is a `BARRIER`, every other stream has drained. Its **start** = `max(t_now, latest predecessor finish)`; its **finish** = `start + estimated_time_ms`. Put it in `active`. The `start = max(deps)` rule is what makes a downstream op wait for the slowest input regardless of which stream produced it.
- Advance the clock to the **smallest finish time** in `active` (event-driven jump, no fixed step). Move every op finishing at that instant into `completed`, advance the head of the stream whose head just finished. Repeat until no work remains. If nothing is eligible but work remains, that is a stall (a malformed graph) and it **raises** — turning a modeling bug into a loud failure rather than a wrong number.

The **step time** is the maximum finish over all completed ops — the critical path through the multi-stream DAG.

**Exposed (non-overlapped) collective time is measured, not derived:** clone the graph with every `COLLECTIVE` node (and dependencies on them) removed, re-run the identical simulator, and take `full_step − comm_free_step` (floored at 0). Whatever the scheduler could hide behind compute contributes 0; only the part on the critical path survives.

**Phases** are split by summing `estimated_time_ms` grouped by each op's phase tag (forward/backward/optimizer) — a static sum, deliberately *not* the overlap-aware makespan, so the three phase numbers describe *where the work is* while step time describes the *realized critical path*. Conflating the two would double-count the overlap the DES exists to expose.

**Efficiency** is computed last from a value-free analytic FLOP budget. The standard transformer count per layer (attention projections `2·b·s·h·h·4`, scores+context `2 × (2·b·heads·s·s·(h/heads))`, FFN `2·b·s·h·f` × 2 or 3 for SwiGLU) plus the vocab projection gives forward FLOPs; backward is 2× forward; multiply by microbatches/step = `global_batch / (micro_batch · DP)`. Then:

```
achieved_TFLOP/s = model_FLOPs / step_seconds / world_size
MFU = achieved_TFLOP/s / peak_tensor_TFLOP/s
HFU = MFU with recompute surcharge added to the numerator
      (+1.0× forward for full recompute, +0.3× for selective)
```

### Why it works

The makespan of a DAG where each node has a fixed duration and each edge is a hard precedence is exactly the longest weighted path; the event-driven loop computes that without enumerating paths — it is the textbook list-scheduling of an in-order multi-machine system. The stream-FIFO constraint plus the predecessor-completion constraint together are necessary and sufficient to reproduce CUDA stream + cross-stream-event semantics.

Overlap is correct by construction: a collective on stream 1 with no consumer until N compute ops later runs concurrently and only delays things if its finish exceeds the compute it overlaps. The strip-and-resimulate definition of exposed comm is operationally honest — *"how much would the step shrink if communication were free?"* is precisely `full − comm-free` makespan, and it composes correctly when many collectives partially hide behind different compute regions, which reading per-op idle gaps does not.

MFU uses the model FLOP budget (useful math only), which is dtype- and value-independent — identical on fake or real tensors. HFU diverges from MFU only via recompute: re-executing the forward during backward makes the hardware do extra FLOPs that do no additional learning, so `HFU ≥ MFU` exactly when recompute is on.

Costs are frozen on the nodes before this stage runs, keeping the scheduler **estimator-agnostic**: swapping the roofline for a learned model changes only node durations, never the timing logic — the structure/cost/timing separation is preserved exactly here.

### Worked example — step 7

The block's nodes enter the DES with their durations already attached. Each TP all-reduce sits on the comm stream with a `STREAM_SYNC` where the program waits; because the block's forward and backward compute dominate, strip-and-resimulate shows the TP collectives contribute essentially nothing to *exposed* communication — they hide behind compute even though their own makespan is non-trivial. The end-of-step DP all-gather is hung off the last traced op; whatever portion of it finishes after the last backward op is what lands exposed on the critical path. The step time is the makespan of the whole multi-stream DAG — forward, then the heavier backward, with the optimizer step and the un-hidden tail of the gradient sync on top — and MFU/HFU follow from the analytic FLOP budget.

---

## 10. The measured memory model and 1F1B scaling

### The idea

Peak memory is **measured, not computed from a closed-form formula**. SysSim runs exactly **one microbatch** of the real Megatron training stack (DDP + distributed Adam) on the fake-tensor model and watches a dispatch-mode tracker attribute every tensor **storage** that gets allocated into one of seven categories (parameter, buffer, gradient, optimizer-state, activation, backward-temp, other). A running per-device byte total is maintained and its **high-water mark** is the measured single-microbatch footprint. A tiny downstream layer scales that footprint to the true steady-state peak using the pipeline schedule.

### The mechanism

1. **Build the real stack on a fake model.** Swap params/buffers to fake CUDA tensors, then wrap in Megatron's real DDP (which allocates the fp32 gradient-reduce buffer) and the real optimizer (which allocates fp32 master weights + Adam m + v). Identical code path to a real run.
2. **Enter the tracker as a `TorchDispatchMode`.** For each output tensor of every op, walk down to the underlying `UntypedStorage` (deduplicating views, which alias one storage); if it is new, record `(numel × element_size, rounded up to the 512-byte CUDA granularity)` and add to the per-device total. A weak-ref finalizer subtracts the bytes the instant the storage is GC'd, so frees are tracked. `UntypedStorage.resize_` is monkey-patched (the dispatcher never sees resizes).
3. **Categorize by *when and how* a storage was created.** Params/buffers tagged up front; a per-parameter grad hook re-tags a storage as `Gradient` the moment autograd populates `.grad`; new storages during `optimizer.step()` are `Optstate`; during backward (outside recompute) are `Temp`; otherwise `Activation`. A storage can be re-tagged in place, moving bytes between buckets.
4. **Step the optimizer *before* the backward, deliberately.** This forces master/m/v and the grad buffer to be co-resident at the backward high-water mark — the binding peak in the no-recompute case — reproducing *steady state*, not the cold first step where m/v don't yet exist.
5. **Run forward + backward for one microbatch.** Collectives are patched to no-ops but their temporary buffers still allocate and are tracked. The total rises through forward (autograd saves activations), peaks during backward (activations + gradients + temps + persistent state), and the high-water snapshot per device is retained.
6. **Bucket the peak** into `persistent_by_type` (Parameter/Buffer/Gradient/Optstate/Other — resident once), `act_bytes_per_mb` (one microbatch's saved activations), and `temp_bytes` (one backward's transient working set).
7. **Scale with the 1F1B schedule.** Per pipeline stage:

```
in_flight = (pp == 1) ? 1 : min(pp − stage_rank, num_microbatches)
peak      = sum(persistent) + in_flight · act_bytes_per_mb + temp_bytes
num_microbatches = global_batch / (micro_batch · DP)
```

The binding stage is the one with the largest peak (the earliest stage).
8. **Detect OOM** by comparing the binding peak in GB to `gpu_memory_GB`.

### Why it works

Cost-from-structure holds exactly for memory: a storage's size is `numel × dtype_size` — no data needed — and the tracker applies the same 512-byte ceiling the real allocator uses. *Whether* two tensors are simultaneously live is determined by the autograd graph and the optimizer/DDP code, all of which run for real here; only the numeric kernels are skipped. So the live-set and its high-water mark are **reproduced**, not approximated.

Measuring beats a formula because the live-set is subtle: a hand formula must enumerate every framework-internal buffer (fp32 grad-reduce buffer, master weights, Adam m/v, fused scratch, attention workspace, saved norms, recompute buffers) and track exactly when each frees — all version- and flag-dependent. Letting the genuine stack allocate them and intercepting at the dispatcher is simpler and self-correcting: if the framework allocates it, it is counted; when it frees it, the finalizer removes it.

**The 1F1B scaling is a correctness derivation.** In a depth-`pp` 1F1B pipeline, stage `s` must keep a microbatch's forward activations alive from its forward until that same microbatch's backward returns to `s`. Between those events the pipeline injects forwards for `(pp − s − 1)` downstream-warming microbatches plus the current one, so stage `s` holds `min(pp − stage_rank, num_microbatches)` activation sets at once. Earlier stages (small `s`) sit further from the backward turnaround and hold **more** — which is why the earliest stage binds the peak.

**DP and gradient accumulation do not multiply activation memory** — and the model reflects that. Each DP rank processes its own microbatch on its own GPU, so per-GPU activations are one microbatch regardless of `dp`; DP only adds a transient gradient-sync buffer (counted as `Temp`). Gradient accumulation runs microbatches sequentially, freeing each before the next. **Only pipeline depth** forces multiple microbatches in flight — the `in_flight` formula depends on `pp` and `stage_rank`, never on `dp` or accumulation count.

**ZeRO-1 sharding is captured implicitly, for free:** the real distributed optimizer partitions master/m/v across the *fake* DP group during the trace, so per-rank `Optstate` storages are already `1/dp`-sized — no special-case code. And the high-water mark is robust even where category attribution blurs (e.g. activation vs temp during recompute), because peak is a *total-bytes* maximum re-evaluated after every op — only the split between two buckets can be approximate, never the OOM-deciding magnitude.

### Worked example — step 8

Per GPU, for one block: TP=2 halves the resident parameters, which in turn halves the gradient buffer and the optimizer state; ZeRO-1 then shards that optimizer state again across the DP=4 group, making it the lightest of the persistent buckets. On top of that persistent state sits one microbatch's saved activations (dominated by the wide SwiGLU intermediate) plus the backward temporaries.

Because `pp = 1`, the activation term is counted **once** — DP=4 does not multiply it, since each data-parallel rank processes its own microbatch. Had this instead been `pp = 4` with the block on the earliest stage, the in-flight count would multiply the activation term several-fold while the persistent state stayed fixed — the concrete illustration of why **pipeline depth, not data parallelism**, drives the activation peak. Summed across all blocks plus the embedding and lm-head and compared against the GPU's capacity, this is what decides OOM.

---

## 11. The offline profiling and calibration pipeline

### The idea

This is the optional, offline accuracy booster that turns the pure-physics roofline into a per-device cost model (§6 is its inference half; this is how the bundle is built). The roofline gives a ceiling; real kernels run at some fraction of it that depends on tile shapes, dtype, alignment, fusion, and launch/memory-bound regime in ways no closed form captures. The calibrator **measures** that fraction once, on real hardware, for a representative span of operator shapes that actual transformer layers dispatch, and learns the per-family residual. A separate, smaller piece measures real collective bus-bandwidth and latency and derives the topology block the flow simulator can consume.

### The mechanism

1. **Measure (on a real GPU).** Build one actual Megatron GPT layer on CUDA (via `get_gpt_layer_local_spec()`, the explicit-attention path — so the score `baddbmm`, `_softmax`, `masked_fill`, and context `bmm` are all dispatched and measured as themselves) and run a real forward+backward. **Pass 1** (a dispatch mode) records every dispatched aten op's exact signature — op name, per-arg shapes/dtypes (including fp32 upcasts and bool masks), kwargs, output shape/dtype — keyed by `(op-name, tensor-arg-shapes)`. **Pass 2** (`torch.profiler` with `record_shapes`) records each op's real self-GPU-time keyed by the same key. The two are joined so each row is the **exact op** the simulator will later reconstruct, paired with its true *in-context* runtime.
2. **Profile TP sharding directly.** The profiler builds the per-rank layer — dividing heads/kv-groups/ffn/vocab by the TP degree (preserving `head_dim`) — so the per-rank GEMM/attention shapes the simulator will see under TP are measured on real silicon. (Collective time comes from the network model, not this profiler — a single rank can't run TP collectives.)
3. **Sweep the shape space, not models.** There is no notion of a named model in the data. Each architectural field (hidden, ffn, heads, query-groups, head_dim, vocab) is swept over its own list while the others stay at a representative base — 1-D coverage of every axis, because the residual tree *interpolates* op size, making the full cartesian product (~80k configs) unnecessary. `(S, b)` points come from a token range; any whose attention scores `[b,h,S,S]` or lm-head logits `[tokens,vocab]` would exceed a memory budget are dropped; coverage above the cap is reached by interpolation.
4. **Measurement hygiene.** Warmup pins the GPU boost clock; each `(S, b)` group's reps run back-to-back *without* an inner sync so they pipeline at the sustained clock (a per-step sync idles the GPU, drops the clock, over-measures); sync once per group for clean self-time attribution. Profile single-worker (concurrent workers contend for the node power/thermal envelope and inflate kernel time ~15%). Each row reports `per_instance_ns = self_device_time / count`.
5. **Reconstruct + route (on CPU).** Deserialize each signature into **meta tensors** (shape/dtype only — values never mattered) and rebuild the precise `(func, args, kwargs, out)`. Route each op to a family by the **same** rule-based router the estimator uses. View/create ops are ignored.
6. **Anchor + featurize.** Compute the roofline anchor (binding pipeline demand + measured launch floor `t_launch`) and build the feature row with the **same** `featurize()` the estimator calls.
7. **Fit one residual tree per family.** Target `y = log(measured_ns) − log(anchor_ns)`; a LightGBM regressor (MAE objective) per family fits `y` from the features. `t_launch` = per-family minimum observed latency (the launch floor). **The split is held out by unique op signature** (op_subtype + log-anchor bucket), seeded — so no shape in train leaks into validation, making the reported APE a genuine generalization estimate.
8. **Pin the feature schema in the bundle.** Each family's tree is saved alongside a manifest whose load-bearing job is *feature parity*: it pins the exact feature-column order, the categorical integer codes (extended to every op_subtype the real layer dispatched — e.g. `baddbmm`, `_softmax`, `masked_fill`, `native_dropout`, the `*_backward` ops), and the per-family `t_launch`, behind a schema version asserted at load. This is what guarantees the inference-time feature row is the *same point in feature space* the tree was fit on, not an inventory for its own sake.
9. **Network profiling (separate).** A `torch.distributed` program runs a ring all-reduce over an intra-node group (and, multi-node, a strided one-GPU-per-node inter-node group) across a message-size sweep. The largest message fixes saturated busBW (`busBW = 2(n−1)/n × bytes / time`); a 1 KB message isolates the pure latency floor. The derivation inverts these into a YAML-shaped topology block: per-dimension bandwidth = `busBW × degree`, per-hop latency = `floor / (2(n−1))` hops — turning measured fabric reality into the constants the flow simulator divides over flows. On GH200 this produced `data/gh200/network.json` (measured intra busBW ≈ 331 GB/s, inter ≈ 23 GB/s, 60.5 µs intra latency floor → derived per-dimension bandwidth ≈ 994 / 23 GB/s).

### Why it works

Learning a residual against a physically-correct anchor is far easier and safer than regressing absolute time: the anchor already spans the orders of magnitude (a 1 KB op vs a 400-GFLOP GEMM differ by ~10⁵×), so the tree fits only a bounded efficiency factor near 1 — which is why a shallow tree suffices and median APE lands at 5–6%.

**Feature parity is enforced mechanically, not by discipline:** both the offline fit and inference call the same `route()`, `roofline()` (with the same per-family `t_launch`), and `featurize()` with the same feature-column order from the manifest. The train-time and inference-time feature rows for a given op are bit-identical, so there is zero train/serve skew — the classic failure mode of learned cost models is eliminated by construction.

**Layer-level in-context measurement** is the right ground truth because a kernel's real runtime depends on state an isolated microbenchmark destroys: the actual residency of inputs in cache/HBM, framework fusion, and the sustained boost clock. The memory-bound attention ops (the explicit-path `_softmax`, `masked_fill`, the norms, dropout) are exactly where isolated benchmarks mispredict — and they are measured here in their true layer context. **Profiling the per-rank sharded shapes directly** means the sharded GEMM is ground-truth, not extrapolated.

**Held-out-by-shape splitting** defeats duplicate-shape leakage (the same `(op,shape)` recurs across configs; a random split would report a memorization score). The **OOD rail** (`max` with roofline) makes the booster monotone-safe: an over-extrapolated tree reverts to pure physics, so a calibrated run is never *less* trustworthy than the uncalibrated roofline. **Meta-tensor reconstruction is valid** precisely because of the founding bet — cost is a function of structure, not values — confirming the bet end-to-end on the calibration side. The anchor is corrected for known kernel semantics (`baddbmm` `beta=0` doesn't charge the unread accumulator; `copy_` charges src-read + self-write) so the residual learns *efficiency*, not accounting errors. The network derivation is grounded in collective physics: the standard busBW metric is the bandwidth a ring all-reduce actually achieves, and inverting the measured floor by the `2(n−1)` hop count recovers a per-hop latency the flow simulator can re-compose for any collective size.

### Worked example — step 9

For the representative GEMM, the offline calibrator and the live estimator compute the *identical* roofline anchor and the *identical* feature row — that train/inference parity is the entire point — so the committed GEMM tree applies the same small upward correction in both places. Every per-op compute time the DES replayed for this walkthrough is that kind of estimate: a roofline anchor nudged by the learned residual from this device's trees.

The fabric constants the flow simulator used came from the hardware YAML's topology block — datasheet-derived, the values the training path actually reads; the measured-and-inverted `data/gh200/network.json` is the calibrated alternative this section builds, not what drove the walkthrough. Threaded through all nine steps, the example stays a qualitative story — which operators bind on which resource, which collectives ride which links, and what sets the memory peak — illustrating the mechanism rather than reporting measurements.
