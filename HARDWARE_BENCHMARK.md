# Hardware Benchmark — Texas Hold'em AI on Apple M5 Max

Benchmark run date: 2026-04-23

## Project context

This repo is a multi-agent Texas Hold'em poker simulator with a pluggable bot interface. It includes a shared game engine (`core/engine.py`) plus ten bot implementations spanning different families: Monte Carlo rollout, CFR (counterfactual regret minimization), PPO-trained reinforcement learning, supervised ML (MLP classifier), GTO mixed strategy, ICM-aware tournament play, heuristic/rule-based, and an opponent-modeling bot. The workload is well-suited to hardware benchmarking because it exercises both sides of the machine: the simulation engine is pure CPU (hand evaluation, Monte Carlo rollouts, CFR traversal), while the RL and ML bots train on PyTorch.

A custom benchmark script (`benchmark_m5.py`) was written to stress-test this machine against the actual code paths that get used during training and tournament play, rather than a generic synthetic benchmark.

## Machine

| Spec | Value |
| --- | --- |
| Chip | Apple M5 Max |
| CPU | 18 cores (6 performance + 12 efficiency) |
| GPU | 40-core integrated (Metal, accessible via PyTorch MPS) |
| RAM | 64 GB unified memory |
| OS | macOS 26.4 |
| Python | 3.12.13 |
| PyTorch | 2.11.0 (MPS backend available) |

## Benchmark phases

The script runs three phases: CPU scaling via parallel poker tournaments, GPU matmul + training throughput comparing CPU vs MPS, and a 60-second sustained thermal test to check for throttling. Two configurations were run: a default "medium" run (~4 min) and a `--heavy` run (~15 min) with larger workloads and the thermal test added.

## Phase 2 — CPU scaling

Each worker runs a 4-bot tournament (Monte Carlo 200-sim, Monte Carlo 100-sim, GTO, ICM) to completion or a hand cap. The Monte Carlo bots dominate the CPU time, which is representative of real self-play training.

**Heavy run, 36 tournaments, 800-hand cap:**

| Workers | Wall (s) | Hands/sec | Speedup vs 1 |
| --- | --- | --- | --- |
| 1 | 448.39 | 31.1 | 1.00x |
| 6 | 97.70 | 142.7 | 4.59x |
| 12 | 58.92 | 236.6 | 7.61x |
| 18 | 43.08 | 323.6 | **10.41x** |

The scaling curve reflects the 6P + 12E core layout exactly. The jump from 1 to 6 workers delivers 4.59x — the six performance cores at roughly full efficiency. Going from 6 to 12 adds another 1.65x (the first six efficiency cores kicking in at ~40% of P-core speed), and 12 to 18 adds 1.37x more. A 10x speedup on 18 asymmetric cores is near the theoretical ceiling for this silicon.

## Phase 3 — GPU matmul (fp32)

Square matrix multiplication, 30 reps per size, compared between CPU and Metal Performance Shaders (MPS):

| Size | CPU (TFLOPS) | MPS (TFLOPS) | Speedup |
| --- | --- | --- | --- |
| 1024 | 1.79 | 5.00 | 2.79x |
| 2048 | 1.84 | 14.73 | 8.01x |
| 4096 | 1.72 | **15.02** | 8.73x |
| 8192 | 1.71 | 14.39 | 8.44x |

Peak GPU throughput is around 15 TFLOPS fp32 at 2048–4096, which is what this class of Apple GPU is designed to deliver. At 1024 the speedup drops to 2.79x because kernel-launch overhead starts to dominate relative to the actual compute. At 8192 the number dips slightly, likely memory-bandwidth bound.

## Phase 3 — Model training

### PokerMLP (256 hidden, ~72K params, batch 1024)

| Device | Steps/sec | Samples/sec |
| --- | --- | --- |
| CPU | 1138.5 | 1,165,797 |
| MPS | 1586.0 | 1,624,026 |

Only a 1.39x win for MPS. The model is too small to amortize GPU kernel-launch overhead.

### BigModel (4 × 2048 hidden, ~17M params, batch 4096)

| Device | Steps/sec | Samples/sec |
| --- | --- | --- |
| CPU | 5.4 | 22,187 |
| MPS | **41.5** | **169,871** |

**7.66x MPS speedup** once the model is large enough to feed the GPU properly. This is the practical takeaway for deep RL: a larger policy/value network actually trains faster in wall-clock time on MPS than a small one on CPU, despite doing dramatically more work per step.

## Phase 4 — Sustained thermal test

4096×4096 fp32 matmul run continuously for 60 seconds per device. Throughput is measured over the first 10 seconds (cold) vs the last 10 seconds (hot) to detect throttling.

| Device | First 10s (TFLOPS) | Last 10s (TFLOPS) | Sustained |
| --- | --- | --- | --- |
| CPU | 1.72 | 1.72 | **100.0%** |
| MPS | 13.22 | 11.36 | 85.9% |

The CPU is not thermally limited at all — it holds full throughput indefinitely. The GPU settles at about 86% of its cold-start performance after 50 seconds of sustained load, a ~14% throttle. This is mild and expected for a laptop; it means overnight training runs will be slightly slower than a burst benchmark but won't collapse.

## Local LLM inference (bonus)

Separate from the poker benchmark, the same machine was tested running the Qwen2.5 family locally through Ollama, prompting each with `"Write a haiku about FPV drones"`. All models are Q4_K_M quantized. This tests the practical ceiling of on-device LLM inference.

| Model | Eval rate (tok/s) | Notes |
| --- | --- | --- |
| qwen2.5:7b | **111.9** | Fits entirely in GPU memory, sub-second response |
| qwen2.5:14b | 54.6 | Comfortable for interactive use |
| qwen2.5:32b | 27.1 | Usable for longer queries |
| qwen2.5:72b | 8.8 | 61 GB model, near the 64 GB RAM limit; 87% GPU / 13% CPU split |

The 72B result is interesting: because the model size (61 GB) pushes against the 64 GB unified memory ceiling, Ollama offloads about 13% of the layers to the CPU, which becomes the bottleneck. Throughput drops from roughly 27 tok/s at 32B to 8.8 tok/s at 72B — not quite linear, which confirms the CPU fallback is the limiter. Running models up to ~30B is comfortable on this machine; 72B is possible but noticeably slower.

## Engineering takeaways for this project

Three things matter for how this codebase should be trained and run:

**Use 18 workers for self-play and tournament training.** The CPU scales nearly linearly and there is zero thermal penalty for sustained load, so `multiprocessing.Pool(18)` is the right default for any simulation-heavy workload. The current training scripts are single-threaded and leave ~90% of the machine idle.

**Scale the RL model up before switching to MPS.** At the current `HIDDEN_SIZE = 512` in `train_multi_deep_rl_bot.py`, moving training to MPS gives roughly 1.4x. At `HIDDEN_SIZE = 2048` with wider batches, the speedup jumps to ~8x. The GPU is strictly wasted on small models.

**Expect a ~14% steady-state GPU dip for long training sessions.** Planning a training run that needs X hours on MPS? Multiply by ~1.16 to get the realistic wall-clock estimate for an unplugged laptop.

## How to reproduce

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python benchmark_m5.py          # medium run, ~4 min
python benchmark_m5.py --heavy  # full run with thermal test, ~15 min
```

Results land in `output/m5_benchmark_results.txt`.

## Talking points for interviews

This benchmark itself isn't the headline; the poker AI codebase is. But it's a useful supporting artifact because it demonstrates three things an interviewer actually cares about beyond "I can write Python":

The first is that you understand the *shape* of your workload. You know whether your code is CPU-bound or GPU-bound, and you can back that up with numbers. When you say "the RL training loop is simulation-bound, not gradient-bound," you have data proving it.

The second is that you can reason about hardware. Apple Silicon's P/E core split, unified memory, MPS kernel-launch overhead, thermal throttling — these are all things you now have first-hand numbers on rather than vague impressions. Interviewers in ML infra roles will ask "how do you know your training is actually using the GPU?" and you can answer with specifics.

The third is engineering judgment. You identified that scaling the model up would make the GPU useful, that 18-worker parallelism is the free performance win, and that the machine can sustain overnight training. These are the exact kinds of decisions that separate engineers who ship training runs from ones who wait on them.

A good one-liner if the project comes up: *"I built a multi-agent Texas Hold'em simulator with bots spanning CFR, PPO, Monte Carlo, and supervised ML approaches, then wrote my own hardware benchmark against the actual code paths to figure out where to parallelize and when GPU acceleration was worth it."* That sentence does a lot of lifting — it tells them you built something real, you understand multiple RL/ML paradigms, and you think quantitatively about performance.
