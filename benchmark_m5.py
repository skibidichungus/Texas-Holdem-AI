"""
benchmark_m5.py
───────────────
Stress test for M5 Max (64GB RAM / 18-core CPU / 40-core GPU) using this
project's own bots and PokerMLP model.

Runs three phases:
  1. System info  — prints machine specs
  2. CPU phase    — parallel tournaments at 1 / 6 / 12 / 18 workers
  3. GPU phase    — PyTorch matmul + PokerMLP training (CPU vs MPS)

Results are saved to output/m5_benchmark_results.txt

Usage:
    python benchmark_m5.py                 # full run (~5-10 min)
    python benchmark_m5.py --quick         # tiny run for smoke-test
"""

import argparse
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

# Make sure we can import the project packages
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "m5_benchmark_results.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sh(cmd):
    """Run a shell command, return stdout as a string (empty on error)."""
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return ""


def log(fp, msg=""):
    """Print to console AND write to the results file."""
    print(msg)
    fp.write(msg + "\n")
    fp.flush()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — System info
# ──────────────────────────────────────────────────────────────────────────────

def phase_system_info(fp):
    log(fp, "=" * 70)
    log(fp, "PHASE 1 — SYSTEM INFO")
    log(fp, "=" * 70)

    log(fp, f"Date:            {datetime.now().isoformat(timespec='seconds')}")
    log(fp, f"Platform:        {platform.platform()}")
    log(fp, f"Python:          {sys.version.split()[0]}")

    # Mac-specific hardware info
    if sys.platform == "darwin":
        chip   = sh("sysctl -n machdep.cpu.brand_string")
        cores  = sh("sysctl -n hw.ncpu")
        pcores = sh("sysctl -n hw.perflevel0.physicalcpu")
        ecores = sh("sysctl -n hw.perflevel1.physicalcpu")
        mem    = sh("sysctl -n hw.memsize")

        log(fp, f"Chip:            {chip}")
        log(fp, f"Total cores:     {cores}  "
                f"(performance: {pcores}, efficiency: {ecores})")
        if mem:
            log(fp, f"RAM:             {int(mem) / 1024**3:.1f} GB")

    # PyTorch + accelerator detection
    try:
        import torch
        log(fp, f"PyTorch:         {torch.__version__}")
        log(fp, f"  MPS available: {torch.backends.mps.is_available()}")
        log(fp, f"  MPS built:     {torch.backends.mps.is_built()}")
        log(fp, f"  CUDA:          {torch.cuda.is_available()}")
    except ImportError:
        log(fp, "PyTorch:         NOT installed")

    log(fp, "")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — CPU stress via parallel tournaments
# ──────────────────────────────────────────────────────────────────────────────

def _run_one_tournament(args):
    """Worker: run a single tournament to completion. Returns (hands, seconds)."""
    seed, hands_cap = args
    # Import inside the worker so each process gets its own module state
    import random
    from core.engine import Table, Seat
    from bots import parse_players, escalate_blinds

    random.seed(seed)

    # Light-but-CPU-heavy lineup (Monte-Carlo bots dominate the work)
    spec = "mc200,mc100,gto,icm"
    players   = parse_players(spec)
    seats     = [Seat(player_id=pid, chips=500) for pid, _, _ in players]
    bots      = {pid: adapter for pid, _, adapter in players}

    table = Table()
    dealer_index = 0
    t0 = time.perf_counter()
    hands_played = 0

    while True:
        active = [s for s in seats if s.chips > 0]
        if len(active) <= 1 or hands_played >= hands_cap:
            break
        hands_played += 1
        sb, bb = escalate_blinds(hands_played, 1, 2, 50)
        table.play_hand(
            seats=active,
            small_blind=sb,
            big_blind=bb,
            dealer_index=dealer_index % len(active),
            bot_for={s.player_id: bots[s.player_id] for s in active},
            on_event=None,
        )
        dealer_index += 1

    return hands_played, time.perf_counter() - t0


def phase_cpu(fp, quick=False, heavy=False):
    log(fp, "=" * 70)
    log(fp, "PHASE 2 — CPU STRESS (parallel tournaments)")
    log(fp, "=" * 70)

    # How many parallel workers to try. Should saturate the 18-core M5 Max.
    worker_steps = [1, 6, 12, 18]
    total_games  = 18           # fixed workload, more workers = faster
    hands_cap    = 300          # per-game safety cap
    if quick:
        worker_steps = [1, 4]
        total_games  = 4
        hands_cap    = 50
    elif heavy:
        total_games  = 36        # twice the workload
        hands_cap    = 800       # longer per-tournament

    log(fp, f"Workload: {total_games} tournaments (4 bots each, cap {hands_cap} hands)")
    log(fp, f"{'Workers':>8}  {'Wall sec':>10}  {'Total hands':>12}  "
            f"{'Hands/sec':>10}  {'Speedup vs 1':>14}")
    log(fp, "-" * 70)

    baseline = None
    for workers in worker_steps:
        jobs = [(i, hands_cap) for i in range(total_games)]
        t0 = time.perf_counter()
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_run_one_tournament, jobs)
        wall = time.perf_counter() - t0

        total_hands = sum(h for h, _ in results)
        hps = total_hands / wall
        if baseline is None:
            baseline = wall
            speedup  = "1.00x"
        else:
            speedup = f"{baseline / wall:.2f}x"

        log(fp, f"{workers:>8}  {wall:>10.2f}  {total_hands:>12}  "
                f"{hps:>10.1f}  {speedup:>14}")

    log(fp, "")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — GPU benchmark (matmul + PokerMLP training loop)
# ──────────────────────────────────────────────────────────────────────────────

def phase_gpu(fp, quick=False, heavy=False):
    log(fp, "=" * 70)
    log(fp, "PHASE 3 — GPU BENCHMARK (PyTorch CPU vs MPS)")
    log(fp, "=" * 70)

    try:
        import torch
        from bots.poker_mlp import PokerMLP
    except ImportError as e:
        log(fp, f"Skipping GPU phase — import failed: {e}")
        return

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    else:
        log(fp, "MPS unavailable — only CPU will be benchmarked.")

    # ── Matmul sweep ────────────────────────────────────────────────────────
    log(fp, "")
    log(fp, "-- Matrix multiplication (square NxN @ NxN, fp32) --")
    if quick:
        sizes, reps = [256, 1024], 5
    elif heavy:
        sizes, reps = [1024, 2048, 4096, 8192], 30
    else:
        sizes, reps = [512, 1024, 2048, 4096], 20

    log(fp, f"{'Size':>6}  " + "  ".join(f"{d.upper():>14}" for d in devices)
            + "  Speedup")
    for n in sizes:
        row_times = {}
        for dev in devices:
            a = torch.randn(n, n, device=dev)
            b = torch.randn(n, n, device=dev)
            # warmup
            for _ in range(3):
                _ = a @ b
            if dev == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                c = a @ b
            if dev == "mps":
                torch.mps.synchronize()
            else:
                # force realization on cpu
                _ = c.sum().item()
            row_times[dev] = (time.perf_counter() - t0) / reps

        tflops_strs = []
        for dev in devices:
            s = row_times[dev]
            # 2 * n^3 FLOPs for an n x n matmul
            tflops = (2 * n**3) / s / 1e12
            tflops_strs.append(f"{s*1000:>7.2f}ms/{tflops:4.2f}TF")

        if len(devices) == 2:
            speedup = row_times["cpu"] / row_times["mps"]
            log(fp, f"{n:>6}  " + "  ".join(f"{t:>14}" for t in tflops_strs)
                    + f"  {speedup:.2f}x")
        else:
            log(fp, f"{n:>6}  " + "  ".join(f"{t:>14}" for t in tflops_strs))

    # ── PokerMLP training loop ──────────────────────────────────────────────
    log(fp, "")
    log(fp, "-- PokerMLP (256 hidden, 6-way classification) training --")

    input_dim  = 26     # matches rl_bot state vector
    batch_size = 1024
    steps      = 200 if not quick else 30

    log(fp, f"batch={batch_size}, steps={steps}")
    log(fp, f"{'Device':>6}  {'Wall sec':>10}  {'Steps/sec':>10}  "
            f"{'Samples/sec':>12}")
    for dev in devices:
        model = PokerMLP(input_dim=input_dim, hidden=256, num_classes=6).to(dev)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.randn(batch_size, input_dim, device=dev)
        y = torch.randint(0, 6, (batch_size,), device=dev)

        # warmup
        for _ in range(5):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        if dev == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        if dev == "mps":
            torch.mps.synchronize()
        wall = time.perf_counter() - t0

        sps = steps / wall
        samp = sps * batch_size
        log(fp, f"{dev.upper():>6}  {wall:>10.2f}  {sps:>10.1f}  {samp:>12.0f}")

    log(fp, "")

    # ── Heavy-only extras: bigger model + sustained thermal test ────────────
    if heavy:
        _heavy_big_model(fp, devices, torch)
        _heavy_thermal(fp, devices, torch)


def _heavy_big_model(fp, devices, torch):
    """Train a ~17M-param MLP so the GPU actually has real work to do."""
    log(fp, "-- BigModel MLP (4 x 2048 hidden, ~17M params) training --")
    batch_size = 4096
    steps      = 300

    class BigModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(26, 2048),   torch.nn.ReLU(),
                torch.nn.Linear(2048, 2048), torch.nn.ReLU(),
                torch.nn.Linear(2048, 2048), torch.nn.ReLU(),
                torch.nn.Linear(2048, 2048), torch.nn.ReLU(),
                torch.nn.Linear(2048, 6),
            )
        def forward(self, x):
            return self.net(x)

    log(fp, f"batch={batch_size}, steps={steps}")
    log(fp, f"{'Device':>6}  {'Wall sec':>10}  {'Steps/sec':>10}  "
            f"{'Samples/sec':>12}")
    for dev in devices:
        model   = BigModel().to(dev)
        opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        x = torch.randn(batch_size, 26, device=dev)
        y = torch.randint(0, 6, (batch_size,), device=dev)

        for _ in range(5):  # warmup
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        if dev == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        if dev == "mps":
            torch.mps.synchronize()
        wall = time.perf_counter() - t0

        sps  = steps / wall
        samp = sps * batch_size
        log(fp, f"{dev.upper():>6}  {wall:>10.2f}  {sps:>10.1f}  {samp:>12.0f}")
    log(fp, "")


def _heavy_thermal(fp, devices, torch):
    """Hammer a 4096x4096 matmul for 60s per device — first-10s vs last-10s
    throughput tells you if the machine is throttling under sustained load."""
    log(fp, "-- Sustained matmul stress (4096x4096 fp32, 60s per device) --")
    duration = 60.0
    size     = 4096

    log(fp, f"{'Device':>6}  {'First 10s TF/s':>15}  {'Last 10s TF/s':>15}  "
            f"{'Sustained':>10}")
    for dev in devices:
        a = torch.randn(size, size, device=dev)
        b = torch.randn(size, size, device=dev)
        for _ in range(3):  # warmup
            _ = a @ b
        if dev == "mps":
            torch.mps.synchronize()

        early_flops = late_flops = 0
        early_time  = late_time  = 0.0
        t_start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed >= duration:
                break
            t0 = time.perf_counter()
            c = a @ b
            if dev == "mps":
                torch.mps.synchronize()
            else:
                _ = c.sum().item()
            dt = time.perf_counter() - t0
            flops = 2 * size ** 3

            if elapsed < 10.0:
                early_flops += flops
                early_time  += dt
            elif elapsed >= duration - 10.0:
                late_flops  += flops
                late_time   += dt

        early_tf = (early_flops / early_time) / 1e12 if early_time else 0
        late_tf  = (late_flops  / late_time ) / 1e12 if late_time  else 0
        ratio    = (late_tf / early_tf) if early_tf else 0
        log(fp, f"{dev.upper():>6}  {early_tf:>14.2f}  {late_tf:>14.2f}  "
                f"{ratio:>10.1%}")
    log(fp, "")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Short smoke-test run (~30s)")
    parser.add_argument("--heavy", action="store_true",
                        help="Long stress run (~15-20 min): bigger model, "
                             "more hands, 60s sustained thermal test per device")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    overall_start = time.perf_counter()
    with open(OUTPUT_PATH, "w") as fp:
        phase_system_info(fp)
        if not args.skip_cpu:
            phase_cpu(fp, quick=args.quick, heavy=args.heavy)
        if not args.skip_gpu:
            phase_gpu(fp, quick=args.quick, heavy=args.heavy)

        total = time.perf_counter() - overall_start
        log(fp, "=" * 70)
        log(fp, f"DONE — total wall time: {total:.1f}s")
        log(fp, f"Full log: {OUTPUT_PATH}")
        log(fp, "=" * 70)


if __name__ == "__main__":
    # macOS needs 'spawn' start method when mixing torch + multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
