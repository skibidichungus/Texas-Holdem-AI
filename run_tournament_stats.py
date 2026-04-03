# run_tournament_stats.py — Run multiple tournaments and track detailed statistics

import argparse
import csv
import io
import os
import random
from collections import defaultdict
from contextlib import redirect_stdout
from multiprocessing import Pool

from core.engine import Table, Seat
from bots import parse_players, escalate_blinds, create_bot


# ── Single silent tournament ──────────────────────────────────────────────────

def run_silent_tournament(args_tuple):
    """Run one tournament silently. Accepts a tuple for multiprocessing.Pool.map.

    Returns dict with: winner, hand_count, finish_order [(pid, position, hand#, chips_at_elim)].
    """
    player_specs, chips, base_sb, base_bb, blind_increase_every, max_hands, seed = args_tuple

    if seed is not None:
        random.seed(seed)

    # Rebuild bots in this process (can't pickle adapters across processes)
    bots = {}
    for pid, btype, _ in player_specs:
        bots[pid] = create_bot(btype)

    seats = [Seat(player_id=pid, chips=chips) for pid, _, _ in player_specs]
    table = Table()
    dealer_index = 0
    hand_count = 0
    total_players = len(seats)
    finish_order: list[tuple[str, int, int, int]] = []  # (pid, pos, hand#, chips_at_elim)

    with redirect_stdout(io.StringIO()):
        while True:
            active_players = [s for s in seats if s.chips > 0]
            if len(active_players) <= 1:
                break

            hand_count += 1
            sb, bb = escalate_blinds(hand_count, base_sb, base_bb, blind_increase_every)
            active_seats = [s for s in seats if s.chips > 0]
            active_bots = {s.player_id: bots[s.player_id] for s in active_seats}

            table.play_hand(
                seats=active_seats,
                small_blind=sb,
                big_blind=bb,
                dealer_index=dealer_index % len(active_seats),
                bot_for=active_bots,
                on_event=None,
            )
            dealer_index = (dealer_index + 1) % len(seats)

            # Track eliminations
            for s in seats:
                if s.chips <= 0 and not any(e[0] == s.player_id for e in finish_order):
                    pos = total_players - len(finish_order)
                    finish_order.append((s.player_id, pos, hand_count, 0))

            if hand_count >= max_hands:
                break

    # Winner / survivors
    for s in seats:
        if s.chips > 0 and not any(e[0] == s.player_id for e in finish_order):
            finish_order.append((s.player_id, 1, hand_count, s.chips))

    # If we hit max_hands, assign remaining positions by chip count
    unfinished = [s for s in seats
                  if not any(e[0] == s.player_id for e in finish_order)]
    unfinished.sort(key=lambda s: s.chips, reverse=True)
    next_pos = total_players - len(finish_order)
    for s in unfinished:
        finish_order.append((s.player_id, next_pos, hand_count, s.chips))
        next_pos -= 1

    winner = None
    for pid, pos, _, _ in finish_order:
        if pos == 1:
            winner = pid
            break
    if winner is None and finish_order:
        winner = finish_order[-1][0]

    return {
        "winner": winner,
        "hand_count": hand_count,
        "finish_order": finish_order,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_tournament_batch(player_spec_str, num_tournaments, chips, base_sb, base_bb,
                         blind_increase_every, max_hands, parallel, output_csv, seed):
    player_specs = parse_players(player_spec_str)
    if len(player_specs) < 2:
        print("Error: need at least 2 players.")
        return

    pids = [pid for pid, _, _ in player_specs]
    bot_types = {pid: btype for pid, btype, _ in player_specs}

    print("=" * 75)
    print(f"RUNNING {num_tournaments} TOURNAMENTS")
    print("=" * 75)
    print(f"Players: {', '.join(f'{pid}={btype}' for pid, btype, _ in player_specs)}")
    print(f"Chips: {chips}  |  Blinds: {base_sb}/{base_bb}  |  "
          f"Escalation every {blind_increase_every} hands")
    if parallel > 1:
        print(f"Parallel workers: {parallel}")
    print("=" * 75)
    print()

    # Build args tuples for each tournament
    tasks = []
    for i in range(num_tournaments):
        t_seed = (seed + i) if seed is not None else None
        tasks.append((player_specs, chips, base_sb, base_bb,
                      blind_increase_every, max_hands, t_seed))

    # Run tournaments
    results = []
    if parallel > 1:
        with Pool(processes=parallel) as pool:
            for i, res in enumerate(pool.imap_unordered(run_silent_tournament, tasks), 1):
                results.append(res)
                if i % 5 == 0 or i == num_tournaments:
                    print(f"  Completed {i}/{num_tournaments}...")
    else:
        for i, task in enumerate(tasks, 1):
            res = run_silent_tournament(task)
            results.append(res)
            if i % 5 == 0 or i == num_tournaments:
                winner = res["winner"]
                hands = res["hand_count"]
                print(f"  Tournament {i}/{num_tournaments} — Winner: {winner} ({hands} hands)")

    # ── Aggregate statistics ──────────────────────────────────────────────────

    wins = defaultdict(int)
    finish_positions = defaultdict(list)       # pid -> [positions]
    chips_at_elimination = defaultdict(list)   # pid -> [chips when eliminated]
    hands_survived = defaultdict(list)         # pid -> [hand# when eliminated]
    h2h_wins = defaultdict(lambda: defaultdict(int))  # pid_a -> pid_b -> count a beat b

    hand_counts = []

    for res in results:
        hand_counts.append(res["hand_count"])
        wins[res["winner"]] += 1

        fo = res["finish_order"]  # [(pid, pos, hand#, chips)]

        for pid, pos, hand, elim_chips in fo:
            finish_positions[pid].append(pos)
            hands_survived[pid].app