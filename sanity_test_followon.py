"""
Sanity test for the two follow-on fixes:

  1. PlayerView.opponents must exclude folded players (on streets after the fold).
     Preflop: P0 acts before P1, so P1 is legitimately still live when P0 acts
     → P1 appearing in P0's preflop opponents is CORRECT.
     Flop/turn/river: P1 has folded → must be absent.

  2. KeyboardInterrupt during training triggers a bot.save().
"""
import sys, os, signal, time, pickle, tempfile, threading

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

from core.engine import Table, Seat, InProcessBot

PASS = True

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — PlayerView.opponents excludes folded players on postflop streets
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: opponents excludes folded players postflop")
print("=" * 60)

class CheckCallBot:
    def act(self, state):
        for t in ("check", "call", "fold"):
            for a in state.legal_actions:
                if a["type"] == t:
                    return {"type": t}
        return {"type": state.legal_actions[0]["type"]}

class PreflopFoldBot:
    def act(self, state):
        if state.street == "preflop":
            for a in state.legal_actions:
                if a["type"] == "fold":
                    return {"type": "fold"}
        for t in ("check", "call"):
            for a in state.legal_actions:
                if a["type"] == t:
                    return {"type": t}
        return {"type": state.legal_actions[0]["type"]}

# Capture every .opponents list seen on each street, keyed by (street, acting_pid)
import random
import collections

all_views = []   # list of (street, acting_pid, opponents_list)

class SpyBot:
    def __init__(self, pid, inner):
        self.pid = pid
        self.inner = inner
    def act(self, state):
        all_views.append((state.street, self.pid, list(state.opponents)))
        return self.inner.act(state)

rng = random.Random(42)
table = Table(rng=rng)
seats = [Seat(f"P{i}", 1000) for i in range(6)]

inner = {
    "P0": CheckCallBot(),
    "P1": PreflopFoldBot(),   # folds preflop
    "P2": CheckCallBot(),
    "P3": CheckCallBot(),
    "P4": CheckCallBot(),
    "P5": CheckCallBot(),
}
bot_for = {pid: InProcessBot(SpyBot(pid, inner[pid])) for pid in inner}
table.play_hand(seats=seats, small_blind=10, big_blind=20,
                dealer_index=0, bot_for=bot_for)

streets_seen = list(dict.fromkeys(s for s, _, _ in all_views))
print(f"Streets seen: {streets_seen}")
print()

# On preflop: P1 may legitimately appear (P0 acts before P1 has folded) — SKIP preflop
# On flop/turn/river: P1 MUST be absent
postflop_failures = []
for street, pid, opps in all_views:
    if street == "preflop":
        continue
    if "P1" in opps:
        postflop_failures.append((street, pid, opps))

if postflop_failures:
    PASS = False
    for street, pid, opps in postflop_failures:
        print(f"  FAIL ❌ [{street}] {pid} sees P1 (folded) in opponents={opps}")
else:
    # Print a clean summary per postflop street
    postflop = [(s, pid, opps) for s, pid, opps in all_views if s != "preflop"]
    if not postflop:
        print("  NOTE: hand ended preflop — no postflop data to check.")
    else:
        by_street = collections.defaultdict(list)
        for s, pid, opps in postflop:
            by_street[s].append(opps)
        for s, opp_lists in by_street.items():
            sample = opp_lists[0]
            print(f"  [{s}] sample opponents (P0's view) = {sample}  "
                  f"P1 absent = {'P1' not in sample}  → PASS ✅")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Ctrl+C triggers bot.save()
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 2: KeyboardInterrupt triggers save")
print("=" * 60)

from training.train_cfr_bot_multiway import train_cfr_bot_multiway

with tempfile.TemporaryDirectory() as tmpdir:
    profile = os.path.join(tmpdir, "cfr_test.pkl")

    def _send_sigint():
        time.sleep(2)
        os.kill(os.getpid(), signal.SIGINT)

    t = threading.Thread(target=_send_sigint, daemon=True)
    t.start()

    try:
        train_cfr_bot_multiway(
            num_tournaments=999_999,
            chips_per_player=1_000,
            iterations=5,
            save_every=999_999,   # disable periodic saves; only Ctrl+C save fires
            profile_path=profile,
        )
    except SystemExit:
        pass

    if os.path.exists(profile):
        size = os.path.getsize(profile)
        print(f"  Checkpoint exists after Ctrl+C: YES ({size:,} bytes)  → PASS ✅")
        try:
            with open(profile, "rb") as f:
                data = pickle.load(f)
            print(f"  Checkpoint is valid pickle with "
                  f"{len(data.get('nodes', {}))} info-set nodes  → PASS ✅")
        except Exception as e:
            print(f"  Checkpoint is CORRUPT: {e}  → FAIL ❌")
            PASS = False
    else:
        print("  Checkpoint does NOT exist after Ctrl+C  → FAIL ❌")
        PASS = False

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"OVERALL: {'ALL CHECKS PASSED ✅' if PASS else 'SOME CHECKS FAILED ❌'}")
print("=" * 60)
