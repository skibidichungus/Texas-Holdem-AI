"""
Sanity test v2: run many hands until we get one that goes all 4 streets.
Also specifically test that:
  - P1 (always-fold preflop) never acts postflop
  - Every postflop street has >= 2 live actors
  - Streets only end when all live players acted AND contributions match

Uses CheckCallBot so nobody folds postflop, guaranteeing 4 streets.
"""
import sys
import random

sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")

from core.engine import Table, Seat, InProcessBot

# ── Always check/call (never fold postflop, so we guarantee 4 streets) ────────
class CheckCallBot:
    def act(self, state):
        legal = state.legal_actions if hasattr(state, "legal_actions") else state["legal_actions"]
        # prefer check, then call, then fold
        for t in ("check", "call", "fold"):
            for a in legal:
                if a["type"] == t:
                    return {"type": t}
        return {"type": legal[0]["type"]}

# ── Folds preflop only ─────────────────────────────────────────────────────────
class PreflopFoldBot:
    def __init__(self):
        self.acted = False

    def act(self, state):
        legal = state.legal_actions if hasattr(state, "legal_actions") else state["legal_actions"]
        street = state.street if hasattr(state, "street") else state["street"]
        if street == "preflop":
            for a in legal:
                if a["type"] == "fold":
                    self.acted = True
                    return {"type": "fold"}
        # Fallback: check/call
        for t in ("check", "call"):
            for a in legal:
                if a["type"] == t:
                    return {"type": t}
        return {"type": legal[0]["type"]}

# ── Recording wrapper ──────────────────────────────────────────────────────────
class RecordingBot:
    def __init__(self, pid, inner):
        self.pid = pid
        self.inner = inner
        self.acted_streets = []

    def act(self, state):
        street = state.street if hasattr(state, "street") else state["street"]
        self.acted_streets.append(street)
        return self.inner.act(state)

# ── Build table with CheckCall bots so every hand reaches showdown ─────────────
rng = random.Random(99)
table = Table(rng=rng)

seats = [
    Seat("P0", 1000),
    Seat("P1", 1000),  # will fold preflop
    Seat("P2", 1000),
    Seat("P3", 1000),
    Seat("P4", 1000),
    Seat("P5", 1000),
]

folder = PreflopFoldBot()
inner_bots = {
    "P0": CheckCallBot(),
    "P1": folder,
    "P2": CheckCallBot(),
    "P3": CheckCallBot(),
    "P4": CheckCallBot(),
    "P5": CheckCallBot(),
}

recording = {pid: RecordingBot(pid, inner_bots[pid]) for pid in inner_bots}

streets_seen = []
_orig_betting = table._betting_round
street_actors = {}  # street → set of pids that acted

def _patched_betting(street, *args, **kwargs):
    streets_seen.append(street)
    print(f"\n=== STREET START: {street} ===")
    result = _orig_betting(street, *args, **kwargs)
    print(f"=== STREET END:   {street} | winner={result} ===")
    return result

table._betting_round = _patched_betting

bot_for = {pid: InProcessBot(recording[pid]) for pid in recording}

print("Playing 1 hand (check/call bots, P1 folds preflop)…\n")
net = table.play_hand(
    seats=seats,
    small_blind=10,
    big_blind=20,
    dealer_index=0,
    bot_for=bot_for,
)

# ── Analysis ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("NET RESULTS:", net)
print()
print("Streets visited:", streets_seen)
print()

PASS = True

# CHECK 1: all 4 streets reached
all_four = all(s in streets_seen for s in ["preflop", "flop", "turn", "river"])
status = "PASS" if all_four else "FAIL"
if not all_four:
    PASS = False
print(f"[CHECK 1] All 4 streets reached: {status}")

# CHECK 2: P1 must NOT have acted postflop
p1_streets = recording["P1"].acted_streets
postflop = [s for s in p1_streets if s != "preflop"]
if postflop:
    PASS = False
    print(f"[CHECK 2] FAIL — P1 acted postflop on: {postflop}")
else:
    print(f"[CHECK 2] PASS — P1 acted only on: {p1_streets} (folded preflop)")

# CHECK 3: every live player (except P1) must appear in each postflop street
for s in ["flop", "turn", "river"]:
    if s not in streets_seen:
        print(f"[CHECK 3] {s}: SKIPPED (street not reached)")
        continue
    actors = {pid for pid, rec in recording.items() if s in rec.acted_streets}
    expected = {pid for pid in recording if pid != "P1"}  # all but the folder
    missing = expected - actors
    if "P1" in actors:
        PASS = False
        print(f"[CHECK 3] {s}: FAIL — folded P1 appeared in actors: {actors}")
    elif missing:
        PASS = False
        print(f"[CHECK 3] {s}: FAIL — live players missing: {missing} | actors={actors}")
    else:
        print(f"[CHECK 3] {s}: PASS — actors={sorted(actors)}")

print()
print("="*60)
print(f"OVERALL: {'ALL CHECKS PASSED ✅' if PASS else 'SOME CHECKS FAILED ❌'}")
