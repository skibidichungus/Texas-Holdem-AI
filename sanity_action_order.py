"""
Verify: does P0 act BEFORE or AFTER P1 preflop?
ring[0]=BTN, ring[1]=SB, ring[2]=BB, ring[3]=UTG ... preflop_start=3
dealer_index=0 → ring order is [0,1,2,3,4,5] = [P0,P1,P2,P3,P4,P5]
  ring[0]=P0=BTN, ring[1]=P1=SB, ring[2]=P2=BB
  UTG (preflop_start=3) = ring[3] = P3
  Action order preflop: P3, P4, P5, P0(BTN), P1(SB), P2(BB)
So P0 acts BEFORE P1 on preflop → P1 hasn't folded yet when P0 acts → correct to see P1.
"""
import sys
sys.path.insert(0, "/Users/jaroslavaupart/Desktop/Projects/Texas-Holdem-AI")
from core.engine import Table, Seat, InProcessBot
import random

action_log = []

class LogBot:
    def __init__(self, pid):
        self.pid = pid
    def act(self, state):
        action_log.append((state.street, self.pid, list(state.opponents)))
        for t in ("check", "call", "fold"):
            for a in state.legal_actions:
                if a["type"] == t:
                    return {"type": t}
        return {"type": state.legal_actions[0]["type"]}

seats = [Seat(f"P{i}", 1000) for i in range(6)]
bots = {f"P{i}": InProcessBot(LogBot(f"P{i}")) for i in range(6)}
Table(rng=random.Random(42)).play_hand(seats=seats, small_blind=10, big_blind=20,
                                        dealer_index=0, bot_for=bots)

print("Preflop action order (pid → opponents):")
for street, pid, opps in action_log:
    if street == "preflop":
        print(f"  {pid} sees opponents={opps}")
print()
print("P1 preflop position in action order:",
      [pid for s, pid, _ in action_log if s == "preflop"].index("P1")
      if "P1" in [pid for s, pid, _ in action_log if s == "preflop"] else "did not act")
