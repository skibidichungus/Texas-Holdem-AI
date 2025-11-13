from core.engine import Table, TournamentManager, InProcessBot, RandomBot, Seat
from bots.poker_mind_bot import SmartBot

if __name__ == "__main__":
    seats = [Seat(player_id="P1", chips=200),
             Seat(player_id="P2", chips=200),
             Seat(player_id="P3", chips=200)]

    bots = {s.player_id: InProcessBot(SmartBot()) for s in seats}

    tbl = Table()
    tm = TournamentManager(tbl)
    nets = tm.run(seats, bots, small_blind=1, big_blind=2, hands=500)

    print("\nFinal stacks:")
    for s in seats: print(f"  {s.player_id}: {s.chips:.2f}")
    print("Net chips:")
    for pid, v in nets.items(): print(f"  {pid}: {v:+.2f}")
