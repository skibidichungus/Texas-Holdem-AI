# run_local_match.py

from core.engine import Table, TournamentManager, Seat, InProcessBot
from bots.poker_mind_bot import PokerMindBot


def main():
    # 3-player demo table with 200 chips each
    seats = [Seat(player_id=f"P{i+1}", chips=200) for i in range(3)]

    # Use PokerMindBot for all seats (you can later mix bots)
    bots = {s.player_id: InProcessBot(PokerMindBot()) for s in seats}

    table = Table()
    tm = TournamentManager(table)

    # Run, say, 20 hands for a quick test
    nets = tm.run(
        seats=seats,
        bot_for=bots,
        small_blind=1,
        big_blind=2,
        hands=20,
        dealer_index=0
    )

    print("Final stacks:")
    for s in seats:
        print(f"  {s.player_id}: {s.chips:.2f}")

    print("Net chips:")
    for pid, v in nets.items():
        print(f"  {pid}: {v:+.2f}")


if __name__ == "__main__":
    main()
