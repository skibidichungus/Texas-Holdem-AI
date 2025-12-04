# run_local_match.py

from core.engine import Table, TournamentManager, Seat, InProcessBot
from bots.poker_mind_bot import SmartBot        # Imports smartbot (v1 rn)
from bots.monte_carlo_bot import MonteCarloBot      # Imports a monte carlo bot
from bots.ml_bot import MLBot


def main():
    # 3-player demo table with 200 chips each
    seats = [Seat(player_id=f"P{i+1}", chips=200) for i in range(3)]

    # Use PokerMindBot for all seats (you can later mix bots)
    #bots = {
    #    "P1": InProcessBot(PokerMindBot()),
    #    "P2": InProcessBot(PokerMindBot()),
    #    "P3": InProcessBot(PokerMindBot()),
    #}

    # Use MonteCarloBot
    bots = {
        "P1": MLBot(),
        "P2": SmartBot(),
        "P3": MonteCarloBot(),
    }


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
