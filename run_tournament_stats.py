# run_tournament_stats.py
# Run multiple tournaments and track win rates

from core.engine import Table, TournamentManager, Seat, InProcessBot
from core.bot_api import BotAdapter, PlayerView, Action
from bots.poker_mind_bot import SmartBot
from bots.monte_carlo_bot import MonteCarloBot
from bots.ml_bot import MLBot
from bots.rl_bot import RLBot  # Add RLBot
from collections import defaultdict
import sys


def run_silent_tournament(seats, bots, small_blind, big_blind):
    """
    Run tournament silently (no output) and return winner.
    """
    table = Table()
    tm = TournamentManager(table)
    
    hand_count = 0
    dealer_index = 0
    
    # Suppress print output
    import io
    from contextlib import redirect_stdout
    
    with redirect_stdout(io.StringIO()):
        while True:
            active_players = [s for s in seats if s.chips > 0]
            
            if len(active_players) <= 1:
                winner = active_players[0].player_id if active_players else None
                break
            
            hand_count += 1
            
            active_seats = [s for s in seats if s.chips > 0]
            
            class PlayerViewAdapter(BotAdapter):
                def __init__(self, bot):
                    self.bot = bot
                def act(self, view: PlayerView) -> Action:
                    return self.bot.act(view)
            
            active_bots = {}
            for s in active_seats:
                bot = bots[s.player_id]
                # Handle MLBot and RLBot (both need InProcessBot)
                if isinstance(bot, (MLBot, RLBot)):
                    active_bots[s.player_id] = InProcessBot(bot)
                else:
                    active_bots[s.player_id] = PlayerViewAdapter(bot)
            
            res = table.play_hand(
                seats=active_seats,
                small_blind=small_blind,
                big_blind=big_blind,
                dealer_index=dealer_index % len(active_seats),
                bot_for=active_bots,
                on_event=None
            )
            
            dealer_index = (dealer_index + 1) % len(seats)
            
            if hand_count > 10000:
                # Safety limit - return player with most chips
                winner = max(seats, key=lambda s: s.chips).player_id
                break
    
    return winner, hand_count


def run_tournament_batch(num_tournaments=30, chips_per_player=500):
    """
    Run multiple tournaments and track statistics.
    """
    print("=" * 70)
    print(f"RUNNING {num_tournaments} TOURNAMENTS")
    print("=" * 70)
    print(f"Chip stack per player: {chips_per_player}")
    print(f"Bots: P1=RLBot, P2=MLBot, P3=SmartBot, P4=MonteCarloBot, P5=MonteCarloBot")
    print("=" * 70)
    print()
    
    wins = defaultdict(int)
    total_hands = 0
    hand_counts = []
    
    for tournament_num in range(1, num_tournaments + 1):
        # Reset seats for each tournament - NOW 5 PLAYERS
        seats = [Seat(player_id=f"P{i+1}", chips=chips_per_player) for i in range(5)]
        
        # Set up 5 bots (fresh instances)
        bots = {
            "P1": RLBot(training_mode=False),  # Trained RL bot
            "P2": MLBot(),                     # ML bot
            "P3": SmartBot(),                  # Smart bot
            "P4": MonteCarloBot(),             # MonteCarlo bot
            "P5": MonteCarloBot(simulations=150),  # Slightly different MonteCarlo
        }
        
        # Debug: Check RLBot status (only print once)
        if tournament_num == 1:
            rl_bot = bots["P1"]
            if hasattr(rl_bot, 'model_loaded'):
                if rl_bot.model_loaded:
                    print(f"RLBot: Model loaded successfully")
                else:
                    print(f"RLBot: Model NOT loaded, using fallback strategy")
        
        # Run tournament
        winner, hands = run_silent_tournament(seats, bots, small_blind=1, big_blind=2)
        
        wins[winner] += 1
        total_hands += hands
        hand_counts.append(hands)
        
        # Progress update every 5 tournaments
        if tournament_num % 5 == 0 or tournament_num == num_tournaments:
            print(f"Tournament {tournament_num}/{num_tournaments} - Winner: {winner} ({hands} hands)")
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("TOURNAMENT STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal tournaments: {num_tournaments}")
    print(f"Total hands played: {total_hands}")
    print(f"Average hands per tournament: {total_hands / num_tournaments:.1f}")
    
    print("\n" + "-" * 70)
    print("WIN RATES:")
    print("-" * 70)
    
    bot_names = {
        "P1": "RLBot",
        "P2": "MLBot",
        "P3": "SmartBot", 
        "P4": "MonteCarloBot",
        "P5": "MonteCarloBot2"
    }
    
    # Iterate over all 5 players
    for player_id in ["P1", "P2", "P3", "P4", "P5"]:
        wins_count = wins[player_id]
        win_rate = (wins_count / num_tournaments) * 100
        bot_name = bot_names[player_id]
        print(f"  {player_id} ({bot_name:15s}): {wins_count:3d} wins ({win_rate:5.1f}%)")
    
    print("\n" + "-" * 70)
    print("HAND COUNT STATISTICS:")
    print("-" * 70)
    if hand_counts:
        print(f"  Shortest tournament: {min(hand_counts)} hands")
        print(f"  Longest tournament:  {max(hand_counts)} hands")
        print(f"  Average:            {sum(hand_counts) / len(hand_counts):.1f} hands")
    
    print("\n" + "=" * 70)
    
    return wins, hand_counts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run multiple tournaments and track win rates")
    parser.add_argument("--tournaments", type=int, default=30, 
                        help="Number of tournaments to run (default: 30)")
    parser.add_argument("--chips", type=int, default=500,
                        help="Starting chips per player (default: 500)")
    
    args = parser.parse_args()
    
    run_tournament_batch(num_tournaments=args.tournaments, chips_per_player=args.chips)
