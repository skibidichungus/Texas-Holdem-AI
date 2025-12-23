"""
Train the RL bot through self-play or against other bots.
"""
from core.engine import Table, TournamentManager, Seat, InProcessBot
from core.bot_api import BotAdapter, PlayerView, Action
from bots.rl_bot import RLBot
from bots.monte_carlo_bot import MonteCarloBot
from bots.poker_mind_bot import SmartBot
import argparse

class PlayerViewAdapter(BotAdapter):
    """Adapter for bots that expect PlayerView."""
    def __init__(self, bot):
        self.bot = bot
    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)

def train_rl_bot(num_episodes=1000, chips_per_player=500, opponent_type="montecarlo"):
    """
    Train RL bot through multiple tournaments.
    
    Args:
        num_episodes: Number of tournaments to play
        chips_per_player: Starting chips
        opponent_type: "self" (self-play), "montecarlo", or "smart"
    """
    print("=" * 70)
    print(f"TRAINING RL BOT")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Chip stack per player: {chips_per_player}")
    print(f"Opponent: {opponent_type}")
    print("=" * 70)
    print()
    
    # Create RL bot with higher learning rate
    rl_bot = RLBot(training_mode=True, learning_rate=3e-4)
    
    table = Table()
    tm = TournamentManager(table)
    
    wins = 0
    total_chips = 0
    recent_wins = 0
    
    for episode in range(1, num_episodes + 1):
        # Reset for new tournament
        rl_bot.end_episode()
        rl_bot.opponent_stats = {}
        
        # Set up opponents
        if opponent_type == "self":
            seats = [
                Seat(player_id="P1", chips=chips_per_player),
                Seat(player_id="P2", chips=chips_per_player)
            ]
            bots = {
                "P1": InProcessBot(RLBot(training_mode=False)),
                "P2": InProcessBot(rl_bot)
            }
        elif opponent_type == "montecarlo":
            seats = [
                Seat(player_id="P1", chips=chips_per_player),
                Seat(player_id="P2", chips=chips_per_player)
            ]
            # Use FULL strength MonteCarlo (200 sims) for harder training
            bots = {
                "P1": PlayerViewAdapter(MonteCarloBot(simulations=200)),  # Full strength!
                "P2": InProcessBot(rl_bot)
            }
        else:  # smart
            seats = [
                Seat(player_id="P1", chips=chips_per_player),
                Seat(player_id="P2", chips=chips_per_player)
            ]
            bots = {
                "P1": PlayerViewAdapter(SmartBot()),
                "P2": InProcessBot(rl_bot)
            }
        
        # Play tournament until winner
        hand_count = 0
        dealer_index = 0
        initial_chips_p2 = chips_per_player
        
        while True:
            active_seats = [s for s in seats if s.chips > 0]
            if len(active_seats) <= 1:
                winner = active_seats[0].player_id if active_seats else None
                break
            
            # Play hand
            result = table.play_hand(
                seats=active_seats,
                small_blind=1,
                big_blind=2,
                dealer_index=dealer_index % len(active_seats),
                bot_for={s.player_id: bots[s.player_id] for s in active_seats},
                on_event=None
            )
            
            # Record reward with better scaling
            if "P2" in result:
                reward = result["P2"] * 0.1  # Scale down per-hand rewards
                rl_bot.record_reward(reward)
            
            dealer_index = (dealer_index + 1) % len(seats)
            hand_count += 1
            
            if hand_count > 10000:  # Safety limit
                winner = max(seats, key=lambda s: s.chips).player_id
                break
        
        # Better final reward structure
        final_chips_p2 = sum(s.chips for s in seats if s.player_id == "P2")
        chip_change = final_chips_p2 - initial_chips_p2
        
        if winner == "P2":
            wins += 1
            recent_wins += 1
            # Bigger bonus for winning
            final_reward = 100 + (chip_change * 0.1)
        else:
            # Penalty based on how badly we lost
            final_reward = -50 + (chip_change * 0.1)
        
        rl_bot.record_reward(final_reward)
        rl_bot.end_episode()
        
        # Track stats
        total_chips += final_chips_p2
        
        # Progress update
        if episode % 50 == 0 or episode == num_episodes:
            win_rate = (wins / episode) * 100
            recent_win_rate = (recent_wins / 50) * 100 if episode >= 50 else win_rate
            avg_chips = total_chips / episode
            print(f"Episode {episode}/{num_episodes} | "
                  f"Win Rate: {win_rate:.1f}% | "
                  f"Recent (50): {recent_win_rate:.1f}% | "
                  f"Avg Final Chips: {avg_chips:.1f}")
            recent_wins = 0  # Reset recent wins counter
    
    # Save trained model
    rl_bot.save_model()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final Win Rate: {(wins/num_episodes)*100:.1f}%")
    print(f"Model saved to: bots/models/rl_model.pt")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL bot")
    parser.add_argument("--episodes", type=int, default=50000,  # Increased from 5000
                       help="Number of training episodes (default: 50000)")
    parser.add_argument("--chips", type=int, default=500,
                       help="Starting chips per player")
    parser.add_argument("--opponent", type=str, default="montecarlo",
                       choices=["self", "montecarlo", "smart"],
                       help="Opponent type")
    
    args = parser.parse_args()
    
    train_rl_bot(
        num_episodes=args.episodes,
        chips_per_player=args.chips,
        opponent_type=args.opponent
    )
