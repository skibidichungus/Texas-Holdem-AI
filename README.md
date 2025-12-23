# ♠️ Texas Hold'em AI Bot Engine

A fully functional **Texas Hold'em poker engine** featuring:
- **Stable betting logic** (blinds, raises, legal actions)
- **Plug-in bot architecture**
- **Four bot types**:
  | Bot | Strategy | Performance |
  |-----|----------|------------|
  | `PokerMindBot` | Heuristic + board strength + position awareness | Moderate |
  | `MonteCarloBot` | Simulated rollouts vs random hands + position awareness | **Strongest** |
  | `MLBot` | PyTorch supervised learning (trainable) | Learning |
  | `RLBot` | Reinforcement learning (REINFORCE) | Learning |

**Note**: MonteCarloBot is currently the strongest bot, consistently winning 40-50% in multi-player tournaments. MLBot and RLBot are trainable but require extensive training to compete.

All engines support logging, replaying, and ML training.

## Requirements
- Python 3.7+
- PyTorch (for MLBot and RLBot)
- Matplotlib (for tournament visualization)
- Standard library only (no other external dependencies)

Install dependencies:
```bash
pip install torch matplotlib
```

## Project Structure

```
Texas Hold'em Bot/
├── core/                    # Engine + logging + events
│   ├── engine.py            # Core game logic
│   ├── bot_api.py           # Bot interface
│   ├── logger.py            # Decision logging
│   └── __init__.py
│
├── bots/                    # Pluggable bot implementations
│   ├── poker_mind_bot.py    # SmartBot (heuristic)
│   ├── monte_carlo_bot.py   # MonteCarloBot (simulation-based)
│   ├── ml_bot.py            # MLBot (supervised learning)
│   ├── rl_bot.py            # RLBot (reinforcement learning)
│   ├── train_ml_bot.py      # MLBot training script
│   └── models/
│       ├── ml_model.pt       # Trained MLBot model (optional)
│       ├── rl_model.pt       # Trained RLBot model (optional)
│       └── poker_mlp.py    # Neural network architecture
│
├── logs/                    # Auto-generated decision logs (ignored by git)
├── run_local_match.py       # Single tournament runner
├── run_tournament_stats.py  # Batch tournament statistics
├── train_rl_bot.py          # RLBot training script
└── README.md
```

## How to Run

### Tournament Mode (Play Until One Winner)
The default mode runs a tournament until only one player remains:
```bash
python3 run_local_match.py
```
This will:
- Play hands until one player has all the chips
- Display real-time chip stacks after each hand
- Generate a visualization chart showing tournament progress
- Save the chart as `tournament_progress.png`

### Running Multiple Tournaments (Win Rate Tracking)
To evaluate bot performance over many tournaments:
```bash
python3 run_tournament_stats.py --tournaments 100 --chips 500
```
This will:
- Run multiple tournaments silently
- Track win rates for each bot
- Report average hands per tournament
- Useful for comparing bot performance

You can swap which bots are used inside `run_local_match.py` or `run_tournament_stats.py`:

```python
bots = {
    "P1": RLBot(training_mode=False),
    "P2": MLBot(),
    "P3": SmartBot(),
    "P4": MonteCarloBot(),
    "P5": MonteCarloBot(simulations=150),
}
```

## Bot Descriptions

### PokerMindBot (SmartBot)
**Location**: `bots/poker_mind_bot.py`

A heuristic-based bot with:
- **Hand Strength Estimation**: Uses `approx_score` to evaluate hand strength
- **Position Awareness**: Adjusts strategy based on position (early/middle/late)
- **Pot Odds Calculation**: Considers pot size and bet sizing

**Preflop Strategy**:
- Premium hands: AA, KK, QQ, JJ, AK, AQ → Raise
- Good hands: Broadway, suited broadway, medium pairs → Call/Check
- Trash hands: Fold when facing a bet, otherwise Check

**Postflop Strategy**:
- Strong hands → bet/raise (~50% pot)
- Medium hands → call small bets
- Weak hands → check/fold
- 10% chance to bluff

### MonteCarloBot ⭐ (Currently Strongest)
**Location**: `bots/monte_carlo_bot.py`

**Performance**: Consistently wins 40-50% in multi-player tournaments. The strongest bot in the current implementation.

**Features**:
- **Monte Carlo Simulation**: Simulates thousands of random outcomes (default: 200 simulations)
- **Equity Estimation**: Calculates win probability against random hands
- **Pot Odds**: Compares equity to pot odds for decision making
- **Position Awareness**: Adjusts winrate thresholds based on position

**How it works**:
1. Simulates random opponent hands and board completions
2. Calculates win rate from simulations
3. Compares win rate to pot odds
4. Makes decisions based on equity vs. pot odds

**Recommended for**: One-and-done tournaments, competitive play

### MLBot (Supervised Learning)
**Location**: `bots/ml_bot.py`

A PyTorch-based neural network that learns from logged game decisions.

**Features**:
- **26-Dimensional Feature Vector**: Includes street, pot, stacks, hole cards, board, hand strength, pot odds, position, and opponent memory
- **Short-Term Memory**: Tracks opponent aggression, tightness, and VPIP during the game
- **Adaptive Strategy**: Adjusts predictions based on opponent behavior
- **Fallback Strategy**: Uses heuristic-based play when model confidence is low or model is untrained

**Train the ML Model**:
```bash
python3 bots/train_ml_bot.py --log_dir logs --epochs 8
```

**Advanced Training Options**:
```bash
# Train only on specific bot's decisions (e.g., learn from MonteCarloBot)
python3 bots/train_ml_bot.py --log_dir logs --epochs 8 --filter_players P3

# Train only on winning hands
python3 bots/train_ml_bot.py --log_dir logs --epochs 8 --filter_winners
```

**Note**: If the model file is missing or has an incompatible format, the bot will use an untrained model (random weights) and continue running normally.

### RLBot (Reinforcement Learning)
**Location**: `bots/rl_bot.py`

A reinforcement learning bot using the REINFORCE algorithm. Learns optimal strategies through trial and error.

**Features**:
- **Policy Gradient Learning**: Uses REINFORCE algorithm
- **512-Hidden Unit Network**: Deep neural network for complex strategy learning
- **26-Dimensional Features**: Same feature set as MLBot
- **Opponent Memory**: Tracks opponent behavior during tournaments
- **Fallback Strategy**: Uses heuristic play when model is untrained

**Train the RL Bot**:
```bash
# Train against MonteCarloBot (recommended)
python3 train_rl_bot.py --episodes 50000 --opponent montecarlo

# Train through self-play
python3 train_rl_bot.py --episodes 50000 --opponent self

# Train against SmartBot (easier)
python3 train_rl_bot.py --episodes 50000 --opponent smart
```

**Training Notes**:
- Training takes significant time (50,000+ episodes recommended)
- Current performance: ~10% win rate in 1v1 against MonteCarloBot
- Still under development - MonteCarloBot remains stronger
- Model saved to `bots/models/rl_model.pt`

**Note**: RLBot is experimental. For competitive play, use MonteCarloBot.

## Decision Logging
Every hand is automatically logged to:
```bash
logs/session_YYYYMMDD_HHMMSS.jsonl
```

Each bot's decisions are stored as newline-separated JSON objects. These logs are used to train MLBot and can be analyzed for strategy insights.

### Example log entry
```json
{
  "player": "P2",
  "street": "turn",
  "hole": [["A","h"], ["8","s"]],
  "board": [["J","d"],["7","h"],["2","c"],["Q","s"]],
  "pot": 34.0,
  "to_call": 4.0,
  "legal": [{"type":"fold"},{"type":"call"},{"type":"raise","min":12,"max":188}],
  "chosen_action": {"type":"call","amount":4.0},
  "stacks": {"P1":195, "P2":144, "P3":161},
  "folded": false,
  "hand_id": 27
}
```

## About the Engine
**Location**: `core/engine.py`

### Engine Features:
- Correct blinds posting
- Correct betting cycle with safety-breaks
- Fixed all-live-equal logic (rounds end at correct time)
- Street transitions: preflop → flop → turn → river
- Showdown using approximate scoring
- Fold-wins return immediately
- Tournament support via TournamentManager
- Supports 2-10 players
- Safety limit: 10,000 hands per tournament

The engine is stable and deterministic across 3–6 players for long simulations.

## Performance Comparison

Based on 5-player tournament statistics (100 tournaments, 500 chips each):

| Bot | Win Rate | Notes |
|-----|----------|-------|
| MonteCarloBot (200 sims) | ~40% | **Strongest** - simulation-based |
| MonteCarloBot (150 sims) | ~40% | Slightly weaker but still strong |
| SmartBot | ~20% | Solid heuristic-based play |
| MLBot | ~3% | Requires extensive training |
| RLBot | ~1% | Still learning, experimental |

**Recommendation**: For competitive play, use **MonteCarloBot**.

## Adding Your Own Bot
Create a new file under `bots/` and implement the `act()` method:

```python
from core.bot_api import Action, PlayerView

class MyBot:
    def act(self, state: PlayerView) -> Action:
        # Your strategy here
        legal = state.legal_actions
        # ... make decision ...
        return Action("fold")  # or "call", "raise", etc.
```

Bots interact with the engine via `core/bot_api.py`.

## Future Work
- Full 7-card evaluator (replace approx_score)
- Side pot support for multi-all-in
- Advanced opponent modeling + bluff balancing
- Improved RL algorithms (PPO, A3C)
- Tournament ranking with ELO
- Multi-table tournament support

## License
MIT License. Use, modify, and extend freely. Attribution appreciated.

## Special Notes
This engine is fully deterministic and safe:
- No illegal actions
- No infinite betting loops
- Always respects to_call, raise min/max, and blinds
- Logs 100% of bot decisions

Perfect foundation for research, ML training, or poker AI competitions.
