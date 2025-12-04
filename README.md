# ♠️ Texas Hold'em AI Bot Engine

A fully functional **Texas Hold’em poker engine** featuring:
- **Stable betting logic** (blinds, raises, legal actions)
- **Plug-in bot architecture**
- **Three bot types**:
  | Bot | Strategy |
  |-----|----------|
  | `PokerMindBot` | Heuristic + board strength |
  | `MonteCarloBot` | Simulated rollouts vs random hands |
  | `MLBot` | PyTorch model (trainable) |

All engines support logging, replaying, and future ML training.

---

## Project Structure

```
Texas Hold'em Bot/
├── core/ # Engine + logging + events
│ ├── engine.py
│ ├── bot_api.py
│ ├── logger.py
│ └── init.py
│
├── bots/ # Pluggable bot implementations
│ ├── poker_mind_bot.py # Smart heuristic bot
│ ├── monte_carlo_bot.py # Monte Carlo bot
│ ├── ml_bot.py # Safe ML inference bot
│ └── models/
│ ├── ml_model.pt # (optional, ignored if missing)
│ └── train_ml_bot.py
│
├── logs/ # Auto-generated decision logs (ignored by git)
├── run_local_match.py
└── README.md
```

## How to Run
Make sure you are in the project root:
```bash
python3 run_local_match.py
```
or specify number of hands:
```bash
python3 run_local_match.py --hands 50
```

You can swap which bots are used inside run_local_match.py.

```bash
seats = {
    "P1": MLBot(),
    "P2": MonteCarloBot(),
    "P3": PokerMindBot()
}
```

## Decision Logging
Every hand is now logged automatically to:
```bash
logs/YYYY-MM-DD_HHMMSS/
```
Each bot’s decisions are stored as newline-separated JSON objects in:
```bash
logs/<timestamp>/hand_<N>.jsonl
```
### Example log entry
```bash
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
These data are used to train future models.

### What gets logged
- Hole cards (P1 never sees opponents’ hole cards)
- Board cards at time of action
- Pot size
- Amount needed to call
- Legal actions (fold/check/call/raise info)
- Final chosen action
- Stack sizes
- Which opponents are in the hand
- Whether the bot is folded on this street
- hand_id (stable across all logs)

### Hand Result Logs
At the end of each hand:
```bash
{"hand_id": 27, "result": {"player": "P1", "net": +12.0}}
```

## Finding the Most Recent Log
Logs are stored in timestamped folders:
```bash
logs/2025-11-19_1458/
logs/2025-11-19_1523/
logs/2025-11-19_1650/
```
The most recent folder is the one with the latest timestamp.
Use macOS Terminal:
```bash
ls -1 logs | sort
```
Or newest first:
```bash
ls -1t logs
```

## Log format for ML Training
ML bot will train on .jsonl files where each line is one decision.
The recommended ML dataset fields:
| field           | meaning                               |
| --------------- | ------------------------------------- |
| `hole`          | 2-card private hand                   |
| `board`         | visible board cards                   |
| `pot`           | pot size before action                |
| `to_call`       | how much needed to call               |
| `legal`         | formatted bet/call/fold/check options |
| `stacks`        | chip counts of all players            |
| `chosen_action` | the target label                      |
| `folded`        | if player is already folded           |
| `street`        | preflop / flop / turn / river         |
| `hand_id`       | join results + decisions              |
You can load it with Python:
```bash
import json

with open("logs/.../hand_12.jsonl") as f:
    data = [json.loads(line) for line in f]
```

## PokerMindBot (SmartBot)
Located at:
```bash
bots/poker_mind_bot.py
```
### Preflop Strategy
- Premium hands: AA, KK, QQ, JJ, AK, AQ → Raise
- Good hands: Broadway, suited broadway, medium pairs → Call/Check
- Trash hands: Fold when facing a bet, otherwise Check

### Postflop Strategy
Uses:
```bash
approx_score(hole + board)
```
- Strong hands → bet/raise (~50% pot)
- Medium hands → call small bets
- Weak hands → check/fold
- 10% chance to bluff

### Safety Guarantees
SmartBot ALWAYS:
- Never makes illegal actions
- Respects to_call
- Never checks when facing a bet
- Never raises below min or above stack

## MonteCarloBot (Rollout Bot)
Located in:
```bash
bots/monte_carlo_bot.py
```

## MLBot (Neural Bot)
- Uses a PyTorch MLP classifer
- Loads models weights from:
```bash
bots/models/ml_model.pt
```
If the model file is missing, the bot still runs using a fallback safe policy, and game continues normally.

### Train the ML Model
```bash
python3 bots/models/train_ml_bot.py --log_dir logs --epochs 8
```
A new model will be saved automatically.

### How it works:
- Simulates random future boards
- Estimates win-rate by rolling out 200–500 trials
- Chooses the action with the best expected value
- Adapts to board texture, number of players, stack depth
- Handles all legal bet/call/raise options
Note: strongest bot so far...

## About the Engine
Located in:
```bash
core/engine.py
```
### Engine Features:
- Correct blinds posting
- Correct betting cycle with safety-breaks
- Fixed all-live-equal logic (rounds end at correct time)
- Street transitions: preflop → flop → turn → river
- Showdown using approximate scoring
- Fold-wins return immediately
- Tournament support via TournamentManager
The engine is now stable across 3–6 players for long simulations.

## Example output
```bash
Final stacks:
  P1: 327.25
  P2: 0.00
  P3: 72.75

Net chips:
  P1: +127.25
  P2: -200.00
  P3: +72.75
```

## Adding Your Own Bot
Create a new file under:
```bash
bots/
```
Example:
```bash
class MyBot:
    def act(self, state):
        return {"type": "fold"}
```
Bots interact with the engine via:
```bash
core/bot_api.py
```

## Future Work
- Full 7-card evaluator (replace approx_score)
- Side pot support for multi-all-in
- Real opponent modeling + bluff balancing
- Reinforcement learning pipeline
- Tournament ranking with ELO

## License
MIT License.
Use, modify, and extend freely. Attribution appreciated.

## Special Notes
This engine is fully deterministic and safe:
- No illegal actions
- No infinite betting loops
- Always respects to_call, raise min/max, and blinds
- Logs 100% of bot decisions
Perfect foundation for research, ML training, or poker AI competitions.