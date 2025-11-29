# Texas Hold’em Bot Project

This project contains:
- A complete poker engine (betting rounds, blinds, pot distribution)
- A configurable AI bot system (plug-in architecture)
- A SmartBot (PokerMindBot) with heuristics for preflop + postflop
- MonteCarloBot that makes decisions using simulated rollouts
- A tournament runner for simulating many hands locally
This version includes major engine stability fixes, legal action correctness, and full MonteCarloBot integration.

## Structure

```
Texas Hold'em Bot/
├── core/
│   ├── engine.py           # Poker game engine (stable, fixed version)
│   ├── bot_api.py          # PlayerView, Action, adapter wrappers
│   └── __init__.py
│
├── bots/
│   ├── poker_mind_bot.py   # SmartBot (heuristics + bluffing logic)
│   ├── monte_carlo_bot.py  # MonteCarloBot (rollout simulation bot)
│   ├── train.py            # Placeholder for future RL training
│   ├── model.pkl           # Placeholder model
│   └── __init__.py
│
├── run_local_match.py      # Script to run a match or tournament
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
  "hole": [["A", "h"], ["8", "s"]],
  "board": [["J","d"],["7","h"],["2","c"],["Q","s"]],
  "pot": 34.0,
  "to_call": 4.0,
  "legal": [
    {"type": "fold"},
    {"type": "call"},
    {"type": "raise", "min": 12.0, "max": 188.0}
  ],
  "chosen_action": {"type": "call", "amount": 4.0},
  "stacks": {"P1": 195, "P2": 144, "P3": 161},
  "opponents": ["P1","P3"],
  "folded": false,
  "hand_id": 27
}
```
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

### Engine Improvements
- Replace approx_score with full 7-card hand evaluator
- Add side-pot support for multi-all-in
- Add deterministic RNG seeding
- Improve board texture analysis (paired boards, flush draws)
- Add optional “fast engine mode” for 50k+ simulations

### Bots
- RL training pipeline inside bots/train.py
- Opponent modeling / tracking tendencies
- GTO-inspired balanced strategy bot
- Memory-based adapting bot
- Difficulty modes (“Easy”, “Normal”, “Smart”, “Insane”)

### Tooling & Testing
- Unit tests for the engine
- Integration tests for full tournaments
- GitHub Actions CI for auto-running matches
- Logging to JSON/CSV
- Plot stack graphs, showdown charts, win-rate curves

### Tournament System
- Swiss, Round Robin, Knockout
- Leaderboards with ELO ratings
- Blind-structure presets (turbo, deepstack)
- Remote bots over HTTP / WebSocket

### Developer Experience
- CONTRIBUTING.md for bot authors
- Example bots (tight, loose, random)
- CLI:
```bash
python3 run_local_match.py --bot1 SmartBot --bot2 MonteCarloBot --hands 500
```
- Documentation website (MkDocs)
