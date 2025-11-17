# Texas Hold’em Bot Project

This project contains:
- **A complete poker engine (betting rounds, blinds, pot distribution)
- **A configurable AI bot system
- **A SmartBot (PokerMindBot) with preflop + postflop heuristics
- **A tournament runner for simulating many hands locally
This version includes major improvements to engine stability, game correctness, and bot intelligence.

## Structure

```
Texas Hold'em Bot/
├── core/
│   ├── engine.py         # Poker game engine (stable, fixed version)
│   ├── bot_api.py        # PlayerView, Action, BotAdapter wrappers
│   └── __init__.py
│
├── bots/
│   ├── poker_mind_bot.py # SmartBot (heuristic + bluffing)
│   ├── train.py          # Placeholder for future RL training
│   ├── model.pkl         # Placeholder model file
│   └── __init__.py
│
├── run_local_match.py    # Entry script for running matches
└── README.md             # (this file)
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

## About the PokerMindBot (SmartBot)
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
Behavior:
- Strong hands → bet/raise (~50% pot)
- Medium hands → call small bets
- Weak hands → check/fold
- 10% chance to bluff

### Safety Guarantees
SmartBot ALWAYS:
- Chooses legal actions
- Respects to_call
- Avoids illegal checks/raises
- Does not cause infinite loops

## About the Engine
Located in:
```bash
core/engine.py
```
### Features:
- Correct blinds posting
- Stable betting rounds with safety breakers
- all_live_equal() logic ensures rounds terminate
- Street handling:
 - Preflop
 - Flop (deal 3)
 - Turn (deal 1)
 - River (deal 1)
- Pot settlement:
 - Fold win (immediate)
 - Showdown (approx hand scoring)
- Tournament mode via TournamentManager
The engine is now fully deterministic and stable across 3+ player simulations.

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
Create new files in:
```bash
bots/
```
Example:
```bash
class MyBot:
    def act(self, state):
        return {"type": "fold"}
```
Bots interact with the engine via the adapter system in:
```bash
core/bot_api.py
```

## Future Work
Below are planned enhancements and stretch goals for the project:

### Engine Improvements
- Add full 7-card hand evaluator for accurate showdowns
- Improve approx_score to consider:
 - draws (flush/straight)
 - blockers
 - board texture
- Add full support for side pots in multi-all-in scenarios
- Add deterministic seeding for reproducible simulations
- Add “heads-up mode” optimizations

### Bot Development
- Add Monte Carlo rollout bot (simulate random future boards)
- Integrate RL training (policy gradient / Q-learning) in bots/train.py
- Add opponent modeling (track player tendencies over many hands)
- Implement a GTO-inspired bot (solvers, balanced strategies)
- Add memory-based bot that adapts during a tournament
- Add configurable difficulty modes (“Easy”, “Normal”, “Smart”, “Insane”)

### Tooling & Testing
- Add unit tests for engine, bot actions, and edge cases
- Add integration tests simulating full tournaments
- Add GitHub Actions to auto-run matches on PRs
- Add visualization of:
 - stack progression
 - win-rate charts
 - showdown frequency
- Add logging to JSON/CSV to help bot developers debug strategies

### Tournament System
- Implement Swiss, Round Robin, and Knockout tournament formats
- Add leaderboard output (ELO rating, profit graphs)
- Allow remote bots via HTTP/WebSocket API
- Add configurable blind structures (turbo, deepstack)

### Developer Experience
- Add a clear CONTRIBUTING.md for bot authors
- Add example bots (tight, loose, aggressive, random, etc.)
- Add a CLI for:
```bash
python3 run_local_match.py --bot1 MyBot --bot2 PokerMindBot --hands 500
```
- **Add documentation website (MkDocs / GitHub Pages)