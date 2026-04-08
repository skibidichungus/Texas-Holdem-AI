"""
Train the CFR bot for multi-player deep-stack conditions.

Four CFRBot instances (all sharing the same regret table) are placed on
seats P1–P4 simultaneously.  Every hand updates regrets from all four
perspectives, giving the table broader multi-way situation coverage.

This mirrors the UI tournament format:
  * 4 players per table
  * 1 000 chip starting stacks
  * 5/10 blinds with 1.5× escalation every 50 hands

Saves to a *separate* profile (models/cfr_regret_deep.pkl) so the existing
heads-up profile (models/cfr_regret.pkl) is never touched.

Equity-cache optimisation
-------------------------
``_quick_equity`` is computed **once** per decision point (before the
iteration loop in ``CFRBot._run_iterations``), not per-action per-iteration.
This is already the design in CFRBot; this script simply inherits it.

Convergence note
----------------
In 4-way self-play the expected win-rate for each seat is ~25 %.  That is the
sign of healthy convergence, not a bug.  Track ``info_sets`` and
``total_iters`` (printed every 1 000 episodes).

Checkpoint
----------
* Loads  ``--profile``  (default: models/cfr_regret_deep.pkl) on startup.
* Saves every ``--save_every`` episodes (default: 500) and at the end of
  training.

Usage
-----
    python training/train_cfr_bot_multiway.py
    python training/train_cfr_bot_multiway.py --tournaments 100000 --iterations 200
    python training/train_cfr_bot_multiway.py --profile models/cfr_deep_v2.pkl
"""

import os
import sys

# Add project root so imports work from any working directory.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse

from core.engine import Table, Seat, InProcessBot
from core.bot_api import BotAdapter, PlayerView, Action
from bots.cfr_bot import CFRBot
from bots import escalate_blinds


# ---------------------------------------------------------------------------
#  Thin adapter so CFRBot plugs into the engine's InProcessBot / bot_for dict
# ---------------------------------------------------------------------------

class _CFRAdapter(BotAdapter):
    """Wraps a CFRBot to satisfy the BotAdapter interface."""

    def __init__(self, bot: CFRBot):
        self.bot = bot

    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)


# ---------------------------------------------------------------------------
#  Main training function
# ---------------------------------------------------------------------------

PLAYER_IDS = ["P1", "P2", "P3", "P4"]
NUM_PLAYERS = len(PLAYER_IDS)

BASE_SB = 5
BASE_BB = 10
BLIND_ESCALATION_EVERY = 50   # hands


def train_cfr_bot_multiway(
    num_tournaments: int = 50_000,
    chips_per_player: int = 1_000,
    iterations: int = 200,
    save_every: int = 500,
    profile_path: str = "models/cfr_regret_deep.pkl",
) -> CFRBot:
    """
    Run multi-player CFR self-play for ``num_tournaments`` episodes.

    Args:
        num_tournaments:  Number of 4-player tournament episodes.
        chips_per_player: Starting chip stack per seat (default 1 000).
        iterations:       MCCFR rollouts per decision point.
        save_every:       Persist the regret table every N episodes.
        profile_path:     Path for regret-table persistence.

    Returns:
        The trained CFRBot instance.
    """
    print("=" * 70)
    print("TRAINING CFR BOT  (4-player deep-stack self-play)")
    print("=" * 70)
    print(f"Episodes:         {num_tournaments}")
    print(f"Players:          {NUM_PLAYERS}  ({', '.join(PLAYER_IDS)})")
    print(f"Chips per player: {chips_per_player}")
    print(f"Base blinds:      {BASE_SB}/{BASE_BB}")
    print(f"Blind escalation: every {BLIND_ESCALATION_EVERY} hands (×1.5)")
    print(f"Iterations/pt:    {iterations}  (MCCFR rollouts per decision)")
    print(f"Save every:       {save_every} episodes")
    print(f"Profile path:     {profile_path}")
    print("=" * 70)
    print()

    # ── Build bot (constructor auto-loads profile if it exists) ──────────────
    bot = CFRBot(
        iterations=iterations,
        profile_path=profile_path,
        use_average=True,
    )

    loaded_stats = bot.stats()
    if loaded_stats["info_sets"] > 0:
        print(
            f"Resumed from {profile_path}: "
            f"{loaded_stats['info_sets']} info sets, "
            f"{loaded_stats['total_iterations']} total iterations.\n"
        )
    else:
        print("No existing profile found — starting fresh.\n")

    # ── One shared adapter: all 4 seats reference the same CFRBot ────────────
    adapter = _CFRAdapter(bot)

    table = Table()
    wins = {pid: 0 for pid in PLAYER_IDS}

    # ── Main training loop ───────────────────────────────────────────────────
    for episode in range(1, num_tournaments + 1):

        # Fresh chip stacks each episode
        seats = [Seat(player_id=pid, chips=chips_per_player) for pid in PLAYER_IDS]

        # All seats share the same adapter (and therefore the same regret table)
        bots = {pid: InProcessBot(adapter) for pid in PLAYER_IDS}

        # ── Play hands until one player remains ──────────────────────────────
        dealer_index = 0
        hand_count = 0
        winner = None

        while True:
            active_seats = [s for s in seats if s.chips > 0]
            if len(active_seats) <= 1:
                winner = active_seats[0].player_id if active_seats else None
                break

            # Blind escalation: 1.5× every BLIND_ESCALATION_EVERY hands
            sb, bb = escalate_blinds(
                hand_count + 1,
                BASE_SB,
                BASE_BB,
                BLIND_ESCALATION_EVERY,
            )

            table.play_hand(
                seats=active_seats,
                small_blind=sb,
                big_blind=bb,
                dealer_index=dealer_index % len(active_seats),
                bot_for={s.player_id: bots[s.player_id] for s in active_seats},
                on_event=None,
                log_decisions=False,
            )

            dealer_index = (dealer_index + 1) % len(active_seats)
            hand_count += 1

            if hand_count > 10_000:      # safety cap
                winner = max(seats, key=lambda s: s.chips).player_id
                break

        # ── Episode bookkeeping ──────────────────────────────────────────────
        if winner and winner in wins:
            wins[winner] += 1

        # ── Periodic save ────────────────────────────────────────────────────
        if episode % save_every == 0:
            bot.save(profile_path)

        # ── Progress report every 1 000 episodes ─────────────────────────────
        if episode % 1_000 == 0:
            s = bot.stats()
            win_rates = "  ".join(
                f"{pid}={wins[pid]/episode:.1%}" for pid in PLAYER_IDS
            )
            print(
                f"  ep={episode:>7}  "
                f"info_sets={s['info_sets']:<7}  "
                f"total_iters={s['total_iterations']:<10}  "
                f"{win_rates}"
            )

    # ── End of training ───────────────────────────────────────────────────────
    bot.save(profile_path)

    final_stats = bot.stats()
    print(f"\n{'=' * 70}")
    print(f"Training complete.")
    print(f"  Episodes:       {num_tournaments}")
    for pid in PLAYER_IDS:
        wr = wins[pid] / num_tournaments if num_tournaments > 0 else 0.0
        print(f"  {pid} wins:      {wins[pid]} / {num_tournaments}  ({wr:.1%})")
    print(f"  Info sets:      {final_stats['info_sets']}")
    print(f"  Total iters:    {final_stats['total_iterations']}")
    print(f"  Profile saved:  {profile_path}")
    print(f"{'=' * 70}")

    return bot


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CFR bot via 4-player deep-stack self-play (MCCFR)"
    )
    parser.add_argument(
        "--tournaments", type=int, default=50_000,
        help="Number of 4-player tournament episodes (default: 50000)"
    )
    parser.add_argument(
        "--chips", type=int, default=1_000,
        help="Starting chips per player (default: 1000)"
    )
    parser.add_argument(
        "--iterations", type=int, default=200,
        help="MCCFR rollouts per decision point (default: 200)"
    )
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="Save regret table every N episodes (default: 500)"
    )
    parser.add_argument(
        "--profile", type=str, default="models/cfr_regret_deep.pkl",
        help="Path for regret-table persistence (default: models/cfr_regret_deep.pkl)"
    )
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    train_cfr_bot_multiway(
        num_tournaments=args.tournaments,
        chips_per_player=args.chips,
        iterations=args.iterations,
        save_every=args.save_every,
        profile_path=args.profile,
    )
