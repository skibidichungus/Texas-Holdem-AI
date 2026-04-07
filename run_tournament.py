"""
Texas Hold'em Tournament UI
----------------------------
Run this file to open the tournament viewer.
Click "Play" to start the game and watch chip stacks update live.

Speed controls:
  + / =   Speed up (less delay between hands)
  - / _   Slow down (more delay)
  Space   Pause / resume
"""

import sys
import time
import argparse
import threading
import matplotlib

# Auto-detect a working interactive backend
if sys.platform == "darwin":
    try:
        matplotlib.use("macosx")
    except Exception:
        matplotlib.use("TkAgg")
else:
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg")
    except ImportError:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

from core.engine import Table, Seat
from bots import parse_players, escalate_blinds

# ─── DEFAULTS ────────────────────────────────────────────────────────────────

DEFAULT_PLAYERS = "mc200,smart,ml,rl,cfr,icm,exploitative,gto,opponentmodel"
DEFAULT_CHIPS   = 1000
DEFAULT_SB      = 5
DEFAULT_BB      = 10
DEFAULT_DELAY   = 0.05
DEFAULT_BLIND_INCREASE_EVERY = 50

COLOURS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
           "#9C27B0", "#00BCD4", "#E91E63", "#8BC34A"]

SPEED_STEPS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]

# ─────────────────────────────────────────────────────────────────────────────


class TournamentUI:
    def __init__(self, players, starting_chips, base_sb, base_bb,
                 hand_delay, blind_increase_every):
        self.player_specs = players  # [(pid, btype, adapter), ...]
        self.player_ids   = [pid for pid, _, _ in players]
        self.bot_types    = {pid: btype for pid, btype, _ in players}
        self.bots         = {pid: adapter for pid, _, adapter in players}
        self.colours      = {pid: COLOURS[i % len(COLOURS)]
                             for i, pid in enumerate(self.player_ids)}

        self.starting_chips = starting_chips
        self.base_sb = base_sb
        self.base_bb = base_bb
        self.blind_increase_every = blind_increase_every

        self.hand_delay   = hand_delay
        self._speed_idx   = SPEED_STEPS.index(hand_delay) if hand_delay in SPEED_STEPS else 3
        self._paused      = False

        self.chip_history: list[dict] = []
        self.running      = False
        self.finished     = False
        self._current_blinds = (base_sb, base_bb)
        self._cancel_event   = threading.Event()
        self._cancelled      = False
        self._btn_mode       = "play"  # "play" | "cancel" | "restart"
        self._eliminations   = {}  # pid -> finishing position

        self._build_figure()

    # ── Figure setup ──────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 7), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("Texas Hold'em Tournament")

        # Main chart area (narrowed to make room for leaderboard sidebar)
        self.ax = self.fig.add_axes([0.08, 0.18, 0.57, 0.72])
        self.ax.set_facecolor("#16213e")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#444")
        self.ax.set_xlabel("Hand", fontsize=12)
        self.ax.set_ylabel("Chips", fontsize=12)
        self.ax.set_title("Texas Hold'em Tournament", fontsize=15,
                          fontweight="bold", pad=12)
        self.ax.grid(True, alpha=0.15, color="white")

        # Draw initial flat lines at starting chips
        self.lines = {}
        for pid in self.player_ids:
            label = f"{pid} ({self.bot_types[pid]})"
            line, = self.ax.plot(
                [0], [self.starting_chips],
                label=label,
                color=self.colours[pid],
                linewidth=2.5,
                alpha=0.9,
            )
            self.lines[pid] = line

        total_chips = self.starting_chips * len(self.player_ids)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, total_chips * 1.05)
        self.ax.legend(loc="upper left", facecolor="#1a1a2e",
                       labelcolor="white", edgecolor="#444", fontsize=9)

        # Leaderboard sidebar
        self.lb_ax = self.fig.add_axes([0.69, 0.18, 0.29, 0.72])
        self.lb_ax.set_facecolor("#16213e")
        self.lb_ax.set_xticks([])
        self.lb_ax.set_yticks([])
        for spine in self.lb_ax.spines.values():
            spine.set_visible(False)
        self._draw_leaderboard()

        # Status label
        self.status_text = self.fig.text(
            0.5, 0.105, "Press  Play  to start  |  +/- speed  |  Space pause",
            ha="center", va="center", fontsize=13,
            color="#aaaaaa", style="italic",
        )

        # Blinds label (top-right of chart area)
        self.blinds_text = self.ax.text(
            0.98, 0.97,
            f"Blinds: {self.base_sb}/{self.base_bb}",
            transform=self.ax.transAxes,
            ha="right", va="top", fontsize=11,
            color="#ffcc00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor="#ffcc00", alpha=0.8),
        )

        # Play button
        btn_ax = self.fig.add_axes([0.42, 0.02, 0.16, 0.07])
        self.play_btn = Button(
            btn_ax, "Play",
            color="#0f3460", hovercolor="#e94560",
        )
        self.play_btn.label.set_color("white")
        self.play_btn.label.set_fontsize(13)
        self.play_btn.on_clicked(self._on_button_click)

        # Keyboard handler
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Keyboard / speed control ──────────────────────────────────────────────

    def _on_key(self, event):
        if event.key in ("+", "="):
            self._speed_idx = max(0, self._speed_idx - 1)
            self.hand_delay = SPEED_STEPS[self._speed_idx]
            self._update_speed_label()
        elif event.key in ("-", "_"):
            self._speed_idx = min(len(SPEED_STEPS) - 1, self._speed_idx + 1)
            self.hand_delay = SPEED_STEPS[self._speed_idx]
            self._update_speed_label()
        elif event.key == " ":
            self._paused = not self._paused
            self._update_speed_label()

    def _update_speed_label(self):
        if self._paused:
            state = "PAUSED"
        else:
            state = f"delay={self.hand_delay:.2f}s"
        self.status_text.set_text(f"Running  |  {state}  |  +/- speed  |  Space pause")
        self.fig.canvas.draw_idle()

    # ── Button dispatcher ─────────────────────────────────────────────────────

    def _on_button_click(self, event=None):
        if self._btn_mode == "play":
            self._start_tournament()
        elif self._btn_mode == "cancel":
            self._request_cancel()
        elif self._btn_mode == "restart":
            self._reset_to_play()

    def _start_tournament(self):
        if self.running or self.finished:
            return
        self._cancel_event.clear()
        self._cancelled = False
        self.running = True
        self._dirty = False
        self._winner_info = None
        # Switch to Cancel mode
        self._btn_mode = "cancel"
        self.play_btn.label.set_text("Cancel")
        self.play_btn.color = "#7a1a1a"
        self.play_btn.hovercolor = "#c0392b"
        self.play_btn.ax.set_facecolor("#7a1a1a")
        self.status_text.set_text("Tournament in progress...  |  +/- speed  |  Space pause")
        self.status_text.set_color("#aaaaaa")
        self.status_text.set_style("italic")
        self.fig.canvas.draw_idle()

        self._timer = self.fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._poll_redraw)
        self._timer.start()

        t = threading.Thread(target=self._run_tournament, daemon=True)
        t.start()

    def _request_cancel(self):
        """Signal the tournament thread to stop; UI reset happens in _poll_redraw."""
        self._cancel_event.set()
        self.play_btn.label.set_text("Stopping...")
        self.play_btn.color = "#444444"
        self.play_btn.ax.set_facecolor("#444444")
        self.status_text.set_text("Cancelling...")
        self.fig.canvas.draw_idle()

    def _reset_to_play(self):
        """Reset all state and restore the chart to the initial ready state."""
        # Clear data state first so leaderboard reads clean values
        self.chip_history  = []
        self._eliminations = {}
        self.running       = False
        self.finished      = False
        self._cancelled    = False
        self._cancel_event.clear()
        self._winner_info  = None
        self._current_blinds = (self.base_sb, self.base_bb)
        # Restore button to Play
        self._btn_mode = "play"
        self.play_btn.label.set_text("Play")
        self.play_btn.color = "#0f3460"
        self.play_btn.hovercolor = "#e94560"
        self.play_btn.ax.set_facecolor("#0f3460")
        # Restore status text
        self.status_text.set_text(
            "Press  Play  to start  |  +/- speed  |  Space pause")
        self.status_text.set_color("#aaaaaa")
        self.status_text.set_style("italic")
        # Restore chart lines to flat starting chips
        for pid, line in self.lines.items():
            line.set_data([0], [self.starting_chips])
        total_chips = self.starting_chips * len(self.player_ids)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, total_chips * 1.05)
        self.blinds_text.set_text(
            f"Blinds: {self.base_sb}/{self.base_bb}")
        # Explicitly wipe the leaderboard axes so no stale artists remain,
        # then re-apply sidebar styling before the fresh draw
        self.lb_ax.clear()
        self.lb_ax.set_facecolor("#16213e")
        self.lb_ax.set_xticks([])
        self.lb_ax.set_yticks([])
        for spine in self.lb_ax.spines.values():
            spine.set_visible(False)
        self._draw_leaderboard()
        self.fig.canvas.draw_idle()

    # ── Tournament loop ───────────────────────────────────────────────────────

    def _run_tournament(self):
        seats = [Seat(player_id=pid, chips=self.starting_chips)
                 for pid in self.player_ids]
        by_pid = {s.player_id: s for s in seats}

        snap = {"hand": 0, **{pid: self.starting_chips for pid in self.player_ids}}
        self.chip_history.append(snap)
        self._mark_dirty()

        active_seats = list(seats)
        table        = Table()
        dealer       = 0
        hand_num     = 0
        finishing     = []
        total        = len(seats)

        while len(active_seats) > 1:
            # Cancellation check (runs before every hand)
            if self._cancel_event.is_set():
                self._signal_cancelled()
                return
            # Pause loop — also exits immediately on cancel
            while self._paused and not self._cancel_event.is_set():
                time.sleep(0.05)

            hand_num += 1
            sb, bb = escalate_blinds(hand_num, self.base_sb, self.base_bb,
                                     self.blind_increase_every)
            self._current_blinds = (sb, bb)
            dealer_i = dealer % len(active_seats)

            active_bots = {s.player_id: self.bots[s.player_id]
                           for s in active_seats}

            try:
                table.play_hand(
                    active_seats, sb, bb,
                    dealer_i, active_bots,
                )
            except Exception as e:
                print(f"[hand {hand_num}] error: {e}")
                break

            snap = {"hand": hand_num}
            for pid in self.player_ids:
                snap[pid] = by_pid[pid].chips
            self.chip_history.append(snap)
            self._mark_dirty()

            eliminated = [s for s in active_seats if s.chips <= 0]
            for s in eliminated:
                pos = total - len(finishing)
                finishing.append((s.player_id, pos))
                self._eliminations[s.player_id] = pos
                active_seats.remove(s)
                print(f"  [OUT] {s.player_id} — position {pos}")

            dealer = (dealer + 1) % max(len(active_seats), 1)
            time.sleep(self.hand_delay)

        if active_seats:
            finishing.append((active_seats[0].player_id, 1))
            winner = active_seats[0].player_id
        else:
            winner = "?"

        self._signal_finish(winner, hand_num)

    # ── Leaderboard ───────────────────────────────────────────────────────────

    _ORDINALS = ["1st", "2nd", "3rd", "4th", "5th",
                 "6th", "7th", "8th", "9th", "10th"]

    def _ordinal(self, n: int) -> str:
        if 1 <= n <= len(self._ORDINALS):
            return self._ORDINALS[n - 1]
        return f"{n}th"

    def _draw_leaderboard(self):
        ax = self.lb_ax
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Title
        ax.text(0.5, 0.97, "Leaderboard",
                ha="center", va="top",
                fontsize=12, fontweight="bold", color="white",
                transform=ax.transAxes)

        # Separator line under title
        ax.axhline(y=0.93, xmin=0.05, xmax=0.95,
                   color="#444", linewidth=0.8)

        # Build current chip snapshot
        if self.chip_history:
            snap = self.chip_history[-1]
        else:
            snap = {pid: self.starting_chips for pid in self.player_ids}

        total_chips = self.starting_chips * len(self.player_ids)

        # Sort: active players descending by chips, then eliminated by position asc
        active  = [(pid, snap.get(pid, 0))
                   for pid in self.player_ids
                   if pid not in self._eliminations]
        active.sort(key=lambda x: x[1], reverse=True)

        eliminated = [(pid, self._eliminations[pid])
                      for pid in self.player_ids
                      if pid in self._eliminations]
        eliminated.sort(key=lambda x: x[1])  # best finish first

        rows = [(pid, snap.get(pid, 0), False) for pid, _ in active] + \
               [(pid, 0,                True)  for pid, _ in eliminated]

        n = len(rows)
        # Vertical layout: spread rows between y=0.90 and y=0.02
        row_h = 0.88 / max(n, 1)

        for rank, (pid, chips, is_out) in enumerate(rows, start=1):
            y_center = 0.90 - (rank - 0.5) * row_h
            color    = self._eliminations_color(pid, is_out)
            btype    = self.bot_types[pid]

            # Rank label
            ax.text(0.03, y_center, self._ordinal(rank),
                    ha="left", va="center",
                    fontsize=8, color="#888888",
                    transform=ax.transAxes)

            # Colored dot
            dot_color = color if not is_out else "#555555"
            ax.plot(0.18, y_center, "s",
                    color=dot_color, markersize=7,
                    transform=ax.transAxes, clip_on=False)

            # Player name + bot type
            name_color = color if not is_out else "#555555"
            ax.text(0.25, y_center, f"{pid} ({btype})",
                    ha="left", va="center",
                    fontsize=8, color=name_color,
                    transform=ax.transAxes)

            if is_out:
                # Finishing position
                elim_pos = self._eliminations[pid]
                ax.text(0.97, y_center,
                        f"OUT ({self._ordinal(elim_pos)})",
                        ha="right", va="center",
                        fontsize=7.5, color="#555555",
                        transform=ax.transAxes)
            else:
                # Chip count
                ax.text(0.97, y_center + row_h * 0.18,
                        f"{chips:,}",
                        ha="right", va="center",
                        fontsize=8, color="white",
                        transform=ax.transAxes)
                # Progress bar
                bar_w = max(chips / total_chips, 0.0) * 0.72
                bar_y = y_center - row_h * 0.25
                bar_h = row_h * 0.18
                # Background track
                ax.add_patch(mpatches.FancyBboxPatch(
                    (0.25, bar_y), 0.72, bar_h,
                    boxstyle="round,pad=0",
                    facecolor="#2a2a4a", edgecolor="none",
                    transform=ax.transAxes, clip_on=False))
                # Fill
                if bar_w > 0:
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (0.25, bar_y), bar_w, bar_h,
                        boxstyle="round,pad=0",
                        facecolor=self.colours[pid],
                        edgecolor="none", alpha=0.75,
                        transform=ax.transAxes, clip_on=False))

    def _eliminations_color(self, pid: str, is_out: bool) -> str:
        return self.colours.get(pid, "#ffffff")

    # ── Data flag ─────────────────────────────────────────────────────────────

    def _mark_dirty(self):
        self._dirty = True

    def _signal_cancelled(self):
        """Called from the tournament thread when the cancel event fires."""
        self._cancelled = True
        self._dirty = True

    def _signal_finish(self, winner: str, hands_played: int):
        self._winner_info = (winner, hands_played)
        self._dirty = True

    # ── Main-thread timer callback ────────────────────────────────────────────

    def _poll_redraw(self):
        if not self._dirty:
            return
        self._dirty = False

        hands = [e["hand"] for e in self.chip_history]
        for pid, line in self.lines.items():
            y = [e.get(pid, 0) for e in self.chip_history]
            line.set_data(hands, y)
        self.ax.set_xlim(0, max(hands) + 1 if hands else 10)

        # Update blinds display
        sb, bb = self._current_blinds
        self.blinds_text.set_text(f"Blinds: {sb}/{bb}")

        # Cancellation: reset everything back to ready state
        if self._cancelled:
            self._timer.stop()
            self.running = False
            self._reset_to_play()
            return

        if self._winner_info is not None:
            winner, hands_played = self._winner_info
            self._timer.stop()
            self.running = False
            self.finished = True
            colour = self.colours.get(winner, "white")
            self.status_text.set_text(
                f"{winner} wins!   ({hands_played} hands played)")
            self.status_text.set_color(colour)
            self.status_text.set_style("normal")
            # Switch to Restart mode
            self._btn_mode = "restart"
            self.play_btn.label.set_text("Restart")
            self.play_btn.color = "#0f3460"
            self.play_btn.hovercolor = "#e94560"
            self.play_btn.ax.set_facecolor("#0f3460")
            print(f"\nWinner: {winner}  ({hands_played} hands)")

        self._draw_leaderboard()
        self.fig.canvas.draw_idle()

    # ── Entry point ───────────────────────────────────────────────────────────

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Texas Hold'em tournament viewer")
    parser.add_argument("--players", type=str, default=DEFAULT_PLAYERS,
                        help=f"Comma-separated bot types (default: {DEFAULT_PLAYERS})")
    parser.add_argument("--chips", type=int, default=DEFAULT_CHIPS,
                        help=f"Starting chips (default: {DEFAULT_CHIPS})")
    parser.add_argument("--sb", type=int, default=DEFAULT_SB,
                        help=f"Starting small blind (default: {DEFAULT_SB})")
    parser.add_argument("--bb", type=int, default=DEFAULT_BB,
                        help=f"Starting big blind (default: {DEFAULT_BB})")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Initial delay between hands in seconds (default: {DEFAULT_DELAY})")
    parser.add_argument("--blind-increase-every", type=int,
                        default=DEFAULT_BLIND_INCREASE_EVERY,
                        help="Increase blinds 1.5x every N hands, 0 to disable (default: 50)")
    parser.add_argument("--rl_model", type=str, default=None,
                        help="Path to RL model weights (e.g. models/rl_model_run3.pt). "
                             "Rewrites any 'rl' entry in --players to use this model.")
    args = parser.parse_args()

    if args.rl_model:
        import re
        args.players = re.sub(r'(?<![:\w])rl(?![\w:])', f'rl:{args.rl_model}', args.players)

    players = parse_players(args.players)
    if len(players) < 2:
        print("Error: need at least 2 players. Check your --players spec.")
        return

    print(f"Players: {', '.join(f'{pid}={btype}' for pid, btype, _ in players)}")
    print(f"Chips: {args.chips}  |  Blinds: {args.sb}/{args.bb}  |  "
          f"Escalation every {args.blind_increase_every} hands")

    ui = TournamentUI(
        players=players,
        starting_chips=args.chips,
        base_sb=args.sb,
        base_bb=args.bb,
        hand_delay=args.delay,
        blind_increase_every=args.blind_increase_every,
    )
    ui.show()


if __name__ == "__main__":
    main()
