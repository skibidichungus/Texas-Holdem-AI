"""
Texas Hold'em Engine (Stable Fixed Version)
------------------------------------------
Now with:
- Correct indentation
- Safety breaker to avoid infinite loops
- Auto-reset of stacks if fewer than 2 players remain
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque

from .bot_api import Action, PlayerView, BotAdapter

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}
Card = Tuple[str, str]

def full_deck() -> List[Card]:
    return [(r, s) for r in RANKS for s in SUITS]

def approx_score(hole: List[Card], board: List[Card]) -> int:
    cards = hole + board
    ranks = [c[0] for c in cards]
    counts = defaultdict(int)
    for r in ranks:
        counts[r] += 1
    freq = sorted(counts.values(), reverse=True)
    base = sum(RANK_TO_INT[r] for r in ranks)
    if freq[0] == 4:
        base += 200
    elif freq[0] == 3 and 2 in freq:
        base += 180
    elif freq[0] == 3:
        base += 120
    elif freq[0] == 2 and freq.count(2) >= 2:
        base += 80
    elif freq[0] == 2:
        base += 40
    branks = sorted(RANK_TO_INT[r] for r, _ in board)
    gaps = [branks[i + 1] - branks[i] for i in range(len(branks) - 1)] if len(branks) >= 2 else []
    base += 5 * sum(1 for g in gaps if g <= 2)
    return base

@dataclass
class Seat:
    player_id: str
    chips: float
    is_sitting_out: bool = False

class InProcessBot(BotAdapter):
    def __init__(self, bot_obj: Any):
        self.bot = bot_obj

    def act(self, view: PlayerView) -> Action:
        state = {
            "street": view.street,
            "position": view.position,
            "hole_cards": view.hole_cards,
            "board": view.board,
            "pot": view.pot,
            "to_call": view.to_call,
            "min_raise": view.min_raise,
            "max_raise": view.max_raise,
            "legal_actions": view.legal_actions,
            "stacks": view.stacks,
            "me": view.me,
            "opponents": view.opponents,
            "history": view.history,
        }
        a = self.bot.act(state)
        t = a.get("type") if isinstance(a, dict) else getattr(a, "type", None)
        amt = a.get("amount") if isinstance(a, dict) else getattr(a, "amount", None)

        return Action(t, amt)

class RandomBot:
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        legal = state["legal_actions"]
        choice = random.choice(legal)
        if choice["type"] in ("bet", "raise"):
            lo, hi = choice["min"], choice["max"]
            amt = lo + random.random() * max(0.0, hi - lo)
            return {"type": choice["type"], "amount": round(amt, 2)}
        return {"type": choice["type"]}

class Table:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random(7331)

    def play_hand(self, seats: List[Seat | Dict[str, Any]], small_blind: float, big_blind: float,
                  dealer_index: int, bot_for: Dict[str, BotAdapter], on_event=None) -> Dict[str, float]:
        # Normalize seats
        seats = [s if isinstance(s, Seat) else Seat(**s) for s in seats]
        by_pid = {s.player_id: s for s in seats}
        start_chips = {s.player_id: s.chips for s in seats}

        # Ensure at least 2 active players
        active = [s for s in seats if s.chips > 0 and not s.is_sitting_out]
        if len(active) < 2:
            # reset stacks if game needs reset
            for s in seats:
                s.chips = 200
            active = [s for s in seats if s.chips > 0]
        assert 2 <= len(active) <= 10

        # Determine play order (dealer rotation)
        order = deque(range(len(seats)))
        order.rotate(-dealer_index)
        ring = [i for i in order if seats[i].chips > 0 and not seats[i].is_sitting_out]

        # Assign positions
        positions = self._positions(len(ring))
        pos_by_pid = {seats[idx].player_id: pos for pos, idx in zip(positions, ring)}

        # Initialize per-player contributions (for current street)
        contrib = defaultdict(float, {s.player_id: 0.0 for s in seats if not s.is_sitting_out})

        def post_blind(kind: str, seat_index: int, amount: float):
            seat = seats[seat_index]
            amt = min(seat.chips, amount)
            seat.chips -= amt
            contrib[seat.player_id] += amt

        # Post blinds
        if len(ring) == 2:
            sb_idx, bb_idx = ring[0], ring[1]
        else:
            sb_idx, bb_idx = ring[1], ring[2 % len(ring)]
        post_blind("SB", sb_idx, small_blind)
        post_blind("BB", bb_idx, big_blind)

        # Shuffle and deal
        deck = full_deck()
        self.rng.shuffle(deck)
        hole = {seats[idx].player_id: [deck.pop(), deck.pop()] for idx in ring}
        board: List[Card] = []
        history: List[Any] = []

        streets = [
            ("preflop", self._betting_round),
            ("flop", self._deal_flop_then_bet),
            ("turn", self._deal_turn_then_bet),
            ("river", self._deal_river_then_bet),
        ]

        # Total pot accumulated from completed streets
        pot_total = 0.0  # IMPORTANT: start at 0, not blinds

        # --- main street loop ---
        for street_name, fn in streets:
            winner = fn(
                street_name,
                seats,
                ring,
                pos_by_pid,
                hole,
                board,
                contrib,
                pot_total,      # we don't actually mutate this inside
                big_blind,
                bot_for,
                history,
                on_event,
            )

            # If someone wins by everyone else folding
            if isinstance(winner, str):
                total_pot = pot_total + sum(contrib.values())
                # Award full pot to winner
                by_pid[winner].chips += total_pot

                # Compute per-player net change vs start of hand
                return {
                    pid: by_pid[pid].chips - start_chips.get(pid, by_pid[pid].chips)
                    for pid in start_chips
                }

            # No winner yet: move this street's contributions into pot_total
            pot_total += sum(contrib.values())

            # Reset contrib for next street (only active ring players matter)
            contrib = defaultdict(float, {seats[i].player_id: 0.0 for i in ring})

        # --- showdown ---
        total_pot = pot_total + sum(contrib.values())

        # Distribute pot among remaining players
        share_net = self._showdown_and_settle(hole, board, total_pot)

        # Apply showdown results
        for pid, delta in share_net.items():
            by_pid[pid].chips += delta

        # Final per-player net for this hand
        net = {
            pid: by_pid[pid].chips - start_chips.get(pid, by_pid[pid].chips)
            for pid in start_chips
        }
        return net

    def _deal_flop_then_bet(self, *a, **k):
        _, seats, ring, pos_by_pid, hole, board, contrib, pot, bb, bot_for, history, on_event = a
        board.extend([self._pop_card(), self._pop_card(), self._pop_card()])
        return self._betting_round(*a, **k)

    def _deal_turn_then_bet(self, *a, **k):
        _, seats, ring, pos_by_pid, hole, board, contrib, pot, bb, bot_for, history, on_event = a
        board.append(self._pop_card())
        return self._betting_round(*a, **k)

    def _deal_river_then_bet(self, *a, **k):
        _, seats, ring, pos_by_pid, hole, board, contrib, pot, bb, bot_for, history, on_event = a
        board.append(self._pop_card())
        return self._betting_round(*a, **k)

    def _pop_card(self):
        deck = getattr(self, "_deck", None)
        if not deck or len(deck) < 10:
            self._deck = full_deck()
            random.shuffle(self._deck)
            deck = self._deck
        return deck.pop()

    def _betting_round(self, street, seats, ring, pos_by_pid, hole, board, contrib, pot, bb, bot_for, history, on_event):

        if not contrib:
            contrib = defaultdict(float, {s.player_id: 0.0 for s in seats if not s.is_sitting_out})
        else:
            for s in seats:
                contrib.setdefault(s.player_id, 0.0)

        folded = defaultdict(bool)
        allin = defaultdict(bool)
        idx = 0
        safety = 0

        def all_live_equal():
            live_contribs = {
            contrib.get(seats[i].player_id, 0.0)
            for i in ring
            if not folded[seats[i].player_id]
        }
            return len(live_contribs) <= 1

        while True:
            safety += 1
            if safety > 100:
                break

            si = ring[idx]
            seat = seats[si]
            pid = seat.player_id

            # Skip folded or all-in players
            if seat.chips <= 0 or folded[pid] or allin[pid]:
                idx = (idx + 1) % len(ring)
                # End round early if everyone left has equal bets
                if all_live_equal():
                    break
                continue

            to_call = max(0.0, max(contrib.values()) - contrib.get(pid, 0.0))
            min_raise_unit = bb
            max_raise = seat.chips + contrib.get(pid, 0.0)
            legal = []

            if to_call <= 1e-9:
                legal.append({"type": "check"})
                if seat.chips > 0:
                    legal.append({"type": "bet", "min": bb, "max": seat.chips})
            else:
                legal.append({"type": "fold"})
                legal.append({"type": "call"})
                if seat.chips > to_call:
                    legal.append({"type": "raise", "min": to_call + min_raise_unit, "max": seat.chips})

            view = PlayerView(
                me=pid,
                street=street,
                position=pos_by_pid[pid],
                hole_cards=hole[pid],
                board=list(board),
                pot=sum(contrib.values()),
                to_call=to_call,
                min_raise=to_call + min_raise_unit if to_call > 0 else min_raise_unit,
                max_raise=max_raise,
                legal_actions=legal,
                stacks={seats[i].player_id: seats[i].chips for i in ring},
                opponents=[seats[i].player_id for i in ring if seats[i].player_id != pid],
                history=list(history)
            )

            action = bot_for[pid].act(view)

            # --- Handle chosen action ---
            if action.type == "fold":
                folded[pid] = True
                # Mark player as no longer eligible for showdown
                hole[pid] = []
            elif action.type == "call":
                need = min(seat.chips, to_call)
                seat.chips -= need
                contrib[pid] += need
                if seat.chips <= 0:
                    allin[pid] = True
            elif action.type in ("bet", "raise"):
                amt = min(max(action.amount or 0.0, bb), seat.chips + contrib.get(pid, 0.0))
                need = max(0.0, amt - contrib.get(pid, 0.0))
                seat.chips -= need
                contrib[pid] += need
                if seat.chips <= 0:
                    allin[pid] = True

            # End round if everyone left has matched contributions <-
            if all_live_equal:
                break

            idx = (idx + 1) % len(ring)

        alive = [seats[i].player_id for i in ring if not folded[seats[i].player_id]]
        if len(alive) == 1:
            return alive[0]
        return None


    def _showdown_and_settle(self, hole, board, total_pot):
        # If no money in pot, nothing to pay
        if total_pot <= 0:
            return {pid: 0.0 for pid in hole}

        # Filter out players who folded or never acted
        eligible = {pid: cards for pid, cards in hole.items() if cards and len(cards) == 2}
        if not eligible:
            return {pid: 0.0 for pid in hole}

        # Compute rough hand strength
        ranks = {pid: approx_score(cards, board) for pid, cards in eligible.items()}
        best = max(ranks.values())
        winners = [pid for pid, r in ranks.items() if r == best]

        # Split pot among winners
        share = total_pot / len(winners)
        net = {pid: 0.0 for pid in hole}
        for w in winners:
            net[w] += share
        return net



    def _positions(self, n):
        if n == 2:
            return ["BTN", "BB"]
        tags = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO"]
        return tags[:n]

class TournamentManager:
    def __init__(self, table: Table):
        self.table = table

    def run(self, seats, bot_for, small_blind, big_blind, hands, dealer_index=0, on_event=None):
        seats = [s if isinstance(s, Seat) else Seat(**s) for s in seats]
        n = len(seats)
        dealer = dealer_index
        nets = defaultdict(float)
        for _ in range(hands):
            res = self.table.play_hand(seats, small_blind, big_blind, dealer, bot_for, on_event=on_event)
            for pid, v in res.items():
                nets[pid] += v
            dealer = (dealer + 1) % n
        return dict(nets)

if __name__ == "__main__":
    from .bot_api import PlayerView
    seats = [Seat(player_id=f"P{i+1}", chips=200) for i in range(3)]
    bots = {s.player_id: InProcessBot(RandomBot()) for s in seats}
    tbl = Table()
    tm = TournamentManager(tbl)
    nets = tm.run(seats, bots, 1, 2, 5)
    print("Final stacks:")
    for s in seats:
        print(f"  {s.player_id}: {s.chips:.2f}")
    print("Net chips:")
    for pid, v in nets.items():
        print(f"  {pid}: {v:+.2f}")