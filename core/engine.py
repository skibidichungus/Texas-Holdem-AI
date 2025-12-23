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
from core.logger import DecisionLogger # imports logger.py

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
    # Safety check: only sum valid ranks
    base = sum(RANK_TO_INT.get(r, 0) for r in ranks)
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
    # Safety check for board ranks
    branks = sorted(RANK_TO_INT.get(r, 0) for r, _ in board)
    gaps = [branks[i + 1] - branks[i] for i in range(len(branks) - 1)] if len(branks) >= 2 else []
    base += 5 * sum(1 for g in gaps if g <= 2)
    return base

def _compute_to_call(contrib, alive_pids, pid):
    """
    contrib: dict[player_id] -> contribution this street
    alive_pids: list of players still in the hand (not folded)
    pid: acting player

    Returns (to_call, highest_contrib).
    """
    highest = 0.0
    for p in alive_pids:
        c = contrib.get(p, 0.0)
        if c > highest:
            highest = c

    player_c = contrib.get(pid, 0.0)
    to_call = highest - player_c
    if to_call < 0:
        to_call = 0.0
    return to_call, highest

def _legal_actions_for(
    pid,
    seat,          # seat object for this player
    contrib,       # dict[player_id] -> contrib this street
    alive_pids,    # list of active players
    big_blind,     # numeric BB
):
    """
    Returns a list of action dicts like:
      {"type": "check"}
      {"type": "bet", "min": X, "max": Y}
      {"type": "call"}
      {"type": "raise", "min": X, "max": Y}
      {"type": "fold"}
    """
    legal = []
    to_call, highest = _compute_to_call(contrib, alive_pids, pid)
    chips = seat.chips

    # === CASE: player is NOT facing a bet (to_call == 0) ===
    if abs(to_call) <= 1e-9:
        # CASE A: no bet at all on this street yet (everyone at 0)
        everyone_zero = all(contrib.get(p, 0.0) == 0.0 for p in alive_pids)
        if everyone_zero:
            # Check or new bet
            legal.append({"type": "check"})
            if chips > 0:
                legal.append({"type": "bet", "min": big_blind, "max": chips})
        else:
            # CASE B: bet exists but this player has already matched it.
            # They may check, but NOT bet again at same level.
            legal.append({"type": "check"})

    # === CASE: facing a bet (to_call > 0) ===
    else:
        # fold is always legal
        legal.append({"type": "fold"})

        # call (possibly all-in)
        if chips <= to_call:
            # calling puts them all in
            legal.append({"type": "call"})  # your engine can interpret as all-in
        else:
            legal.append({"type": "call"})

            # raise only if they have more than to_call
            # New *total* contribution must be at least (highest + big_blind)
            min_total = highest + big_blind
            max_total = contrib.get(pid, 0.0) + chips  # everything they have

            if min_total > contrib.get(pid, 0.0) and max_total > min_total:
                legal.append({"type": "raise", "min": min_total, "max": max_total})

    return legal

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
        self.hand_counter = 0

    def play_hand(self, seats: List[Seat | Dict[str, Any]], small_blind: float, big_blind: float,
                  dealer_index: int, bot_for: Dict[str, BotAdapter], on_event=None) -> Dict[str, float]:
        
        logger = DecisionLogger(enabled=True)

        # NEW — set hand ID
        logger.start_hand(self.hand_counter)
        self.hand_counter += 1

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
        running_pot = 0.0

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
                pot_total,
                big_blind,
                bot_for,
                history,
                on_event,
                logger=logger,   # <===== PASS LOGGER HERE
            )

            # If someone wins by everyone else folding
            if isinstance(winner, str):
                total_pot = pot_total + sum(contrib.values())
                by_pid[winner].chips += total_pot

                return {
                    pid: by_pid[pid].chips - start_chips.get(pid, by_pid[pid].chips)
                    for pid in start_chips
                }

            # No winner yet — move this street's contribs into pot_total
            pot_total += sum(contrib.values())

            # Reset contrib for next street
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

        # Log final results for ML training
        for pid, delta in net.items():
            logger.log_result(pid, delta)

        logger.flush()

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

    def _betting_round(
        self, street, seats, ring, pos_by_pid, hole, board, contrib, pot, bb,
        bot_for, history, on_event, logger
    ):
        print(f"\n=== BETTING ROUND START: {street} ===")
        print(f"Pot before street: {pot:.2f}")
        print("Ring order:", [seats[i].player_id for i in ring])
        print("Initial contrib:", {s.player_id: float(contrib.get(s.player_id, 0.0)) for s in seats})

        # Ensure contrib entries
        if not contrib:
            contrib = defaultdict(float, {s.player_id: 0.0 for s in seats if not s.is_sitting_out})
        else:
            for s in seats:
                contrib.setdefault(s.player_id, 0.0)

        folded = defaultdict(bool)
        allin = defaultdict(bool)

        current_bet = max(contrib.values()) if contrib else 0.0
        last_raise_size = bb if current_bet > 0 else bb

        # ---- helpers ----
        def num_players_can_act():
            cnt = 0
            for i in ring:
                s = seats[i]
                pid = s.player_id
                if folded[pid] or allin[pid] or s.chips <= 0:
                    continue
                cnt += 1
            return cnt

        def all_live_equal():
            live = []
            contribs = set()
            for i in ring:
                s = seats[i]
                pid = s.player_id
                if folded[pid]:
                    continue
                if allin[pid]:
                    continue
                if s.chips <= 0:
                    allin[pid] = True
                    continue
                live.append(pid)
                contribs.add(float(contrib[pid]))
            if len(live) <= 1:
                return True
            return len(contribs) == 1

        idx = 0
        safety = 0

        # -------- MAIN LOOP ----------
        while True:
            safety += 1
            if safety > 500:
                print("!!! SAFETY BREAK in betting_round")
                break

            if num_players_can_act() == 0:
                print("No players able to act → ending round")
                break

            si = ring[idx]
            seat = seats[si]
            pid = seat.player_id

            # Skip dead players
            if folded[pid] or allin[pid] or seat.chips <= 0:
                idx = (idx + 1) % len(ring)
                if all_live_equal():
                    print("All live equal → ending street")
                    break
                continue

            # Recompute current bet / call amount
            current_bet = max(contrib.values()) if contrib else 0.0
            to_call = max(0.0, current_bet - contrib[pid])

            # LEGAL ACTIONS
            legal = []
            if to_call <= 1e-9:
                legal.append({"type": "check"})
                if seat.chips > 0:
                    min_bet = min(bb, seat.chips)
                    max_bet = seat.chips
                    if max_bet >= min_bet:
                        legal.append({"type": "bet", "min": float(min_bet), "max": float(max_bet)})
            else:
                legal.append({"type": "fold"})
                call_amt = min(seat.chips, to_call)
                if call_amt > 0:
                    legal.append({"type": "call"})
                if seat.chips > to_call:
                    max_total = seat.chips + contrib[pid]
                    min_total = current_bet + last_raise_size
                    min_total = max(min_total, current_bet + bb)
                    if max_total + 1e-9 >= min_total:
                        legal.append({
                            "type": "raise",
                            "min": float(min_total),
                            "max": float(max_total),
                        })

            print(f"[{street}] Acting: {pid} | chips={seat.chips} contrib={contrib[pid]} to_call={to_call}")
            print("    Legal:", legal)

            # PlayerView
            if to_call <= 1e-9:
                pv_min_raise = bb
                pv_max_raise = seat.chips
            else:
                pv_min_raise = max(0.0, (current_bet + last_raise_size) - contrib[pid])
                pv_max_raise = seat.chips

            view = PlayerView(
                me=pid,
                street=street,
                position=pos_by_pid[pid],
                hole_cards=hole[pid],
                board=list(board),
                pot=pot + sum(contrib.values()),
                to_call=to_call,
                min_raise=pv_min_raise,
                max_raise=pv_max_raise,
                legal_actions=legal,
                stacks={seats[i].player_id: seats[i].chips for i in ring},
                opponents=[seats[i].player_id for i in ring if seats[i].player_id != pid],
                history=list(history),
            )

            # BOT ACTION
            raw_action = bot_for[pid].act(view)

            # ---- ML LOGGING (only here, once per actual action) ----
            if logger is not None:
                logger.log_decision({
                    "player": pid,
                    "street": street,
                    "hole": hole[pid],
                    "board": list(board),
                    "pot": view.pot,
                    "to_call": to_call,
                    "legal": view.legal_actions,
                    "chosen_action": {"type": raw_action.type, "amount": raw_action.amount},
                    "stacks": view.stacks,
                    "opponents": view.opponents,
                    "folded": False,
                    "hand_id": getattr(self, "hand_index", None)
                })

            # Sanitize illegal action type
            legal_types = {a["type"] for a in legal}
            action_type = raw_action.type
            action_amt = raw_action.amount

            if action_type not in legal_types:
                print(f"    [WARN] Illegal action '{action_type}', fixing...")
                if "call" in legal_types:
                    action_type = "call"; action_amt = None
                elif "check" in legal_types:
                    action_type = "check"; action_amt = None
                else:
                    action_type = "fold"; action_amt = None

            # Sanitize bet/raise amount
            if action_type in ("bet", "raise"):
                spec = next(a for a in legal if a["type"] == action_type)
                lo, hi = spec["min"], spec["max"]
                if action_amt is None:
                    action_amt = lo
                amt = float(action_amt)
                if amt < lo: amt = lo
                if amt > hi: amt = hi
                action_amt = amt
            else:
                action_amt = None

            action = Action(action_type, action_amt)
            print(f"    Chosen action: {action.type} {action.amount}")

            # Add to history BEFORE modifying contrib
            history.append({
                "street": street,
                "pid": pid,
                "type": action.type,
                "amount": action.amount,
                "to_call_before": to_call,
            })

            # APPLY ACTION
            if action.type == "fold":
                folded[pid] = True
                hole[pid] = []
                print(f"    {pid} FOLDS")

            elif action.type == "call":
                need = min(seat.chips, to_call)
                seat.chips -= need
                contrib[pid] += need
                if seat.chips <= 0:
                    allin[pid] = True
                print(f"    {pid} CALLS {need}")

            elif action.type == "check":
                print(f"    {pid} CHECKS")

            elif action.type in ("bet", "raise"):
                prev_bet = max(contrib.values())
                target_total = float(action.amount or 0.0)
                need = max(0.0, target_total - contrib[pid])
                if need > seat.chips:
                    need = seat.chips
                    target_total = contrib[pid] + need

                seat.chips -= need
                contrib[pid] += need
                if seat.chips <= 0:
                    allin[pid] = True

                new_bet = max(contrib.values())
                if street == "preflop" and prev_bet == 0.0:
                    last_raise_size = new_bet
                else:
                    raise_sz = new_bet - prev_bet
                    if raise_sz > 1e-9:
                        last_raise_size = raise_sz

                print(f"    {pid} {action.type.upper()} to {target_total} (paid {need})")

            if all_live_equal():
                print("All live equal → ending street")
                break
            if num_players_can_act() == 0:
                print("No one left who can act → ending street")
                break

            idx = (idx + 1) % len(ring)

        alive = [seats[i].player_id for i in ring if not folded[seats[i].player_id]]
        print(f"=== BETTING ROUND END: {street} | Alive: {alive} ===")

        if len(alive) == 1:
            print(f"--> Winner by fold on {street}: {alive[0]}")
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