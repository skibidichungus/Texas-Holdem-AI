# bots/poker_mind_bot.py

import random
from typing import Dict, Any
from core.engine import approx_score


class PokerMindBot:
    """
    Fixed Smart Bot — now respects to_call properly.

    Prevents infinite loops by:
    - never betting when facing a bet
    - never checking when check is illegal
    - choosing valid, legal actions based on to_call logic
    """

    def __init__(self):
        self.strong_threshold = 40
        self.medium_threshold = 20
        self.bluff_chance = 0.10

    def act(self, state: Dict[str, Any]):
        legal = state["legal_actions"]
        hole = state["hole_cards"]
        board = state["board"]
        pot = state["pot"]
        to_call = state["to_call"]
        me = state["me"]

        # If already folded
        if not hole:
            return {"type": "check"}

        street = state["street"]

        # ====== PRE-FLOP ======
        if street == "preflop":
            ranks = [c[0] for c in hole]
            suits = [c[1] for c in hole]
            order = "23456789TJQKA"
            r1 = order.index(ranks[0])
            r2 = order.index(ranks[1])
            high = max(r1, r2)
            suited = suits[0] == suits[1]

            # Facing a bet?
            if to_call > 0:
                # Strong → raise
                if set(ranks) in ({"A","K"}, {"A","Q"}) or ranks in (["A","A"],["K","K"],["Q","Q"],["J","J"]):
                    for a in legal:
                        if a["type"] == "raise":
                            amt = max(a["min"], min(a["max"], 6.0))
                            return {"type": "raise", "amount": amt}
                    # fallback:
                    return {"type": "call"}

                # Decent → call if cheap
                if high >= order.index("T") or (suited and high >= order.index("9")) or ranks[0]==ranks[1]:
                    for a in legal:
                        if a["type"] == "call":
                            return {"type": "call"}
                    return {"type": "fold"}

                # Weak → fold
                return {"type": "fold"}

            else:
                # No bet facing us → CHECK or BET
                if ranks in (["A","A"],["K","K"],["Q","Q"],["J","J"]) or set(ranks) in ({"A","K"}, {"A","Q"}):
                    # Raise first-in
                    for a in legal:
                        if a["type"] in ("bet","raise"):
                            amt = max(a["min"], min(a["max"], 6.0))
                            return {"type": a["type"], "amount": amt}
                # Otherwise check
                return {"type": "check"}

        # ====== POST-FLOP ======
        strength = approx_score(hole, board)
        bluff = random.random() < self.bluff_chance

        # Facing a bet?
        if to_call > 0:
            # Weak hand → fold
            if strength < self.medium_threshold and not bluff:
                for a in legal:
                    if a["type"] == "fold":
                        return {"type": "fold"}
                # fallback (call if fold missing)
                return {"type": "call"}

            # Medium → call if reasonable
            call_limit = pot * 0.25
            if to_call <= call_limit or strength >= self.medium_threshold:
                for a in legal:
                    if a["type"] == "call":
                        return {"type": "call"}

            # Strong → raise
            if strength >= self.strong_threshold or bluff:
                for a in legal:
                    if a["type"] == "raise":
                        target = pot * 0.5
                        amt = max(a["min"], min(a["max"], target))
                        return {"type": "raise", "amount": amt}

            # Fallback → call
            return {"type": "call"}

        # Not facing a bet → we can check or bet
        if strength >= self.strong_threshold or bluff:
            # Bet 50% pot
            for a in legal:
                if a["type"] == "bet":
                    target = pot * 0.5
                    amt = max(a["min"], min(a["max"], target))
                    return {"type": "bet", "amount": amt}

        # Medium → check
        if strength >= self.medium_threshold:
            for a in legal:
                if a["type"] == "check":
                    return {"type": "check"}

        # Weak → check
        for a in legal:
            if a["type"] == "check":
                return {"type": "check"}

        # Fallback (very rare)
        return {"type": "fold"}


# Backward alias
SmartBot = PokerMindBot
