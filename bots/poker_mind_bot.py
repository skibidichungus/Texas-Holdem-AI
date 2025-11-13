# ai_bot/poker_mind_bot.py
import random
from typing import Dict, Any
from core.engine import approx_score


class SmartBot:
    def __init__(self):
        self.strong_threshold = 40
        self.medium_threshold = 20
        self.bluff_chance = 0.1  # 10% chance to bluff randomly

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        legal = state["legal_actions"]
        hole = state["hole_cards"]
        board = state["board"]
        pot = state["pot"]
        to_call = state["to_call"]
        stacks = state["stacks"]
        me = state["me"]

        # Player has folded earlier — engine marks hole_cards = []
        if not state["hole_cards"]:
            return {"type": "check"}  # engine ignores it anyway
        
        # -----------------------------
        # PRE-FLOP STRATEGY SECTION
        # -----------------------------
        if state["street"] == "preflop":
            ranks = [c[0] for c in hole]
            suits = [c[1] for c in hole]
            order = "23456789TJQKA"

            r1 = order.index(ranks[0])
            r2 = order.index(ranks[1])
            high = max(r1, r2)
            low = min(r1, r2)
            suited = suits[0] == suits[1]

            # Premium hands → RAISE
            if ranks in (["A","A"],["K","K"],["Q","Q"],["J","J"]) or \
                set(ranks) in ({"A","K"}, {"A","Q"}):
                    return {"type": "raise", "amount": 6}

            # Good hands (Broadway, suited connectors, medium pairs) → CALL
            if high >= order.index("T") or \
                (suited and high >= order.index("9")) or \
            ranks[0] == ranks[1]:
                for a in legal:
                    if a["type"] in ("call","check"):
                        return {"type": a["type"]}
                return {"type": "fold"}

            # Everything else → FOLD
            return {"type": "fold"}

        # Estimate hand strength based on hole + board cards
        strength = approx_score(hole, board)

        # Occasional bluff trigger
        bluff = random.random() < self.bluff_chance

        # -----------------------------------------------------
        # DECISION LOGIC (but do NOT return yet)
        # -----------------------------------------------------
        decision = None

        # Strong hand OR bluffing
        if strength >= self.strong_threshold or bluff:
            for a in legal:
                if a["type"] in ("bet", "raise"):       # NEW BETTING SIZE FIX
                    pot = state["pot"]
                    street = state["street"]

                    if street == "preflop":
                        # Standard preflop raise — 3× to 4× BB
                        amt = max(a["min"], min(a["max"], 6))   # 6 chips = 3BB if BB=2
                    else:
                        # Postflop bet — 50% of pot
                        amt = max(a["min"], min(a["max"], pot * 0.5))

                    decision = {"type": a["type"], "amount": round(amt, 2)}
                    break 
            if decision is None:
                for a in legal:
                    if a["type"] in ("call", "check"):
                        decision = {"type": a["type"]}
                        break

        # Medium hand
        elif strength >= self.medium_threshold:
            if to_call <= pot * 0.25:
                for a in legal:
                    if a["type"] in ("call", "check"):
                        decision = {"type": a["type"]}
                        break
            else:
                for a in legal:
                    if a["type"] == "fold":
                        decision = {"type": "fold"}
                        break

        # Weak hand
        else:
            for a in legal:
                if a["type"] == "check":
                    decision = {"type": "check"}
                    break
            if decision is None:
                for a in legal:
                    if a["type"] == "fold":
                        decision = {"type": "fold"}
                        break

        # Fallback
        if decision is None:
            decision = random.choice(legal)

        # -----------------------------------------------------
        # DEBUG LOG — NOW we can print!
        # -----------------------------------------------------
        if hole:
            print(
                f"[{me}] street={state['street']} "
                f"hand={hole} board={board} "
                f"strength={strength} to_call={to_call} pot={pot} "
                f"→ action={decision}"
            )

        # -----------------------------------------------------
        # RETURN THE DECISION
        # -----------------------------------------------------
        return decision
