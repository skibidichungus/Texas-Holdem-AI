import random
from typing import Dict, Any, List
from core.engine import approx_score, full_deck


class MonteCarloBot:
    """
    Monte Carlo rollout bot:
    - Simulates random future boards
    - Estimates win probability vs random opponent hands
    - Acts using EV-aware policy
    """

    def __init__(self, simulations=300):
        self.simulations = simulations
        self.bluff_chance = 0.05  # small random bluff chance

    # --------------------------------------------------------
    # UTILITY: sample opponents' hole cards
    # --------------------------------------------------------
    def _sample_opponent_holes(self, used_cards, num_opponents):
        deck = [c for c in full_deck() if c not in used_cards]
        random.shuffle(deck)
        return [ [deck.pop(), deck.pop()] for _ in range(num_opponents) ]

    # --------------------------------------------------------
    # UTILITY: finish board to 5 cards
    # --------------------------------------------------------
    def _sample_future_board(self, used_cards, board):
        deck = [c for c in full_deck() if c not in used_cards]
        random.shuffle(deck)

        missing = 5 - len(board)
        return board + [deck.pop() for _ in range(missing)]

    # --------------------------------------------------------
    # CORE MONTE CARLO SIMULATION
    # --------------------------------------------------------
    def estimate_winrate(self, hole, board, num_opponents):
        used = set(hole + board)

        wins = 0
        ties = 0

        for _ in range(self.simulations):
            # Sample other players' hands
            opp_hands = self._sample_opponent_holes(used, num_opponents)

            # Sample full board
            sim_board = self._sample_future_board(used | set(sum(opp_hands, [])), board)

            # Score my hand
            my_score = approx_score(hole, sim_board)

            # Score all opponents
            opp_scores = [
                approx_score(h, sim_board) for h in opp_hands
            ]

            # Compare
            if my_score > max(opp_scores):
                wins += 1
            elif my_score == max(opp_scores):
                ties += 1

        return (wins + ties * 0.5) / self.simulations

    # --------------------------------------------------------
    # PUBLIC: act() for engine compatibility
    # --------------------------------------------------------
    def act(self, state: Dict[str, Any]):
        legal = state["legal_actions"]
        hole = state["hole_cards"]
        board = state["board"]
        pot = state["pot"]
        to_call = state["to_call"]
        me = state["me"]
        opponents = state["opponents"]

        if not hole:
            return self._sanitize({"type": "check"}, legal)

        # Monte Carlo estimate
        winrate = self.estimate_winrate(hole, board, len(opponents))

        # Bluff
        if random.random() < self.bluff_chance:
            winrate += 0.15

        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0
        call_threshold = pot_odds
        raise_threshold = 0.60

        # -----------------------------
        # FACING A BET
        # -----------------------------
        if to_call > 0:

            # Fold if losing
            if winrate < call_threshold:
                return self._sanitize({"type": "fold"}, legal)

            # Call if decent
            if winrate < raise_threshold:
                return self._sanitize({"type": "call"}, legal)

            # Raise if strong > 60%
            for a in legal:
                if a["type"] == "raise":
                    target = pot * 0.75
                    amt = max(a["min"], min(a["max"], target))
                    return self._sanitize({"type": "raise", "amount": round(amt, 2)}, legal)

            return self._sanitize({"type": "call"}, legal)

        # -----------------------------
        # NO ONE BET YET
        # -----------------------------
        if winrate > 0.65:
            for a in legal:
                if a["type"] == "bet":
                    target = pot * 0.5
                    amt = max(a["min"], min(a["max"], target))
                    return self._sanitize({"type": "bet", "amount": round(amt, 2)}, legal)

        # Check if possible
        for a in legal:
            if a["type"] == "check":
                return self._sanitize({"type": "check"}, legal)

        # fallback
        return self._sanitize({"type": "fold"}, legal)

    def _sanitize(self, choice, legal_actions):
        desired_type = choice["type"]
        desired_amount = choice.get("amount", None)

        # Filter legal actions that match this type
        same_type = [a for a in legal_actions if a["type"] == desired_type]

        if not same_type:
            # Desired action type not allowed → fallback
            # 1. Prefer call
            calls = [a for a in legal_actions if a["type"] == "call"]
            if calls:
                return {"type": "call", "amount": None}

            # 2. Then check
            checks = [a for a in legal_actions if a["type"] == "check"]
            if checks:
                return {"type": "check", "amount": None}

            # 3. Otherwise fold if legal
            folds = [a for a in legal_actions if a["type"] == "fold"]
            if folds:
                return {"type": "fold", "amount": None}

            # 4. LAST RESORT: choose first legal action
            a = legal_actions[0]
            return {"type": a["type"], "amount": a.get("min")}

        a = same_type[0]

        # Bet/Raise: clamp amount
        if a["type"] in ("bet", "raise"):
            # If Monte Carlo didn't supply amount → default min
            if desired_amount is None:
                amt = a["min"]
            else:
                amt = max(a["min"], min(desired_amount, a["max"]))
            return {"type": a["type"], "amount": round(amt, 2)}

        # fold, call, check
        return {"type": a["type"], "amount": None}

# Backwards-compat alias
SmartBot = MonteCarloBot
