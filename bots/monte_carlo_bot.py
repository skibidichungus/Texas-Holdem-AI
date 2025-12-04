import random
from core.bot_api import Action, PlayerView
from core.engine import approx_score


class MonteCarloBot:
    """
    Monte Carlo rollout bot updated to work with PlayerView objects.
    """

    def __init__(self, simulations=200):
        self.simulations = simulations

    # ----------------------------------------------------
    # PUBLIC: act() for engine compatibility
    # ----------------------------------------------------
    def act(self, state: PlayerView):
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        street = state.street
        opponents = state.opponents

        # If no hole cards (folded)
        if not hole:
            for a in legal:
                if a["type"] == "check":
                    return Action("check")
            return Action("fold")

        # Monte Carlo equity estimate
        winrate = self._estimate_equity(hole, board, sims=self.simulations)

        # -----------------------------
        # FACING A BET
        # -----------------------------
        if to_call > 0:
            pot_odds = to_call / (pot + to_call)

            # Weak hand → fold
            if winrate < pot_odds:
                return self._choose("fold", legal)

            # Medium hand → call
            if winrate < 0.60:
                return self._choose("call", legal)

            # Strong → raise
            return self._raise(pot, legal)

        # -----------------------------
        # NO BET YET
        # -----------------------------
        if winrate > 0.65:
            return self._bet(pot, legal)

        # Medium → check
        for a in legal:
            if a["type"] == "check":
                return Action("check")

        # fallback
        return self._choose("fold", legal)

    # ----------------------------------------------------
    # Monte Carlo equity estimation
    # ----------------------------------------------------
    def _estimate_equity(self, hole, board, sims=200):
        wins = 0
        ties = 0

        for _ in range(sims):
            opp = self._random_opponent_hand(hole, board)
            full_board = self._random_board(board, hole, opp)

            my_score = approx_score(hole, full_board)
            opp_score = approx_score(opp, full_board)

            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1

        return (wins + ties * 0.5) / sims

    # ----------------------------------------------------
    # Randomize opponent hole cards
    # ----------------------------------------------------
    def _random_opponent_hand(self, hole, board):
        deck = self._remaining_deck(hole, board)
        return random.sample(deck, 2)

    # ----------------------------------------------------
    # Complete the board to 5 cards randomly
    # ----------------------------------------------------
    def _random_board(self, board, hole, opp):
        deck = self._remaining_deck(hole, board, opp)
        need = 5 - len(board)
        cards = random.sample(deck, need)
        return board + cards

    # ----------------------------------------------------
    # Remaining deck helper
    # ----------------------------------------------------
    def _remaining_deck(self, hole, board, opp=None):
        ranks = "23456789TJQKA"
        suits = "cdhs"

        full = [(r, s) for r in ranks for s in suits]

        used = set(tuple(c) for c in hole)
        used |= set(tuple(c) for c in board)
        if opp:
            used |= set(tuple(c) for c in opp)

        return [c for c in full if c not in used]

    # ----------------------------------------------------
    # Helper: choose legal action
    # ----------------------------------------------------
    def _choose(self, typ, legal):
        for a in legal:
            if a["type"] == typ:
                return Action(typ)
        # fallback if that action type isn't available
        for a in legal:
            if a["type"] in ("call", "check"):
                return Action(a["type"])
        return Action("fold")

    # ----------------------------------------------------
    # Raise helper
    # ----------------------------------------------------
    def _raise(self, pot, legal):
        for a in legal:
            if a["type"] == "raise":
                amt = max(a["min"], min(a["max"], pot * 0.75))
                return Action("raise", round(amt, 2))
        return self._choose("call", legal)

    # ----------------------------------------------------
    # Bet helper
    # ----------------------------------------------------
    def _bet(self, pot, legal):
        for a in legal:
            if a["type"] == "bet":
                amt = max(a["min"], min(a["max"], pot * 0.5))
                return Action("bet", round(amt, 2))
        return self._choose("check", legal)


# Backwards alias
SmartBot = MonteCarloBot