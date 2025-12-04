# bots/poker_mind_bot.py

from core.bot_api import Action, PlayerView


class SmartBot:
    """
    Updated SmartBot compatible with PlayerView object.
    Uses simple heuristic poker logic for preflop and postflop.
    """

    def act(self, state: PlayerView):
        """
        Main decision function.
        'state' is a PlayerView, NOT a dict.
        """
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        stacks = state.stacks
        street = state.street

        # ===========================
        #        PREFLOP LOGIC
        # ===========================
        if street == "preflop":
            ranks = sorted([c[0] for c in hole], reverse=True)
            r1, r2 = ranks

            # --- Premium hands ---
            if (r1 == "A" and r2 in ("K", "Q", "J")) or (r1 == r2 and r1 in ("A", "K", "Q", "J")):
                return self._raise_or_call(legal, pot)

            # --- Medium-strength hands ---
            if r1 in ("A", "K", "Q", "J"):
                return self._call_or_check(legal)

            # --- Trash hands ---
            return self._fold_or_check(legal, to_call)

        # ===========================
        #        POSTFLOP LOGIC
        # ===========================
        strength = self._approx_strength(hole, board)

        if strength >= 0.75:
            return self._raise_or_call(legal, pot)

        if strength >= 0.40:
            return self._call_or_check(legal)

        return self._fold_or_check(legal, to_call)

    # -----------------------------------------------------
    # HELPER DECISION FUNCTIONS
    # -----------------------------------------------------

    def _fold_or_check(self, legal, to_call):
        """Check if possible, otherwise fold."""
        if to_call > 0:
            return Action("fold")

        # check if allowed
        for a in legal:
            if a["type"] == "check":
                return Action("check")

        return Action("fold")

    def _call_or_check(self, legal):
        """Call if facing a bet, otherwise check."""
        for a in legal:
            if a["type"] == "call":
                return Action("call")

        for a in legal:
            if a["type"] == "check":
                return Action("check")

        return Action("fold")

    def _raise_or_call(self, legal, pot):
        """Raise about 50% pot if possible, else call."""
        for a in legal:
            if a["type"] in ("raise", "bet"):
                size = max(a["min"], min(a["max"], pot * 0.5))
                return Action(a["type"], size)

        # fallback to call
        for a in legal:
            if a["type"] == "call":
                return Action("call")

        # fallback to check
        for a in legal:
            if a["type"] == "check":
                return Action("check")

        return Action("fold")

    # -----------------------------------------------------
    # Temporary strength estimator (replace later)
    # -----------------------------------------------------
    def _approx_strength(self, hole, board):
        return 0.5


# backward compatibility
PokerMindBot = SmartBot