import torch
import random
from core.bot_api import Action
from core.engine import approx_score
from bots.models.poker_mlp import PokerMLP



# CARD ENCODING ------------------------------------------------------

RANKS = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
         "9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
SUITS = {"c":0, "d":1, "h":2, "s":3}

def encode_card(card):
    rank, suit = card
    return [RANKS[rank], SUITS[suit]]


STREET_MAP = {"preflop":0, "flop":1, "turn":2, "river":3}


# ML BOT -------------------------------------------------------------

class MLBot:
    def __init__(self, model_path="bots/models/ml_model.pt", device="cpu"):
        self.device = device
        self.model = PokerMLP(input_dim=20, hidden=128, num_classes=6)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using untrained model (random weights).")
            self.model.eval()

    def _make_features(self, state):
        """
        Produce EXACT 20-DIM feature vector like training.
        """

        hole = state.hole_cards
        board = state.board

        # ---- Encode hole ----
        hole_enc = []
        for c in hole:
            hole_enc += encode_card(c)   # 4 numbers

        # ---- Encode board ----
        board_enc = []
        for c in board:
            board_enc += encode_card(c)  # up to 10 numbers
        while len(board_enc) < 10:
            board_enc.append(0)

        street = STREET_MAP[state.street]
        pot = float(state.pot)
        to_call = float(state.to_call)

        stacks = state.stacks
        me = state.me
        hero_stack = float(stacks[me])
        eff_stack = min(float(v) for v in stacks.values())
        n_players = len([v for v in stacks.values() if v > 0])

        # FULL 20-feature vector
        features = (
            [street, pot, to_call, hero_stack, eff_stack, n_players]
            + hole_enc
            + board_enc
        )

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return x.to(self.device)

    # ----------------------------------------------------------
    # ACT ------------------------------------------------------
    # ----------------------------------------------------------
    def act(self, state):
        legal = state.legal_actions

        # Build 20-dim features
        x = self._make_features(state)

        # Predict class
        logits = self.model(x)
        pred = int(logits.argmax(dim=1).item())

        # -------- handle buckets --------
        if pred == 0:
            return self._choose("fold", legal)
        if pred == 1:
            return self._choose("check", legal)
        if pred == 2:
            return self._choose("call", legal)

        # RAISE BUCKETS (3,4,5)
        return self._raise_bucket(pred - 3, legal)

    # ----------------------------------------------------------
    def _choose(self, typ, legal):
        for a in legal:
            if a["type"] == typ:
                return Action(typ)
        for fallback in ("call", "check", "fold"):
            for a in legal:
                if a["type"] == fallback:
                    return Action(fallback)
        a = legal[0]
        return Action(a["type"], a.get("min"))

    # ----------------------------------------------------------
    def _raise_bucket(self, bucket, legal):
        raises = [a for a in legal if a["type"] == "raise"]
        bets = [a for a in legal if a["type"] == "bet"]

        if raises:
            a = raises[0]
        elif bets:
            a = bets[0]
        else:
            return self._choose("call", legal)

        lo, hi = a["min"], a["max"]
        amt = lo + (hi - lo) * (bucket / 2)   # 3 buckets

        return Action(a["type"], round(amt, 2))