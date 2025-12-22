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
    def __init__(self, model_path="bots/models/ml_model.pt", device="cpu", use_fallback=True):
        self.device = device
        self.use_fallback = use_fallback
        self.model = PokerMLP(input_dim=23, hidden=128, num_classes=6)
        self.model_trained = False
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Check if model dimensions match
            first_layer_weight = checkpoint.get('net.0.weight', None)
            if first_layer_weight is not None:
                expected_input_dim = first_layer_weight.shape[1]
                if expected_input_dim == 20:
                    # Old model with 20 features - can't use it
                    print(f"Warning: Model file has old 20-feature format. Need to retrain with 23 features.")
                    print("Using fallback strategy until model is retrained.")
                    self.model_trained = False
                elif expected_input_dim == 23:
                    # New model with 23 features - load it
                    self.model.load_state_dict(checkpoint)
                    self.model.eval()
                    self.model_trained = True
                else:
                    print(f"Warning: Model has unexpected input dimension {expected_input_dim}. Using fallback.")
                    self.model_trained = False
            else:
                # Try loading anyway
                self.model.load_state_dict(checkpoint)
                self.model.eval()
                self.model_trained = True
        except (FileNotFoundError, OSError, RuntimeError) as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using untrained model (random weights).")
            self.model.eval()

    def _make_features(self, state):
        """
        Produce feature vector with hand strength, pot odds, and position.
        Now 23 dimensions: 20 original + 3 new features
        """
        # Handle both PlayerView and dict
        if isinstance(state, dict):
            class DictView:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            state = DictView(state)

        hole = state.hole_cards
        board = state.board

        # ---- Encode hole ----
        hole_enc = []
        for c in hole:
            hole_enc += encode_card(c)   # 4 numbers
        # Pad to 4 numbers if missing (player folded)
        while len(hole_enc) < 4:
            hole_enc.append(0)

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
        
        # NEW FEATURES: Hand strength, pot odds, position
        # Hand strength
        if hole and len(hole) >= 2:
            score = approx_score(hole, board)
            hand_strength = min(1.0, score / 400.0)  # Normalize to 0-1
        else:
            hand_strength = 0.0
        
        # Pot odds
        if pot + to_call > 0:
            pot_odds = to_call / (pot + to_call)
        else:
            pot_odds = 0.0
        
        # Position encoding (0.0 = early/tight, 1.0 = late/loose)
        position_order = {
            "UTG": 0.0, "UTG+1": 0.1, "MP": 0.3, "LJ": 0.4,
            "HJ": 0.6, "CO": 0.8, "BTN": 1.0, "SB": 0.7, "BB": 0.5
        }
        position_value = position_order.get(state.position, 0.5)

        # FULL 23-feature vector (was 20, now +3)
        features = (
            [street, pot, to_call, hero_stack, eff_stack, n_players]
            + hole_enc
            + board_enc
            + [hand_strength, pot_odds, position_value]  # NEW FEATURES
        )

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return x.to(self.device)

    def _estimate_hand_strength(self, hole, board):
        """Quick hand strength estimate for fallback."""
        if not hole or len(hole) < 2:
            return 0.0
        # Simple approximation using approx_score
        score = approx_score(hole, board)
        # Normalize roughly (this is approximate)
        return min(1.0, score / 500.0)

    # ----------------------------------------------------------
    # ACT ------------------------------------------------------
    # ----------------------------------------------------------
    def act(self, state):
        # Handle both PlayerView objects and dicts (from InProcessBot adapter)
        if isinstance(state, dict):
            # Convert dict to PlayerView-like access
            class DictView:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            state = DictView(state)
        
        legal = state.legal_actions

        # If model not trained and fallback enabled, use fallback
        if not self.model_trained and self.use_fallback:
            return self._fallback_strategy(state)

        # Build 20-dim features
        x = self._make_features(state)

        # Predict class
        logits = self.model(x)
        pred = int(logits.argmax(dim=1).item())
        
        # Get confidence (softmax probability)
        probs = torch.softmax(logits, dim=1)[0]
        confidence = float(probs[pred].item())

        # If low confidence and fallback enabled, use fallback
        if self.use_fallback and confidence < 0.3:
            return self._fallback_strategy(state)

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

    def _fallback_strategy(self, state):
        """Fallback to simple hand strength logic when model is untrained."""
        # Handle both PlayerView and dict
        if isinstance(state, dict):
            class DictView:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            state = DictView(state)
            
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions

        if not hole:
            return self._choose("fold", legal)

        strength = self._estimate_hand_strength(hole, board)

        # Facing a bet
        if to_call > 0:
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.5
            if strength < pot_odds:
                return self._choose("fold", legal)
            if strength < 0.55:
                return self._choose("call", legal)
            return self._raise(pot, legal)

        # No bet yet
        if strength > 0.60:
            return self._bet(pot, legal)
        return self._choose("check", legal)