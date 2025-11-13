# bot_api.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Action:
    type: str                # "fold" | "check" | "call" | "bet" | "raise"
    amount: Optional[float] = None

@dataclass
class PlayerView:
    me: str
    street: str              # "preflop" | "flop" | "turn" | "river"
    position: str
    hole_cards: list
    board: list
    pot: float
    to_call: float
    min_raise: float
    max_raise: float
    legal_actions: List[Dict[str, Any]]
    stacks: Dict[str, float]
    opponents: List[str]
    history: List[Dict[str, Any]]

class BotAdapter:
    def act(self, view: PlayerView) -> Action:
        raise NotImplementedError
