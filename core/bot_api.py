# bot_api.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Action:
    type: str                # "fold" | "check" | "call" | "bet" | "raise"
    amount: Optional[int] = None

@dataclass
class PlayerView:
    me: str
    street: str              # "preflop" | "flop" | "turn" | "river"
    position: str
    hole_cards: list
    board: list
    pot: int
    to_call: int
    min_raise: int
    max_raise: int
    legal_actions: List[Dict[str, Any]]
    stacks: Dict[str, int]
    opponents: List[str]
    history: List[Dict[str, Any]]

class BotAdapter:
    def act(self, view: PlayerView) -> Action:
        raise NotImplementedError
