# core/logger.py

import json
import os
from datetime import datetime

class DecisionLogger:
    def __init__(self, enabled=True, directory="logs"):
        self.enabled = enabled
        self.directory = directory
        self.file = None
        self.hand_id = 0  # increments every hand

        if enabled:
            os.makedirs(directory, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file = open(f"{directory}/session_{ts}.jsonl", "w")

    def start_hand(self, hand_id: int):
        """Set hand ID at start of each hand."""
        self.hand_id = hand_id

    def log_decision(self, entry: dict):
        if not self.enabled:
            return

        entry = dict(entry)  # copy
        entry["hand_id"] = self.hand_id

        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()

    def log_result(self, pid: str, net: float):
        """Log the hand outcome for each player."""
        if not self.enabled:
            return

        entry = {
            "hand_id": self.hand_id,
            "result": {
                "player": pid,
                "net": net
            }
        }
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()

    def flush(self):
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
