import json
import os
from datetime import datetime

class DecisionLogger:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.buffer = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = f"logs/session_{self.session_id}.jsonl"

        # Make sure logs/ exists
        os.makedirs("logs", exist_ok=True)

    def log_decision(self, record):
        """Record one decision example."""
        if not self.enabled:
            return

        self.buffer.append(record)

        # Write to file every 200 entries
        if len(self.buffer) >= 200:
            self.flush()

    def flush(self):
        """Write buffered data to disk."""
        if not self.buffer:
            return

        with open(self.path, "a") as f:
            for r in self.buffer:
                f.write(json.dumps(r) + "\n")

        self.buffer = []

    def close(self):
        """Flush on shutdown."""
        self.flush()