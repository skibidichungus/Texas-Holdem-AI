import os
import sys
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bots.models.poker_mlp import PokerMLP

import torch.nn as nn

RANKS = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
         "9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
SUITS = {"c":0, "d":1, "h":2, "s":3}

STREET_MAP = {"preflop":0, "flop":1, "turn":2, "river":3}


def encode_card(card):
    """Convert ['A','h'] → [14, 2]"""
    rank, suit = card
    return [RANKS[rank], SUITS[suit]]


def bucket_raise(chosen, legal):
    t = chosen["type"]

    if t == "fold":
        return 0
    if t == "check":
        return 1
    if t == "call":
        return 2

    if t in ("bet", "raise"):
        amt = chosen.get("amount")
        mn = legal.get("min", None)
        mx = legal.get("max", None)

        # if ANYTHING is missing or weird → put everything into one bucket (label 3)
        if amt is None or mn is None or mx is None or mn >= mx:
            return 3

        # scale raise sizes across 5 buckets (labels 3–8)
        bucket = int((amt - mn) / max(1, (mx - mn)) * 5)
        bucket = max(0, min(bucket, 4))  # clamp 0–4
        return 3 + bucket

    # fallback bucket instead of returning None
    return 0

def load_decision_logs(root):
    """
    Recursively loads every .jsonl file inside logs/
    and returns a list of decision-dictionaries.
    """
    decisions = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".jsonl"):
                continue

            full = os.path.join(dirpath, fname)
            try:
                with open(full, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)

                        # Skip final result rows
                        if "chosen_action" not in obj:
                            continue

                        decisions.append(obj)

            except Exception as e:
                print(f"[WARN] Could not read {full}: {e}")

    return decisions

class PokerDataset(Dataset):
    ACTION_MAP = {
        "fold": 0,
        "check": 1,
        "call": 2,
        # raises get bucketed into 3 bins
        "raise_small": 3,
        "raise_medium": 4,
        "raise_large": 5
    }

    def __init__(self, log_folder):
        self.samples = []

        for path in glob.glob(f"{log_folder}/**/*.jsonl", recursive=True):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    if "chosen_action" not in row:
                        continue

                    chosen = row["chosen_action"]
                    act_type = chosen["type"]
                    act_amt = chosen["amount"]

                    # --------- LABEL ENCODING ---------
                    if act_type in ("fold", "check", "call"):
                        label = self.ACTION_MAP[act_type]

                    elif act_type in ("raise", "bet"):
                        # We need min/max to bucket
                        legal = next(
                            (la for la in row["legal"]
                             if la["type"] in ("raise", "bet")),
                            None
                        )
                        if legal is None:
                            # shouldn't happen, fallback to call
                            label = self.ACTION_MAP["call"]
                        else:
                            lo, hi = legal["min"], legal["max"]

                            # bucket the raise amount
                            if act_amt <= lo + (hi - lo) * 0.33:
                                label = self.ACTION_MAP["raise_small"]
                            elif act_amt <= lo + (hi - lo) * 0.66:
                                label = self.ACTION_MAP["raise_medium"]
                            else:
                                label = self.ACTION_MAP["raise_large"]
                    else:
                        # Safety fallback
                        continue

                    # --------- FEATURE ENCODING ---------
                    hole = row.get("hole", [])
                    hole_enc = []

                    for c in hole:
                        hole_enc += encode_card(c)

                    # pad missing hole cards
                    while len(hole_enc) < 4:
                        hole_enc.append(0)

                    board = row["board"]
                    board_enc = []
                    for c in board:
                        board_enc += encode_card(c)
                    while len(board_enc) < 10:
                        board_enc.append(0)

                    street = STREET_MAP[row["street"]]
                    pot = float(row["pot"])
                    to_call = float(row["to_call"])

                    stacks = row["stacks"]
                    me = row["player"]
                    hero_stack = float(stacks[me])
                    eff_stack = min(float(v) for v in stacks.values())
                    n_players = len([v for v in stacks.values() if v > 0])
                    
                    # NEW FEATURES: Hand strength, pot odds, position
                    # Hand strength
                    if hole and len(hole) >= 2:
                        from core.engine import approx_score
                        score = approx_score(hole, board)
                        hand_strength = min(1.0, score / 400.0)
                    else:
                        hand_strength = 0.0
                    
                    # Pot odds
                    if pot + to_call > 0:
                        pot_odds = to_call / (pot + to_call)
                    else:
                        pot_odds = 0.0
                    
                    # Position encoding
                    position_order = {
                        "UTG": 0.0, "UTG+1": 0.1, "MP": 0.3, "LJ": 0.4,
                        "HJ": 0.6, "CO": 0.8, "BTN": 1.0, "SB": 0.7, "BB": 0.5
                    }
                    position = row.get("position", "MP")  # Default to MP if missing
                    position_value = position_order.get(position, 0.5)

                    features = [
                        street, pot, to_call,
                        hero_stack, eff_stack, n_players
                    ] + hole_enc + board_enc + [hand_strength, pot_odds, position_value]  # +3 features

                    self.samples.append(
                        (
                            torch.tensor(features, dtype=torch.float32),
                            torch.tensor(label)
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class PokerMLP(nn.Module):
    def __init__(self, input_dim, hidden=256, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
def train_model(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    epochs=8,
    device="cpu"
):
    model = model.to(device)
    
    # Multi-class classification (fold, call, raise, check)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # ---- Validation ----
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f} | "
              f"Val Acc:  {accuracy:.2f}%")

    return model

if __name__ == "__main__":
    import argparse
    import sys
    from torch.utils.data import DataLoader, random_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Folder containing JSONL logs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("Loading logs...")
    logs = load_decision_logs(args.log_dir)
    print(f"Loaded {len(logs)} decisions.")

    dataset = PokerDataset(args.log_dir)

    if len(dataset) == 0:
        print(f"Error: No training data found in {args.log_dir}")
        print("Please run some games first to generate logs.")
        sys.exit(1)  # Changed from 'return' to 'sys.exit(1)'

    # 90% training / 10% validation
    val_size = max(1, int(0.10 * len(dataset)))
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch)

    print(f"Training samples: {train_size}  |  Validation samples: {val_size}")

    sample_x, _ = dataset[0]
    input_size = sample_x.shape[0]  # Will be 23 now

    model = PokerMLP(input_dim=input_size, hidden=128, num_classes=6)

    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device
    )

    # ---- SAVE MODEL ----
    save_path = "bots/models/ml_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"\nModel saved to: {save_path}")