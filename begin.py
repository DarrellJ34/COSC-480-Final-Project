"""
LSTM stock direction model -- my architecture notes

What goes in:
  - a sequence: the last 120 trading days for one stock, with F features per day
    (returns, volatility, volume, sentiment, ...)        -> shape [batch, 120, F]
  - a static vector: stuff that doesn't change day to day, like fundamental ranks,
    macro features, and the sector one-hot                -> shape [batch, S]

The layers:
  1. LSTM layer with 64 hidden units. This is a real LSTM cell, so it has the gates:
       - forget gate: how much of yesterday's memory (the cell state c) to keep
       - input gate:  how much of today's new info to write into memory
       - output gate: how much of the memory to show as the hidden state h
     The forget gate bias starts at 1 so the model remembers by default and
     learns what to forget, instead of the other way around.
  2. I take the hidden state from the LAST day (that is the summary of the whole
     120-day window) and concat it with the static vector.
  3. Dropout, then 2 fully connected layers:
       fc1: (64 + S) -> 32, ReLU
       fc2: 32 -> 1 logit  (sigmoid of this = probability the stock goes "up")

Loss is BCEWithLogitsLoss (binary, not 10 classes like the CNN).
Metrics are AUC and accuracy, compared against always predicting "up".
Train/test split is by year (walk-forward), never random.

NOTE: I did independent research to understand the concpets behind LSTM, and used Claude Code to help implement it and see the differences between our CNN code.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)

# use the Mac GPU if available, otherwise CPU (cuda if this ever runs on a PC with one)
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# making the LSTM
class HybridLSTM(nn.Module):
    def __init__(self, n_seq_features, n_static_features, hidden=64, dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=n_seq_features, hidden_size=hidden,
                            num_layers=1, batch_first=True)   # [B, T, F] -> [B, T, hidden]
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden + n_static_features, 32)  # hidden state + static vector
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(32, 1)                            # one logit, not ten
        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self):
        # PyTorch stacks the 4 gate blocks in one vector as [input, forget, candidate, output],
        # so the forget gate is rows hidden:2*hidden. There are two bias vectors (ih and hh)
        # that get added together, so set one block to 1 and the other to 0 -> total bias = 1.
        f = slice(self.hidden, 2 * self.hidden)
        with torch.no_grad():
            self.lstm.bias_ih_l0[f].fill_(1.0)
            self.lstm.bias_hh_l0[f].fill_(0.0)

    def forward(self, x_seq, x_static):
        out, (h_n, c_n) = self.lstm(x_seq)        # out: [B, T, hidden], one hidden state per day
        h = out[:, -1, :]                          # [B, hidden]: last day's state = window summary
        h = torch.cat([self.drop(h), x_static], dim=1)   # [B, hidden + S]
        h = self.relu(self.fc1(h))                 # [B, 32]
        return self.fc2(h).squeeze(1)              # [B] logits


# wrapping numpy arrays into a loader (no custom Dataset class needed)
def make_loader(X_seq, X_static, y, batch_size=64, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),      # [N, T, F]
        torch.tensor(X_static, dtype=torch.float32),   # [N, S]
        torch.tensor(y, dtype=torch.float32),          # [N]  0/1
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    probs, labels = [], []
    for x_seq, x_static, y in loader:
        logits = model(x_seq.to(device), x_static.to(device))
        probs.extend(torch.sigmoid(logits).cpu().tolist())   # keep probabilities, don't argmax
        labels.extend(y.tolist())
    return np.array(probs), np.array(labels)


def choose_threshold_from_probs(probs, labels):
    """Pick the probability cutoff on the VALIDATION set, never on test.
    0.5 is arbitrary when ~60% of labels are "up"; we take the cutoff that maximizes
    balanced accuracy, so the model has to be right about downs as well as ups."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(labels, probs > t) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def choose_threshold(model, val_loader):
    probs, labels = predict_proba(model, val_loader)
    return choose_threshold_from_probs(probs, labels)


def metrics_from_probs(probs, labels, threshold=0.5):
    probs, labels = np.asarray(probs), np.asarray(labels)
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "threshold": float(threshold),
        "always_up_acc": float(labels.mean()),      # base rate = accuracy of predicting "up" always
        # threshold-free: how well the model RANKS, and how honest its probabilities are
        "auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "log_loss": float(log_loss(labels, probs, labels=[0, 1])),
        # confusion matrix at the threshold, positive class = "up"
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "acc": float(accuracy_score(labels, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }


def evaluate(model, loader, threshold=0.5):
    probs, labels = predict_proba(model, loader)
    return metrics_from_probs(probs, labels, threshold)


def print_metrics(m, title="Results", detailed=False):
    """Compact by default. detailed=True prints the full confusion-matrix report for final runs."""
    base = m["always_up_acc"]
    print(f"\n{title}  ({m['n']} samples, threshold = {m['threshold']:.2f})")
    print(f"  ROC AUC:   {m['auc']:.3f}    (0.500 = random)")
    print(f"  Accuracy: {m['acc'] * 100:5.1f}%    (always-up {base * 100:.1f}%, edge {(m['acc'] - base) * 100:+.1f} pts)")
    print(f"  MCC:      {m['mcc']:+.3f}")
    print(f"  Confusion: tp={m['tp']}  fp={m['fp']}  fn={m['fn']}  tn={m['tn']}")
    if not detailed:
        return
    print("  Confusion matrix (rows = actual, cols = predicted)")
    print("                   pred down    pred up")
    print(f"    actual down   {m['tn']:9d}  {m['fp']:9d}")
    print(f"    actual up     {m['fn']:9d}  {m['tp']:9d}")
    print("  Ranking and calibration (threshold-free)")
    print(f"    ROC AUC:           {m['auc']:.3f}     (0.500 = random)")
    print(f"    PR AUC:            {m['pr_auc']:.3f}     ({base:.3f} = random, i.e. the base rate)")
    print(f"    Brier score:       {m['brier']:.4f}    (lower is better; {base * (1 - base):.4f} = always say the base rate)")
    print(f"    Log loss:          {m['log_loss']:.4f}")
    print("  At the threshold")
    print(f"    Accuracy:          {m['acc'] * 100:5.1f}%    (always-up {base * 100:.1f}%, edge {(m['acc'] - base) * 100:+.1f} pts)")
    print(f"    Balanced accuracy: {m['balanced_acc'] * 100:5.1f}%    (50.0% = random)")
    print(f"    Precision:         {m['precision'] * 100:5.1f}%    of predicted ups that actually went up")
    print(f"    Recall:            {m['recall'] * 100:5.1f}%    of actual ups we flagged")
    print(f"    Specificity:       {m['specificity'] * 100:5.1f}%    of actual downs we avoided")
    print(f"    F1:                {m['f1']:.3f}")
    print(f"    MCC:               {m['mcc']:+.3f}    (-1 to +1, 0 = random; fair under class imbalance)")


def train_model(model, train_loader, val_loader=None, num_epochs=10, lr=1e-3, weight_decay=1e-4,
                patience=None, verbose=True):
    """Adam + gradient clipping. With a val_loader: tracks validation AUC every epoch, stops early
    once it hasn't improved for `patience` epochs, and rolls back to the best epoch at the end.
    Returns (model, history); history has one dict per epoch for learning-curve plots."""
    criterion = nn.BCEWithLogitsLoss()          # binary loss on raw logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)

    history = []
    best_auc, best_epoch, best_state, since_best = -1.0, None, None, 0
    for epoch in range(num_epochs):
        model.train()
        total_loss, n = 0.0, 0
        for x_seq, x_static, y in train_loader:               # three tensors per batch
            x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
            logits = model(x_seq, x_static)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # standard for RNNs
            optimizer.step()

            total_loss += loss.item() * len(y)
            n += len(y)

        rec = {"epoch": epoch + 1, "train_loss": total_loss / n}
        msg = f"Epoch {epoch + 1:>2}/{num_epochs} | train loss {total_loss / n:.4f}"
        if val_loader is not None:
            m = evaluate(model, val_loader)
            rec.update(val_auc=m["auc"], val_acc=m["acc"], val_log_loss=m["log_loss"])
            msg += f" | val AUC {m['auc']:.3f} | val acc {m['acc'] * 100:5.1f}%"
            if m["auc"] > best_auc:   # remember the best epoch (noisy data overfits fast)
                best_auc, best_epoch, since_best = m["auc"], epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
        history.append(rec)
        if verbose:
            print(msg)
        if patience is not None and val_loader is not None and since_best >= patience:
            if verbose:
                print(f"Stopping early at epoch {epoch + 1}: no val improvement for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)   # roll back to the best validation epoch
        if verbose:
            print(f"Kept epoch {best_epoch} (best val AUC {best_auc:.3f})")
    return model, history


# running model (smoke test with random data until the real features/labels exist)
if __name__ == "__main__":
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N, T, F, S = 768, 120, 20, 10
    X_seq    = rng.standard_normal((N, T, F)).astype(np.float32)
    X_static = rng.standard_normal((N, S)).astype(np.float32)
    y        = rng.integers(0, 2, N).astype(np.float32)

    # train / val / test in time order (with real data these are year blocks, see walk_forward_split)
    train_loader = make_loader(X_seq[:512], X_static[:512], y[:512], shuffle=True)
    val_loader   = make_loader(X_seq[512:640], X_static[512:640], y[512:640])
    test_loader  = make_loader(X_seq[640:], X_static[640:], y[640:])

    lstm_model = HybridLSTM(n_seq_features=F, n_static_features=S)
    print("=== Model ===")
    print(lstm_model)
    print(f"parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    print(f"forget-gate bias, first 3 units: {lstm_model.lstm.bias_ih_l0[64:67].tolist()}")
    print(f"device: {device}")

    print("\n=== Training ===")
    lstm_model, history = train_model(lstm_model, train_loader, val_loader, num_epochs=10)

    thr = choose_threshold(lstm_model, val_loader)        # cutoff chosen on validation only
    print_metrics(evaluate(lstm_model, val_loader, thr), title="Validation results")
    print_metrics(evaluate(lstm_model, test_loader, thr), title="Test results")   # test touched once
