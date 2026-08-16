"""
Member 03 (Fesal) - DL model: PyTorch Bidirectional LSTM with an attention layer.
Class imbalance handled with a weighted cross-entropy loss during training.

Why this and not another plain BiLSTM: Member 01's model is a Keras BiLSTM with
Word2Vec embeddings. This one is implemented in PyTorch, uses packed sequences
(so padding never enters the recurrence) and adds an ATTENTION layer that learns
which words in a posting drive the decision - giving the report a genuine
architectural comparison rather than a duplicate.

The architecture itself lives in src/bilstm_model.py so the Flask app builds the
identical network at inference time.

Run:  python member3_fesal/train_bilstm.py --data dataset/fake_job_postings.csv

Works on CPU (defaults drop to a lighter config automatically, a few minutes);
a Colab T4 GPU makes it roughly 10x faster.
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import load_and_clean, train_test_split_stratified
from src.evaluate import evaluate_model, plot_confusion_matrix
from src.bilstm_model import BiLSTMAttention, build_vocab, collate, encode, save_bundle

try:
    from tqdm.auto import tqdm
except ImportError:                                   # tqdm optional
    def tqdm(x, **kw):
        return x

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class PostingDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.seqs = [encode(t, vocab, max_len) for t in texts]
        self.labels = [int(l) for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


@torch.no_grad()
def predict_probs(model, loader, device, criterion=None):
    model.eval()
    probs, total_loss, n = [], 0.0, 0
    for ids, lengths, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids, lengths)
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * labels.size(0)
            n += labels.size(0)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs), (total_loss / n if n else float("nan"))


def best_f1_threshold(y_true, y_prob):
    """Decision threshold that maximises fraud-class F1 on the VALIDATION set."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.5, 0.0
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i])


def main(args):
    torch.manual_seed(42)
    np.random.seed(42)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ---- hardware-aware defaults (any flag you pass overrides these) -------
    if use_cuda:
        epochs = 8 if args.epochs is None else args.epochs
        batch_size = 64 if args.batch_size is None else args.batch_size
        max_len = 300 if args.max_len is None else args.max_len
        layers = 2 if args.layers is None else args.layers
        hidden_dim = 128 if args.hidden_dim is None else args.hidden_dim
    else:
        epochs = 5 if args.epochs is None else args.epochs
        batch_size = 128 if args.batch_size is None else args.batch_size
        max_len = 150 if args.max_len is None else args.max_len
        layers = 1 if args.layers is None else args.layers
        hidden_dim = 96 if args.hidden_dim is None else args.hidden_dim
        torch.set_num_threads(os.cpu_count() or 2)
        print("No GPU detected - using lighter CPU settings.")
        print("For the full configuration: Runtime > Change runtime type > T4 GPU.")

    print(f"Device: {device} | epochs={epochs} batch={batch_size} "
          f"max_len={max_len} layers={layers} hidden={hidden_dim}")

    # ---- data (same cleaning + same stratified split as every other member)
    df = load_and_clean(args.data)
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    # validation slice carved out of TRAIN, so the test set stays untouched
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    print(f"train={len(X_tr)}  val={len(X_val)}  test={len(X_test)}  "
          f"| scam rate={y_train.mean():.2%}")

    vocab = build_vocab(X_tr, min_freq=args.min_freq, max_size=args.max_vocab)
    print("Vocabulary size:", len(vocab))

    train_loader = DataLoader(PostingDataset(X_tr, y_tr, vocab, max_len),
                              batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PostingDataset(X_val, y_val, vocab, max_len),
                            batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(PostingDataset(X_test, y_test, vocab, max_len),
                             batch_size=batch_size * 2, shuffle=False, collate_fn=collate)

    # ---- model ------------------------------------------------------------
    model = BiLSTMAttention(
        vocab_size=len(vocab), embed_dim=args.embed_dim, hidden_dim=hidden_dim,
        num_layers=layers, dropout=args.dropout, attn_dim=args.attn_dim,
    ).to(device)
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- weighted loss for class imbalance ---------------------------------
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    weights = torch.tensor([len(y_tr) / (2 * n_neg), len(y_tr) / (2 * n_pos)],
                           dtype=torch.float, device=device)
    print(f"Loss class weights: legitimate={weights[0]:.3f}  scam={weights[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=0)

    # ---- training loop -----------------------------------------------------
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for ids, lengths, labels in bar:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{running / seen:.4f}")

        val_prob, val_loss = predict_probs(model, val_loader, device, criterion)
        val_f1 = f1_score(y_val, (val_prob >= 0.5).astype(int), zero_division=0)
        scheduler.step(val_loss)
        print(f"epoch {epoch}/{epochs}  train_loss={running / seen:.4f}  "
              f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  ({time.time() - t0:.0f}s)")

        if val_loss < best_loss - 1e-4:
            best_loss, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            print("   ^ best so far, checkpointed")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- evaluation --------------------------------------------------------
    val_prob, _ = predict_probs(model, val_loader, device)
    thr, thr_f1 = best_f1_threshold(y_val, val_prob)
    print(f"\nBest validation threshold: {thr:.3f} (validation F1={thr_f1:.4f})")

    y_prob, _ = predict_probs(model, test_loader, device)

    # headline result at 0.5, directly comparable with the other five models
    m = evaluate_model("BiLSTM", y_test, (y_prob >= 0.5).astype(int), y_prob)
    plot_confusion_matrix("BiLSTM", m["confusion_matrix"])

    # extra: threshold tuned for the 5%-positive class (report Section 4 note).
    # save=False so compare_models.py still shows ONE row per model.
    evaluate_model(f"BiLSTM (tuned thr={thr:.2f})", y_test,
                   (y_prob >= thr).astype(int), y_prob, save=False)

    # ---- save --------------------------------------------------------------
    out_dir = os.path.join(MODELS_DIR, "bilstm_final")
    config = {
        "vocab_size": len(vocab), "embed_dim": args.embed_dim,
        "hidden_dim": hidden_dim, "num_layers": layers, "dropout": args.dropout,
        "attn_dim": args.attn_dim, "max_len": max_len, "threshold": thr,
    }
    save_bundle(out_dir, model, vocab, config)
    print(f"Saved -> {os.path.abspath(out_dir)} "
          f"(bilstm.pt + vocab.json + config.json)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/fake_job_postings.csv")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--attn-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-vocab", type=int, default=30_000)
    p.add_argument("--patience", type=int, default=2)
    main(p.parse_args())
