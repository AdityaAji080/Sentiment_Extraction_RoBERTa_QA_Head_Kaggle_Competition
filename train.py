"""Fine-tune RoBERTa for sentiment-span extraction and report validation Jaccard.

Each tweet is encoded as a QA pair, (sentiment, tweet), and the model learns start/end positions of
`selected_text` with cross-entropy over the start and end logits. The best checkpoint (by validation
loss) is scored with the competition's word-level Jaccard on the full validation split.

Usage:
    python train.py                                    # roberta-base
    python train.py --model roberta-large --lr 7e-6 --batch_size 8 --grad_accum 2
"""

import argparse
import inspect
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

from span_qa import encode_example, jaccard, predict


class TweetSpanDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.items = [encode_example(tokenizer, s, t, sel, max_len)
                      for s, t, sel in zip(df["sentiment"], df["text"], df["selected_text"])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--train_csv", default="data/train.csv")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_len", type=int, default=96)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = os.path.join(args.out_dir, args.model.replace("/", "_"))

    df = pd.read_csv(args.train_csv).fillna("")
    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed,
                                        stratify=df["sentiment"])

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    train_ds = TweetSpanDataset(train_df, tokenizer, args.max_len)
    val_ds = TweetSpanDataset(val_df, tokenizer, args.max_len)

    # 10% linear warmup: transformers 5 takes a float `warmup_steps`; 4.x used `warmup_ratio`.
    warmup = ({"warmup_ratio": 0.1} if "warmup_ratio" in inspect.signature(TrainingArguments).parameters
              else {"warmup_steps": 0.1})
    training_args = TrainingArguments(
        output_dir=run_dir, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum, weight_decay=0.01, **warmup,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=torch.cuda.is_available(), logging_steps=50, seed=args.seed, report_to="none")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()

    best_dir = os.path.join(run_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = predict(trainer.model.to(device), tokenizer, val_df["sentiment"].tolist(),
                    val_df["text"].tolist(), max_len=args.max_len, device=device)
    scores = np.array([jaccard(p, t) for p, t in zip(preds, val_df["selected_text"])])
    per_sentiment = {s: float(scores[(val_df["sentiment"] == s).to_numpy()].mean())
                     for s in sorted(val_df["sentiment"].unique())}
    results = {"model": args.model, "val_jaccard": float(scores.mean()), "per_sentiment": per_sentiment,
               "n_val": int(len(val_df))}
    print(json.dumps(results, indent=2))
    with open(os.path.join(run_dir, "val_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    val_df.assign(predicted=preds, jaccard=scores).to_csv(os.path.join(run_dir, "val_predictions.csv"),
                                                         index=False)


if __name__ == "__main__":
    main()
