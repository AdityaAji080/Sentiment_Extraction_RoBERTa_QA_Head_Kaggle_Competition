"""Write a Kaggle submission from a fine-tuned span-extraction model.

Usage:
    python predict.py outputs/roberta-base/best --test_csv data/test.csv --out submission.csv
"""

import argparse

import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from span_qa import predict


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model_dir", help="Folder with config, weights and tokenizer (train.py writes <run>/best).")
    p.add_argument("--test_csv", default="data/test.csv")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--max_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir).to(device)

    test = pd.read_csv(args.test_csv).fillna("")
    preds = predict(model, tokenizer, test["sentiment"].tolist(), test["text"].tolist(),
                    max_len=args.max_len, batch_size=args.batch_size, device=device)
    pd.DataFrame({"textID": test["textID"], "selected_text": preds}).to_csv(args.out, index=False)
    print(f"Wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
