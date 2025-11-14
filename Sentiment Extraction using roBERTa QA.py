import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_name = "roberta-large"
train_csv = "train.csv"
test_csv = "test.csv"
text_col = "text"
span_col = "selected_text"
output_dir = "roberta-large-pretrained"
EPOCHS = 10
BATCH_SIZE = 8
learning_rate = 7e-6
MAX_LEN = 96

df = pd.read_csv(train_csv).fillna('')
test = pd.read_csv(test_csv).fillna('')

# Training and Validation Splits
train_df, valid_df = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

encode_kwargs = dict(max_length=MAX_LEN,
                     padding="max_length",
                     truncation=True,
                     return_offsets_mapping=True,
                     )

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN
        self.text_col = text_col
        self.span_col = span_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        target = str(row[self.span_col])
        sentiment = str(row["sentiment"]).lower().strip()

        enc = self.tokenizer(sentiment,
                             text,
                             max_length = self.max_len,
                             padding="max_length",
                             truncation=True,
                             return_offsets_mapping=True,
        )

        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        offsets = enc["offset_mapping"]

        start_char = text.find(target.strip())
        if start_char == -1 or target.strip() == "":
            start_char, end_char = 0, len(text)
        else:
            end_char = start_char + len(target.strip())

        start_tok = end_tok = 0
        found_s = found_e = False
        for idx, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if not found_s and s <= start_char < e:
                start_tok = idx
                found_s = True
            if not found_e and s < end_char <= e:
                end_tok = idx
                found_e = True
            if found_s and found_e:
                break
        if not found_s:
            for idx, (s, e) in enumerate(offsets):
                if s < e and s >= start_char:
                    start_tok = idx
                    break
        if not found_e:
            for idx, (s, e) in enumerate(offsets):
                if e >= end_char and e > 0:
                    end_tok = idx
                    break
        if end_tok < start_tok:
            end_tok = start_tok

        return {"input_ids": ids,
                "attention_mask": attention_mask,
                "start_positions": torch.tensor(start_tok),
                "end_positions": torch.tensor(end_tok),
                "offset_mapping": torch.tensor(offsets),
                "text": text,
                "selected_text": target,
                "sentiment": sentiment,
                }

train_ds = TweetDataset(train_df)
valid_ds = TweetDataset(valid_df)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def collate(batch):
    keys = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    learning_rate=learning_rate,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    warmup_ratio=0.1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset = train_ds,
    eval_dataset=valid_ds,
    data_collator=collate,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

BEST_DIR = f"{output_dir}-best"
trainer.save_model(BEST_DIR)        # saves current (best) model
tokenizer.save_pretrained(BEST_DIR) # save tokenizer alongside


model.to(device).eval()

@torch.no_grad()
def extract_span(text: str, sentiment: str, tokenizer, model, max_len=96):
    """
    Returns the model-extracted substring for a single (sentiment, text) pair.
    """
    enc = tokenizer(
        sentiment.strip().lower(), text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()  # list of (start_char, end_char)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    start_logits = out.start_logits[0].cpu()
    end_logits   = out.end_logits[0].cpu()

    # naive pick: argmax each, then fix if inverted
    start_idx = int(torch.argmax(start_logits))
    end_idx   = int(torch.argmax(end_logits))
    if end_idx < start_idx:
        end_idx = start_idx

    # gather char bounds from offsets, skipping specials (0,0)
    valid = [(i, s, e) for i, (s, e) in enumerate(offsets) if e > 0]
    # clamp to last valid token if needed
    if start_idx >= len(offsets) or offsets[start_idx][1] == 0:
        start_idx = valid[0][0]
    if end_idx >= len(offsets) or offsets[end_idx][1] == 0:
        end_idx = valid[-1][0]

    start_char = offsets[start_idx][0]
    end_char   = offsets[end_idx][1]
    # safety
    if end_char < start_char:
        end_char = start_char

    return text[start_char:end_char]

# quick sanity check on a few validation rows
for i in range(3):
    row = valid_df.iloc[i]
    pred = extract_span(row[text_col], row["sentiment"], tokenizer, model, max_len=MAX_LEN)
    print(f"sentiment: {row['sentiment']}")
    print("TEXT:     ", row[text_col])
    print("TARGET:   ", row[span_col])
    print("PREDICTED:", pred)
    print("---")

def jaccard(a: str, b: str) -> float:
    A = set(a.lower().split())
    B = set(b.lower().split())
    if not A and not B: return 1.0
    return len(A & B) / max(1, len(A | B))

# sample score over a few examples
scores = []
for i in range(50):  # small sample
    row = valid_df.iloc[i]
    pred = extract_span(row[text_col], row["sentiment"], tokenizer, model, max_len=MAX_LEN)
    scores.append(jaccard(pred, row[span_col]))
print("Sample Jaccard:", sum(scores)/len(scores))


BEST_DIR = f"{output_dir}-best"   # from earlier steps

# 1) Load best model + tokenizer
best_tok = AutoTokenizer.from_pretrained(BEST_DIR, use_fast=True)
best_model = AutoModelForQuestionAnswering.from_pretrained(BEST_DIR).to(device).eval()

@torch.no_grad()
def predict_batch(sentiments, texts, tokenizer, model, max_len=96, batch_size=64):
    preds = []
    N = len(texts)
    for i in range(0, N, batch_size):
        s_batch = sentiments[i:i+batch_size]
        t_batch = texts[i:i+batch_size]

        enc = tokenizer(
            s_batch, t_batch,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"]  # shape: (B, L, 2) on CPU tensors

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = out.start_logits    # (B, L)
        end_logits   = out.end_logits      # (B, L)

        # decode each sequence
        for b in range(start_logits.size(0)):
            sl = start_logits[b].cpu()
            el = end_logits[b].cpu()
            offs = offsets[b].tolist()     # list of (s,e)

            # mask out special tokens (offset (0,0)) so they aren't picked
            valid_idx = [k for k,(a,b_) in enumerate(offs) if b_ > 0]
            if not valid_idx:
                preds.append("")  # degenerate row
                continue

            # best span by maximizing start[i] + end[j] with i<=j (simple O(L^2) is fine for L~100)
            best_score, best_i, best_j = -1e9, valid_idx[0], valid_idx[0]
            for i2 in valid_idx:
                for j2 in valid_idx:
                    if j2 < i2:
                        continue
                    score = sl[i2].item() + el[j2].item()
                    if score > best_score:
                        best_score, best_i, best_j = score, i2, j2

            s_char, _ = offs[best_i]
            _, e_char = offs[best_j]
            text = t_batch[b]
            if e_char < s_char:
                e_char = s_char
            preds.append(text[s_char:e_char])
    return preds

# 2) Run predictions over your test set
test_sent = test["sentiment"].astype(str).str.lower().str.strip().tolist()
test_text = test[text_col].astype(str).tolist()
test_preds = predict_batch(test_sent, test_text, best_tok, best_model, max_len=MAX_LEN, batch_size=64)

# 3) (Optional) Build a Kaggle-style submission if you have textID
submission = pd.DataFrame({
    "textID": test.get("textID", pd.RangeIndex(len(test))),
    "selected_text": test_preds,
})
submission_path = f"{output_dir}-submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)












