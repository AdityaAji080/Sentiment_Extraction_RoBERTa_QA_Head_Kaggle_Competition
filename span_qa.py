"""Sentiment-span extraction framed as extractive question answering.

The tokenizer encodes each example as a pair: (question = sentiment word, context = tweet).
RoBERTa then predicts start/end logits over the tokens, and the span is mapped back to
characters with the offset mapping.

Both labelling and decoding are restricted to *context* tokens (sequence_ids == 1). Offsets of
question tokens are relative to the sentiment word, not the tweet, so letting them through would
point labels at the question and slice the wrong characters out of the tweet.
"""

import torch


def context_token_range(sequence_ids):
    """Indices of the tokens that belong to the context (the tweet)."""
    return [i for i, s in enumerate(sequence_ids) if s == 1]


def encode_example(tokenizer, sentiment, text, selected_text=None, max_len=96):
    """Tokenise one (sentiment, tweet) pair; add start/end token labels when `selected_text` is given."""
    enc = tokenizer(str(sentiment).lower().strip(), str(text), max_length=max_len, padding="max_length",
                    truncation="only_second", return_offsets_mapping=True)
    item = {"input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"])}
    if selected_text is None:
        return item

    text, target = str(text), str(selected_text).strip()
    start_char = text.find(target) if target else -1
    if start_char == -1:  # label missing or not found verbatim: fall back to the whole tweet
        start_char, end_char = 0, len(text)
    else:
        end_char = start_char + len(target)

    offsets = enc["offset_mapping"]
    ctx = [i for i in context_token_range(enc.sequence_ids()) if offsets[i][1] > offsets[i][0]]
    starts = [i for i in ctx if offsets[i][1] > start_char]
    ends = [i for i in ctx if offsets[i][0] < end_char]
    if not ctx or not starts or not ends:  # span truncated away: point at <s>, as standard QA does
        start_tok = end_tok = 0
    else:
        start_tok, end_tok = starts[0], ends[-1]
        end_tok = max(end_tok, start_tok)
    item["start_positions"] = torch.tensor(start_tok)
    item["end_positions"] = torch.tensor(end_tok)
    return item


def decode_span(start_logits, end_logits, offsets, sequence_ids, text, max_span_tokens=None):
    """Best character span (i <= j, both context tokens) maximising start[i] + end[j]."""
    ctx = [i for i in context_token_range(sequence_ids) if offsets[i][1] > offsets[i][0]]
    if not ctx:
        return ""
    idx = torch.tensor(ctx)
    s, e = start_logits[idx].float(), end_logits[idx].float()
    scores = s[:, None] + e[None, :]
    valid = torch.ones_like(scores, dtype=torch.bool).triu()
    if max_span_tokens is not None:
        valid &= ~torch.ones_like(valid).triu(max_span_tokens)
    scores = scores.masked_fill(~valid, float("-inf"))
    flat = int(torch.argmax(scores))
    i, j = ctx[flat // len(ctx)], ctx[flat % len(ctx)]
    return str(text)[offsets[i][0]:offsets[j][1]]


def jaccard(a, b):
    """Word-level Jaccard similarity, as defined by the Tweet Sentiment Extraction competition."""
    a, b = set(str(a).lower().split()), set(str(b).lower().split())
    if not a and not b:  # undefined (0/0) in the official metric; two empty strings match
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


@torch.no_grad()
def predict(model, tokenizer, sentiments, texts, max_len=96, batch_size=64, device="cpu"):
    """Predict the selected text for each (sentiment, tweet) pair."""
    model.eval()
    preds = []
    for k in range(0, len(texts), batch_size):
        sent = [str(s).lower().strip() for s in sentiments[k:k + batch_size]]
        txt = [str(t) for t in texts[k:k + batch_size]]
        enc = tokenizer(sent, txt, max_length=max_len, padding=True, truncation="only_second",
                        return_offsets_mapping=True, return_tensors="pt")
        out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
        for b in range(len(txt)):
            preds.append(decode_span(out.start_logits[b].cpu(), out.end_logits[b].cpu(),
                                     enc["offset_mapping"][b].tolist(), enc.sequence_ids(b), txt[b]))
    return preds
