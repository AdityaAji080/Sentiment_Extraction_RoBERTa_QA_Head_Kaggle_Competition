# Tweet Sentiment Extraction with a RoBERTa QA Head

Given a tweet and its sentiment, find the words that carry that sentiment. This is the task of Kaggle's [Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction) competition. For example, with sentiment *negative*, the tweet "my boss is bullying me" should yield "bullying me".

The approach treats it as **extractive question answering**. The sentiment word is the question, the tweet is the context, and a RoBERTa model with a QA head predicts the start and end tokens of the answer span.

```
<s> negative </s></s> my boss is bullying me </s>
                      └─ start/end logits over tweet tokens ─┘  →  "bullying me"
```

## Method

- **Encoding.** `(sentiment, tweet)` pairs, truncating only the tweet (`max_len=96`).
- **Labels.** `selected_text` is located in the tweet and mapped to the first and last *tweet* tokens covering it, using the tokenizer's offset mapping and `sequence_ids`.
- **Training.** Cross-entropy over start and end logits with 🤗 `Trainer`: AdamW, 10% linear warmup, weight decay 0.01, fp16 on GPU, early stopping on validation loss. The 80/20 split is stratified by sentiment.
- **Decoding.** Choose the tweet-token pair *(i ≤ j)* that maximises `start[i] + end[j]`, then slice the original tweet with the character offsets.
- **Metric.** The competition's word-level Jaccard score, computed on the full validation split. Results are reported overall and per sentiment.

### Design note: spans come only from tweet tokens

With pair inputs, the question tokens (the sentiment word) also carry character offsets, but those offsets index into the *sentiment word*, not the tweet. Both labelling and decoding are therefore restricted to context tokens (`sequence_ids == 1`). This keeps a span that starts at the first character of a tweet labelled on the tweet's first token, and it guarantees that predictions are always substrings of the tweet.

The tests in `tests/` check this with a RoBERTa-style tokenizer built on the fly, so they run offline.

## Usage

```bash
pip install -r requirements.txt
# Download train.csv and test.csv from the competition page into data/

python train.py                                                  # roberta-base
python train.py --model roberta-large --lr 7e-6 --batch_size 8 --grad_accum 2
python predict.py outputs/roberta-base/best --out submission.csv
```

`train.py` saves the best checkpoint to `outputs/<model>/best`, and writes `val_metrics.json` (overall and per-sentiment Jaccard) and `val_predictions.csv` for error analysis.

`notebooks/kaggle_inference.ipynb` is the self-contained inference notebook for Kaggle's notebook-only submissions. It uses the same decoding.

## Repository layout

```
span_qa.py        # labelling, span decoding, batched prediction, Jaccard metric
train.py          # fine-tuning + validation Jaccard
predict.py        # submission file from a trained model
notebooks/        # Kaggle inference notebook
tests/            # offline tests for labelling and decoding
```

## Testing

```bash
pip install pytest && pytest -q tests/
```

## Ideas for improvement

- For *neutral* tweets the target is usually the whole tweet, which a rule can handle.
- Ensemble several folds and seeds by averaging logits.
- Replace the token-boundary post-processing with character-level span prediction.
