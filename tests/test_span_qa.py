"""CPU-only tests for span labelling and decoding; no model download needed.

A small byte-level BPE tokenizer is trained on the fly and wrapped as RobertaTokenizerFast, so pair
encoding, special tokens and offsets behave exactly like roberta-base/large.

    pytest -q tests/
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from span_qa import decode_span, encode_example, jaccard  # noqa: E402

CORPUS = ["positive negative neutral", "Last session of the day", "So excited for tomorrow!!",
          "I`d have responded, if I were going", "this is so sad", "sooo happy right now",
          "my boss is bullying me", "what interview! leave me alone"] * 20


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import ByteLevelBPETokenizer, processors
    from transformers import RobertaTokenizerFast

    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    bpe = ByteLevelBPETokenizer(trim_offsets=True)
    bpe.train_from_iterator(CORPUS, vocab_size=400, min_frequency=1, special_tokens=specials)
    bpe._tokenizer.post_processor = processors.RobertaProcessing(
        ("</s>", bpe.token_to_id("</s>")), ("<s>", bpe.token_to_id("<s>")), trim_offsets=True)
    return RobertaTokenizerFast(tokenizer_object=bpe._tokenizer, bos_token="<s>", eos_token="</s>",
                                sep_token="</s>", cls_token="<s>", unk_token="<unk>", pad_token="<pad>",
                                mask_token="<mask>")


def span_of(tok, sentiment, text, selected):
    item = encode_example(tok, sentiment, text, selected, max_len=32)
    enc = tok(sentiment, text, max_length=32, padding="max_length", truncation="only_second",
              return_offsets_mapping=True)
    s, e = int(item["start_positions"]), int(item["end_positions"])
    return enc, s, e


@pytest.mark.parametrize("text,selected", [
    ("So excited for tomorrow!!", "So excited"),           # span at the very start of the tweet
    ("this is so sad", "so sad"),                          # span at the end
    ("I`d have responded, if I were going", "I`d have responded, if I were going"),  # whole tweet
    (" sooo happy right now", "sooo happy"),               # leading space in the raw tweet
])
def test_labels_point_at_context_tokens_and_recover_the_span(tokenizer, text, selected):
    enc, s, e = span_of(tokenizer, "positive", text, selected)
    seq = enc.sequence_ids()
    assert seq[s] == 1 and seq[e] == 1, "label points outside the tweet (e.g. at the sentiment word)"
    offs = enc["offset_mapping"]
    assert text[offs[s][0]:offs[e][1]].strip() == selected.strip()


def test_decode_never_returns_question_tokens(tokenizer):
    text = "my boss is bullying me"
    enc = tokenizer("negative", text, return_offsets_mapping=True)
    n = len(enc["input_ids"])
    start, end = torch.zeros(n), torch.zeros(n)
    start[1] = end[1] = 100.0                 # model "prefers" the sentiment token...
    ctx = [i for i, sid in enumerate(enc.sequence_ids()) if sid == 1]
    start[ctx[2]], end[ctx[-1]] = 5.0, 5.0    # ...but only context tokens are allowed
    pred = decode_span(start, end, enc["offset_mapping"], enc.sequence_ids(), text)
    assert pred == "is bullying me"


def test_jaccard_matches_competition_metric():
    assert jaccard("So excited", "so EXCITED") == 1.0
    assert jaccard("so sad", "this is so sad") == 0.5
    assert jaccard("", "") == 1.0
