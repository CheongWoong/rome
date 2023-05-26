"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def compute_rewrite_quality_LAMA_TREx(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    neighborhood_prompts = [prompt['prompt'] for prompt in record["neighborhood_prompts"]]
    neighborhood_ids = [prompt['uid'] for prompt in record["neighborhood_prompts"]]

    # Form a list of lists of prefixes to test.
    all_prompts = [
        rewrite_prompts,
        neighborhood_prompts
    ]
    all_prompts = list(chain(*all_prompts))

    ret_preds, ret_probs = get_batch_prediction(model, tok, all_prompts, target_new["str"])

    ret = {
        "rewrite_prompts_preds": ret_preds[:len(rewrite_prompts)],
        "rewrite_prompts_probs": ret_probs[:len(rewrite_prompts)],
        "neighborhood_ids": neighborhood_ids,
        "neighborhood_prompts_preds": ret_preds[len(rewrite_prompts):],
        "neighborhood_prompts_probs": ret_probs[len(rewrite_prompts):],
    }
    
    return ret

def get_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
):
    """ """

    prompts = [prefix for prefix in prefixes]
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        probs = torch.softmax(gathered, dim=1)
        ans = torch.argmax(gathered, dim=1)
        preds = tok.batch_decode(ans.detach().cpu())
        pred_probs = []
        for max_idx, prob in zip(ans, probs):
            pred_probs.append(prob[max_idx].item())

    return preds, pred_probs