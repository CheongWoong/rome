import json
from pathlib import Path
from collections import defaultdict
import random

import torch
from transformers import AutoTokenizer

from util.globals import *


class LAMA_TREx_Dataset:
    """
    Dataset of factual knowledge based on LAMA_TREx.
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, *args, **kwargs):
        random.seed(0)
        data_dir = Path(data_dir)
        LAMA_TREx_loc = data_dir / "LAMA_TREx_test.json"
        assert LAMA_TREx_loc.exists()

        with open(LAMA_TREx_loc, "r") as f:
            raw = json.load(f)

        rel_records = defaultdict(list)
        for record in raw:
            rel_records[record["rel_id"]].append(record)

        data = []
        for i, record in enumerate(raw):
            neighborhood_candidates = random.sample(rel_records[record["rel_id"]], 10)
            neighborhood_records = []
            for cand in neighborhood_candidates:
                if cand["uid"] != record["uid"]:
                    neighborhood_records.append(cand)

            data.append(
                {
                    "case_id": record["uid"],
                    "rel_id": record["rel_id"],
                    "requested_rewrite": {
                        "prompt": record["truncated_input"].replace(record["subj"], "{}"),
                        "subject": record["subj"],
                        "target_new": {"str": record["output"]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "neighborhood_prompts": [
                        {
                            "prompt": neighborhood_record["truncated_input"],
                            "uid": neighborhood_record["uid"]
                        }
                        for neighborhood_record in neighborhood_records
                    ],
                    "paraphrase_prompts": [],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data

        with open(str(LAMA_TREx_loc).replace('.json', '_processed.json'), 'w') as fout:
            json.dump(data, fout)

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
