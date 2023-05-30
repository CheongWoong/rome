import json
import os
import argparse
from tqdm.auto import tqdm
from collections import defaultdict

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    args = parser.parse_args()

    with open('data/LAMA_TREx_test.json', 'r') as fin:
        data = json.load(fin)

    rel_map = {}

    for instance in data:
        uid = instance['uid']
        rel_id = instance['rel_id']
        rel_map[uid] = rel_id

    efficacy = []
    specificity = []
    rel_efficacy = defaultdict(list)
    rel_specificity = defaultdict(list)

    fnames = os.listdir(args.dir_name)
    for fname in tqdm(fnames):
        if not fname.startswith('case'):
            continue
        with open(os.path.join(args.dir_name, fname)) as fin:
            ret = json.load(fin)
            rel = rel_map[ret['case_id']]

            target_new = ret['requested_rewrite']['target_new']['str']

            post_rewrite_prompts_preds = ret['post']['rewrite_prompts_preds']
            # pre_rewrite_prompts_preds = ret['pre']['rewrite_prompts_preds']

            post_neighborhood_prompts_preds = ret['post']['neighborhood_prompts_preds']
            pre_neighborhood_prompts_preds = ret['pre']['neighborhood_prompts_preds']
        
            for pred in post_rewrite_prompts_preds:
                if target_new.strip().lower() == pred.strip().lower():
                    efficacy.append(1)
                    rel_efficacy[rel].append(1)
                else:
                    efficacy.append(0)
                    rel_efficacy[rel].append(0)

            for post_pred, pre_pred in zip(post_neighborhood_prompts_preds, pre_neighborhood_prompts_preds):
                if post_pred == pre_pred:
                    specificity.append(1)
                    rel_specificity[rel].append(1)
                else:
                    specificity.append(0)
                    rel_specificity[rel].append(0)

    efficacy_mean, efficacy_std = np.mean(efficacy), np.std(efficacy)
    specificity_mean, specificity_std = np.mean(specificity), np.std(specificity)
        
    print('Efficacy:', efficacy_mean, '+-', efficacy_std)
    print('Specificity:', specificity_mean, '+-', specificity_std)

    rels = sorted(list(rel_efficacy.keys()))

    for rel in rels:
        efficacy_mean, efficacy_std = np.mean(rel_efficacy[rel]), np.std(rel_efficacy[rel])
        specificity_mean, specificity_std = np.mean(rel_specificity[rel]), np.std(rel_specificity[rel])
        print('rel_id:', rel)
        print('Efficacy:', efficacy_mean, '+-', efficacy_std)
        print('Specificity:', specificity_mean, '+-', specificity_std)