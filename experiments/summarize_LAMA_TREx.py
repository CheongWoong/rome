import json
import os
import argparse
from tqdm.auto import tqdm

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    args = parser.parse_args()

    efficacy = []
    specificity = []

    fnames = os.listdir(args.dir_name)
    for fname in tqdm(fnames):
        if not fname.startswith('case'):
            continue
        with open(os.path.join(args.dir_name, fname)) as fin:
            ret = json.load(fin)

            target_new = ret['requested_rewrite']['target_new']['str']

            post_rewrite_prompts_preds = ret['post']['rewrite_prompts_preds']
            # pre_rewrite_prompts_preds = ret['pre']['rewrite_prompts_preds']

            post_neighborhood_prompts_preds = ret['post']['neighborhood_prompts_preds']
            pre_neighborhood_prompts_preds = ret['pre']['neighborhood_prompts_preds']
        
            for pred in post_rewrite_prompts_preds:
                if target_new.strip().lower() == pred.strip().lower():
                    efficacy.append(1)
                else:
                    efficacy.append(0)

            for post_pred, pre_pred in zip(post_neighborhood_prompts_preds, pre_neighborhood_prompts_preds):
                if post_pred == pre_pred:
                    specificity.append(1)
                else:
                    specificity.append(0)

    efficacy_mean, efficacy_std = np.mean(efficacy), np.std(efficacy)
    specificity_mean, specificity_std = np.mean(specificity), np.std(specificity)
    
    print('Efficacy:', efficacy_mean, '+-', efficacy_std)
    print('Specificity:', specificity_mean, '+-', specificity_std)