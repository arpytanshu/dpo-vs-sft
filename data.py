import os
import fire
import random
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset


COLS_TO_REMOVE = ['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 
                  'endings', 'source_id', 'split', 'split_type', 'label']

MAPPING= {'0':'A', '1':'B', '2':'C', '3':'D'}


def getHellaSwag(cache_path):
    try:
        dataset = load_from_disk(cache_path)
    except:
        dataset = load_dataset('Rowan/hellaswag')
        dataset.save_to_disk(cache_path)
    return dataset

def get_prompt(sample):
    return f'''Task: Choose the most appropriate ending for the statement from the given options.
Context: {sample['activity_label']}
Statement: {sample['ctx']}
Options:
A. {sample['endings'][0]}
B. {sample['endings'][1]}
C. {sample['endings'][2]}
D. {sample['endings'][3]}
Answer: '''

def sft_format_sample(sample):
    prompt = get_prompt(sample)
    if sample['label'] != '':
        ans_choice = MAPPING[sample['label']]
        response = f'''{ans_choice}. {sample['endings'][int(sample['label'])]}'''
        return dict(prompt=prompt, response=response, sequence=prompt+response)
    else:
        return dict(prompt=prompt)

def prepSFT(cache_path, dev_size):

    # load hellaswag dataset, from cache, or hub
    orig_dataset_cache_path = Path(cache_path) / 'hellaswag/'
    dataset = getHellaSwag(orig_dataset_cache_path)

    # A seperate validation set for monitoring generalization losses.
    eval_set_ix = random.sample(range(len(dataset['validation'])), dev_size)
    dataset['eval'] = dataset['validation'].select(eval_set_ix)
    
    dataset = dataset.map(sft_format_sample, remove_columns=COLS_TO_REMOVE)
    save_path = Path(cache_path) / 'hellaswag_sft/'
    dataset.save_to_disk(save_path)
    print(f"saved dataset processed for SFT at {save_path}")

def dpo_format_sample_3x(sample):
    prompt = get_prompt(sample)

    if sample['label'] != '':
        list_3x = []
        chosen_ix = sample['label']
        chosen_char = MAPPING[chosen_ix]
        chosen = f'''{chosen_char}. {sample['endings'][int(chosen_ix)]}'''
        rejected_ixs = list(set(list(MAPPING.keys())) - set([chosen_ix]))
        for ix in rejected_ixs:
            rejected_char = MAPPING[ix]
            rejected = f'''{rejected_char}. {sample['endings'][int(ix)]}'''
            list_3x.append(dict(prompt=prompt, chosen=chosen, rejected=rejected))
        return list_3x
    else:
        return [dict(prompt=prompt)]

def dpo_format_sample(sample):
    prompt = get_prompt(sample)
    
    if sample['label'] != '':
        list_3x = []
        chosen_ix = sample['label']
        chosen_char = MAPPING[chosen_ix]
        chosen = f'''{chosen_char}. {sample['endings'][int(chosen_ix)]}'''
        rejected_ixs = list(set(list(MAPPING.keys())) - set([chosen_ix]))
        rejected_ix = random.sample(rejected_ixs, 1)[0]
        rejected_char = MAPPING[rejected_ix]
        rejected = f'''{rejected_char}. {sample['endings'][int(rejected_ix)]}'''
        return dict(prompt=prompt, chosen=chosen, rejected=rejected)
    else:
        return dict(prompt=prompt)

def prepDPO(cache_path, dev_size, dpo_3x):
    cache_path = 'dataset_cache/'
    # load hellaswag dataset, from cache, or hub
    orig_dataset_cache_path = Path(cache_path) / 'hellaswag/'
    dataset = getHellaSwag(orig_dataset_cache_path)

    if dpo_3x:
        samples_list = []
        for raw_sample in dataset['train']:
            sample_3x = dpo_format_sample_3x(raw_sample)
            for sample in sample_3x:
                samples_list.append(sample)
        dataset['train'] = Dataset.from_list(samples_list)
        save_path = Path(cache_path) / 'hellaswag_dpo_3x/'
    else:
        dataset['train'] = dataset['train'].map(dpo_format_sample, remove_columns=COLS_TO_REMOVE)
        save_path = Path(cache_path) / 'hellaswag_dpo/'

    # A seperate validation set for monitoring generalization losses.
    eval_set_ix = random.sample(range(len(dataset['validation'])), dev_size)
    dataset['eval'] = dataset['validation'].select(eval_set_ix)

    dataset['eval'] = dataset['eval'].map(dpo_format_sample, remove_columns=COLS_TO_REMOVE)
    dataset['test'] = dataset['test'].map(dpo_format_sample, remove_columns=COLS_TO_REMOVE)
    dataset['validation'] = dataset['validation'].map(dpo_format_sample, remove_columns=COLS_TO_REMOVE)

    dataset.save_to_disk(save_path)
    print(f"saved dataset processed for DPO at {save_path}")

def prepSFT(cache_path, dev_size):

    cache_path = 'dataset_cache/'
    # load hellaswag dataset, from cache, or hub
    orig_dataset_cache_path = Path(cache_path) / 'hellaswag/'
    dataset = getHellaSwag(orig_dataset_cache_path)
    
    # A seperate validation set for monitoring generalization losses.
    eval_set_ix = random.sample(range(len(dataset['validation'])), dev_size)
    dataset['eval'] = dataset['validation'].select(eval_set_ix)
    
    dataset = dataset.map(sft_format_sample, remove_columns=COLS_TO_REMOVE)
    save_path = Path(cache_path) / 'hellaswag_sft/'
    dataset.save_to_disk(save_path)
    print(f"saved dataset processed for SFT at {save_path}")

def main(cache_path='dataset_cache/', 
         dev_size=1024,
         prep_sft=False, 
         dpo_3x=False,
         prep_dpo=False,
         all=True):
    
    os.makedirs(cache_path, exist_ok=True)

    if prep_sft:
        prepSFT(cache_path, dev_size)
    if prep_dpo:
        prepDPO(cache_path, dev_size, dpo_3x=dpo_3x)
    if all:
        prepSFT(cache_path, dev_size)
        prepDPO(cache_path, dev_size, dpo_3x=True)
        prepDPO(cache_path, dev_size, dpo_3x=False)


if __name__ == '__main__':
    fire.Fire(main)
