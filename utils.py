
import os
import torch
import random
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import TrainerCallback
import sys

class SFTCollater:
    '''
    collater = Collater(tokenizer, ctx_labels=False, mode='train')
    batch = collater.get_debug_batch()
    out = collater.train_collater(batch)
    print(out['input_ids'])
    print(out['attention_mask'])
    print(out['labels'])
    '''
    def __init__(self, tokenizer, ctx_labels=False, mode='train'):
        self.tokenizer = tokenizer
        self.mode = mode
        self.ctx_labels = ctx_labels
        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id
        self.pad_id = tokenizer.pad_token_id
        
    def __call__(self, batch):
        if self.mode == 'train':
            return self.train_collater(batch)
        else:
            return self.custom_eval_collater(batch)
    
    def get_debug_batch(self):
        return [
            {
                'prompt': 'Task: Choose the date.\nAnswer: ',
                'response': 'A. However'
            },
            {
                'prompt': 'Task: How to clean an ultra boost sole.\nAnswer: ',
                'response': 'D. After you use.'
            }
        ]
        
    def custom_eval_collater(self, batch):      
        prompt = [x['prompt'] for x in batch]
        responses = [x['response'][0] for x in batch]
        inputs = self.tokenizer(prompt, padding=True, return_tensors='pt')
        responses = self.tokenizer(responses, add_special_tokens=False, return_tensors='pt')['input_ids']
        return inputs, responses

    def train_collater(self, batch):        
        response = [x['response'] for x in batch]
        prompt = [x['prompt'] for x in batch]
        max_length = 0
        
        inputs = []
        labels = []
        for p, r in zip(prompt, response):
            p_ids = self.tokenizer(p, add_special_tokens=False)['input_ids']
            r_ids = self.tokenizer(r, add_special_tokens=False)['input_ids']
            input = [self.bos_id] + p_ids + r_ids + [self.eos_id]
            label = [-100] * (len(p_ids) + 1) + r_ids + [self.eos_id]
            if len(input) > max_length:
                max_length = len(input)
            inputs.append(input)
            labels.append(label)
            
        stacked_inputs = []
        stacked_labels = []
        stacked_attention_mask = []
        for input, label in zip(inputs, labels):
            pad_len = (max_length-len(input))
            stacked_inputs.append([self.pad_id]*pad_len + input)
            stacked_attention_mask.append([0]*pad_len + [1]*len(input))
            if self.ctx_labels:
                stacked_labels.append([-100]*(pad_len+1) + input[1:])
            else:
                stacked_labels.append([-100]*pad_len + label)
        
        return dict(
            input_ids=torch.tensor(stacked_inputs),
            attention_mask=torch.tensor(stacked_attention_mask),
            labels=torch.tensor(stacked_labels))




class CustomDataset(Dataset):
    '''
    The default Random Sampler samples each sample exactly once per epoch.
    Wrapping the dataset inside this class ensures that the samples
     returned from the dataset are truly stochastic.
    '''
    def __init__(self, dataset):
        self.ds = dataset
    def __getitem__(self, index):
        ix = random.randint(0, len(self.ds)-1)
        return self.ds[ix]
    def __len__(self):
        return len(self.ds)

def sft_custom_eval(model, tokenizer, dataset):
    
    collater = SFTCollater(tokenizer, ctx_labels=None, mode=None)
    dl = DataLoader(dataset, batch_size=24, shuffle=False, collate_fn=collater)
    
    model.eval()
    correct = 0
    total = 0
    for ix, batch in enumerate(dl):
        inputs, responses = batch
        with torch.no_grad():
            out = model(**inputs.to(model.device))
        logits = out.logits[:,-1,:]
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        for gt, pred in zip(responses.ravel().tolist(), preds.cpu().ravel().tolist()):
            total += 1
            if gt == pred:
                correct += 1
        sys.stdout.write("\rbatch:{} correct={} total={} accuracy={}".format\
                         (ix, correct, total, round(correct/total, 3)))

    return dict(correct=correct, total=total, accuracy=round(correct/total, 3))


class DPOCollater:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id
        self.pad_id = tokenizer.pad_token_id
        
    def __call__(self, batch):
        return self.custom_eval_collater(batch)
    
    def get_debug_batch(self):
        return [
            {
                'prompt': 'Task: Choose the date.\nAnswer: ',
                'chosen': 'A. However',
                'rejected': 'B. Somewhere'
            },
            {
                'prompt': 'Task: How to clean an ultra boost sole.\nAnswer: ',
                'chosen': 'D. After you use.',
                'rejected': 'C. Kill them all.'
            }
        ]
        
    def custom_eval_collater(self, batch):      
        prompt = [x['prompt'] for x in batch]
        responses = [x['chosen'][0] for x in batch]
        inputs = self.tokenizer(prompt, padding=True, return_tensors='pt')
        responses = self.tokenizer(responses, add_special_tokens=False, return_tensors='pt')['input_ids']
        return inputs, responses

def dpo_custom_eval(model, tokenizer, dataset):
    # collater = SFTCollater(tokenizer, ctx_labels=None, mode=None)
    collater = DPOCollater(tokenizer)
    dl = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collater)
    
    model.eval()
    correct = 0
    total = 0
    for ix, batch in enumerate(dl):
        inputs, responses = batch
        with torch.no_grad():
            out = model(**inputs.to(model.device))
        logits = out.logits[:,-1,:]
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        for gt, pred in zip(responses.ravel().tolist(), preds.cpu().ravel().tolist()):
            total += 1
            if gt == pred:
                correct += 1
        sys.stdout.write("\rbatch:{} correct={} total={} accuracy={}".format\
                         (ix, correct, total, round(correct/total, 3)))

    return dict(correct=correct, total=total, accuracy=round(correct/total, 3))

class MyCallback(TrainerCallback):
    def __init__(self, eval_dataset, eval_fn, logger):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.eval_fn = eval_fn
        self.logger = logger

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training...")
    
    def on_log(self, args, state, control, **kwargs):
        self.logger.info(kwargs['logs'])
    
    def on_evaluate(self, args, state, control, **kwargs):
        result = self.eval_fn(model=kwargs['model'], 
                              tokenizer=kwargs['tokenizer'],
                              dataset=self.eval_dataset)
        self.logger.info(result)

        ix = random.randint(0, len(self.eval_dataset)-1)
        input_ids = torch.tensor(self.eval_dataset[ix]['prompt_input_ids']).view(1, -1).to(kwargs['model'].device)
        out = kwargs['model'].generate(inputs=input_ids)
        print(kwargs['tokenizer'].decode(out.ravel()))


def create_logger(checkpoint_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    os.makedirs(checkpoint_path, exist_ok=True)
    f_handler = logging.FileHandler(Path(checkpoint_path) / 'logs.log')
    f_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    logger.addHandler(f_handler)
    return logger
