
#%%
import fire
import torch
import random
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import sft_custom_eval, create_logger

def main(checkpoint_path, run_eval=True, device='cuda'):
    
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, padding_side='left')
    logger = create_logger(Path(checkpoint_path).parent)
    if run_eval:
        dataset = load_from_disk('dataset_cache/hellaswag_sft')['validation']
        res = sft_custom_eval(model=model, tokenizer=tokenizer, dataset=dataset)
        logger.info(res)
    else:
        dataset = load_from_disk('dataset_cache/hellaswag_dpo_sft_precomputed')['eval']
        ix = random.randint(0, len(dataset)-1)
        input_ids = torch.tensor(dataset[ix]['prompt_input_ids']).view(1, -1).to(model.device)
        out = model.generate(inputs=input_ids)

        print(tokenizer.decode(out.ravel()))


if __name__ == '__main__':
    fire.Fire(main)
