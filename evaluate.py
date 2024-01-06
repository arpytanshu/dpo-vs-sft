
import fire
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import custom_eval, create_logger

def main(checkpoint_path):
    dataset = load_from_disk('dataset_cache/hellaswag_sft')['validation']
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, padding_side='left')
    logger = create_logger(Path(checkpoint_path).parent)

    res = custom_eval(model=model, tokenizer=tokenizer, dataset=dataset)
    logger.info(res)


if __name__ == '__main__':
    fire.Fire(main)
