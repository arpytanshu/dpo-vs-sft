
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from torch.utils.data import DataLoader
import fire
from utils import custom_eval, create_logger
from pathlib import Path

def main(checkpoint_path):
    dataset = load_from_disk('dataset_cache/hellaswag_sft')['eval']
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, padding_side='left')
    logger = create_logger(Path(checkpoint_path).parent)

    res = custom_eval(model=model, tokenizer=tokenizer, dataset=dataset)
    logger.info(str({'message': 'Logging from evaluate.py'}.update(res)))

if __name__ == '__main__':
    fire.Fire(main)
