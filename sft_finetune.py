
#%%

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from data import getHellaSwag
from utils import Collater, CustomDataset, MyCallback, create_logger


MODEL_STR = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"



output_dir = "checkpoints/run5"
checkpoint_path=None
lora=False
lora_r=32
lora_alpha=32



# MODEL
if lora:
    model = AutoModelForCausalLM.from_pretrained(MODEL_STR, load_in_8bit=True)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_STR, torch_dtype=torch.bfloat16)

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_STR, use_fast=True, padding_side='left')
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# DATASET
dataset = getHellaSwag('dataset_cache/hellaswag_sft')
train_dataset = CustomDataset(dataset['train'])
eval_dataset = dataset['eval']
collater = Collater(tokenizer, mode='train')

# TRAINER
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1.41e-5,
    logging_steps=10,
    num_train_epochs=3,
    max_steps=2500,
    report_to="tensorboard",
    evaluation_strategy='steps',
    eval_steps=100,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    gradient_checkpointing=True,
    bf16=True,
    remove_unused_columns=False
)

custom_callbacks = [MyCallback(eval_dataset=dataset['eval'], 
                                logger=create_logger(training_args.output_dir))]

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=collater,
    callbacks=custom_callbacks,
)



if checkpoint_path is not None:
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()



# main(checkpoint_path='checkpoints/run1', lora=True, lora_r=64, lora_alpha=64)
# main(output_dir="checkpoints/run2", checkpoint_path='checkpoints/run2/checkpoint-1750')



# %%
