
#%%

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer
from data import getHellaSwag
from utils import MyCallback, create_logger, dpo_custom_eval


MODEL_STR = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "checkpoints/dpo_run14"
checkpoint_path=None

# model = AutoModelForCausalLM.from_pretrained(MODEL_STR, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("checkpoints/sft_run6/checkpoint-1000", torch_dtype=torch.bfloat16)

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_STR, use_fast=True, padding_side='left')
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# DATASET
dataset = getHellaSwag('dataset_cache/hellaswag_dpo_sft_precomputed')
train_dataset = dataset['train']
eval_dataset = dataset['eval']


# TRAINER
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    logging_steps=10,
    num_train_epochs=3,
    max_steps=2000,
    report_to="tensorboard",
    evaluation_strategy='steps',
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    gradient_checkpointing=True,
    bf16=True,
    remove_unused_columns=False,
    warmup_steps=150,
    optim="rmsprop",
)

custom_callbacks = [MyCallback(eval_dataset=dataset['eval'],
                               eval_fn=dpo_custom_eval,
                               logger=create_logger(training_args.output_dir))]


trainer = DPOTrainer(
    model=model,
    ref_model=model,
    args=training_args,
    beta=0.1, # DPO beta
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    generate_during_eval=False,
    precompute_ref_log_probs=True,
    callbacks=custom_callbacks,
)


if not trainer.train_dataset.features.get('reference_rejected_logps'):
    
    trainer.get_eval_dataloader()
    eval_dataset = trainer.eval_dataset
    print('precomputed eval dataset')
    trainer.get_train_dataloader()
    train_dataset = trainer.train_dataset
    print('precomputed train dataset')

    dataset['train'] = train_dataset
    dataset['eval'] = eval_dataset
    
    dataset.save_to_disk('dataset_cache/hellaswag_dpo_sft_precomputed')

else:
    trainer._precomputed_train_ref_log_probs = True
    trainer._precomputed_eval_ref_log_probs = True
    

# trainer.evaluate()
    
if checkpoint_path is not None:
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()



# %%
