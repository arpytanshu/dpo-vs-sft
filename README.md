
### This repo logs the training runs, mistakes and learnings from finetuning LMs on QA type datasets using 2 approaches: SFT and DPO.

Why QA type data? (specifically, multiple choice answers)
A Multiple choice QA data sample typically has a context and/or a question, and multiple possible options for answer. One out of which is a valid answer to the question.  
`<...Context...> <...Question...> <...opt1...> <...opt2...> <...opt3...> <...opt4...> Answer: <...opt3...>`

This type of dataset is ideal for being trained in a Supervised Finetuning (SFT) setting where the sequence containing the Answer to the context forms the label for supervision.  
context : `<...Context...> <...Question...> <...opt1...> <...opt2...> <...opt3...> <...opt4...>`  
labels  : `Answer: <...opt3...>`  


The wrong options do not take part in SFT.
However in a Preference Optimization Finetuning setting like DPO, the correct answer forms the preferred completion choice, and the wrong answer forms the un-preferred choice.  
context     : `<...Context...> <...Question...> <...opt1...> <...opt2...> <...opt3...> <...opt4...>`  
preferred   : `Answer: <...opt2...>`  
un-preferred: `Answer: <...opt4...>`  




### Dataset
`HellaSwag: Rowan/hellaswag`  
train size: 39k, validation size: 10k

### Step 1. Get a baseline using SFT.

Best Baseline using SFT  
initial LR: 1.41e-5

| steps     | epoch     | train_loss    | eval_loss | eval_metrics(2k)  | val_set_metrics(10k)  |
|-----------|-----------|---------------|-----------|-------------------|-----------------------|
| 500       | 0.8       |  0.0369       |   0.0373  |   acc=0.275       |   acc=0.264           |
| 1000      | 1.6       |  0.0133       |   0.0160  |   acc=0.773       |   acc=0.778           |
| 1400      | 2.24      |  0.0065       |   0.0135  |   acc=0.818       |   acc=0.821           |


### Step 2. Fail multiple times

There were several failed attempts when training with DPO.
Amongst the failed attempts, the factors that were varied were:
- using "rmsprop"   (default=adamW)
- different betas   (default=0.1, 0.3, 0.5)
- different LR         (default=1.41e-5, 1e-5, 1e-4)
- beginning from a non-finetuned model.
- beginning from the SFT model from last step.


The checkpoint around ~500th step from SFT model gave accuracies of ~25%, which is the random guess number.
However the generations are aligned, i.e. the model will always generate one of the 4 given options in the context.

All of the above tries with DPO gave ~0 accuracies. i.e. the model predictions were completely off and werent' aligned in the expected format.

The reason was that the log_ps from the reference model were precomputed using a non SFT model. 
The reference log_ps were recomputed from the SFT model trained for 1000 steps, which reported accuracy of 0.778 on validation set.
The DPO model trained using these updated log_ps, were aligned in the expected format.

### Step 3. Fix issues, and try again.
Using the updated precomputed reference log_ps, adding a warmup for 150 steps and a slower LR from 1.41e-5 to 1e-5,
allowed the DPO method to improve the results marginally, before overfitting again by 700th step.


| steps     | epoch     | train_loss    | eval_loss | eval_metrics(2k)  | val_set_metrics(10k)  |
|-----------|-----------|---------------|-----------|-------------------|-----------------------|
| 0         | 0         |  0.6927       |   0.6933  |   acc=0.786       |   acc=0.778           |
| 500       | 0.3       |  0.2074       |   0.2513  |   acc=0.792       |   acc=0.789           |
