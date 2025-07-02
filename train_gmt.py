# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:44:43 2025

@author: Hosik
"""

# trainer.py
from transformers import BertTokenizer, BertForMaskedLM, T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset
import wandb
import os
from torch.utils.tensorboard import SummaryWriter

class NumericDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {"labels": torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

class NumericEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, gmt_tokenizer, inputs):
        self.tokenizer = tokenizer
        self.gmt = gmt_tokenizer
        self.inputs = inputs

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs['model']
        model.eval()
        gen = model.generate(self.inputs['input_ids'], max_length=10)
        decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        numeric_preds = []
        for val in decoded:
            try:
                num = float(val.strip())
                approx = self.gmt.token2value(self.gmt.value2token(num))
                numeric_preds.append(approx)
            except:
                numeric_preds.append(None)
        wandb.log({"sample_numeric_predictions": numeric_preds})
        return control

def train_t5_model(train_dataset, eval_dataset, model_path='t5-small', output_dir='./results_t5', tokenizer=None, gmt=None, val_inputs=None):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        report_to=["wandb", "tensorboard"],
        run_name="t5_numeric"
    )
    wandb.init(project="gmt-numeric", name="t5_run", dir=os.getenv("WANDB_DIR", "."))
    callbacks = []
    if tokenizer and gmt and val_inputs:
        callbacks.append(NumericEvalCallback(tokenizer, gmt, val_inputs))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks
    )
    trainer.train()
    return model


