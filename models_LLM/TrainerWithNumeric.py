class TrainerWithNumeric(Trainer):
    def __init__(self, *args, numeric_cols=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.numeric_cols = numeric_cols

    def compute_loss(self, model, inputs, return_outputs=False):
        df_numerics = inputs.pop("df_numerics", None)
        outputs = model(**inputs, df_numerics=df_numerics)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss
from transformers import TrainingArguments
from my_module import TrainerWithNumeric  # 위 정의된 클래스

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,  # ✅ 중요
)

trainer = TrainerWithNumeric(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    numeric_cols=["x1", "x2"]  # 반드시 명시
)

trainer.train()
