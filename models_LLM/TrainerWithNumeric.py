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


