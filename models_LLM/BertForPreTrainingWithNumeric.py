# BertForPreTrainingWithNumeric (MLM용)
class BertForPreTrainingWithNumeric(BertPreTrainedModel):
    def __init__(self, config, gmt_embedder, numeric_cols):
        super().__init__(config)
        self.bert_with_numeric = BertModelWithNumeric(config, gmt_embedder, numeric_cols)
        self.cls = BertOnlyMLMHead(config)  # Masked LM head only
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        df_numerics=None,
    ):
        outputs = self.bert_with_numeric(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            df_numerics=df_numerics,
        )

        sequence_output = outputs["last_hidden_state"]
        prediction_scores = self.cls(sequence_output)  # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))

        return {
            "loss": loss,
            "logits": prediction_scores,
            "hidden_states": outputs["last_hidden_state"],
        }

