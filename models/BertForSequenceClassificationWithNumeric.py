###############################################################################################
#BertForSequenceClassificationWithNumeric: 텍스트 + 수치형 입력을 받아 분류를 수행하는 통합 모델
#DataCollatorForNumericMLM: 수치형 입력이 포함된 MLM pretraining용 데이터 콜레이터
###############################################################################################
# BertForSequenceClassificationWithNumeric
class BertForSequenceClassificationWithNumeric(BertPreTrainedModel):
    def __init__(self, config, gmt_embedder, numeric_cols):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert_with_numeric = BertModelWithNumeric(config, gmt_embedder, numeric_cols)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
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

        pooled_output = outputs["pooled_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs["last_hidden_state"],
            attentions=None,
        )

# DataCollatorForNumericMLM
class DataCollatorForNumericMLM:
    def __init__(self, tokenizer, gmt_embedder, numeric_cols, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.gmt_embedder = gmt_embedder
        self.numeric_cols = numeric_cols
        self.mlm_prob = mlm_prob

    def __call__(self, examples):
        text_inputs = self.tokenizer([ex["text"] for ex in examples], padding=True, truncation=True, return_tensors="pt")
        df_numerics = pd.DataFrame([{col: ex[col] for col in self.numeric_cols} for ex in examples])

        # MLM 마스킹 처리
        input_ids = text_inputs["input_ids"].clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, self.mlm_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in input_ids.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return {
            "input_ids": input_ids,
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
            "df_numerics": df_numerics
        }
