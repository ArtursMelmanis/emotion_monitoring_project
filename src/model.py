import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int = 3):
        super(SentimentClassifier, self).__init__()
        config = BertConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        self.bert_model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=config
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
