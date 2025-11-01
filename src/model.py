from transformers import AutoModelForSequenceClassification, PreTrainedModel


def create_bert_model(
        model_name: str,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        freeze_bert_first_n_layers: int = 0
) -> PreTrainedModel:

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    if freeze_bert_first_n_layers > 0:
        for layer in model.bert.encoder.layer[:freeze_bert_first_n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    return model
