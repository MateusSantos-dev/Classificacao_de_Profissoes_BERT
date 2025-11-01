from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding


def prepare_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    return tokenizer


def tokenize_function(
        examples: dict[str, list[str]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int
) -> BatchEncoding:
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        stride=128,
        return_overflowing_tokens=False,
           )


def encode_labels(dataset: DatasetDict) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    labels = sorted(set(dataset["train"]["label"]))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    def map_labels(example: dict) -> dict:
        example["label"] = label2id[example["label"]]
        return example

    dataset = dataset.map(map_labels)
    return dataset, label2id, id2label
