from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from datasets import DatasetDict
from src.eval import compute_metrics
from src.callbacks import EarlyStoppingCallback


def train_model(
        model: PreTrainedModel,
        dataset: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: str = "./results/models/bert/freeze_8",
        num_train_epochs: int = 10,
        learning_rate: float = 2e-5,
        use_early_stopping: bool = False,
        patience_steps: int = 300,
        min_steps: int = 500,
        improvement_threshold: float = 0
) -> Trainer:

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,

        num_train_epochs=num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        save_total_limit=2,
        report_to="none",
        fp16=True,

        logging_dir="./results/logs",
        logging_steps=10,
        log_level="info",
        log_level_replica="warning",
        log_on_each_node=True,
        output_dir=output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    if use_early_stopping:
        early_stopping = EarlyStoppingCallback(
            patience_steps=patience_steps,
            min_steps=min_steps,
            improvement_threshold=improvement_threshold
        )
        trainer.add_callback(early_stopping)

    trainer.train()
    return trainer
