from src.analysis import (train_logistic_model, generate_global_interpretation, generate_local_interpretation,
                          analyze_model_errors, print_confusion_analysis)
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from src.data.load import load_dataset, make_splits
from src.preprocessing import tokenize_function, encode_labels

if __name__ == "__main__":

    # modelo regressão logistica com tf-idf
    model, x_test, y_test = train_logistic_model(data_path="dataset/ep2-train.csv")
    feature_names = model.named_steps['tfidfvectorizer'].get_feature_names_out()
    generate_global_interpretation(model, feature_names)

    for i in range(3):
        random_idx = random.randint(0, len(x_test) - 1)
        print(f"Explicando amostra aleatória: índice {random_idx}")
        generate_local_interpretation(model, x_test.iloc[random_idx],
                                      output_path=f"results/analysis/local_explanation{i+1}.html")

    # modelo bert final
    model_path = "results/models/bert/freeze_8/checkpoint-4000"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    trainer = Trainer(model=model)

    df = load_dataset("dataset/ep2-train.csv")

    dataset = make_splits(df, test_size=0.15, val_size=0.15)
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer, max_length=512), batched=True)
    tokenized_datasets, label2id, id2label = encode_labels(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    misclassified_df = analyze_model_errors(
        trainer=trainer,
        tokenized_dataset=tokenized_datasets["test"],
        id2label=id2label,
        original_texts=dataset["test"]["text"]
    )

    print_confusion_analysis(misclassified_df)
