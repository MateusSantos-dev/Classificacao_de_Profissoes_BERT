import eli5
import os
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd


def train_logistic_model(data_path: str) -> tuple:
    df = pd.read_csv(data_path, encoding="latin-1", sep=";")
    x_train, x_test, y_train, y_test = train_test_split(
        df["req_text"], df["profession"], test_size=0.2, stratify=df["profession"], random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    model = make_pipeline(vectorizer, clf)
    model.fit(x_train, y_train)

    return model, x_test, y_test


def generate_global_interpretation(
        model,
        feature_names: str,
        output_path: str = "results/analysis/global_explanation.html"
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html = eli5.explain_weights(model.named_steps["logisticregression"], top=15, feature_names=feature_names)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(eli5.format_as_html(html))


def generate_local_interpretation(
        model,
        text_example: str,
        output_path: str = "results/analysis/local_explanation.html"
) -> None:
    html = eli5.explain_prediction(
        model.named_steps["logisticregression"],
        text_example,
        vec=model.named_steps["tfidfvectorizer"])

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(eli5.format_as_html(html))


def analyze_model_errors(
        trainer,
        tokenized_dataset,
        id2label,
        original_texts: list[str],
        num_samples: int = 7
) -> pd.DataFrame:
    predictions = trainer.predict(tokenized_dataset)
    pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)
    pred_labels = predictions.predictions.argmax(axis=-1)
    true_labels = predictions.label_ids
    class_probs = pred_probs.numpy()

    error_data = []
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        error_data.append({
            'text': original_texts[i],
            'true_label': id2label[true],
            'pred_label': id2label[pred],
            'true_encoded': true,
            'pred_encoded': pred,
            'confidence': class_probs[i][pred],
            'all_probabilities': class_probs[i],
            'is_correct': true == pred
        })

    error_df = pd.DataFrame(error_data)
    misclassified = error_df[~error_df['is_correct']]

    print_errors_analysis(misclassified, id2label, num_samples)

    print(f"Total errors: {len(misclassified)}/{len(error_df)}")
    print(f"Error rate: {len(misclassified) / len(error_df) * 100:.2f}%")

    return misclassified


def print_errors_analysis(misclassified_df: pd.DataFrame, id2label, num_samples: int) -> None:
    sample_errors = misclassified_df.head(num_samples)
    class_names = list(id2label.values())

    print("\n" + "=" * 80)
    print("BERT MODEL ERROR ANALYSIS")
    print("=" * 80)

    for idx, row in sample_errors.iterrows():
        print(f"\n--- Error Case {idx} ---")
        print(f"True label: {row['true_label']}")
        print(f"Predicted label: {row['pred_label']}")
        print(f"Prediction confidence: {row['confidence']:.3f}")
        print(f"Text preview: {row['text'][:150]}...")

        probs = list(zip(class_names, row['all_probabilities']))
        probs_sorted = sorted(probs, key=lambda x: x[1], reverse=True)

        print("Top 3 class probabilities:")
        for class_name, prob in probs_sorted[:3]:
            print(f"  {class_name}: {prob:.3f}")

        print("-" * 50)


def print_confusion_analysis(misclassified_df: pd.DataFrame) -> None:
    confusion_matrix = pd.crosstab(
        misclassified_df['true_label'],
        misclassified_df['pred_label'],
        margins=True
    )

    print("\n" + "=" * 80)
    print("CONFUSION ANALYSIS")
    print("=" * 80)
    print("Most common confusion patterns:")
    print(confusion_matrix)
