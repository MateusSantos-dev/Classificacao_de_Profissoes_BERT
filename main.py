import torch
from src.data.load import load_dataset, make_splits
from src.preprocessing import prepare_tokenizer, tokenize_function, encode_labels
from src.model import create_bert_model
from src.train import train_model
from src.eval import detailed_report
from src.explainability import TextExplainer

df = load_dataset("dataset/ep2-train.csv")

dataset = make_splits(df, test_size=0.15, val_size=0.15)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


model_name = "neuralmind/bert-large-portuguese-cased"
tokenizer = prepare_tokenizer(model_name=model_name)
tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer, max_length=512), batched=True)
tokenized_datasets, label2id, id2label = encode_labels(tokenized_datasets)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

model = create_bert_model(model_name=model_name,
                          num_labels=len(label2id),
                          freeze_bert_first_n_layers=8,
                          id2label=id2label,
                          label2id=label2id)
model.to(device)

trainer = train_model(model, tokenized_datasets, tokenizer, num_train_epochs=4)

print("📊 Avaliando modelo no conjunto de teste...\n")
predictions = trainer.predict(tokenized_datasets["test"])
preds = predictions.predictions.argmax(axis=-1)
y_true = predictions.label_ids.tolist()
y_pred = preds.tolist()

report = detailed_report(y_true, y_pred, list(label2id.keys()))

print("\n--- MÉTRICAS GERAIS ---")
for cls, metrics in report["report"].items():
    if isinstance(metrics, dict):
        print(f"{cls}: acc={metrics['precision']:.3f}, rec={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")

print("\nMatriz de confusão:")
for row in report["confusion_matrix"]:
    print(row)


print("\nGerando explicação para um exemplo de texto...\n")
explainer = TextExplainer(model=model, tokenizer=tokenizer, class_names=list(label2id.keys()))

sample_text = df["text"].iloc[0]
explanation = explainer.explain(sample_text)

print(f"Texto: {sample_text}\n")
print("Palavras mais influentes:")
for word, weight in explanation["weights"]:
    print(f"{word}: {weight:.3f}")

with open("results/explanation_sample.html", "w", encoding="utf-8") as f:
    f.write(explanation["html"])
print("\nExplicação salva em results/explanation_sample.html")
