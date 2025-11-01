import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from lime.lime_text import LimeTextExplainer


class TextExplainer:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, class_names: list[str]) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Fazer predição
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return probabilities.cpu().numpy()

    def explain(self, text: str, num_features: int = 10, num_samples: int = 1000) -> dict:
        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self._predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )

        return {
            "text": text,
            "weights": exp.as_list(),
            "html": exp.as_html()
        }
