# src/model.py
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelConfig:
    model_type: str = "bert-base-uncased"
    num_labels: int = 2

class DummyModel:
    """
    Hugging Face–like dummy model for sequence classification.
    Returns logits of shape [batch_size, num_labels].
    """
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.config = ModelConfig(model_type=model_name, num_labels=num_labels)

    def forward(self, input_ids):
        """
        Simulates a forward pass: produces random logits with the right shape.
        """
        batch_size = len(input_ids)
        logits = np.random.rand(batch_size, self.config.num_labels)
        return {"logits": logits}

    def predict(self, texts):
        """
        Tokenizes text (simple split) and calls forward() to get logits.
        """
        if isinstance(texts, str):
            texts = [texts]
        # simulate token IDs
        input_ids = [list(range(len(t.split()))) for t in texts]
        return self.forward(input_ids)


def get_model(model_name="bert-base-uncased", num_labels=2):
    """
    Factory function mimicking AutoModelForSequenceClassification.from_pretrained.
    """
    return DummyModel(model_name=model_name, num_labels=num_labels)

