# src/model.py
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelConfig:
    model_type: str = "bert-base-uncased"
    num_labels: int = 2

class DummyModel:
    
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.config = ModelConfig(model_type=model_name, num_labels=num_labels)

    def forward(self, input_ids):
        
        batch_size = len(input_ids)
        logits = np.random.rand(batch_size, self.config.num_labels)
        return {"logits": logits}

    def predict(self, texts):
        
        if isinstance(texts, str):
            texts = [texts]
        # simulate token IDs
        input_ids = [list(range(len(t.split()))) for t in texts]
        return self.forward(input_ids)


def get_model(model_name="bert-base-uncased", num_labels=2):
    
    return DummyModel(model_name=model_name, num_labels=num_labels)

