import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.model import get_model
import numpy as np

def test_model_initializes():
    model = get_model()
    assert model.config.num_labels == 2
    assert model.config.model_type == "bert-base-uncased"

def test_model_forward_outputs_logits_shape():
    model = get_model()
    dummy_input = [[1, 2, 3], [4, 5]]  # two samples
    out = model.forward(dummy_input)
    logits = out["logits"]
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (2, 2)  # batch=2, num_labels=2

def test_model_predict_works_with_text():
    model = get_model()
    outputs = model.predict(["good app", "bad experience"])
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 2)
