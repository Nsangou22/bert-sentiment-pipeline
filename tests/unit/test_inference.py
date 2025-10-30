# tests/unit/test_inference.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.inference import predict_sentiment


def test_single_text_prediction():
    result = predict_sentiment("I love this app")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in ["Positive", "Negative"]


def test_multiple_text_predictions():
    texts = ["Good app", "Bad experience", "Average performance"]
    result = predict_sentiment(texts)
    assert isinstance(result, list)
    assert len(result) == len(texts)
    for label in result:
        assert label in ["Positive", "Negative"]
