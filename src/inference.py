# src/inference.py

from src.model import get_model


def predict_sentiment(texts):
    # Import and initialize the model exactly as in the teacher's instructions
    model = get_model(model_name="bert-base-uncased", num_labels=2)

    # Call the model's built-in predict() method
    outputs = model.predict(texts)

    # Extract logits (like BERT's output)
    logits = outputs["logits"]

    #  Convert logits to human-readable labels
    predictions = []
    for logit in logits:
        label_index = 1 if logit[1] >= logit[0] else 0
        label = "Positive" if label_index == 1 else "Negative"
        predictions.append(label)

    return predictions


if __name__ == "__main__":
    # Example usage for manual testing
    samples = ["I love this app!", "This update is terrible."]
    preds = predict_sentiment(samples)
    for text, label in zip(samples, preds):
        print(f"Text: {text} → Sentiment: {label}")
