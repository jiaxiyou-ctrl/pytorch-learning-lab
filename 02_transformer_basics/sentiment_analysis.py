"""Sentiment analysis using a pre-trained HuggingFace pipeline (DistilBERT)."""

from transformers import pipeline


def run_sentiment_analysis():
    """Classify example texts as POSITIVE/NEGATIVE with confidence scores."""
    print("Loading sentiment-analysis model...")
    sentiment_analyzer = pipeline("sentiment-analysis")

    texts = [
        "I love this movie, it's absolutely fantastic!",
        "This is the worst experience I've ever had.",
        "The weather is okay today.",
        "PyTorch makes deep learning research incredibly accessible.",
        "I'm not sure whether I like this or not.",
    ]

    print("=" * 60)
    print("  Sentiment Analysis Results")
    print("=" * 60)

    for text in texts:
        result = sentiment_analyzer(text)[0]
        print(f'\n"{text}"')
        print(f'   -> {result["label"]}  |  Confidence: {result["score"]:.4f}')


if __name__ == "__main__":
    run_sentiment_analysis()
