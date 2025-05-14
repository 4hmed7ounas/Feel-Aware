from transformers import pipeline

# Force use of PyTorch to avoid TensorFlow/Keras issues
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")

def get_sentiment_score(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    if label == "POSITIVE":
        return 1
    elif label == "NEGATIVE":
        return -1
    else:
        return 0

# Example usage
live_transcripts = [
    "I'm really disappointed.",
    "That was fantastic!",
    "It's fine, I guess.",
    "I hate this so much.",
    "I'm very happy with the result!"
]

for text in live_transcripts:
    score = get_sentiment_score(text)
    print(f"Text: \"{text}\" → Sentiment Score: {score}")
