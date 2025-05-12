import time
from transformers import pipeline

class TextSentimentChecker:
    def __init__(self):
        # Force use of PyTorch to avoid TensorFlow/Keras issues
        print("⏳ Loading sentiment analysis model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english", 
            framework="pt"
        )
        self.last_check_time = 0
        self.check_interval = 2  # Check sentiment every 2 seconds
        print("📝 Text Sentiment Checker initialized")
    
    def get_sentiment_score(self, text):
        """
        Analyzes text sentiment and returns a score from -1 to +1
        -1: Very negative
        0: Neutral
        +1: Very positive
        """
        if not text or text.strip() == "":
            return 0  # Neutral for empty text
            
        result = self.sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']  # Confidence score
        
        # Convert to -1 to +1 scale with confidence weighting
        if label == "POSITIVE":
            return score  # 0.5 to 1.0 range
        elif label == "NEGATIVE":
            return -score  # -0.5 to -1.0 range
        else:
            return 0
    
    def should_check_sentiment(self):
        """Returns True if enough time has passed since the last check."""
        current_time = time.time()
        if current_time - self.last_check_time >= self.check_interval:
            self.last_check_time = current_time
            return True
        return False
    
    def analyze_transcript(self, transcript):
        """
        Analyzes the transcript if it's time to check sentiment.
        Returns a tuple of (sentiment_score, sentiment_label)
        """
        if not self.should_check_sentiment():
            return None, None
            
        score = self.get_sentiment_score(transcript)
        
        # Convert score to label
        if score >= 0.7:
            label = "very_positive"
        elif score >= 0.3:
            label = "positive"
        elif score <= -0.7:
            label = "very_negative"
        elif score <= -0.3:
            label = "negative"
        else:
            label = "neutral"
            
        return score, label

# Example usage
if __name__ == "__main__":
    checker = TextSentimentChecker()
    
    # Example transcripts to analyze
    test_transcripts = [
        "I'm really disappointed with the service today.",
        "This is absolutely amazing! I love it!",
        "It's okay, I guess. Nothing special.",
        "I'm furious about how this was handled.",
        "I'm so grateful for your help today."
    ]
    
    for transcript in test_transcripts:
        score, label = checker.analyze_transcript(transcript)
        print(f"Text: \"{transcript}\"")
        print(f"Sentiment Score: {score:.2f}, Label: {label}\n")
        time.sleep(0.5)  # Just for demonstration
