import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from textblob import TextBlob

class SentimentAnalyzer:
    """Class for sentiment analysis of comments."""
    
    def __init__(self, method='textblob'):
        """
        Initialize sentiment analyzer.
        
        Parameters:
        method (str): Method to use for sentiment analysis.
                     Options: 'textblob', 'custom'
        """
        self.method = method
        self.model = None
        
    def train_custom_model(self, texts, labels):
        """
        Train a custom sentiment classifier.
        
        Parameters:
        texts (list): List of text samples
        labels (list): List of sentiment labels (positive, negative, neutral)
        """
        if self.method != 'custom':
            print("Warning: Switching to custom method for training")
            self.method = 'custom'
        
        # Create a simple pipeline with TF-IDF and Logistic Regression
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        
        # Train the model
        self.model.fit(texts, labels)
        print("Custom model trained successfully")
        
    def analyze_sentiment(self, texts):
        """
        Analyze sentiment of texts.
        
        Parameters:
        texts (list): List of text comments
        
        Returns:
        DataFrame with original texts and sentiment analysis results
        """
        results = pd.DataFrame({'text': texts})
        
        if self.method == 'textblob':
            # Use TextBlob for sentiment analysis
            sentiments = [TextBlob(text).sentiment for text in texts]
            results['polarity'] = [s.polarity for s in sentiments]
            results['subjectivity'] = [s.subjectivity for s in sentiments]
            
            # Categorize sentiment
            results['sentiment'] = results['polarity'].apply(
                lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
            )
            
        elif self.method == 'custom' and self.model is not None:
            # Use the custom trained model
            predictions = self.model.predict(texts)
            results['sentiment'] = predictions
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(texts)
                for i, class_label in enumerate(self.model.classes_):
                    results[f'prob_{class_label}'] = probas[:, i]
        
        else:
            raise ValueError("Invalid method or model not trained")
        
        return results