# src/models/sentiment.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from textblob import TextBlob
from transformers import pipeline

class SentimentAnalyzer:
    """Class for sentiment analysis of comments."""
    
    def __init__(self, method='textblob', model_path=None):
        """
        Initialize sentiment analyzer.
        
        Parameters:
        method (str): Method to use for sentiment analysis.
                     Options: 'textblob', 'custom', 'transformer'
        model_path (str): Path to saved model (for 'custom' method)
        """
        self.method = method
        self.model = None
        self.transformer = None
        
        # Load custom model if method is 'custom' and path is provided
        if method == 'custom' and model_path and os.path.exists(model_path):
            try:
                self.model = pickle.load(open(model_path, 'rb'))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Initialize transformer model if method is 'transformer'
        if method == 'transformer':
            try:
                self.transformer = pipeline("sentiment-analysis", 
                                           model="nlptown/bert-base-multilingual-uncased-sentiment")
                print("Loaded transformer model")
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                self.method = 'textblob'  # Fallback to TextBlob
    
    def train_custom_model(self, texts, labels, save_path=None):
        """
        Train a custom sentiment classifier.
        
        Parameters:
        texts (list): List of text samples
        labels (list): List of sentiment labels (positive, negative)
        save_path (str): Path to save the trained model
        """
        if self.method != 'custom':
            print("Warning: Switching to custom method for training")
            self.method = 'custom'
        
        # Create a pipeline with TF-IDF and SVM
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LinearSVC(C=10, max_iter=2000))
        ])
        
        # Train the model
        self.model.fit(texts, labels)
        print("Custom model trained successfully")
        
        # Save the model if path is provided
        if save_path:
            pickle.dump(self.model, open(save_path, 'wb'))
            print(f"Model saved to {save_path}")
        
        return self
    
    def analyze_sentiment(self, texts):
        """
        Analyze sentiment of texts.
        
        Parameters:
        texts (list or Series): List of text comments
        
        Returns:
        DataFrame with original texts and sentiment analysis results
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        results = pd.DataFrame({'text': texts})
        
        if self.method == 'textblob':
            # Use TextBlob for sentiment analysis
            sentiments = [TextBlob(str(text)).sentiment for text in texts]
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
            
            # If using a model that provides probability scores
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(texts)
                for i, class_label in enumerate(self.model.classes_):
                    results[f'prob_{class_label}'] = probas[:, i]
            
        elif self.method == 'transformer' and self.transformer is not None:
            # Use the transformer model
            # Need to truncate texts to 512 tokens for BERT
            predictions = [self.transformer(text[:512])[0] for text in texts]
            
            # Extract ratings (1-5 stars)
            results['rating'] = [int(pred['label'].split()[0]) for pred in predictions]
            results['score'] = [pred['score'] for pred in predictions]
            
            # Convert ratings to sentiment categories
            results['sentiment'] = results['rating'].apply(
                lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive')
            )
        
        else:
            # Fallback to TextBlob if method is invalid or model not loaded
            sentiments = [TextBlob(str(text)).sentiment for text in texts]
            results['polarity'] = [s.polarity for s in sentiments]
            results['sentiment'] = results['polarity'].apply(
                lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
            )
        
        return results
    
    def compare_sentiment_methods(self, texts):
        """
        Compare different sentiment analysis methods on the same texts.
        
        Parameters:
        texts (list): List of text comments
        
        Returns:
        DataFrame with results from multiple methods
        """
        # Save current method
        current_method = self.method
        
        # Get TextBlob results
        self.method = 'textblob'
        textblob_results = self.analyze_sentiment(texts)
        
        # Try to get transformer results
        transformer_results = None
        try:
            self.method = 'transformer'
            self.transformer = pipeline("sentiment-analysis", 
                                      model="nlptown/bert-base-multilingual-uncased-sentiment")
            transformer_results = self.analyze_sentiment(texts)
        except Exception as e:
            print(f"Error with transformer: {e}")
        
        # Restore original method
        self.method = current_method
        
        # Combine results
        comparison = pd.DataFrame({
            'text': texts,
            'textblob_sentiment': textblob_results['sentiment'],
            'textblob_polarity': textblob_results['polarity']
        })
        
        if transformer_results is not None:
            comparison['transformer_sentiment'] = transformer_results['sentiment']
            comparison['transformer_rating'] = transformer_results['rating']
        
        # If custom model is available, add its predictions
        if self.model is not None:
            custom_results = self.analyze_sentiment(texts)
            comparison['custom_sentiment'] = custom_results['sentiment']
        
        return comparison

def get_sentiment_distribution(df, sentiment_column='sentiment', group_by=None):
    """
    Get the distribution of sentiment categories, optionally grouped by another column.
    
    Parameters:
    df (DataFrame): DataFrame with sentiment data
    sentiment_column (str): Column containing sentiment categories
    group_by (str): Column to group by (e.g., 'movie')
    
    Returns:
    DataFrame with sentiment distribution
    """
    if group_by:
        return df.groupby([group_by, sentiment_column]).size().unstack().fillna(0)
    else:
        return df[sentiment_column].value_counts().to_frame()

def get_top_comments(df, sentiment_category, n=5, sentiment_column='sentiment', text_column='text'):
    """
    Get top comments for a specific sentiment category based on polarity strength.
    
    Parameters:
    df (DataFrame): DataFrame with sentiment data
    sentiment_category (str): Category to filter by ('positive', 'negative', 'neutral')
    n (int): Number of comments to return
    sentiment_column (str): Column containing sentiment categories
    text_column (str): Column containing the comment text
    
    Returns:
    DataFrame with top comments
    """
    # Filter by sentiment category
    filtered_df = df[df[sentiment_column] == sentiment_category].copy()
    
    # If polarity column exists, sort by absolute polarity
    if 'polarity' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='polarity', 
                                             ascending=(sentiment_category == 'negative'),
                                             key=abs if sentiment_category == 'negative' else lambda x: x)
    
    # If rating exists (from transformer), sort by rating
    elif 'rating' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='rating', 
                                             ascending=(sentiment_category == 'negative'))
    
    # Select top n comments
    top_comments = filtered_df.head(n)[[text_column, sentiment_column]]
    
    return top_comments

def sentiment_over_time(df, date_column='date', sentiment_column='sentiment'):
    """
    Analyze sentiment trends over time.
    
    Parameters:
    df (DataFrame): DataFrame with sentiment data
    date_column (str): Column containing date information
    sentiment_column (str): Column containing sentiment categories
    
    Returns:
    DataFrame with sentiment breakdown by time period
    """
    # Ensure date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by date and sentiment
    time_sentiment = df.groupby([pd.Grouper(key=date_column, freq='D'), sentiment_column]).size().unstack().fillna(0)
    
    # Calculate proportions
    for col in time_sentiment.columns:
        time_sentiment[f'{col}_pct'] = time_sentiment[col] / time_sentiment.sum(axis=1)
    
    return time_sentiment