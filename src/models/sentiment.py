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
                self.method = 'textblob'  # Fallback to TextBlob
        
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
            sentiments = []
            for text in texts:
                try:
                    # Handle None or empty strings
                    if text is None or not isinstance(text, str) or text.strip() == '':
                        sentiments.append((0, 0))  # neutral polarity and subjectivity
                    else:
                        sentiments.append(TextBlob(text).sentiment)
                except Exception as e:
                    print(f"Error analyzing text with TextBlob: {e}")
                    sentiments.append((0, 0))  # neutral fallback
            
            results['polarity'] = [s[0] for s in sentiments]
            results['subjectivity'] = [s[1] for s in sentiments]
            
            # Categorize sentiment with improved thresholds
            # Positive: > 0.05, Negative: < -0.02, Neutral: between
            results['sentiment'] = results['polarity'].apply(
                lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.02 else 'neutral')
            )
            
        elif self.method == 'custom' and self.model is not None:
            # Use the custom trained model
            try:
                # Filter out None values or empty strings
                valid_texts = [t if t is not None and isinstance(t, str) else "" for t in texts]
                predictions = self.model.predict(valid_texts)
                results['sentiment'] = predictions
                
                # If using a model that provides probability scores
                if hasattr(self.model, 'predict_proba'):
                    probas = self.model.predict_proba(valid_texts)
                    for i, class_label in enumerate(self.model.classes_):
                        results[f'prob_{class_label}'] = probas[:, i]
            except Exception as e:
                print(f"Error with custom model: {e}")
                # Fallback to TextBlob
                self._fallback_to_textblob(results)
            
        elif self.method == 'transformer' and self.transformer is not None:
            # Use the transformer model
            try:
                # Process texts in batches to avoid memory issues
                batch_size = 32
                all_predictions = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    batch_predictions = []
                    
                    for text in batch:
                        # Handle None or empty strings
                        if text is None or not isinstance(text, str) or text.strip() == '':
                            batch_predictions.append({'label': '3', 'score': 1.0})
                        else:
                            # Truncate text to 512 tokens for BERT and ensure it's not empty
                            truncated_text = text[:512] if len(text) > 512 else text
                            if truncated_text.strip():
                                pred = self.transformer(truncated_text)[0]
                                batch_predictions.append(pred)
                            else:
                                batch_predictions.append({'label': '3', 'score': 1.0})  # Neutral fallback
                    
                    all_predictions.extend(batch_predictions)
                
                # Extract ratings (1-5 stars) and handle different label formats
                results['rating'] = []
                results['score'] = []
                
                for pred in all_predictions:
                    try:
                        # Handle both "1 star" and "1" formats
                        label = pred['label']
                        if ' ' in label:
                            rating = int(label.split()[0])
                        else:
                            rating = int(label)
                        
                        results['rating'].append(rating)
                        results['score'].append(pred['score'])
                    except (ValueError, IndexError):
                        # Fallback for unexpected format
                        results['rating'].append(3)  # Neutral fallback
                        results['score'].append(1.0)
                
                # Convert ratings to sentiment categories
                results['sentiment'] = results['rating'].apply(
                    lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive')
                )
            except Exception as e:
                print(f"Error in transformer analysis: {e}")
                # Fallback to TextBlob
                self._fallback_to_textblob(results)
        
        else:
            # Fallback to TextBlob if method is invalid or model not loaded
            self._fallback_to_textblob(results)
        
        return results
    
    def _fallback_to_textblob(self, results):
        """Helper method to provide TextBlob sentiment as fallback"""
        sentiments = []
        for text in results['text']:
            try:
                if text is None or not isinstance(text, str) or text.strip() == '':
                    sentiments.append((0, 0))  # neutral polarity and subjectivity
                else:
                    sentiments.append(TextBlob(text).sentiment)
            except:
                sentiments.append((0, 0))  # neutral fallback
                
        results['polarity'] = [s[0] for s in sentiments]
        results['subjectivity'] = [s[1] for s in sentiments]
        results['sentiment'] = results['polarity'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.02 else 'neutral')
        )
    
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
            if self.transformer is None:
                # Initialize transformer if needed
                self.transformer = pipeline("sentiment-analysis", 
                                         model="nlptown/bert-base-multilingual-uncased-sentiment")
            transformer_results = self.analyze_sentiment(texts)
        except Exception as e:
            print(f"Error with transformer: {e}")
            transformer_results = None
        
        # Restore original method
        self.method = current_method
        
        # Combine results
        comparison = pd.DataFrame({
            'text': texts,
            'textblob_sentiment': textblob_results['sentiment'],
            'textblob_polarity': textblob_results['polarity']
        })
        
        if transformer_results is not None and 'sentiment' in transformer_results.columns:
            comparison['transformer_sentiment'] = transformer_results['sentiment']
            if 'rating' in transformer_results.columns:
                comparison['transformer_rating'] = transformer_results['rating']
        else:
            # Add placeholder columns to avoid KeyError
            comparison['transformer_sentiment'] = 'neutral'
            comparison['transformer_rating'] = 3
        
        # If custom model is available, add its predictions
        if self.model is not None:
            try:
                self.method = 'custom'
                custom_results = self.analyze_sentiment(texts)
                comparison['custom_sentiment'] = custom_results['sentiment']
            except Exception as e:
                print(f"Error with custom model: {e}")
                comparison['custom_sentiment'] = 'neutral'
        
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
    try:
        # Check for valid data
        if df is None or sentiment_column not in df.columns:
            return pd.DataFrame()
        
        if group_by:
            if group_by not in df.columns:
                print(f"Column '{group_by}' not found in DataFrame")
                return pd.DataFrame()
                
            sentiment_counts = df.groupby([group_by, sentiment_column]).size().unstack().fillna(0)
            
            # Ensure all sentiment categories are present
            for cat in ['positive', 'negative', 'neutral']:
                if cat not in sentiment_counts.columns:
                    sentiment_counts[cat] = 0
            
            return sentiment_counts
        else:
            return df[sentiment_column].value_counts().to_frame()
    except Exception as e:
        print(f"Error in get_sentiment_distribution: {e}")
        return pd.DataFrame()

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
    try:
        # Check for valid data
        if df is None or sentiment_column not in df.columns or text_column not in df.columns:
            print(f"Missing required columns in DataFrame")
            return pd.DataFrame({text_column: [], sentiment_column: []})
        
        # Filter by sentiment category
        filtered_df = df[df[sentiment_column] == sentiment_category].copy()
        
        if filtered_df.empty:
            return pd.DataFrame({text_column: [], sentiment_column: []})
        
        # If polarity column exists, sort by absolute polarity
        if 'polarity' in filtered_df.columns:
            if sentiment_category == 'negative':
                # For negative, sort by most negative (lowest polarity)
                filtered_df = filtered_df.sort_values(by='polarity', ascending=True)
            else:
                # For positive/neutral, sort by highest polarity
                filtered_df = filtered_df.sort_values(by='polarity', ascending=False)
        
        # If rating exists (from transformer), sort by rating
        elif 'rating' in filtered_df.columns:
            if sentiment_category == 'negative':
                # For negative, sort by lowest rating
                filtered_df = filtered_df.sort_values(by='rating', ascending=True)
            else:
                # For positive/neutral, sort by highest rating
                filtered_df = filtered_df.sort_values(by='rating', ascending=False)
        
        # Select top n comments
        top_comments = filtered_df.head(n)[[text_column, sentiment_column]]
        
        return top_comments
    except Exception as e:
        print(f"Error in get_top_comments: {e}")
        return pd.DataFrame({text_column: [], sentiment_column: []})

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
    try:
        # Check for valid data
        if df is None or date_column not in df.columns or sentiment_column not in df.columns:
            print(f"Missing required columns in DataFrame")
            return pd.DataFrame()
        
        # Ensure date column is in datetime format
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=[date_column])
        
        # Group by date and sentiment
        time_sentiment = df.groupby([pd.Grouper(key=date_column, freq='D'), sentiment_column]).size().unstack().fillna(0)
        
        # Calculate proportions
        for col in time_sentiment.columns:
            time_sentiment[f'{col}_pct'] = time_sentiment[col] / time_sentiment.sum(axis=1)
        
        return time_sentiment
    except Exception as e:
        print(f"Error in sentiment_over_time: {e}")
        return pd.DataFrame()