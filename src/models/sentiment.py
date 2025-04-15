# src/models/sentiment.py
import pandas as pd
import re
import numpy as np
from textblob import TextBlob
from collections import Counter

class SentimentAnalyzer:
    """Improved class for sentiment analysis of movie trailer comments with context awareness."""
    
    def __init__(self):
        """Initialize sentiment analyzer using TextBlob with domain-specific adjustments."""
        # Movie positive terms that might be misclassified
        self.domain_positive_terms = [
            'villain', 'evil', 'brutal', 'fight', 'battle', 'explosion', 'dark',
            'intense', 'scary', 'kill', 'die', 'death', 'worst enemy', 'legendary',
            'epic', 'insane', 'crazy', 'sick', 'favourite', 'favorite', 'hype',
            'hyped', 'cant wait', "can't wait", 'finally', 'omg', 'exciting',
            'peak', 'fire', 'go hard', 'hard', 'badass', 'savage', 'beast',
            'awesome', 'cool', 'lit', 'amazing', 'dope', 'goat', 'greatest'
        ]
        
        # Context indicators that should neutralize negative words
        self.context_modifiers = [
            'scene', 'part', 'character', 'quote', 'quotes', 'moment', 'moments',
            'reference', 'easter egg', 'callback', 'reference', 'throwback',
            'trailer', 'teaser', 'preview', 'clip', 'ending', 'beginning'
        ]
        
        # Genuine negative sentiment indicators for movies
        self.movie_negative_indicators = [
            'disappointing', 'letdown', 'waste', 'terrible movie', 'awful movie',
            'bad acting', 'poor effects', 'boring', 'lame', 'skip', 'not watching',
            'wont watch', 'wont see', 'cash grab', 'money grab', 'sells out', 'sellout',
            'ruin', 'ruined', 'disaster', 'flop', 'cash grab', 'awful', 'horrible',
            'terrible', 'worst movie', 'garbage', 'trash', 'pathetic', 'embarrassing'
        ]
        
        # Excitement/anticipation expressions (often with "can't wait")
        self.excitement_patterns = [
            r"can'?t wait",
            r"so excited",
            r"(looking|excited) (forward to|for)",
            r"(gonna|going to) (be|watch)",
            r"hyped",
            r"this looks (good|great|amazing)",
            r"🔥+"  # Fire emojis often indicate excitement
        ]
        
        # Slang positivity indicators
        self.slang_positive = [
            r"(\b|^)fire(\b|$)",
            r"(\b|^)lit(\b|$)",
            r"(\b|^)peak(\b|$)",
            r"(\b|^)dope(\b|$)",
            r"(\b|^)go(es|ing)? hard(\b|$)",
            r"(\b|^)goat(\b|$)",
            r"(\b|^)🔥+",
            r"(\b|^)W(\b|$)",  # Internet slang for "win"
            r"(\b|^)banger(\b|$)"
        ]
        
        # Movie-specific frequent terms (will be populated during analysis)
        self.movie_specific_terms = {}
        
    def _extract_frequent_terms(self, texts, movie_name=None, min_count=3):
        """Extract frequently occurring terms that might be movie-specific."""
        # Tokenize all texts
        all_words = []
        for text in texts:
            if not isinstance(text, str):
                continue
                
            # Extract words and short phrases
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
            
            # Also look for quoted text
            quotes = re.findall(r'"([^"]+)"', text)
            for quote in quotes:
                if 3 <= len(quote.split()) <= 5:  # Short quotes that might be catchphrases
                    all_words.append(quote.lower())
        
        # Count occurrences
        word_counts = Counter(all_words)
        
        # Filter for words that appear frequently
        frequent_terms = [word for word, count in word_counts.items() 
                          if count >= min_count and len(word) > 2]
        
        # Store for the specific movie if provided
        if movie_name:
            self.movie_specific_terms[movie_name] = frequent_terms
            
        return frequent_terms
    
    def _is_neutral_reference(self, text, movie_name=None):
        """Check if text appears to be a neutral reference to movie content."""
        # Check for quoted content
        has_quotes = '"' in text or "'" in text
        
        # Check for timestamps (common in YouTube comments referencing specific moments)
        has_timestamp = bool(re.search(r'(\d+:\d+)|(\d+\s*min)', text))
        
        # Check for movie-specific terms if available
        has_specific_term = False
        if movie_name and movie_name in self.movie_specific_terms:
            terms = self.movie_specific_terms[movie_name]
            has_specific_term = any(term in text.lower() for term in terms)
        
        # Check for slang positivity
        has_slang_positivity = any(re.search(pattern, text.lower()) for pattern in self.slang_positive)
        
        # Consider it neutral if it has quotes, timestamps, or movie-specific terms
        # unless it has clear negative indicators
        is_neutral = (has_quotes or has_timestamp or has_specific_term) and not any(
            indicator in text.lower() for indicator in self.movie_negative_indicators
        )
        
        # Consider it positive if it has slang positivity indicators
        is_positive_slang = has_slang_positivity
        
        return is_neutral, is_positive_slang
        
    def _check_domain_specific_terms(self, text):
        """Check for domain-specific terms that affect sentiment."""
        text_lower = text.lower()
        
        # Check for positive domain-specific terms
        has_positive_term = any(term in text_lower for term in self.domain_positive_terms)
        
        # Check for context modifiers
        has_context = any(modifier in text_lower for modifier in self.context_modifiers)
        
        # Check for genuine negative indicators
        has_negative_indicator = any(indicator in text_lower for indicator in self.movie_negative_indicators)
        
        # Check for excitement patterns
        has_excitement = any(re.search(pattern, text_lower) for pattern in self.excitement_patterns)
        
        return has_positive_term, has_context, has_negative_indicator, has_excitement
    
    def _adjust_sentiment(self, polarity, subjectivity, text, movie_name=None):
        """Adjust TextBlob sentiment based on domain-specific rules."""
        has_positive_term, has_context, has_negative_indicator, has_excitement = self._check_domain_specific_terms(text)
        is_neutral_reference, is_positive_slang = self._is_neutral_reference(text, movie_name)
        
        adjusted_polarity = polarity
        
        # Rule 1: If there's excitement expressed, ensure it's positive
        if has_excitement and polarity < 0.3:
            adjusted_polarity = 0.5
        
        # Rule 2: If it has slang positivity markers, make it positive
        if is_positive_slang and polarity < 0.3:
            adjusted_polarity = 0.7
        
        # Rule 3: If it appears to be a neutral reference to movie content, neutralize extreme sentiment
        if is_neutral_reference and not has_negative_indicator:
            if polarity < 0:  # If it was classified as negative
                adjusted_polarity = 0.1  # Make it slightly positive
        
        # Rule 4: If contains domain-specific positive terms AND context modifiers,
        # but was classified as negative, neutralize or make slightly positive
        if has_positive_term and has_context and polarity < 0:
            adjusted_polarity = 0.1
        
        # Rule 5: If mentions characters/scenes with "positive" domain terms, 
        # correct negative sentiment
        if has_positive_term and polarity < 0 and not has_negative_indicator:
            adjusted_polarity = min(polarity + 0.5, 0.3)  # Boost but don't make too positive
        
        # Rule 6: If has genuine negative movie indicators, ensure it's negative
        if has_negative_indicator and polarity > -0.2:
            adjusted_polarity = -0.4
        
        # Rule 7: Quotes or references to movie lines are usually positive or neutral
        if '"' in text or "'" in text and polarity < 0:
            # Check if it's referencing a quote (quotes + context word)
            if has_context:
                adjusted_polarity = max(0.1, polarity)
        
        # Rule 8: Emoji detection - multiple fire or hearts usually mean positive
        if re.search(r'(🔥|❤️|😍|👍){2,}', text) and polarity < 0.3:
            adjusted_polarity = 0.7
            
        return adjusted_polarity, subjectivity
    
    def analyze_sentiment(self, texts, movie_names=None):
        """
        Analyze sentiment of texts with domain-specific adjustments.
        
        Parameters:
        texts (list or Series): List of text comments
        movie_names (list or Series): Optional list of movie names corresponding to each text
        
        Returns:
        DataFrame with original texts and sentiment analysis results
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        if movie_names is not None and isinstance(movie_names, pd.Series):
            movie_names = movie_names.tolist()
        
        results = pd.DataFrame({'text': texts})
        if movie_names is not None:
            results['movie'] = movie_names
            
            # Extract frequent terms for each movie
            for movie in set(movie_names):
                movie_texts = [text for text, m in zip(texts, movie_names) if m == movie]
                self._extract_frequent_terms(movie_texts, movie, min_count=3)
        else:
            # If no movie names provided, extract terms from all texts
            self._extract_frequent_terms(texts, min_count=5)
        
        # Use TextBlob for initial sentiment analysis
        sentiments = []
        for i, text in enumerate(texts):
            try:
                # Handle None or empty strings
                if text is None or not isinstance(text, str) or text.strip() == '':
                    sentiments.append((0, 0))  # neutral polarity and subjectivity
                else:
                    # Get basic TextBlob sentiment
                    initial_sentiment = TextBlob(text).sentiment
                    
                    # Apply domain-specific adjustments
                    movie = movie_names[i] if movie_names is not None and i < len(movie_names) else None
                    adjusted_sentiment = self._adjust_sentiment(
                        initial_sentiment.polarity,
                        initial_sentiment.subjectivity,
                        text,
                        movie
                    )
                    sentiments.append(adjusted_sentiment)
            except Exception as e:
                print(f"Error analyzing text: {e}")
                sentiments.append((0, 0))  # neutral fallback
        
        results['polarity'] = [s[0] for s in sentiments]
        results['subjectivity'] = [s[1] for s in sentiments]
        
        # Categorize sentiment with improved thresholds for movie context
        # More strict for negative classification to avoid false negatives
        results['sentiment'] = results['polarity'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.15 else 'neutral')
        )
        
        return results

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
    # Check for valid data
    if df is None or sentiment_column not in df.columns or text_column not in df.columns:
        print(f"Missing required columns in DataFrame")
        return pd.DataFrame({text_column: [], sentiment_column: []})
    
    # Filter by sentiment category
    filtered_df = df[df[sentiment_column] == sentiment_category].copy()
    
    if filtered_df.empty:
        return pd.DataFrame({text_column: [], sentiment_column: []})
    
    # For negative sentiment, add additional filtering
    if sentiment_category == 'negative':
        # Create a SentimentAnalyzer instance to get negative indicators
        analyzer = SentimentAnalyzer()
        
        # Check for genuine negative indicators
        def has_negative_indicator(text):
            if not isinstance(text, str):
                return False
            return any(indicator in text.lower() for indicator in analyzer.movie_negative_indicators)
        
        # Add a column indicating if the comment has genuine negative indicators
        filtered_df['has_negative'] = filtered_df[text_column].apply(has_negative_indicator)
        
        # If we have enough comments with genuine negative indicators, filter to those
        if filtered_df['has_negative'].sum() >= n:
            filtered_df = filtered_df[filtered_df['has_negative']]
    
    # Sort by polarity (most extreme first)
    if 'polarity' in filtered_df.columns:
        if sentiment_category == 'negative':
            filtered_df = filtered_df.sort_values(by='polarity', ascending=True)
        else:
            filtered_df = filtered_df.sort_values(by='polarity', ascending=False)
    
    # Select top n comments
    top_comments = filtered_df.head(n)[[text_column, sentiment_column]]
    
    return top_comments