# src/preprocessing/clean_text.py
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TextCleaner:
    """Class for cleaning and normalizing YouTube comments."""
    
    def __init__(self, remove_stopwords=False, remove_punctuation=False):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.stopwords = set(stopwords.words('english'))
        
        # Add custom YouTube comment noise words
        self.custom_noise_words = {
            'subscribe', 'like', 'comment', 'notification', 'bell', 'edit',
            'youtube', 'channel', 'video', 'watch', 'click'
        }
        
    def clean_text(self, text):
        """Clean and normalize a single text comment."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions (common in YouTube comments)
        text = re.sub(r'@\w+', '', text)
        
        # Handle contractions
        text = contractions.fix(text)
        
        # Remove timestamps
        text = re.sub(r'\d+:\d+', '', text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        else:
            # Just normalize some punctuation
            text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [word for word in tokens if word.lower() not in self.stopwords 
                     and word.lower() not in self.custom_noise_words]
        
        # Reconstruct text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def clean_comments(self, comments_list):
        """Clean a list of comments."""
        return [self.clean_text(comment) for comment in comments_list]

def clean_text_for_sentiment(text):
    """
    Simplified cleaning function for sentiment analysis.
    Preserves some punctuation and doesn't remove stopwords.
    """
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text) 
    text = ' '.join(text.split())
    return text

def clean_text_for_clustering(text):
    """
    More aggressive cleaning for clustering.
    Removes stopwords and all punctuation.
    """
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    text = ' '.join(tokens)
    return text