# src/preprocessing/clean_text.py
import re
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
    
    def __init__(self):
        """Initialize the text cleaner."""
        self.stopwords = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize a single text comment."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Handle contractions
        text = contractions.fix(text)
        
        # Remove timestamps
        text = re.sub(r'\d+:\d+', '', text)
        
        # Normalize punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

def clean_text_for_sentiment(text):
    """
    Simplified cleaning function for sentiment analysis.
    Preserves punctuation and doesn't remove stopwords.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text) 
    text = ' '.join(text.split())
    return text

def clean_text_for_clustering(text):
    """
    More aggressive cleaning for clustering.
    Removes stopwords and all punctuation.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    text = ' '.join(tokens)
    return text