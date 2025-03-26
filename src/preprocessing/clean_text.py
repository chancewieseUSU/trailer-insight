import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TextCleaner:
    """Class for cleaning and normalizing YouTube comments."""
    
    def __init__(self, remove_stopwords=True, remove_punctuation=True):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.stopwords = set(stopwords.words('english'))
        
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
        text = self._expand_contractions(text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stopwords]
        
        # Reconstruct text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def clean_comments(self, comments_list):
        """Clean a list of comments."""
        return [self.clean_text(comment) for comment in comments_list]
    
    def _expand_contractions(self, text):
        """Expand common contractions."""
        # This is a simplified version - you may want to expand this
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'m": " am",
            "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text