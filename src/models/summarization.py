import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextRankSummarizer:
    """Class for text summarization using TextRank algorithm."""
    
    def __init__(self, n_sentences=3, language='english'):
        """
        Initialize the summarizer.
        
        Parameters:
        n_sentences (int): Number of sentences to include in the summary
        language (str): Language of the text
        """
        self.n_sentences = n_sentences
        self.language = language
        self.vectorizer = TfidfVectorizer(stop_words=language)
    
    def summarize(self, text, n_sentences=None):
        """
        Generate a summary of the text using TextRank.
        
        Parameters:
        text (str): Text to summarize
        n_sentences (int): Number of sentences to include (overrides the default)
        
        Returns:
        str: Summary of the text
        """
        if n_sentences is None:
            n_sentences = self.n_sentences
        
        try:
            # Validate and preprocess text
            if not text or not isinstance(text, str) or text.strip() == '':
                return "No text to summarize."
                
            # Tokenize the text into sentences
            sentences = sent_tokenize(text)
            
            # Remove very short sentences (likely noise)
            sentences = [s for s in sentences if len(s.split()) > 3]
            
            # If there are fewer sentences than requested, return the whole text
            if len(sentences) <= n_sentences:
                return " ".join(sentences)
            
            # Create TF-IDF vectors for each sentence
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Create similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Apply PageRank algorithm
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Rank sentences by score
            ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
            
            # Get the top sentences by score
            top_sentence_indices = sorted([i for _, i, _ in ranked_sentences[:n_sentences]])
            
            # Create the summary
            summary = ' '.join([sentences[i] for i in top_sentence_indices])
            
            return summary
            
        except Exception as e:
            print(f"Error in summarize: {e}")
            # In case of error, return a portion of the original text if possible
            if isinstance(text, str) and text:
                sentences = sent_tokenize(text)
                if sentences:
                    return ' '.join(sentences[:min(n_sentences, len(sentences))])
            return "Could not generate summary due to an error."
    
    def summarize_by_group(self, texts, group_labels, n_sentences_per_group=None):
        """
        Generate summaries for groups of texts.
        
        Parameters:
        texts (list): List of texts
        group_labels (list): List of group labels corresponding to each text
        n_sentences_per_group (int): Number of sentences per group summary
        
        Returns:
        dict: Dictionary mapping group labels to summaries
        """
        if n_sentences_per_group is None:
            n_sentences_per_group = self.n_sentences
        
        try:
            # Group texts by label
            grouped_texts = {}
            for text, label in zip(texts, group_labels):
                if label not in grouped_texts:
                    grouped_texts[label] = []
                grouped_texts[label].append(text)
            
            # Summarize each group
            summaries = {}
            for label, group_texts in grouped_texts.items():
                # Combine texts in the group
                combined_text = ' '.join(group_texts)
                
                # Generate summary
                summary = self.summarize(combined_text, n_sentences_per_group)
                summaries[label] = summary
            
            return summaries
            
        except Exception as e:
            print(f"Error in summarize_by_group: {e}")
            return {}
    
    def extract_keywords(self, text, n_keywords=10):
        """
        Extract key terms from the text.
        
        Parameters:
        text (str): Text to analyze
        n_keywords (int): Number of keywords to extract
        
        Returns:
        list: List of key terms
        """
        try:
            # Validate text
            if not text or not isinstance(text, str) or text.strip() == '':
                return []
                
            # Create TF-IDF matrix for the text
            vectorizer = TfidfVectorizer(
                stop_words=self.language,
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            # Fit TF-IDF on text
            tfidf_matrix = vectorizer.fit_transform([text])
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Sort terms by score
            sorted_indices = tfidf_scores.argsort()[::-1]
            
            # Get top terms
            keywords = [feature_names[i] for i in sorted_indices[:n_keywords]]
            
            return keywords
            
        except Exception as e:
            print(f"Error in extract_keywords: {e}")
            return []
    
    def summarize_comments(self, comments, sentiment_labels=None, n_sentences=None):
        """
        Summarize YouTube comments, optionally grouped by sentiment.
        
        Parameters:
        comments (list): List of comment texts
        sentiment_labels (list): List of sentiment labels (optional)
        n_sentences (int): Number of sentences in each summary
        
        Returns:
        dict: Summaries (either overall or by sentiment)
        """
        if n_sentences is None:
            n_sentences = self.n_sentences
        
        try:
            # Handle empty input
            if not comments:
                return {"overall": "No comments to summarize."}
                
            # If sentiment labels are provided, group by sentiment
            if sentiment_labels and len(sentiment_labels) == len(comments):
                return self.summarize_by_group(comments, sentiment_labels, n_sentences)
            
            # Otherwise, generate overall summary
            combined_text = ' '.join(comments)
            summary = self.summarize(combined_text, n_sentences)
            
            return {"overall": summary}
            
        except Exception as e:
            print(f"Error in summarize_comments: {e}")
            return {"overall": "Could not generate summary due to an error."}