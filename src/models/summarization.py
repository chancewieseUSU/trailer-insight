# src/models/summarization.py
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextRankSummarizer:
    """Improved class for text summarization using TextRank algorithm."""
    
    def __init__(self, n_sentences=3):
        """
        Initialize the summarizer.
        
        Parameters:
        n_sentences (int): Number of sentences to include in the summary
        """
        self.n_sentences = n_sentences
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def _preprocess_text(self, text):
        """Preprocess text for better sentence extraction."""
        # Handle punctuation to improve sentence boundary detection
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\.{2,}', '.', text)  # Replace multiple periods with a single one
        
        # Fix common YouTube comment issues
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)  # Add period between lowercase and uppercase
        
        return text
    
    def _get_sentences(self, text):
        """Extract sentences from text with improved handling."""
        # First try NLTK's sentence tokenizer
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to regex-based splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter sentences
        valid_sentences = []
        for s in sentences:
            # Skip very short sentences or non-sentences
            if len(s.split()) < 3 or len(s) < 15:
                continue
                
            # Skip sentences that are likely noise
            if s.count('@') > 0 or 'http' in s:
                continue
                
            valid_sentences.append(s)
            
        return valid_sentences
    
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
        
        # Validate text
        if not text or not isinstance(text, str) or text.strip() == '':
            return "No text to summarize."
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Extract sentences
        sentences = self._get_sentences(text)
        
        # If there are fewer sentences than requested, return the whole text
        if len(sentences) <= n_sentences:
            return " ".join(sentences)
        
        # Create TF-IDF vectors for each sentence
        try:
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
            
            # Ensure the summary is properly formatted
            summary = re.sub(r'\s+([.,!?])', r'\1', summary)  # Fix spacing before punctuation
            summary = re.sub(r'\.{2,}', '.', summary)  # Replace multiple periods
            
            return summary
        except Exception as e:
            # Fallback to returning top sentences by position
            print(f"Error in TextRank: {e}. Falling back to position-based summary.")
            important_indices = [0]  # First sentence is often important
            
            # Add some sentences from the middle and end
            if len(sentences) >= 3:
                important_indices.append(len(sentences) // 2)  # Middle
                important_indices.append(len(sentences) - 1)  # Last
            
            # Ensure we don't exceed the requested number of sentences
            important_indices = important_indices[:n_sentences]
            
            return " ".join([sentences[i] for i in sorted(important_indices)])

def summarize_comments(comments, n_sentences=3, min_comment_length=20):
    """
    Summarize a list of comments with improved processing.
    
    Parameters:
    comments (list): List of comment texts
    n_sentences (int): Number of sentences in the summary
    min_comment_length (int): Minimum character length for a comment to be included
    
    Returns:
    str: Summary of the comments
    """
    # Handle empty input
    if not comments:
        return "No comments to summarize."
    
    # Filter out very short comments and non-string items
    valid_comments = [
        text for text in comments 
        if text and isinstance(text, str) and len(text) >= min_comment_length
    ]
    
    if not valid_comments:
        return "No significant comments to summarize."
    
    # Initialize summarizer
    summarizer = TextRankSummarizer(n_sentences=n_sentences)
    
    # Process each comment to improve sentence boundaries
    processed_comments = []
    for comment in valid_comments:
        # Ensure sentences end with proper punctuation
        if not comment.rstrip().endswith(('.', '!', '?')):
            comment += '.'
        processed_comments.append(comment)
    
    # Combine comments with proper sentence boundaries
    combined_text = ' '.join(processed_comments)
    
    # Generate summary
    summary = summarizer.summarize(combined_text, n_sentences)
    
    return summary

def summarize_by_sentiment(comments_df, text_column='clean_text', sentiment_column='sentiment', n_sentences=3):
    """
    Generate improved summaries for each sentiment category.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    text_column (str): Column containing text to summarize
    sentiment_column (str): Column containing sentiment labels
    n_sentences (int): Number of sentences per summary
    
    Returns:
    dict: Dictionary mapping sentiment categories to summaries
    """
    # Check if required columns exist
    if text_column not in comments_df.columns or sentiment_column not in comments_df.columns:
        return {"error": "Required columns not found in DataFrame"}
    
    # Group comments by sentiment
    sentiment_summaries = {}
    for sentiment in comments_df[sentiment_column].unique():
        # Get comments for this sentiment
        sentiment_comments = comments_df[comments_df[sentiment_column] == sentiment][text_column].tolist()
        
        # Get comment count
        comment_count = len(sentiment_comments)
        
        # Generate summary
        summary = summarize_comments(sentiment_comments, n_sentences=n_sentences)
        
        # Add metadata to the summary
        sentiment_display = sentiment.title()  # Capitalize first letter
        sentiment_summaries[sentiment] = {
            "summary": summary,
            "count": comment_count,
            "display_name": sentiment_display
        }
    
    return sentiment_summaries

def summarize_by_cluster(comments_df, text_column='clean_text', cluster_column='cluster', 
                         sentiment_column='sentiment', n_sentences=3, 
                         cluster_descriptions=None):
    """
    Generate improved summaries for each cluster with detailed metadata.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    text_column (str): Column containing text to summarize
    cluster_column (str): Column containing cluster labels
    sentiment_column (str): Column containing sentiment labels
    n_sentences (int): Number of sentences per summary
    cluster_descriptions (dict): Dictionary mapping cluster IDs to descriptions
    
    Returns:
    dict: Dictionary mapping cluster IDs to summary data
    """
    # Check if required columns exist
    if text_column not in comments_df.columns or cluster_column not in comments_df.columns:
        return {"error": "Required columns not found in DataFrame"}
    
    # Group comments by cluster
    cluster_summaries = {}
    for cluster_id in sorted(comments_df[cluster_column].unique()):
        # Get comments for this cluster
        cluster_df = comments_df[comments_df[cluster_column] == cluster_id]
        cluster_comments = cluster_df[text_column].tolist()
        
        # Skip clusters with very few comments
        comment_count = len(cluster_comments)
        if comment_count < 3:
            cluster_summaries[cluster_id] = {
                "summary": "Not enough comments for summarization",
                "count": comment_count,
                "description": cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}") if cluster_descriptions else f"Cluster {cluster_id}",
                "sentiment_stats": {}
            }
            continue
        
        # Generate summary
        summary = summarize_comments(cluster_comments, n_sentences=n_sentences)
        
        # Calculate sentiment distribution
        sentiment_stats = {}
        if sentiment_column in cluster_df.columns:
            sentiment_counts = cluster_df[sentiment_column].value_counts()
            total = sentiment_counts.sum()
            
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total) * 100
                sentiment_stats[sentiment] = {
                    "count": int(count),
                    "percentage": float(percentage)
                }
        
        # Get sample comments (for display)
        sample_comments = []
        if comment_count > 0:
            # Try to get diverse samples by including most liked comments if available
            if 'likes' in cluster_df.columns:
                top_comments = cluster_df.sort_values('likes', ascending=False).head(3)
            else:
                # Otherwise just take the first few
                top_comments = cluster_df.head(3)
                
            sample_comments = top_comments[text_column].tolist()
        
        # Store summary data
        cluster_summaries[cluster_id] = {
            "summary": summary,
            "count": comment_count,
            "description": cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}") if cluster_descriptions else f"Cluster {cluster_id}",
            "sentiment_stats": sentiment_stats,
            "sample_comments": sample_comments
        }
    
    return cluster_summaries