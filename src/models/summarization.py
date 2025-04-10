# src/models/summarization.py
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re

# Ensure NLTK resources are available - updated handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextRankSummarizer:
    """Class for text summarization using TextRank algorithm."""
    
    def __init__(self, n_sentences=3):
        """
        Initialize the summarizer.
        
        Parameters:
        n_sentences (int): Number of sentences to include in the summary
        """
        self.n_sentences = n_sentences
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
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
        
        # Validate and preprocess text
        if not text or not isinstance(text, str) or text.strip() == '':
            return "No text to summarize."
        
        # Use regular expressions to split text into sentences
        # This avoids relying on NLTK's sent_tokenize which requires punkt_tab
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
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

def summarize_comments(comments, n_sentences=3):
    """
    Summarize a list of comments.
    
    Parameters:
    comments (list): List of comment texts
    n_sentences (int): Number of sentences in the summary
    
    Returns:
    str: Summary of the comments
    """
    # Handle empty input
    if not comments:
        return "No comments to summarize."
    
    # Initialize summarizer
    summarizer = TextRankSummarizer(n_sentences=n_sentences)
    
    # Combine comments with proper sentence boundaries
    # This helps TextRank identify sentence boundaries correctly
    combined_text = ' . '.join([text.strip() for text in comments if text and isinstance(text, str)])
    
    # Generate summary
    summary = summarizer.summarize(combined_text, n_sentences)
    
    return summary

def summarize_by_sentiment(comments_df, text_column='clean_text', sentiment_column='sentiment'):
    """
    Generate summaries for each sentiment category.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    text_column (str): Column containing text to summarize
    sentiment_column (str): Column containing sentiment labels
    
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
        
        # Generate summary
        summary = summarize_comments(sentiment_comments)
        
        # Store summary
        sentiment_summaries[sentiment] = summary
    
    return sentiment_summaries

def summarize_by_cluster(comments_df, text_column='clean_text', cluster_column='cluster', cluster_descriptions=None):
    """
    Generate summaries for each cluster.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    text_column (str): Column containing text to summarize
    cluster_column (str): Column containing cluster labels
    cluster_descriptions (dict): Dictionary mapping cluster IDs to descriptions
    
    Returns:
    dict: Dictionary mapping cluster IDs to summaries
    """
    # Check if required columns exist
    if text_column not in comments_df.columns or cluster_column not in comments_df.columns:
        return {"error": "Required columns not found in DataFrame"}
    
    # Group comments by cluster
    cluster_summaries = {}
    for cluster in comments_df[cluster_column].unique():
        # Get comments for this cluster
        cluster_comments = comments_df[comments_df[cluster_column] == cluster][text_column].tolist()
        
        # Skip clusters with very few comments
        if len(cluster_comments) < 3:
            cluster_summaries[cluster] = "Not enough comments for summarization"
            continue
        
        # Generate summary
        summary = summarize_comments(cluster_comments)
        
        # Add cluster description if available
        if cluster_descriptions and cluster in cluster_descriptions:
            cluster_summaries[cluster] = f"{cluster_descriptions[cluster]}\n\n{summary}"
        else:
            cluster_summaries[cluster] = summary
    
    return cluster_summaries