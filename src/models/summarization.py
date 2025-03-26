import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

class TextRankSummarizer:
    """Class for text summarization using TextRank algorithm."""
    
    def __init__(self, n_sentences=3):
        """
        Initialize the summarizer.
        
        Parameters:
        n_sentences (int): Number of sentences to include in the summary
        """
        self.n_sentences = n_sentences
    
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
        
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        
        # If there are fewer sentences than requested, return the whole text
        if len(sentences) <= n_sentences:
            return text
        
        # Create sentence vectors (simple word overlap for demonstration)
        sentence_vectors = self._create_sentence_vectors(sentences)
        
        # Create similarity matrix
        similarity_matrix = self._create_similarity_matrix(sentence_vectors)
        
        # Apply PageRank algorithm
        scores = self._apply_pagerank(similarity_matrix)
        
        # Get the top sentences
        top_sentence_indices = np.argsort(scores)[-n_sentences:]
        top_sentence_indices = sorted(top_sentence_indices)
        
        # Create the summary
        summary = ' '.join([sentences[i] for i in top_sentence_indices])
        
        return summary
    
    def _create_sentence_vectors(self, sentences):
        """
        Create vectors for each sentence based on word overlap.
        
        Parameters:
        sentences (list): List of sentences
        
        Returns:
        list: List of sentence vectors
        """
        # Simple implementation - in practice, you'd use more sophisticated embeddings
        vectors = []
        for sentence in sentences:
            # Convert to lowercase and split into words
            words = set(sentence.lower().split())
            vectors.append(words)
        
        return vectors
    
    def _create_similarity_matrix(self, sentence_vectors):
        """
        Create a similarity matrix based on word overlap.
        
        Parameters:
        sentence_vectors (list): List of sentence vectors
        
        Returns:
        array: Similarity matrix
        """
        n = len(sentence_vectors)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate Jaccard similarity
                    intersection = len(sentence_vectors[i].intersection(sentence_vectors[j]))
                    union = len(sentence_vectors[i].union(sentence_vectors[j]))
                    
                    if union > 0:
                        similarity_matrix[i][j] = intersection / union
        
        return similarity_matrix
    
    def _apply_pagerank(self, similarity_matrix):
        """
        Apply PageRank algorithm to the similarity matrix.
        
        Parameters:
        similarity_matrix (array): Similarity matrix
        
        Returns:
        array: Sentence scores
        """
        # Create a graph from the similarity matrix
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank
        scores = nx.pagerank(nx_graph)
        
        # Convert to array
        return np.array(list(scores.values()))