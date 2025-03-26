import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class CommentClusterer:
    """Class for clustering comments into thematic groups."""
    
    def __init__(self, n_clusters=5, max_features=5000):
        """
        Initialize the clusterer.
        
        Parameters:
        n_clusters (int): Number of clusters to create
        max_features (int): Maximum number of features for TF-IDF
        """
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
    def find_optimal_clusters(self, texts, min_clusters=2, max_clusters=10):
        """
        Find the optimal number of clusters using silhouette score.
        
        Parameters:
        texts (list): List of preprocessed comments
        min_clusters (int): Minimum number of clusters to try
        max_clusters (int): Maximum number of clusters to try
        
        Returns:
        int: Optimal number of clusters
        """
        # Get document vectors
        X = self.vectorizer.fit_transform(texts)
        
        # Try different numbers of clusters
        silhouette_scores = []
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {n_clusters}, silhouette score = {silhouette_avg}")
        
        # Find the best number of clusters
        optimal_clusters = min_clusters + np.argmax(silhouette_scores)
        self.n_clusters = optimal_clusters
        
        return optimal_clusters
    
    def fit(self, texts):
        """
        Fit the clustering model.
        
        Parameters:
        texts (list): List of preprocessed comments
        
        Returns:
        self: The fitted model
        """
        # Get document vectors
        X = self.vectorizer.fit_transform(texts)
        
        # Fit KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X)
        
        return self
    
    def predict(self, texts):
        """
        Predict cluster labels for new texts.
        
        Parameters:
        texts (list): List of preprocessed comments
        
        Returns:
        array: Cluster labels
        """
        X = self.vectorizer.transform(texts)
        return self.kmeans.predict(X)
    
    def get_top_terms_per_cluster(self, n_terms=10):
        """
        Get the top terms for each cluster.
        
        Parameters:
        n_terms (int): Number of terms to return per cluster
        
        Returns:
        dict: Dictionary mapping cluster IDs to top terms
        """
        # Get the feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get the cluster centroids
        centroids = self.kmeans.cluster_centers_
        
        # Get top terms for each cluster
        cluster_terms = {}
        for i in range(self.n_clusters):
            # Get the indices of the top terms
            indices = centroids[i].argsort()[-n_terms:][::-1]
            
            # Get the corresponding terms
            terms = [feature_names[j] for j in indices]
            cluster_terms[i] = terms
        
        return cluster_terms