import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class CommentClusterer:
    """Class for clustering comments into thematic groups."""
    
    def __init__(self, n_clusters=5, max_features=5000, random_state=42):
        """
        Initialize the clusterer.
        
        Parameters:
        n_clusters (int): Number of clusters to create
        max_features (int): Maximum number of features for TF-IDF
        random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=3,
            max_df=0.85,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        self.feature_names = None
        self.cluster_centers = None
        
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
        # Clean and validate texts
        valid_texts = self._preprocess_texts(texts)
        
        try:
            # Get document vectors
            X = self.vectorizer.fit_transform(valid_texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # Try different numbers of clusters
            silhouette_scores = []
            for n_clusters in range(min_clusters, max_clusters + 1):
                try:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=self.random_state,
                        n_init=10
                    )
                    cluster_labels = kmeans.fit_predict(X)
                    
                    # Calculate silhouette score
                    silhouette_avg = silhouette_score(X, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                    print(f"For n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.3f}")
                except Exception as e:
                    print(f"Error calculating silhouette score for {n_clusters} clusters: {e}")
                    silhouette_scores.append(-1)  # Use negative value to indicate error
            
            # Find the best number of clusters
            if silhouette_scores:
                optimal_clusters = min_clusters + np.argmax(silhouette_scores)
                self.n_clusters = optimal_clusters
                print(f"Optimal number of clusters: {optimal_clusters}")
                return optimal_clusters
            else:
                print(f"Could not determine optimal clusters, using default: {self.n_clusters}")
                return self.n_clusters
                
        except Exception as e:
            print(f"Error in find_optimal_clusters: {e}")
            return self.n_clusters
    
    def fit(self, texts):
        """
        Fit the clustering model.
        
        Parameters:
        texts (list): List of preprocessed comments
        
        Returns:
        self: The fitted model
        """
        # Clean and validate texts
        valid_texts = self._preprocess_texts(texts)
        
        try:
            # Get document vectors
            X = self.vectorizer.fit_transform(valid_texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # Fit KMeans
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            self.kmeans.fit(X)
            self.cluster_centers = self.kmeans.cluster_centers_
            
            return self
        except Exception as e:
            print(f"Error in fit: {e}")
            # Initialize with empty values so other methods don't crash
            self.feature_names = np.array([])
            self.cluster_centers = np.zeros((self.n_clusters, 1))
            return self
    
    def predict(self, texts):
        """
        Predict cluster labels for new texts.
        
        Parameters:
        texts (list): List of preprocessed comments
        
        Returns:
        array: Cluster labels
        """
        # Clean and validate texts
        valid_texts = self._preprocess_texts(texts)
        
        try:
            # Transform texts to TF-IDF vectors
            X = self.vectorizer.transform(valid_texts)
            
            # Predict cluster labels
            return self.kmeans.predict(X)
        except Exception as e:
            print(f"Error in predict: {e}")
            # Return default labels as fallback
            return np.zeros(len(valid_texts), dtype=int)
    
    def get_top_terms_per_cluster(self, n_terms=10):
        """
        Get the top terms for each cluster.
        
        Parameters:
        n_terms (int): Number of terms to return per cluster
        
        Returns:
        dict: Dictionary mapping cluster IDs to top terms
        """
        try:
            # Check if model has been fitted
            if self.feature_names is None or self.cluster_centers is None:
                raise ValueError("Model has not been fitted yet")
            
            # Get the feature names
            feature_names = self.feature_names
            
            # Get the cluster centroids
            centroids = self.cluster_centers
            
            # Get top terms for each cluster
            cluster_terms = {}
            for i in range(self.n_clusters):
                # Get the indices of the top terms
                indices = centroids[i].argsort()[-n_terms:][::-1]
                
                # Get the corresponding terms
                terms = [feature_names[j] for j in indices]
                cluster_terms[i] = terms
            
            return cluster_terms
        except Exception as e:
            print(f"Error in get_top_terms_per_cluster: {e}")
            # Return empty dictionary as fallback
            return {i: ["term_error"] for i in range(self.n_clusters)}
    
    def visualize_clusters(self, texts, labels=None, save_path=None):
        """
        Visualize clusters using PCA.
        
        Parameters:
        texts (list): List of preprocessed comments
        labels (array): Cluster labels (if None, will be predicted)
        save_path (str): Path to save the visualization
        
        Returns:
        None
        """
        # Clean and validate texts
        valid_texts = self._preprocess_texts(texts)
        
        try:
            # Get document vectors
            X = self.vectorizer.transform(valid_texts)
            
            # Get cluster labels if not provided
            if labels is None:
                labels = self.kmeans.predict(X)
            
            # Reduce dimensionality with PCA
            pca = PCA(n_components=2, random_state=self.random_state)
            pca_result = pca.fit_transform(X.toarray())
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'x': pca_result[:, 0],
                'y': pca_result[:, 1],
                'Cluster': labels
            })
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df, x='x', y='y', hue='Cluster', palette='viridis')
            plt.title(f'Clusters Visualization (n_clusters={self.n_clusters})')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            
            # Save plot if path is provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            # Show plot
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in visualize_clusters: {e}")
    
    def get_cluster_distribution(self, texts, labels=None):
        """
        Get the distribution of texts across clusters.
        
        Parameters:
        texts (list): List of preprocessed comments
        labels (array): Cluster labels (if None, will be predicted)
        
        Returns:
        Counter: Counts of comments in each cluster
        """
        try:
            # Get cluster labels if not provided
            if labels is None:
                labels = self.predict(texts)
            
            # Count labels
            return Counter(labels)
        except Exception as e:
            print(f"Error in get_cluster_distribution: {e}")
            return Counter()
    
    def _preprocess_texts(self, texts):
        """
        Preprocess and validate text data.
        
        Parameters:
        texts (list or Series): List of text comments
        
        Returns:
        list: Cleaned and validated texts
        """
        # Convert Series to list if needed
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Validate and clean texts
        valid_texts = []
        for text in texts:
            # Handle None, non-string, or empty texts
            if text is None or not isinstance(text, str):
                valid_texts.append("")
            else:
                valid_texts.append(text)
        
        return valid_texts
    
    def save_model(self, filepath):
        """
        Save the clustering model.
        
        Parameters:
        filepath (str): Path to save the model
        
        Returns:
        bool: Success flag
        """
        try:
            import pickle
            
            # Create model data
            model_data = {
                'n_clusters': self.n_clusters,
                'max_features': self.max_features,
                'random_state': self.random_state,
                'vectorizer': self.vectorizer,
                'kmeans': self.kmeans,
                'feature_names': self.feature_names,
                'cluster_centers': self.cluster_centers
            }
            
            # Save model
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved clustering model.
        
        Parameters:
        filepath (str): Path to the saved model
        
        Returns:
        CommentClusterer: Loaded model
        """
        try:
            import pickle
            
            # Load model data
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new instance
            model = cls(
                n_clusters=model_data['n_clusters'],
                max_features=model_data['max_features'],
                random_state=model_data['random_state']
            )
            
            # Set model attributes
            model.vectorizer = model_data['vectorizer']
            model.kmeans = model_data['kmeans']
            model.feature_names = model_data['feature_names']
            model.cluster_centers = model_data['cluster_centers']
            
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None