# src/models/clustering.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import re

class CommentClusterer:
    """Improved class for clustering comments into meaningful thematic groups."""
    
    def __init__(self, n_clusters=5, random_state=42):
        """
        Initialize the clusterer.
        
        Parameters:
        n_clusters (int): Number of clusters to create
        random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        self.feature_names = None
        self.cluster_centers = None
        self.theme_patterns = self._create_theme_patterns()
    
    def _create_theme_patterns(self):
        """Create patterns for identifying movie-related comment themes."""
        return {
            # Expectations & Anticipation
            'anticipation': [
                'cant wait', "can't wait", 'looking forward', 'excited', 'hyped',
                'excited for', 'hype', 'finally', 'hope', 'please', 'imagine',
                'gonna be', 'going to be', 'will be', 'should be', 'want to see'
            ],
            
            # Visual/Effects commentary
            'visuals': [
                'looks', 'animation', 'effects', 'graphics', 'cgi', 'beautiful',
                'stunning', 'gorgeous', 'ugly', 'realistic', 'detailed', 'style',
                'visuals', 'visual', 'design', 'cinematography', 'shots', 'camera'
            ],
            
            # Actor & Character focused
            'characters': [
                'actor', 'actress', 'character', 'protagonist', 'villain', 
                'cast', 'casting', 'role', 'voice', 'voices', 'performances',
                'played by', 'star', 'stars', 'starring', 'acting', 'portrayal'
            ],
            
            # Comparing to other media
            'comparison': [
                'better than', 'worse than', 'reminds me', 'similar to', 'like the',
                'just like', 'compared to', 'copy', 'ripoff', 'inspired by',
                'original', 'sequel', 'prequel', 'franchise', 'universe', 'series'
            ],
            
            # Story & Plot Commentary
            'story': [
                'story', 'plot', 'writing', 'narrative', 'ending', 'beginning',
                'twist', 'surprise', 'predictable', 'cliche', 'classic', 'arc',
                'character development', 'script', 'screenplay', 'adaptation'
            ],
            
            # Emotional Reactions
            'emotion': [
                'love', 'hate', 'amazing', 'awesome', 'terrible', 'awful',
                'best', 'worst', 'glad', 'sad', 'laugh', 'cry', 'emotional',
                'hilarious', 'epic', 'shocking', 'touching', 'boring', 'excited'
            ],
            
            # Criticism
            'criticism': [
                'disappointed', 'disappointing', 'bad', 'terrible', 'awful',
                'worst', 'waste', 'trash', 'garbage', 'hate', 'stupid',
                'boring', 'problem', 'issue', 'wrong', 'mistake', 'missed',
                'ruined', 'ruin', 'destroyed', 'cash grab', 'flop', 'fail'
            ],
            
            # Hype & Marketing
            'marketing': [
                'trailer', 'promotion', 'ad', 'advertisement', 'marketing',
                'poster', 'teaser', 'preview', 'reveal', 'announced', 'release date',
                'premiere', 'theater', 'theatre', 'tickets', 'box office'
            ],
            
            # Music & Sound
            'sound': [
                'music', 'soundtrack', 'score', 'sound', 'song', 'theme',
                'audio', 'composer', 'track', 'beat', 'tune', 'sound effects'
            ],
            
            # Director & Production
            'production': [
                'director', 'producer', 'writer', 'studio', 'budget',
                'production', 'release', 'delayed', 'postponed', 'filming',
                'behind the scenes', 'making of', 'crew', 'development'
            ]
        }
    
    def fit_transform(self, texts, additional_features=None):
        """
        Fit the clustering model and transform texts.
        
        Parameters:
        texts (list): List of preprocessed comments
        additional_features (DataFrame): Optional additional features for clustering
        
        Returns:
        tuple: (cluster_labels, cluster_terms, theme_mapping)
        """
        # Clean and validate texts
        valid_texts = self._preprocess_texts(texts)
        
        # Extract theme indicators
        theme_indicators = self._extract_theme_indicators(valid_texts)
        
        # Get document vectors from TF-IDF
        X = self.vectorizer.fit_transform(valid_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # If we have additional features, combine them with TF-IDF
        if additional_features is not None:
            # Normalize additional features
            normalized_features = (additional_features - additional_features.mean()) / additional_features.std()
            # Convert sparse matrix to dense and combine
            X_dense = X.toarray()
            X_combined = np.hstack((X_dense, normalized_features.values))
        else:
            X_combined = X
        
        # Fit KMeans
        self.kmeans.fit(X_combined)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Get cluster assignments
        cluster_labels = self.kmeans.predict(X_combined)
        
        # Get top terms per cluster
        cluster_terms = self._get_top_terms_per_cluster()
        
        # Map clusters to themes
        theme_mapping = self._map_clusters_to_themes(cluster_labels, theme_indicators)
        
        return cluster_labels, cluster_terms, theme_mapping
    
    def _extract_theme_indicators(self, texts):
        """
        Extract theme indicators from texts.
        
        Parameters:
        texts (list): List of preprocessed comments
        
        Returns:
        DataFrame: DataFrame with theme indicators
        """
        theme_indicators = {theme: [] for theme in self.theme_patterns}
        
        for text in texts:
            text_lower = text.lower()
            
            # Check each theme
            for theme, patterns in self.theme_patterns.items():
                # Check if any pattern matches
                has_theme = any(
                    re.search(r'\b' + re.escape(pattern) + r'\b', text_lower) 
                    if not pattern.startswith(r'\\') and not pattern.startswith('"') and not pattern.startswith("'")
                    else re.search(pattern, text_lower) 
                    for pattern in patterns
                )
                theme_indicators[theme].append(1 if has_theme else 0)
        
        return pd.DataFrame(theme_indicators)
    
    def _map_clusters_to_themes(self, cluster_labels, theme_indicators):
        """
        Map clusters to themes based on theme indicators.
        
        Parameters:
        cluster_labels (array): Cluster assignments
        theme_indicators (DataFrame): DataFrame with theme indicators
        
        Returns:
        dict: Dictionary mapping cluster IDs to theme descriptions
        """
        # Add cluster labels to theme indicators
        theme_indicators['cluster'] = cluster_labels
        
        # Calculate theme prevalence for each cluster
        cluster_themes = {}
        
        for cluster_id in range(self.n_clusters):
            # Get theme indicators for this cluster
            cluster_theme_indicators = theme_indicators[theme_indicators['cluster'] == cluster_id]
            
            if cluster_theme_indicators.empty:
                cluster_themes[cluster_id] = "Miscellaneous Comments"
                continue
            
            # Calculate theme prevalence
            theme_prevalence = cluster_theme_indicators.drop('cluster', axis=1).mean()
            
            # Get most prevalent themes (above threshold)
            threshold = max(0.15, theme_prevalence.max() * 0.7)  # At least 15% or 70% of max
            top_themes = theme_prevalence[theme_prevalence >= threshold].sort_values(ascending=False)
            
            if top_themes.empty:
                # If no themes above threshold, use top 2
                top_themes = theme_prevalence.sort_values(ascending=False).head(2)
            
            # Create theme description
            top_theme_names = top_themes.index.tolist()
            if top_theme_names:
                # Create more descriptive theme names based on the theme
                theme_descriptions = {
                    'anticipation': 'Anticipation & Expectations',
                    'visuals': 'Visual Effects & Cinematography',
                    'characters': 'Character & Acting Comments',
                    'comparison': 'Comparisons to Other Media',
                    'story': 'Story & Plot Discussion',
                    'emotion': 'Emotional Reactions',
                    'criticism': 'Critical Comments',
                    'marketing': 'Marketing & Promotion',
                    'sound': 'Music & Sound Design',
                    'production': 'Production & Direction'
                }
                
                # Get better descriptions for the top themes
                formatted_themes = [theme_descriptions.get(name, name.title()) for name in top_theme_names[:2]]
                
                # Special case combinations
                if 'emotion' in top_theme_names and 'criticism' in top_theme_names:
                    cluster_themes[cluster_id] = "Negative Reactions & Criticism"
                elif 'emotion' in top_theme_names and 'anticipation' in top_theme_names:
                    cluster_themes[cluster_id] = "Excited Anticipation"
                elif 'story' in top_theme_names and 'criticism' in top_theme_names:
                    cluster_themes[cluster_id] = "Story & Plot Criticism"
                elif 'visuals' in top_theme_names and 'emotion' in top_theme_names:
                    if theme_prevalence['emotion'] > 0 and 'criticism' not in top_theme_names[:3]:
                        cluster_themes[cluster_id] = "Visual Appreciation"
                    else:
                        cluster_themes[cluster_id] = "Visual Criticism"
                else:
                    cluster_themes[cluster_id] = " & ".join(formatted_themes)
            else:
                cluster_themes[cluster_id] = "Miscellaneous Comments"
        
        return cluster_themes
    
    def _get_top_terms_per_cluster(self, n_terms=10):
        """
        Get the top terms for each cluster.
        
        Parameters:
        n_terms (int): Number of terms to return per cluster
        
        Returns:
        dict: Dictionary mapping cluster IDs to top terms
        """
        # Check if model has been fitted
        if self.feature_names is None or self.cluster_centers is None:
            return {}
        
        # Get the feature names
        feature_names = self.feature_names
        
        # Get the cluster centroids
        centroids = self.cluster_centers
        
        # Get top terms for each cluster
        cluster_terms = {}
        for i in range(self.n_clusters):
            # If centroids is 2D (TF-IDF only)
            if len(centroids.shape) == 2 and centroids.shape[1] >= len(feature_names):
                # Get the indices of the top terms
                indices = centroids[i, :len(feature_names)].argsort()[-n_terms:][::-1]
                # Get the corresponding terms
                terms = [feature_names[j] for j in indices]
            else:
                # If we used additional features, just get top TF-IDF terms
                tfidf_centroid = centroids[i][:len(feature_names)]
                indices = tfidf_centroid.argsort()[-n_terms:][::-1]
                terms = [feature_names[j] for j in indices]
            
            cluster_terms[i] = terms
        
        return cluster_terms
    
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

def create_cluster_visualization(df, cluster_col='cluster', theme_mapping=None):
    """
    Create visualization data for clusters with shortened labels for graphs.
    
    Parameters:
    df (DataFrame): DataFrame with clustered comments
    cluster_col (str): Column with cluster assignments
    theme_mapping (dict): Dictionary mapping cluster IDs to theme descriptions
    
    Returns:
    dict: Visualization data for clusters
    """
    # Count comments per cluster
    cluster_counts = df[cluster_col].value_counts().sort_index()
    
    # Get cluster labels (use theme mapping if available)
    if theme_mapping:
        # Create full labels
        full_labels = [f"{theme_mapping.get(i, f'Cluster {i}')} (Cluster {i})" 
                     for i in cluster_counts.index]
        
        # Create short labels for graphs
        short_labels = []
        for i in cluster_counts.index:
            theme = theme_mapping.get(i, f"Cluster {i}")
            # Extract just the main theme (before the colon)
            main_theme = theme.split(':')[0]
            # If it has "Emotion & Production", just take "Emotion"
            if '&' in main_theme:
                main_theme = main_theme.split('&')[0].strip()
            short_labels.append(f"{main_theme} ({i})")
    else:
        full_labels = [f"Cluster {i}" for i in cluster_counts.index]
        short_labels = [f"Cluster {i}" for i in cluster_counts.index]
    
    # Get sentiment distribution by cluster
    sentiment_by_cluster = None
    if 'sentiment' in df.columns:
        sentiment_by_cluster = pd.crosstab(
            df[cluster_col], df['sentiment'], normalize='index'
        )
    
    return {
        'counts': cluster_counts,
        'labels': full_labels,
        'short_labels': short_labels,
        'sentiment': sentiment_by_cluster
    }

def cluster_comments(comments_df, text_column='clean_text', n_clusters=5, include_sentiment=True):
    """
    Cluster comments and add cluster labels to the dataframe.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    text_column (str): Column containing text to cluster
    n_clusters (int): Number of clusters
    include_sentiment (bool): Whether to include sentiment as a feature
    
    Returns:
    tuple: (DataFrame with cluster labels, cluster descriptions, cluster terms, sentiment by cluster)
    """
    # Initialize and fit clusterer
    clusterer = CommentClusterer(n_clusters=n_clusters)
    
    # Prepare additional features if requested
    additional_features = None
    if include_sentiment and 'polarity' in comments_df.columns:
        # Create features from sentiment data
        additional_features = pd.DataFrame({
            'polarity': comments_df['polarity'],
        })
        
        # Add sentiment category as one-hot features
        if 'sentiment' in comments_df.columns:
            sentiment_dummies = pd.get_dummies(comments_df['sentiment'], prefix='sentiment')
            additional_features = pd.concat([additional_features, sentiment_dummies], axis=1)
    
    # Fit and transform
    cluster_labels, cluster_terms, theme_mapping = clusterer.fit_transform(
        comments_df[text_column], 
        additional_features=additional_features
    )
    
    # Add cluster assignments to DataFrame
    df_with_clusters = comments_df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Get sentiment distribution by cluster (if sentiment column exists)
    sentiment_by_cluster = None
    if 'sentiment' in df_with_clusters.columns:
        sentiment_by_cluster = pd.crosstab(
            df_with_clusters['cluster'], 
            df_with_clusters['sentiment'],
            normalize='index'
        )
    
    # Create expanded cluster descriptions with top terms
    cluster_descriptions = {}
    for cluster_id, theme in theme_mapping.items():
        top_terms = cluster_terms.get(cluster_id, [])[:5]  # Get top 5 terms
        terms_str = ", ".join(top_terms)
        cluster_descriptions[cluster_id] = f"{theme}: {terms_str}"
    
    return df_with_clusters, cluster_descriptions, cluster_terms, sentiment_by_cluster