# src/pipeline.py
from src.preprocessing.clean_text import clean_text_for_sentiment
from src.models.sentiment import SentimentAnalyzer
from src.models.clustering import cluster_comments
from src.visualization.sentiment_viz import analyze_sentiment_revenue_correlation
from src.visualization.dashboard_viz import create_dashboard_metrics

def run_integrated_analysis_pipeline(comments_df, movies_df):
    """Run the integrated analysis pipeline."""
    import time
    
    # Generate a unique identifier for this data set
    data_hash = f"{len(comments_df)}_{len(movies_df)}"
    
    results = {}
    results['data_hash'] = data_hash
    
    # Step 1: Ensure we have clean text
    if 'clean_text' not in comments_df.columns and 'text' in comments_df.columns:
        comments_df['clean_text'] = comments_df['text'].apply(clean_text_for_sentiment)
    
    # Step 2: Ensure we have sentiment analysis
    if 'sentiment' not in comments_df.columns:
        analyzer = SentimentAnalyzer()
        
        # Apply context-aware sentiment analysis
        sentiment_results = analyzer.analyze_sentiment(
            comments_df['clean_text'],
            movie_names=comments_df['movie']
        )
        comments_df['sentiment'] = sentiment_results['sentiment']
        comments_df['polarity'] = sentiment_results['polarity']

    # Step 3: Calculate movie-level sentiment metrics
    movie_sentiment = {}
    
    for movie_title in movies_df['title'].unique():
        movie_comments = comments_df[comments_df['movie'] == movie_title]
        
        if len(movie_comments) == 0:
            continue
            
        # Calculate sentiment metrics
        pos_pct = (movie_comments['sentiment'] == 'positive').mean() * 100
        neg_pct = (movie_comments['sentiment'] == 'negative').mean() * 100
        
        if 'polarity' in movie_comments.columns:
            avg_polarity = movie_comments['polarity'].mean()
        else:
            avg_polarity = (pos_pct - neg_pct) / 100
        
        # Store metrics
        movie_sentiment[movie_title] = {
            'positive_pct': pos_pct,
            'negative_pct': neg_pct,
            'avg_polarity': avg_polarity,
            'comment_count': len(movie_comments)
        }
    
    # Merge with movies dataframe
    movies_with_sentiment = movies_df.copy()
    
    # Add sentiment data to movies dataframe
    for movie_title, sentiment_data in movie_sentiment.items():
        movies_with_sentiment.loc[movies_with_sentiment['title'] == movie_title, 'positive_pct'] = sentiment_data['positive_pct']
        movies_with_sentiment.loc[movies_with_sentiment['title'] == movie_title, 'negative_pct'] = sentiment_data['negative_pct']
        movies_with_sentiment.loc[movies_with_sentiment['title'] == movie_title, 'avg_polarity'] = sentiment_data['avg_polarity']
    
    # Step 4: Filter box office movies and analyze correlation
    all_box_office_movies = movies_with_sentiment[movies_with_sentiment['revenue'] > 0].copy()
    normal_range_movies = filter_box_office_outliers(all_box_office_movies, z_score_threshold=2.5)
    
    # Calculate outlier stats
    total_movies = len(all_box_office_movies)
    normal_range_count = len(normal_range_movies)
    outlier_count = total_movies - normal_range_count
    outlier_pct = (outlier_count / total_movies * 100) if total_movies > 0 else 0
    
    # Run correlation analysis
    all_movies_correlation = analyze_sentiment_revenue_correlation(all_box_office_movies)
    normal_range_correlation = analyze_sentiment_revenue_correlation(normal_range_movies)
    
    # Create dashboard metrics
    dashboard_metrics = create_dashboard_metrics(normal_range_movies, comments_df, normal_range_correlation)
    
    # Step 5: Run clustering analysis
    df_with_clusters, cluster_descriptions, cluster_terms, sentiment_by_cluster = cluster_comments(
        comments_df, text_column='clean_text', n_clusters=6, include_sentiment=True
    )
    
    # Store results
    timestamp = time.time()
    results['comments_df'] = df_with_clusters
    results['movies_df'] = movies_with_sentiment
    results['all_box_office_movies'] = all_box_office_movies
    results['normal_range_movies'] = normal_range_movies
    results['correlation_results'] = normal_range_correlation
    results['all_movies_correlation'] = all_movies_correlation
    results['dashboard_metrics'] = dashboard_metrics
    results['cluster_descriptions'] = cluster_descriptions
    results['cluster_terms'] = cluster_terms
    results['sentiment_by_cluster'] = sentiment_by_cluster
    results['movie_sentiment'] = movie_sentiment
    results['timestamp'] = timestamp
    results['outlier_stats'] = {
        'total_box_office_movies': total_movies,
        'normal_range_count': normal_range_count,
        'outlier_count': outlier_count,
        'outlier_pct': outlier_pct,
        'z_score_threshold': 2.5
    }
    
    return results

def filter_box_office_outliers(movies_df, revenue_column='revenue', budget_column='budget', 
                              z_score_threshold=2.5, min_movies=10):
    """Filter outlier movies based on revenue and budget using z-scores."""
    import numpy as np
    
    # Make a copy to avoid modifying the original
    df = movies_df.copy()
    
    # Filter movies with revenue data
    valid_movies = df[df[revenue_column] > 0].copy()
    
    if len(valid_movies) <= min_movies:
        return valid_movies
    
    # Calculate z-scores for revenue
    valid_movies['revenue_zscore'] = np.abs((valid_movies[revenue_column] - valid_movies[revenue_column].mean()) / valid_movies[revenue_column].std())
    
    # Filter based on revenue z-score
    normal_range_movies = valid_movies[valid_movies['revenue_zscore'] <= z_score_threshold]
    
    # If we filtered too aggressively, adjust threshold
    while len(normal_range_movies) < min_movies and z_score_threshold < 5.0:
        z_score_threshold += 0.5
        normal_range_movies = valid_movies[valid_movies['revenue_zscore'] <= z_score_threshold]
    
    # Drop temporary columns
    if 'revenue_zscore' in normal_range_movies.columns:
        normal_range_movies = normal_range_movies.drop('revenue_zscore', axis=1)
    
    return normal_range_movies