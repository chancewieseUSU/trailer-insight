# src/pipeline.py
import pandas as pd
import numpy as np
from src.models.sentiment import SentimentAnalyzer
from src.models.clustering import cluster_comments
from src.models.summarization import summarize_by_sentiment, summarize_by_cluster
from src.visualization.sentiment_viz import analyze_sentiment_revenue_correlation
from src.visualization.dashboard_viz import create_dashboard_metrics

def run_integrated_analysis_pipeline(comments_df, movies_df, force_refresh=False):
    """
    Run the integrated analysis pipeline.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    movies_df (DataFrame): DataFrame with movie data
    force_refresh (bool): Force refresh of analysis even if cached
    
    Returns:
    dict: Dictionary with all analysis results
    """
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    
    # Generate a unique identifier for this data set to prevent caching issues
    data_hash = f"{len(comments_df)}_{len(movies_df)}"
    
    results = {}
    results['data_hash'] = data_hash
    
    # Step 1: Ensure we have clean text
    if 'clean_text' not in comments_df.columns and 'text' in comments_df.columns:
        from src.preprocessing.clean_text import clean_text_for_sentiment
        comments_df['clean_text'] = comments_df['text'].apply(clean_text_for_sentiment)
    
    # Step 2: Ensure we have sentiment analysis
    print("Checking sentiment analysis...")
    if 'sentiment' not in comments_df.columns:
        print("Running sentiment analysis...")
        from src.models.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        
        # Apply context-aware sentiment analysis
        sentiment_results = analyzer.analyze_sentiment(
            comments_df['clean_text'],
            movie_names=comments_df['movie']  # Pass movie names for context
        )
        comments_df['sentiment'] = sentiment_results['sentiment']
        comments_df['polarity'] = sentiment_results['polarity']
    else:
        print("Sentiment analysis already completed.")

    # Step 3: Ensure we have movie-level sentiment metrics
    print("Calculating movie-level sentiment metrics...")
    # Calculate movie-level sentiment statistics
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
    
    # Step 4: Filter out revenue outliers for better correlation analysis
    print("Filtering outlier movies for better analysis...")
    
    # Get both all valid movies and filtered movies within normal range
    all_box_office_movies = movies_with_sentiment[movies_with_sentiment['revenue'] > 0].copy()
    normal_range_movies = filter_box_office_outliers(
        all_box_office_movies, 
        revenue_column='revenue', 
        budget_column='budget',
        z_score_threshold=2.5,  # 2.5 standard deviations is a common outlier threshold
        min_movies=10  # Ensure we have at least 10 movies after filtering
    )
    
    # Calculate what percentage of movies were filtered as outliers
    total_movies = len(all_box_office_movies)
    normal_range_count = len(normal_range_movies)
    outlier_count = total_movies - normal_range_count
    outlier_pct = (outlier_count / total_movies * 100) if total_movies > 0 else 0
    
    # Add this information to results
    results['outlier_stats'] = {
        'total_box_office_movies': total_movies,
        'normal_range_count': normal_range_count,
        'outlier_count': outlier_count,
        'outlier_pct': outlier_pct
    }
    
    # Add timestamp to ensure box office analysis is fresh
    import time
    timestamp = time.time()
    results['timestamp'] = timestamp
    
    # Step 5: Run correlation analysis for box office using filtered movies
    print("Running correlation analysis...")
    from src.visualization.sentiment_viz import analyze_sentiment_revenue_correlation
    
    # Run correlation analysis on both datasets for comparison
    all_movies_correlation = analyze_sentiment_revenue_correlation(all_box_office_movies)
    normal_range_correlation = analyze_sentiment_revenue_correlation(normal_range_movies)
    
    # Store both results
    results['correlation_results'] = normal_range_correlation  # Use filtered results as primary
    results['all_movies_correlation'] = all_movies_correlation  # Store all movies results for comparison
    
    # Step 6: Create dashboard metrics
    print("Creating dashboard metrics...")
    from src.visualization.dashboard_viz import create_dashboard_metrics
    dashboard_metrics = create_dashboard_metrics(
        normal_range_movies, comments_df, normal_range_correlation
    )
    results['dashboard_metrics'] = dashboard_metrics
    
    # Step 7: Run clustering analysis
    print("Running clustering analysis...")
    from src.models.clustering import cluster_comments
    df_with_clusters, cluster_descriptions, cluster_terms, sentiment_by_cluster = cluster_comments(
        comments_df, text_column='clean_text', n_clusters=6, include_sentiment=True
    )
    
    # Store all results
    results['comments_df'] = df_with_clusters
    results['movies_df'] = movies_with_sentiment
    results['all_box_office_movies'] = all_box_office_movies
    results['normal_range_movies'] = normal_range_movies  # Store filtered movies
    results['cluster_descriptions'] = cluster_descriptions
    results['cluster_terms'] = cluster_terms
    results['sentiment_by_cluster'] = sentiment_by_cluster
    results['movie_sentiment'] = movie_sentiment
    
    print("Analysis pipeline completed!")
    return results

def filter_box_office_outliers(movies_df, revenue_column='revenue', budget_column='budget', 
                              z_score_threshold=2.5, min_movies=10):
    """
    Filter outlier movies based on revenue and budget using z-scores.
    
    Parameters:
    movies_df (DataFrame): DataFrame with movie data
    revenue_column (str): Column containing revenue data
    budget_column (str): Column containing budget data
    z_score_threshold (float): Threshold for z-score filtering (default: 2.5)
    min_movies (int): Minimum number of movies to retain after filtering
    
    Returns:
    DataFrame: Filtered movies DataFrame
    """
    import numpy as np
    import pandas as pd
    
    # Make a copy to avoid modifying the original
    df = movies_df.copy()
    
    # Ensure data types are correct
    if revenue_column in df.columns:
        df[revenue_column] = pd.to_numeric(df[revenue_column], errors='coerce')
    
    if budget_column in df.columns:
        df[budget_column] = pd.to_numeric(df[budget_column], errors='coerce')
    
    # Calculate ROI (Return on Investment) if both revenue and budget are available
    if revenue_column in df.columns and budget_column in df.columns:
        # Only calculate for movies with both values
        mask = (df[revenue_column] > 0) & (df[budget_column] > 0)
        if mask.sum() > 0:
            df.loc[mask, 'roi'] = (df.loc[mask, revenue_column] - df.loc[mask, budget_column]) / df.loc[mask, budget_column]
    
    # Filter movies with revenue data
    valid_movies = df[df[revenue_column] > 0].copy()
    
    if len(valid_movies) <= min_movies:
        # If we have too few movies, don't filter
        return valid_movies
    
    # Calculate z-scores for revenue
    valid_movies['revenue_zscore'] = np.abs((valid_movies[revenue_column] - valid_movies[revenue_column].mean()) / valid_movies[revenue_column].std())
    
    # Filter based on revenue z-score
    normal_range_movies = valid_movies[valid_movies['revenue_zscore'] <= z_score_threshold]
    
    # If we filtered too aggressively, adjust threshold
    while len(normal_range_movies) < min_movies and z_score_threshold < 5.0:
        z_score_threshold += 0.5
        normal_range_movies = valid_movies[valid_movies['revenue_zscore'] <= z_score_threshold]
    
    # Calculate ROI z-scores if ROI is available
    if 'roi' in normal_range_movies.columns:
        roi_mask = normal_range_movies['roi'].notna()
        if roi_mask.sum() > min_movies:
            normal_range_movies.loc[roi_mask, 'roi_zscore'] = np.abs(
                (normal_range_movies.loc[roi_mask, 'roi'] - normal_range_movies.loc[roi_mask, 'roi'].mean()) / 
                normal_range_movies.loc[roi_mask, 'roi'].std()
            )
            
            # Filter based on ROI z-score
            roi_normal_range = normal_range_movies[
                ~normal_range_movies['roi_zscore'].notna() | 
                (normal_range_movies['roi_zscore'] <= z_score_threshold)
            ]
            
            # Only use ROI filtering if we have enough movies left
            if len(roi_normal_range) >= min_movies:
                normal_range_movies = roi_normal_range
    
    # Drop temporary columns
    if 'revenue_zscore' in normal_range_movies.columns:
        normal_range_movies = normal_range_movies.drop('revenue_zscore', axis=1)
    
    if 'roi_zscore' in normal_range_movies.columns:
        normal_range_movies = normal_range_movies.drop('roi_zscore', axis=1)
    
    return normal_range_movies