# src/pipeline.py
import pandas as pd
import numpy as np
from src.models.sentiment import SentimentAnalyzer
from src.models.clustering import cluster_comments
from src.models.summarization import summarize_by_sentiment, summarize_by_cluster
from src.visualization.sentiment_viz import analyze_sentiment_revenue_correlation
from src.visualization.dashboard_viz import create_dashboard_metrics

def run_integrated_analysis_pipeline(comments_df, movies_df):
    """
    Run the simplified integrated analysis pipeline.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    movies_df (DataFrame): DataFrame with movie data
    
    Returns:
    dict: Dictionary with all analysis results
    """
    results = {}
    
    # Step 1: Ensure we have clean text
    if 'clean_text' not in comments_df.columns and 'text' in comments_df.columns:
        from src.preprocessing.clean_text import clean_text_for_sentiment
        comments_df['clean_text'] = comments_df['text'].apply(clean_text_for_sentiment)
    
    # Step 2: Sentiment Analysis
    print("Running sentiment analysis...")

    # Check if sentiment analysis is already done
    if 'sentiment' not in comments_df.columns:
        analyzer = SentimentAnalyzer()
        # Ensure we have clean text
        if 'clean_text' not in comments_df.columns and 'text' in comments_df.columns:
            from src.preprocessing.clean_text import clean_text_for_sentiment
            comments_df['clean_text'] = comments_df['text'].apply(clean_text_for_sentiment)
            
        # Apply context-aware sentiment analysis
        sentiment_results = analyzer.analyze_sentiment(
            comments_df['clean_text'],
            movie_names=comments_df['movie']  # Pass movie names for context
        )
        comments_df['sentiment'] = sentiment_results['sentiment']
        comments_df['polarity'] = sentiment_results['polarity']

    # Calculate movie-level sentiment statistics
    movie_sentiment = comments_df.groupby('movie').agg({
        'polarity': 'mean',
        'sentiment': lambda x: (x == 'positive').mean()
    }).rename(columns={'sentiment': 'positive_ratio'})
    
    # Merge with movies dataframe
    movies_with_sentiment = pd.merge(
        movies_df,
        movie_sentiment,
        left_on='title',
        right_index=True,
        how='left'
    )
    
    # Add sentiment columns
    movies_with_sentiment['avg_polarity'] = movies_with_sentiment['polarity']
    movies_with_sentiment['positive_pct'] = movies_with_sentiment['positive_ratio'] * 100
    
    # Step 3: Clustering
    print("Running clustering analysis...")
    df_with_clusters, cluster_descriptions, cluster_terms, sentiment_by_cluster = cluster_comments(
        comments_df, text_column='clean_text', n_clusters=6, include_sentiment=True
    )

    # Store clustering results
    clustering_results = {
        'comments_df': df_with_clusters,
        'cluster_descriptions': cluster_descriptions,
        'cluster_terms': cluster_terms,
        'sentiment_by_cluster': sentiment_by_cluster
    }
    results['clustering_results'] = clustering_results

    # Get clustered comments dataframe
    comments_df_with_clusters = df_with_clusters
    
    # Step 4: Summarization
    print("Generating summaries...")
    sentiment_summaries = summarize_by_sentiment(
        df_with_clusters, text_column='clean_text', sentiment_column='sentiment'
    )
    
    cluster_summaries = summarize_by_cluster(
        df_with_clusters, text_column='clean_text', cluster_column='cluster',
        cluster_descriptions=cluster_descriptions
    )
    
    # Step 5: Correlation Analysis
    print("Running correlation analysis...")
    correlation_results = analyze_sentiment_revenue_correlation(movies_with_sentiment)
    
    # Step 6: Dashboard Metrics
    print("Creating dashboard metrics...")
    dashboard_metrics = create_dashboard_metrics(
        movies_with_sentiment, df_with_clusters, correlation_results
    )
    
    # Prepare results
    results['comments_df'] = df_with_clusters
    results['movies_df'] = movies_with_sentiment
    results['cluster_descriptions'] = cluster_descriptions
    results['cluster_terms'] = cluster_terms
    results['sentiment_by_cluster'] = sentiment_by_cluster
    results['sentiment_summaries'] = sentiment_summaries
    results['cluster_summaries'] = cluster_summaries
    results['correlation_results'] = correlation_results
    results['dashboard_metrics'] = dashboard_metrics
    
    print("Integrated analysis pipeline completed!")
    return results