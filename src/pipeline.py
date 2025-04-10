# src/pipeline.py

import pandas as pd
import numpy as np
import time
from src.models.sentiment import SentimentAnalyzer
from src.models.clustering import CommentClusterer, enhance_clustering, analyze_cluster_revenue_correlation
from src.models.summarization import TextRankSummarizer, generate_enhanced_summaries, format_summarization_insights
from src.visualization.sentiment_viz import analyze_sentiment_revenue_correlation, create_enhanced_sentiment_visualizations
from src.visualization.dashboard_viz import create_dashboard_metrics, create_professional_visualizations

def run_integrated_analysis_pipeline(comments_df, movies_df):
    """
    Run the full integrated analysis pipeline.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    movies_df (DataFrame): DataFrame with movie data
    
    Returns:
    dict: Dictionary with all analysis results
    """
    results = {}
    
    # Step 1: Sentiment Analysis
    print("Running sentiment analysis...")
    
    # Check if sentiment analysis is already done
    if 'sentiment' not in comments_df.columns:
        analyzer = SentimentAnalyzer()
        # Ensure we have clean text
        if 'clean_text' not in comments_df.columns and 'text' in comments_df.columns:
            from src.preprocessing.clean_text import clean_text_for_sentiment
            comments_df['clean_text'] = comments_df['text'].apply(clean_text_for_sentiment)
            
        sentiment_results = analyzer.analyze_sentiment(comments_df['clean_text'])
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
    
    # Step 2: Correlation Analysis
    print("Running correlation analysis...")
    correlation_results = analyze_sentiment_revenue_correlation(movies_with_sentiment)
    results['correlation_results'] = correlation_results
    
    # Step 3: Clustering
    print("Running clustering analysis...")
    clustering_results = enhance_clustering(comments_df)
    results['clustering_results'] = clustering_results
    
    # Get clustered comments dataframe
    comments_df_with_clusters = clustering_results.get('comments_df', comments_df)
    
    # Step 4: Cluster-Revenue Correlation
    print("Analyzing cluster-revenue correlation...")
    cluster_correlation_results = analyze_cluster_revenue_correlation(
        comments_df_with_clusters,
        movies_with_sentiment
    )
    results['cluster_correlation_results'] = cluster_correlation_results
    
    # Step 5: Summarization
    print("Generating enhanced summaries...")
    summarization_results = generate_enhanced_summaries(
        comments_df_with_clusters,
        clustering_results
    )
    results['summarization_results'] = summarization_results
    
    # Format summarization insights
    formatted_insights = format_summarization_insights(
        summarization_results['cluster_summaries'],
        clustering_results['cluster_descriptions'],
        cluster_correlation_results
    )
    results['formatted_insights'] = formatted_insights
    
    # Step 6: Dashboard Metrics and Visualizations
    print("Creating dashboard metrics and visualizations...")
    dashboard_metrics = create_dashboard_metrics(
        movies_with_sentiment,
        comments_df_with_clusters,
        correlation_results
    )
    results['dashboard_metrics'] = dashboard_metrics
    
    # Create professional visualizations
    visualizations = create_professional_visualizations(
        comments_df_with_clusters,
        movies_with_sentiment,
        clustering_results,
        correlation_results
    )
    results['visualizations'] = visualizations
    
    # Return the updated dataframes along with results
    results['comments_df'] = comments_df_with_clusters
    results['movies_df'] = movies_with_sentiment
    
    print("Integrated analysis pipeline completed!")
    return results