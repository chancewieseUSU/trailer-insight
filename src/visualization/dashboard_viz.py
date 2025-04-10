# src/visualization/dashboard_viz.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def create_dashboard_metrics(movies_df, comments_df, correlation_results=None):
    """
    Create simple dashboard metrics for the Streamlit app.
    
    Parameters:
    movies_df (DataFrame): DataFrame with movie data
    comments_df (DataFrame): DataFrame with comment data
    correlation_results (dict): Results from correlation analysis
    
    Returns:
    dict: Dictionary with dashboard metrics
    """
    metrics = {}
    
    # Movie metrics
    metrics["total_movies"] = len(movies_df)
    if "revenue" in movies_df.columns:
        metrics["avg_revenue"] = movies_df["revenue"].mean()
    
    # Comment metrics
    metrics["total_comments"] = len(comments_df)
    metrics["avg_comments_per_movie"] = len(comments_df) / len(movies_df) if len(movies_df) > 0 else 0
    
    # Sentiment metrics
    if "sentiment" in comments_df.columns:
        sentiment_counts = comments_df["sentiment"].value_counts(normalize=True)
        metrics["positive_pct"] = sentiment_counts.get("positive", 0) * 100
        metrics["negative_pct"] = sentiment_counts.get("negative", 0) * 100
    
    # Correlation metrics
    if correlation_results and "r_squared" in correlation_results:
        metrics["sentiment_correlation"] = correlation_results.get("r_squared", 0)
    
    return metrics

def create_cluster_visualization(comments_df, cluster_descriptions=None):
    """
    Create visualization of comment clusters.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data including clusters
    cluster_descriptions (dict): Dictionary mapping cluster IDs to descriptions
    
    Returns:
    Plotly figure object
    """
    if "cluster" not in comments_df.columns:
        return go.Figure().update_layout(title="No cluster data available")
    
    # Count comments per cluster
    cluster_counts = comments_df["cluster"].value_counts().sort_index()
    
    # Get cluster labels (use descriptions if available)
    if cluster_descriptions:
        labels = [cluster_descriptions.get(i, f"Cluster {i}") for i in cluster_counts.index]
    else:
        labels = [f"Cluster {i}" for i in cluster_counts.index]
    
    # Create bar chart
    fig = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={"x": "Cluster", "y": "Count"},
        title="Comment Clusters Distribution"
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(cluster_counts.index), ticktext=labels),
        template="plotly_white"
    )
    
    return fig

def create_box_office_visualization(movies_df):
    """
    Create visualization of box office performance.
    
    Parameters:
    movies_df (DataFrame): DataFrame with movie data
    
    Returns:
    Plotly figure object
    """
    if "revenue" not in movies_df.columns or "title" not in movies_df.columns:
        return go.Figure().update_layout(title="No box office data available")
    
    # Filter movies with revenue data
    valid_movies = movies_df[movies_df["revenue"] > 0].copy()
    
    if len(valid_movies) == 0:
        return go.Figure().update_layout(title="No movies with box office data")
    
    # Sort by revenue
    valid_movies = valid_movies.sort_values("revenue", ascending=False)
    
    # Limit to top 10 movies for readability
    top_movies = valid_movies.head(10)
    
    # Create bar chart
    fig = px.bar(
        top_movies,
        x="title",
        y="revenue",
        title="Box Office Revenue by Movie",
        labels={"title": "Movie", "revenue": "Revenue ($)"}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    
    return fig