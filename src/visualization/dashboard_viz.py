# src/visualization/dashboard_viz.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dashboard_metrics(movies_df, comments_df, correlation_results):
    """
    Create dashboard metrics for the Streamlit app.
    
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
    metrics["avg_revenue"] = movies_df["revenue"].mean() if "revenue" in movies_df.columns else 0
    
    # Comment metrics
    metrics["total_comments"] = len(comments_df)
    metrics["avg_comments_per_movie"] = len(comments_df) / len(movies_df) if len(movies_df) > 0 else 0
    
    # Sentiment metrics
    if "sentiment" in comments_df.columns:
        sentiment_counts = comments_df["sentiment"].value_counts(normalize=True)
        metrics["positive_pct"] = sentiment_counts.get("positive", 0) * 100
        metrics["negative_pct"] = sentiment_counts.get("negative", 0) * 100
    
    # Correlation metrics
    correlations = correlation_results.get("correlations", {})
    if correlations:
        # Get highest absolute correlation
        top_corr = max(correlations.items(), key=lambda x: abs(x[1]), default=(None, 0))
        metrics["top_correlation_metric"] = top_corr[0]
        metrics["top_correlation_value"] = top_corr[1]
    
    # Predictive accuracy (placeholder for future implementation)
    metrics["predictive_accuracy"] = None
    
    return metrics

def create_professional_visualizations(comments_df, movies_df, cluster_results, correlation_results):
    """
    Create professional visualizations for the Streamlit app.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    movies_df (DataFrame): DataFrame with movie data
    cluster_results (dict): Results from clustering
    correlation_results (dict): Results from correlation analysis
    
    Returns:
    dict: Dictionary with professional visualizations
    """
    visualizations = {}
    
    # 1. Create sentiment distribution pie chart
    if "sentiment" in comments_df.columns:
        sentiment_counts = comments_df["sentiment"].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Overall Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#f1c40f',
                'negative': '#e74c3c'
            },
            hole=0.4
        )
        
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        visualizations["sentiment_pie"] = fig
    
    # 2. Create cluster distribution bar chart
    if "cluster" in comments_df.columns:
        cluster_counts = comments_df["cluster"].value_counts().sort_index()
        
        # Get cluster descriptions if available
        cluster_descriptions = cluster_results.get("cluster_descriptions", {})
        labels = [cluster_descriptions.get(i, f"Cluster {i}") for i in cluster_counts.index]
        
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Comment Clusters Distribution",
            labels={"x": "Cluster", "y": "Count"},
            text=cluster_counts.values
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis=dict(tickmode="array", tickvals=list(cluster_counts.index), ticktext=labels)
        )
        
        visualizations["cluster_bar"] = fig
    
    # 3. Create sentiment by movie heatmap
    if "sentiment" in comments_df.columns and "movie" in comments_df.columns:
        sentiment_by_movie = pd.crosstab(
            comments_df["movie"],
            comments_df["sentiment"],
            normalize="index"
        )
        
        # Sort by positive sentiment
        if "positive" in sentiment_by_movie.columns:
            sentiment_by_movie = sentiment_by_movie.sort_values("positive", ascending=False)
        
        # Take top 10 movies for readability
        sentiment_by_movie = sentiment_by_movie.head(10)
        
        fig = px.imshow(
            sentiment_by_movie,
            title="Sentiment Distribution by Movie (Top 10)",
            color_continuous_scale=["#e74c3c", "#f1c40f", "#2ecc71"],
            aspect="auto",
            labels=dict(x="Sentiment", y="Movie", color="Proportion")
        )
        
        fig.update_layout(template="plotly_white")
        
        visualizations["sentiment_heatmap"] = fig
    
    # 4. Create box office prediction visualization
    if "revenue" in movies_df.columns and "avg_polarity" in movies_df.columns:
        valid_movies = movies_df[movies_df["revenue"] > 0].copy()
        
        # Calculate regression line
        x = valid_movies["avg_polarity"]
        y = valid_movies["revenue"]
        
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Create scatter plot with trend line
        fig = px.scatter(
            valid_movies,
            x="avg_polarity",
            y="revenue",
            text="title",
            size="comment_count" if "comment_count" in valid_movies.columns else None,
            color="positive_pct" if "positive_pct" in valid_movies.columns else None,
            title="Box Office Revenue Prediction Model",
            labels={
                "avg_polarity": "Comment Sentiment Polarity",
                "revenue": "Box Office Revenue ($)",
                "positive_pct": "Positive Comments (%)"
            }
        )
        
        # Add trend line
        x_range = np.linspace(x.min(), x.max(), 100)
        y_range = slope * x_range + intercept
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode="lines",
                name=f"Trend (R²={r_value**2:.2f})",
                line=dict(color="red", dash="dash")
            )
        )
        
        fig.update_layout(
            template="plotly_white",
            annotations=[
                dict(
                    x=0.01,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"R² = {r_value**2:.2f}, p = {p_value:.3f}",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )
        
        visualizations["prediction_model"] = fig
    
    return visualizations