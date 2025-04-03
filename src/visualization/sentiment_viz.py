# src/visualization/sentiment_viz.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_sentiment_bar_chart(sentiment_data, title="Sentiment Distribution"):
    """
    Create a bar chart showing sentiment distribution.
    
    Parameters:
    sentiment_data (DataFrame): DataFrame with columns for movie/category and sentiment counts
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    fig = px.bar(sentiment_data, 
                barmode='group', 
                color_discrete_sequence=['#e74c3c', '#f1c40f', '#2ecc71'],
                title=title)
    
    fig.update_layout(
        xaxis_title="Movie",
        yaxis_title="Count",
        legend_title="Sentiment",
        template="plotly_white"
    )
    
    return fig

def create_sentiment_distribution_pie(df, sentiment_column='sentiment', title="Sentiment Distribution"):
    """
    Create a pie chart showing the distribution of sentiment categories.
    
    Parameters:
    df (DataFrame): DataFrame with sentiment data
    sentiment_column (str): Column containing sentiment categories
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    sentiment_counts = df[sentiment_column].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Define colors for sentiment categories
    color_map = {
        'positive': '#2ecc71',  # Green
        'neutral': '#f1c40f',   # Yellow
        'negative': '#e74c3c'   # Red
    }
    
    # Extract colors in the order they appear in the data
    colors = [color_map.get(sentiment, '#3498db') for sentiment in sentiment_counts['Sentiment']]
    
    fig = px.pie(sentiment_counts, 
                values='Count', 
                names='Sentiment',
                color='Sentiment',
                color_discrete_map=color_map,
                title=title)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white")
    
    return fig

def create_sentiment_comparison(sentiment_data, movies=None, title="Sentiment Comparison"):
    """
    Create a grouped bar chart comparing sentiment across movies.
    
    Parameters:
    sentiment_data (DataFrame): DataFrame with sentiment distribution by movie
    movies (list): List of movies to include (None for all)
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    # Reshape data for plotting if needed
    if isinstance(sentiment_data, pd.DataFrame) and sentiment_data.index.name:
        # If data is in the format from get_sentiment_distribution with group_by
        plot_data = sentiment_data.copy().reset_index()
    else:
        # Convert to long format
        plot_data = pd.melt(sentiment_data.reset_index(), 
                          id_vars='index', 
                          value_vars=['positive', 'negative', 'neutral'],
                          var_name='Sentiment', 
                          value_name='Count')
        plot_data.rename(columns={'index': 'Movie'}, inplace=True)
    
    # Filter by selected movies if specified
    if movies:
        plot_data = plot_data[plot_data['Movie'].isin(movies)]
    
    # Create plot
    fig = px.bar(plot_data, 
                x='Movie', 
                y='Count', 
                color='Sentiment',
                barmode='group',
                color_discrete_map={
                    'positive': '#2ecc71',
                    'neutral': '#f1c40f',
                    'negative': '#e74c3c'
                },
                title=title)
    
    fig.update_layout(
        xaxis_title="Movie",
        yaxis_title="Count",
        legend_title="Sentiment",
        template="plotly_white"
    )
    
    return fig

def create_sentiment_heatmap(sentiment_data, title="Sentiment Heatmap"):
    """
    Create a heatmap showing sentiment distribution across movies.
    
    Parameters:
    sentiment_data (DataFrame): DataFrame with sentiment percentages by movie
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    # Create normalized version of the data (percentages)
    normalized_data = sentiment_data.div(sentiment_data.sum(axis=1), axis=0) * 100
    
    # Create two subplots: counts and percentages
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=["Sentiment Counts", "Sentiment Percentages (%)"],
                       vertical_spacing=0.15)
    
    # Add counts heatmap
    fig.add_trace(
        go.Heatmap(
            z=sentiment_data.values,
            x=sentiment_data.columns,
            y=sentiment_data.index,
            colorscale=[
                [0, '#e74c3c'],    # Red for negative
                [0.5, '#f1c40f'],  # Yellow for neutral
                [1, '#2ecc71']     # Green for positive
            ],
            showscale=True,
            text=sentiment_data.values.astype(int),
            texttemplate="%{text}",
            textfont={"size":10},
        ),
        row=1, col=1
    )
    
    # Add percentages heatmap
    fig.add_trace(
        go.Heatmap(
            z=normalized_data.values,
            x=normalized_data.columns,
            y=normalized_data.index,
            colorscale=[
                [0, '#e74c3c'],    # Red for negative
                [0.5, '#f1c40f'],  # Yellow for neutral
                [1, '#2ecc71']     # Green for positive
            ],
            showscale=True,
            text=normalized_data.values,
            texttemplate="%{text:.1f}%",
            textfont={"size":10},
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text=title,
        template="plotly_white"
    )
    
    return fig

def create_sentiment_timeline(time_sentiment_data, title="Sentiment Over Time"):
    """
    Create a line chart showing sentiment trends over time.
    
    Parameters:
    time_sentiment_data (DataFrame): DataFrame with sentiment by date
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    # Get percentage columns
    pct_columns = [col for col in time_sentiment_data.columns if 'pct' in col]
    
    # Create area chart of percentages
    fig = go.Figure()
    
    # Add trace for each sentiment
    colors = {
        'positive_pct': '#2ecc71',
        'neutral_pct': '#f1c40f',
        'negative_pct': '#e74c3c'
    }
    
    for col in pct_columns:
        base_col = col.replace('_pct', '')
        fig.add_trace(go.Scatter(
            x=time_sentiment_data.index,
            y=time_sentiment_data[col],
            mode='lines',
            name=base_col.capitalize(),
            line=dict(width=0.5, color=colors.get(col, '#3498db')),
            stackgroup='one',
            fill='tonexty'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Proportion",
        yaxis=dict(
            tickformat='.0%',
            range=[0, 1]
        ),
        legend_title="Sentiment",
        template="plotly_white"
    )
    
    return fig

def create_model_comparison_chart(comparison_data, title="Sentiment Model Comparison"):
    """
    Create a chart comparing sentiment predictions from different models.
    
    Parameters:
    comparison_data (DataFrame): DataFrame with sentiment predictions from multiple models
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    # Count agreement between models
    sentiment_columns = [col for col in comparison_data.columns if 'sentiment' in col]
    
    agreements = {}
    for i, col1 in enumerate(sentiment_columns):
        for col2 in sentiment_columns[i+1:]:
            agreement = (comparison_data[col1] == comparison_data[col2]).mean()
            name1 = col1.replace('_sentiment', '')
            name2 = col2.replace('_sentiment', '')
            agreements[f"{name1} vs {name2}"] = agreement
    
    # Create bar chart of agreement percentages
    agreement_df = pd.DataFrame(list(agreements.items()), columns=['Models', 'Agreement'])
    
    fig = px.bar(agreement_df, 
                x='Models', 
                y='Agreement',
                text_auto='.1%',
                color='Agreement',
                color_continuous_scale=['#e74c3c', '#f1c40f', '#2ecc71'],
                title=title)
    
    fig.update_layout(
        xaxis_title="Model Comparison",
        yaxis_title="Agreement",
        yaxis=dict(
            tickformat='.0%',
            range=[0, 1]
        ),
        template="plotly_white"
    )
    
    return fig