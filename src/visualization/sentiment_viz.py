# src/visualization/sentiment_viz.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    # Check if data is valid
    if df is None or sentiment_column not in df.columns:
        return go.Figure().update_layout(title="No data available")
    
    sentiment_counts = df[sentiment_column].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Define colors for sentiment categories
    color_map = {
        'positive': '#2ecc71',  # Green
        'neutral': '#f1c40f',   # Yellow
        'negative': '#e74c3c'   # Red
    }
    
    # Create custom color map based on actual sentiment values
    custom_color_map = {sentiment: color_map.get(sentiment.lower(), '#3498db') 
                       for sentiment in sentiment_counts['Sentiment']}
    
    fig = px.pie(sentiment_counts, 
                values='Count', 
                names='Sentiment',
                color='Sentiment',
                color_discrete_map=custom_color_map,
                title=title)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white")
    
    return fig

def create_sentiment_comparison(sentiment_data, title="Sentiment Comparison"):
    """
    Create a grouped bar chart comparing sentiment across movies.
    
    Parameters:
    sentiment_data (DataFrame): DataFrame with sentiment distribution by movie
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    # Check if data is valid
    if sentiment_data is None or sentiment_data.empty:
        return go.Figure().update_layout(title="No data available")
        
    # Reset index to get movie names as a column
    plot_data = sentiment_data.reset_index()
    plot_data.columns = ['Movie' if i == 0 else col for i, col in enumerate(plot_data.columns)]
    
    # Melt the dataframe for plotting
    plot_data = pd.melt(
        plot_data, 
        id_vars='Movie', 
        value_vars=[col for col in plot_data.columns if col != 'Movie'],
        var_name='Sentiment', 
        value_name='Count'
    )
    
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
    # Check if data is valid
    if sentiment_data is None or sentiment_data.empty:
        return go.Figure().update_layout(title="No data available")
        
    # Create normalized version of the data (percentages)
    normalized_data = sentiment_data.div(sentiment_data.sum(axis=1), axis=0) * 100
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=normalized_data.values,
        x=normalized_data.columns,
        y=normalized_data.index,
        colorscale=[
            [0, '#e74c3c'],    # Red for negative
            [0.5, '#f1c40f'],  # Yellow for neutral
            [1, '#2ecc71']     # Green for positive
        ],
        text=normalized_data.values,
        texttemplate="%{text:.1f}%",
        textfont={"size":10}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Sentiment",
        yaxis_title="Movie",
        template="plotly_white"
    )
    
    return fig

def analyze_sentiment_revenue_correlation(movies_df):
    """
    Analyze correlation between sentiment metrics and box office revenue.
    
    Parameters:
    movies_df (DataFrame): DataFrame with movie and sentiment data
    
    Returns:
    dict: Dictionary with correlation results and scatter plot
    """
    # Check for required columns
    required_cols = ['revenue', 'avg_polarity', 'positive_pct', 'negative_pct']
    missing_cols = [col for col in required_cols if col not in movies_df.columns]
    
    if missing_cols:
        return {
            "error": f"Missing required columns: {', '.join(missing_cols)}",
            "scatter_plot": go.Figure().update_layout(title="Missing data for correlation analysis")
        }
    
    # Filter out movies with missing revenue data
    valid_movies = movies_df[movies_df['revenue'] > 0].copy()
    
    if len(valid_movies) < 3:
        return {
            "error": "Not enough movies with revenue data for correlation analysis",
            "scatter_plot": go.Figure().update_layout(title="Not enough data for correlation analysis")
        }
    
    # Calculate correlations
    correlations = {}
    for metric in ['avg_polarity', 'positive_pct', 'negative_pct']:
        correlations[metric] = valid_movies[metric].corr(valid_movies['revenue'])
    
    # Create scatter plot
    from scipy import stats
    import numpy as np
    
    x = valid_movies['avg_polarity']
    y = valid_movies['revenue']
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Create scatter plot with trend line
    fig = px.scatter(
        valid_movies, 
        x='avg_polarity', 
        y='revenue',
        hover_data=['title'],
        labels={
            'avg_polarity': 'Comment Sentiment (Polarity)',
            'revenue': 'Box Office Revenue ($)',
        },
        title="Sentiment vs. Box Office Revenue"
    )
    
    # Add regression line
    x_range = np.linspace(min(x), max(x), 100)
    y_range = slope * x_range + intercept
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name=f'Trend (R²={r_value**2:.2f})',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Add correlation annotation
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Correlation: {r_value:.2f} (R²: {r_value**2:.2f})",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    return {
        "correlations": correlations,
        "scatter_plot": fig,
        "r_squared": r_value**2,
        "p_value": p_value
    }