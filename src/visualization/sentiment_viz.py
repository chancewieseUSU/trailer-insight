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
    try:
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
    except Exception as e:
        print(f"Error creating sentiment bar chart: {e}")
        # Return empty figure as fallback
        return go.Figure().update_layout(title=f"Error: {str(e)}")

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
    try:
        # Check if data is valid
        if df is None or sentiment_column not in df.columns:
            raise ValueError(f"Missing '{sentiment_column}' column in DataFrame")
        
        sentiment_counts = df[sentiment_column].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Define colors for sentiment categories
        color_map = {
            'positive': '#2ecc71',  # Green
            'neutral': '#f1c40f',   # Yellow
            'negative': '#e74c3c'   # Red
        }
        
        # Extract colors in the order they appear in the data
        colors = [color_map.get(sentiment.lower(), '#3498db') for sentiment in sentiment_counts['Sentiment']]
        
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
    except Exception as e:
        print(f"Error creating sentiment pie chart: {e}")
        # Return empty figure as fallback
        return go.Figure().update_layout(title=f"Error: {str(e)}")

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
    try:
        # Reshape data for plotting if needed
        if isinstance(sentiment_data, pd.DataFrame):
            # Convert dataframe to prepare for plotting
            plot_data = sentiment_data.copy()
            
            # Check if this is a grouped DataFrame with MultiIndex
            if isinstance(plot_data.index, pd.MultiIndex):
                # Reset the index to columns
                plot_data = plot_data.reset_index()
                
                # Rename the index levels if they don't have explicit names
                if plot_data.columns[0] == 'level_0' or plot_data.columns[0] == 0:
                    plot_data.rename(columns={plot_data.columns[0]: 'Movie'}, inplace=True)
                if plot_data.columns[1] == 'level_1' or plot_data.columns[1] == 1:
                    plot_data.rename(columns={plot_data.columns[1]: 'Sentiment'}, inplace=True)
                
                # Convert from wide to long format
                plot_data = pd.melt(
                    plot_data, 
                    id_vars='Movie', 
                    value_vars=plot_data.columns[1:],
                    var_name='Sentiment', 
                    value_name='Count'
                )
            else:
                # Single index dataframe
                if 'movie' in plot_data.index.name.lower() if plot_data.index.name else False:
                    # Already in the right format, just reset index
                    plot_data = plot_data.reset_index()
                    plot_data.rename(columns={plot_data.columns[0]: 'Movie'}, inplace=True)
                    
                    # Melt to long format for plotting
                    plot_data = pd.melt(
                        plot_data, 
                        id_vars='Movie', 
                        value_vars=['positive', 'negative', 'neutral'] if all(col in plot_data.columns for col in ['positive', 'negative']) else plot_data.columns[1:],
                        var_name='Sentiment', 
                        value_name='Count'
                    )
                else:
                    # This is a standard dataframe, hopefully already formatted correctly
                    if 'Movie' not in plot_data.columns and 'movie' in plot_data.columns:
                        plot_data.rename(columns={'movie': 'Movie'}, inplace=True)
                    
                    if 'Sentiment' not in plot_data.columns and 'sentiment' in plot_data.columns:
                        plot_data.rename(columns={'sentiment': 'Sentiment'}, inplace=True)
                    
                    # If we still don't have Movie or Sentiment columns, try to infer them
                    if 'Movie' not in plot_data.columns:
                        # Use the first column as Movie
                        plot_data.rename(columns={plot_data.columns[0]: 'Movie'}, inplace=True)
                    
                    if 'Sentiment' not in plot_data.columns and len(plot_data.columns) > 1:
                        # Try to melt the remaining columns as sentiment categories
                        id_cols = ['Movie']
                        value_cols = [col for col in plot_data.columns if col not in id_cols]
                        
                        plot_data = pd.melt(
                            plot_data,
                            id_vars=id_cols,
                            value_vars=value_cols,
                            var_name='Sentiment',
                            value_name='Count'
                        )
        else:
            raise ValueError("sentiment_data must be a pandas DataFrame")
            
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
    except Exception as e:
        print(f"Error creating sentiment comparison chart: {e}")
        # Return empty figure as fallback
        return go.Figure().update_layout(title=f"Error: {str(e)}")

def create_sentiment_heatmap(sentiment_data, title="Sentiment Heatmap"):
    """
    Create a heatmap showing sentiment distribution across movies.
    
    Parameters:
    sentiment_data (DataFrame): DataFrame with sentiment percentages by movie
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    try:
        # Check if data is valid
        if sentiment_data is None or sentiment_data.empty:
            raise ValueError("Empty sentiment data")
            
        # Create normalized version of the data (percentages)
        normalized_data = sentiment_data.div(sentiment_data.sum(axis=1), axis=0) * 100
        
        # Create two separate heatmaps side by side
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=["Sentiment Counts", "Sentiment Percentages (%)"],
                           horizontal_spacing=0.15)
        
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
                colorbar=dict(title="Count", x=0.46)
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
                colorbar=dict(title="Percentage (%)")
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            width=1000,
            title_text=title,
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        print(f"Error creating sentiment heatmap: {e}")
        # Return empty figure as fallback
        return go.Figure().update_layout(title=f"Error: {str(e)}")

def create_sentiment_timeline(time_sentiment_data, title="Sentiment Over Time"):
    """
    Create a line chart showing sentiment trends over time.
    
    Parameters:
    time_sentiment_data (DataFrame): DataFrame with sentiment by date
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    try:
        # Check if data is valid
        if time_sentiment_data is None or time_sentiment_data.empty:
            raise ValueError("Empty time sentiment data")
            
        # Get percentage columns
        pct_columns = [col for col in time_sentiment_data.columns if 'pct' in col]
        
        if not pct_columns:
            # Calculate percentages if not present
            for col in time_sentiment_data.columns:
                if col not in ['date', 'Date']:
                    time_sentiment_data[f'{col}_pct'] = time_sentiment_data[col] / time_sentiment_data.sum(axis=1)
            
            # Update percentage columns list
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
    except Exception as e:
        print(f"Error creating sentiment timeline: {e}")
        # Return empty figure as fallback
        return go.Figure().update_layout(title=f"Error: {str(e)}")

def create_model_comparison_chart(comparison_data, title="Sentiment Model Comparison"):
    """
    Create a chart comparing sentiment predictions from different models.
    
    Parameters:
    comparison_data (DataFrame): DataFrame with sentiment predictions from multiple models
    title (str): Chart title
    
    Returns:
    Plotly figure object
    """
    try:
        # Check if data is valid
        if comparison_data is None or comparison_data.empty:
            raise ValueError("Empty comparison data")
            
        # Count agreement between models
        sentiment_columns = [col for col in comparison_data.columns if 'sentiment' in col]
        
        if len(sentiment_columns) < 2:
            raise ValueError("Need at least two sentiment columns for comparison")
        
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
    except Exception as e:
        print(f"Error creating model comparison chart: {e}")
        # Return empty figure as fallback
        return go.Figure().update_layout(title=f"Error: {str(e)}")

def analyze_sentiment_revenue_correlation(movies_df):
    """
    Perform advanced correlation analysis between sentiment metrics and box office performance.
    
    Parameters:
    movies_df (DataFrame): DataFrame with movie and sentiment data
    
    Returns:
    dict: Dictionary with correlation results and insights
    """
    # Ensure we have the necessary columns
    required_cols = ['revenue', 'avg_polarity', 'positive_pct', 'negative_pct']
    if not all(col in movies_df.columns for col in required_cols):
        return {"error": "Missing required columns for correlation analysis"}
    
    # Filter out movies with missing revenue data
    valid_movies = movies_df[movies_df['revenue'] > 0].copy()
    
    # Calculate correlations
    correlations = {}
    for metric in ['avg_polarity', 'positive_pct', 'negative_pct', 'comment_count']:
        if metric in valid_movies.columns:
            correlations[metric] = valid_movies[metric].corr(valid_movies['revenue'])
    
    # Identify top positive and negative correlators
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Group by genre if available
    genre_correlations = {}
    if 'genres' in valid_movies.columns:
        # Extract all unique genres
        all_genres = set()
        for genres_str in valid_movies['genres'].dropna():
            genres = [g.strip() for g in genres_str.split(',')]
            all_genres.update(genres)
        
        # Calculate correlation by genre
        for genre in all_genres:
            genre_movies = valid_movies[valid_movies['genres'].str.contains(genre, na=False)]
            if len(genre_movies) >= 5:  # Only consider genres with enough samples
                genre_corr = genre_movies['avg_polarity'].corr(genre_movies['revenue'])
                genre_correlations[genre] = genre_corr
    
    # Generate insights
    insights = []
    
    # Overall correlation insight
    top_metric, top_corr = sorted_corrs[0] if sorted_corrs else (None, 0)
    if abs(top_corr) > 0.3:
        direction = "positive" if top_corr > 0 else "negative"
        insights.append(f"There is a {direction} correlation ({top_corr:.2f}) between {top_metric} and box office revenue")
    else:
        insights.append("No strong correlations found between sentiment metrics and revenue")
    
    # Genre-specific insights
    if genre_correlations:
        # Find genre with strongest correlation
        top_genre, top_genre_corr = max(genre_correlations.items(), key=lambda x: abs(x[1]), default=(None, 0))
        if top_genre and abs(top_genre_corr) > 0.4:
            direction = "positive" if top_genre_corr > 0 else "negative"
            insights.append(f"{top_genre} movies show a strong {direction} correlation ({top_genre_corr:.2f}) between sentiment and revenue")
    
    return {
        "correlations": correlations,
        "genre_correlations": genre_correlations,
        "insights": insights,
        "data": valid_movies
    }

def create_enhanced_sentiment_visualizations(correlation_results):
    """
    Create enhanced visualizations for sentiment-revenue relationship.
    
    Parameters:
    correlation_results (dict): Results from analyze_sentiment_revenue_correlation
    
    Returns:
    dict: Dictionary with plotly figures
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    from scipy import stats
    
    figures = {}
    
    # Get the data
    if "error" in correlation_results or "data" not in correlation_results:
        return {"error": "Invalid correlation results"}
    
    data = correlation_results["data"]
    
    # 1. Create advanced scatter plot with trend line
    if 'avg_polarity' in data.columns and 'revenue' in data.columns:
        # Create scatter plot
        fig = px.scatter(
            data, 
            x='avg_polarity', 
            y='revenue',
            size='comment_count' if 'comment_count' in data.columns else None,
            color='positive_pct' if 'positive_pct' in data.columns else None,
            hover_data=['title', 'comment_count', 'positive_pct', 'negative_pct'],
            labels={
                'avg_polarity': 'Sentiment Polarity',
                'revenue': 'Box Office Revenue ($)',
                'positive_pct': 'Positive Comments (%)'
            },
            title="Sentiment Polarity vs. Box Office Revenue"
        )
        
        # Add regression line
        x = data['avg_polarity']
        y = data['revenue']
        
        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Add regression line to plot
        x_range = np.linspace(x.min(), x.max(), 100)
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
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        figures["scatter_plot"] = fig
    
    # 2. Create correlation heatmap
    correlations = correlation_results.get("correlations", {})
    if correlations:
        corr_items = list(correlations.items())
        metrics = [item[0] for item in corr_items]
        values = [item[1] for item in corr_items]
        
        fig = go.Figure(data=go.Bar(
            x=metrics,
            y=values,
            marker_color=[
                'green' if v > 0.1 else ('red' if v < -0.1 else 'gray') 
                for v in values
            ]
        ))
        
        fig.update_layout(
            title="Correlation with Box Office Revenue",
            xaxis_title="Metric",
            yaxis_title="Correlation Coefficient",
            template="plotly_white"
        )
        
        figures["correlation_bar"] = fig
    
    # 3. Create genre correlation visualization
    genre_correlations = correlation_results.get("genre_correlations", {})
    if genre_correlations:
        genres = list(genre_correlations.keys())
        corr_values = list(genre_correlations.values())
        
        # Sort by absolute correlation
        sorted_indices = sorted(range(len(corr_values)), key=lambda i: abs(corr_values[i]), reverse=True)
        sorted_genres = [genres[i] for i in sorted_indices[:10]]  # Top 10 genres
        sorted_values = [corr_values[i] for i in sorted_indices[:10]]
        
        fig = go.Figure(data=go.Bar(
            x=sorted_genres,
            y=sorted_values,
            marker_color=[
                'green' if v > 0.1 else ('red' if v < -0.1 else 'gray') 
                for v in sorted_values
            ]
        ))
        
        fig.update_layout(
            title="Sentiment-Revenue Correlation by Genre",
            xaxis_title="Genre",
            yaxis_title="Correlation Coefficient",
            template="plotly_white"
        )
        
        figures["genre_correlation"] = fig
    
    return figures