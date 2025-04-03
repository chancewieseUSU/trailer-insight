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
            # Check if this is a grouped DataFrame with MultiIndex
            if isinstance(sentiment_data.index, pd.MultiIndex):
                plot_data = sentiment_data.reset_index()
            elif sentiment_data.index.name:
                # If data is in the format from get_sentiment_distribution with group_by
                plot_data = sentiment_data.copy().reset_index()
                
                # Check if the first column needs renaming for consistency
                if plot_data.columns[0].lower() == 'movie':
                    plot_data.rename(columns={plot_data.columns[0]: 'Movie'}, inplace=True)
                elif plot_data.columns[0] not in ['Movie', 'movie']:
                    plot_data.rename(columns={plot_data.columns[0]: 'Movie'}, inplace=True)
            else:
                # Convert to long format
                plot_data = pd.melt(sentiment_data.reset_index(), 
                                id_vars='index', 
                                value_vars=['positive', 'negative', 'neutral'],
                                var_name='Sentiment', 
                                value_name='Count')
                plot_data.rename(columns={'index': 'Movie'}, inplace=True)
        else:
            raise ValueError("sentiment_data must be a pandas DataFrame")
        
        # Ensure 'Movie' column exists (capital M for consistency)
        if 'movie' in plot_data.columns and 'Movie' not in plot_data.columns:
            plot_data.rename(columns={'movie': 'Movie'}, inplace=True)
            
        # If we still don't have a 'Movie' column, try to find it
        if 'Movie' not in plot_data.columns:
            # Look for any column that might contain movie names
            potential_cols = [col for col in plot_data.columns if plot_data[col].dtype == 'object']
            if potential_cols:
                plot_data.rename(columns={potential_cols[0]: 'Movie'}, inplace=True)
            else:
                raise ValueError("Could not identify a movie column in the data")
                
        # Ensure 'Sentiment' column exists
        if 'Sentiment' not in plot_data.columns:
            # Try to identify sentiment and count columns
            numeric_cols = [col for col in plot_data.columns if col not in ['Movie']]
            if len(numeric_cols) > 0:
                # Create a dummy Sentiment column using column names
                plot_data = pd.melt(plot_data, 
                                   id_vars='Movie',
                                   value_vars=numeric_cols,
                                   var_name='Sentiment',
                                   value_name='Count')
        
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