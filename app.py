# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import config
import time
from datetime import datetime
import os
from src.api.youtube import YouTubeClient
from src.api.movie_db import MovieDBClient
from src.preprocessing.clean_text import TextCleaner, clean_text_for_sentiment
from src.models.sentiment import SentimentAnalyzer, get_sentiment_distribution, get_top_comments
from src.visualization.sentiment_viz import (
    create_sentiment_bar_chart,
    create_sentiment_distribution_pie,
    create_sentiment_comparison,
    create_sentiment_heatmap,
    create_sentiment_timeline,
    create_model_comparison_chart
)

# Import the data collection functions
from data_collection_functions import collect_movie_dataset, process_sentiment_stats

# Initialize API clients
youtube_client = YouTubeClient()
movie_db_client = MovieDBClient()

# Set page configuration
st.set_page_config(
    page_title="TrailerInsight",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state for data persistence
if 'comments_df' not in st.session_state:
    # Try to load data if it exists
    data_path = os.path.join('data', 'processed', 'comments_processed.csv')
    if os.path.exists(data_path):
        st.session_state.comments_df = pd.read_csv(data_path)
    else:
        st.session_state.comments_df = None

if 'movies_df' not in st.session_state:
    # Try to load data if it exists
    data_path = os.path.join('data', 'processed', 'movies.csv')
    if os.path.exists(data_path):
        st.session_state.movies_df = pd.read_csv(data_path)
    else:
        st.session_state.movies_df = None

# Title and introduction
st.title("TrailerInsight")
st.markdown("### YouTube Trailer Comment Analysis for Box Office Prediction")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select a page",
        ["Data Collection", "Sentiment Analysis", "Comment Clusters", "Summaries", "Box Office Insights"]
    )

# Main content based on selected page
if page == "Data Collection":
    st.header("Data Collection")
    
    st.info("This page allows you to collect a dataset of movies with trailer comments and box office data.")
    
    # Main collection form
    with st.form("collection_form"):
        st.subheader("Collection Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_movies = st.number_input("Minimum Number of Movies", min_value=10, max_value=100, value=50)
            min_comments = st.number_input("Minimum Comments per Trailer", min_value=50, max_value=500, value=100)
        
        with col2:
            require_box_office = st.checkbox("Require Box Office Data", value=True)
            save_directory = st.text_input("Save Directory", value="data/processed")
        
        submit_button = st.form_submit_button("Start Collection")

    # Process form submission
    if submit_button:
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Start collection
        status_text.text("Starting data collection process...")
        
        try:
            # Run collection with progress updates
            with st.spinner("Collecting data..."):
                # Run collection function
                comments_df, movies_df = collect_movie_dataset(
                    min_movies=min_movies,
                    min_comments=min_comments,
                    include_box_office=require_box_office,
                    save_path=save_directory
                )
                
                # Update progress bar as data is collected
                for i in range(100):
                    # This simulates the progress
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                # Store data in session state
                if comments_df is not None:
                    st.session_state.comments_df = comments_df
                    
                if movies_df is not None:
                    st.session_state.movies_df = movies_df
                    
                    # Process sentiment stats
                    movies_with_stats = process_sentiment_stats(comments_df, movies_df)
                    st.session_state.movies_with_stats = movies_with_stats
                
                # Show success message
                if comments_df is not None and movies_df is not None:
                    st.success(f"Successfully collected data for {len(movies_df)} movies with {len(comments_df)} total comments!")
                else:
                    st.error("Data collection failed or no movies met the criteria.")
        
        except Exception as e:
            st.error(f"An error occurred during data collection: {str(e)}")
            
    # Show data status if available
    if 'comments_df' in st.session_state and st.session_state.comments_df is not None:
        st.subheader("Current Dataset")
        
        # Display basic stats
        movies_count = len(st.session_state.movies_df) if st.session_state.movies_df is not None else 0
        comments_count = len(st.session_state.comments_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Movies", movies_count)
        
        with col2:
            st.metric("Comments", comments_count)
        
        # Show data download buttons
        st.subheader("Download Current Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            comments_csv = st.session_state.comments_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Comments Data (CSV)",
                comments_csv,
                "trailer_comments.csv",
                "text/csv",
                key="download-comments"
            )
        
        with col2:
            if st.session_state.movies_df is not None:
                if 'movies_with_stats' in st.session_state:
                    movies_csv = st.session_state.movies_with_stats.to_csv(index=False).encode('utf-8')
                else:
                    movies_csv = st.session_state.movies_df.to_csv(index=False).encode('utf-8')
                    
                st.download_button(
                    "Download Movies Data (CSV)",
                    movies_csv,
                    "movie_data.csv",
                    "text/csv",
                    key="download-movies"
                )

elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    
    # Check if data is available
    if st.session_state.comments_df is None:
        st.warning("No data available. Please collect data in the Data Collection page.")
    else:
        # Sidebar controls for sentiment analysis
        st.sidebar.subheader("Sentiment Analysis Options")
        
        # Select movie(s) to analyze
        available_movies = st.session_state.comments_df['movie'].unique()
        selected_movies = st.sidebar.multiselect(
            "Select Movies",
            options=available_movies,
            default=available_movies[0] if len(available_movies) > 0 else None
        )
        
        # Select sentiment analysis method
        sentiment_method = st.sidebar.radio(
            "Sentiment Analysis Method",
            options=["TextBlob", "Transformer", "Compare Methods"],
            index=0
        )
        
        # Filter data by selected movies
        if selected_movies:
            filtered_df = st.session_state.comments_df[st.session_state.comments_df['movie'].isin(selected_movies)]
        else:
            filtered_df = st.session_state.comments_df.copy()
        
        # Analyze sentiment based on selected method
        analyzer = SentimentAnalyzer(
            method='textblob' if sentiment_method == "TextBlob" else 'transformer'
        )
        
        # Container for results
        results_container = st.container()
        
        with results_container:
            if sentiment_method == "Compare Methods":
                st.subheader("Sentiment Method Comparison")
                
                # Compare different sentiment methods
                with st.spinner("Analyzing sentiment with multiple methods..."):
                    # Sample data for comparison (limit to 100 comments for performance)
                    sample_df = filtered_df.head(100)
                    comparison_results = analyzer.compare_sentiment_methods(sample_df['clean_text'])
                    
                    # Show model agreement chart
                    agreement_chart = create_model_comparison_chart(comparison_results)
                    st.plotly_chart(agreement_chart, use_container_width=True)
                    
                    # Show sample of disagreements
                    disagreements = comparison_results[
                        comparison_results['textblob_sentiment'] != comparison_results['transformer_sentiment']
                    ].head(10)
                    
                    if not disagreements.empty:
                        st.subheader("Sample Disagreements Between Models")
                        for i, row in disagreements.iterrows():
                            st.markdown(f"**Comment:** {row['text']}")
                            st.markdown(f"- TextBlob: {row['textblob_sentiment']} (polarity: {row['textblob_polarity']:.2f})")
                            st.markdown(f"- Transformer: {row['transformer_sentiment']} (rating: {row['transformer_rating']})")
                            st.markdown("---")
            else:
                # Analyze sentiment with selected method
                with st.spinner(f"Analyzing sentiment with {sentiment_method}..."):
                    if 'sentiment' not in filtered_df.columns or sentiment_method == "Transformer":
                        sentiment_results = analyzer.analyze_sentiment(filtered_df['clean_text'])
                        filtered_df['sentiment'] = sentiment_results['sentiment']
                        
                        # Add additional columns based on method
                        if sentiment_method == "TextBlob":
                            filtered_df['polarity'] = sentiment_results['polarity']
                            filtered_df['subjectivity'] = sentiment_results['subjectivity']
                        elif sentiment_method == "Transformer":
                            filtered_df['rating'] = sentiment_results['rating']
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(["Overall Sentiment", "Movie Comparison", "Top Comments"])
                    
                    with tab1:
                        st.subheader("Overall Sentiment Distribution")
                        
                        # Create distribution pie chart
                        if len(selected_movies) == 1:
                            title = f"Sentiment Distribution for {selected_movies[0]}"
                        else:
                            title = "Overall Sentiment Distribution"
                        
                        pie_chart = create_sentiment_distribution_pie(filtered_df, title=title)
                        st.plotly_chart(pie_chart, use_container_width=True)
                        
                        # Additional statistics
                        st.subheader("Sentiment Statistics")
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        
                        with stats_col1:
                            st.metric("Total Comments", len(filtered_df))
                        
                        with stats_col2:
                            positive_pct = (filtered_df['sentiment'] == 'positive').mean() * 100
                            st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
                        
                        with stats_col3:
                            negative_pct = (filtered_df['sentiment'] == 'negative').mean() * 100
                            st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
                    
                    with tab2:
                        if len(selected_movies) > 1:
                            st.subheader("Movie Sentiment Comparison")
                            
                            # Get sentiment distribution by movie
                            movie_sentiment = get_sentiment_distribution(filtered_df, group_by='movie')
                            
                            # Create comparison chart
                            comparison_chart = create_sentiment_comparison(movie_sentiment)
                            st.plotly_chart(comparison_chart, use_container_width=True)
                            
                            # Create heatmap
                            heatmap = create_sentiment_heatmap(movie_sentiment)
                            st.plotly_chart(heatmap, use_container_width=True)
                        else:
                            st.info("Select multiple movies in the sidebar to compare sentiment across movies.")
                    
                    with tab3:
                        st.subheader("Top Comments by Sentiment")
                        
                        # Create columns for positive and negative comments
                        pos_col, neg_col = st.columns(2)
                        
                        with pos_col:
                            st.subheader("Top Positive Comments")
                            top_positive = get_top_comments(filtered_df, 'positive', n=5)
                            for i, row in top_positive.iterrows():
                                st.markdown(f"**Comment:** {row['text']}")
                                if 'polarity' in filtered_df.columns:
                                    st.markdown(f"*Polarity: {filtered_df.loc[i, 'polarity']:.2f}*")
                                elif 'rating' in filtered_df.columns:
                                    st.markdown(f"*Rating: {filtered_df.loc[i, 'rating']} stars*")
                                st.markdown("---")
                        
                        with neg_col:
                            st.subheader("Top Negative Comments")
                            top_negative = get_top_comments(filtered_df, 'negative', n=5)
                            for i, row in top_negative.iterrows():
                                st.markdown(f"**Comment:** {row['text']}")
                                if 'polarity' in filtered_df.columns:
                                    st.markdown(f"*Polarity: {filtered_df.loc[i, 'polarity']:.2f}*")
                                elif 'rating' in filtered_df.columns:
                                    st.markdown(f"*Rating: {filtered_df.loc[i, 'rating']} stars*")
                                st.markdown("---")
                        
                        # Show neutral comments if available
                        if 'neutral' in filtered_df['sentiment'].unique():
                            st.subheader("Top Neutral Comments")
                            top_neutral = get_top_comments(filtered_df, 'neutral', n=5)
                            for i, row in top_neutral.iterrows():
                                st.markdown(f"**Comment:** {row['text']}")
                                if 'polarity' in filtered_df.columns:
                                    st.markdown(f"*Polarity: {filtered_df.loc[i, 'polarity']:.2f}*")
                                elif 'rating' in filtered_df.columns:
                                    st.markdown(f"*Rating: {filtered_df.loc[i, 'rating']} stars*")
                                st.markdown("---")

elif page == "Comment Clusters":
    st.header("Comment Clusters")
    st.info("This section will show thematic clusters of comments. Implementation coming in the next update.")
    
    # Example clusters with dummy data
    st.subheader("Cluster Distribution")
    cluster_data = pd.DataFrame({
        'Cluster': ['Actor Mentions', 'Story Elements', 'Visual Effects', 'Comparisons', 'Release Anticipation', 'Music'],
        'Count': [120, 85, 65, 50, 40, 30]
    })
    
    fig = px.pie(cluster_data, values='Count', names='Cluster', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Summaries":
    st.header("Comment Summarization")
    st.info("This section will display auto-generated summaries of key themes. Implementation coming in the next update.")
    
    # Example summaries
    st.subheader("Key Themes by Sentiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Positive Sentiment")
        st.markdown("""
        - **Visual Effects**: Audiences are impressed by the visual effects, particularly the realistic CGI.
        - **Acting Performance**: Lead actor's performance is receiving significant praise.
        - **Music Score**: Original soundtrack is highlighted as enhancing emotional scenes.
        """)
    
    with col2:
        st.markdown("#### Negative Sentiment")
        st.markdown("""
        - **Plot Concerns**: Some viewers express worries about plot similarities to previous films.
        - **Pacing Issues**: Several comments mention concerns about the trailer showing too many action scenes.
        - **Character Development**: Questions about depth of supporting characters.
        """)

elif page == "Box Office Insights":
    st.header("Box Office Prediction Insights")
    
    # Check if data is available
    if st.session_state.comments_df is None or st.session_state.movies_df is None:
        st.warning("No data available. Please collect data in the Data Collection page.")
    else:
        st.info("This section connects comment sentiment to box office performance.")
        
        # Get sentiment stats by movie if not already calculated
        if 'movies_with_stats' not in st.session_state:
            st.session_state.movies_with_stats = process_sentiment_stats(
                st.session_state.comments_df, 
                st.session_state.movies_df
            )
        
        movies_df = st.session_state.movies_with_stats
        
        # Filter to only movies with both sentiment stats and box office data
        if 'revenue' in movies_df.columns:
            has_box_office = movies_df['revenue'].notna() & (movies_df['revenue'] > 0)
            has_sentiment = movies_df['avg_polarity'].notna()
            
            valid_movies = movies_df[has_box_office & has_sentiment]
            
            if len(valid_movies) > 0:
                st.subheader("Sentiment vs. Box Office Revenue")
                
                # Create scatter plot
                fig = px.scatter(
                    valid_movies, 
                    x='avg_polarity', 
                    y='revenue',
                    text='title',
                    size='comment_count',
                    color='positive_pct',
                    hover_data=['positive_pct', 'negative_pct', 'neutral_pct'],
                    title="Sentiment Polarity vs. Box Office Revenue",
                    labels={
                        'avg_polarity': 'Average Sentiment Polarity',
                        'revenue': 'Box Office Revenue ($)',
                        'positive_pct': 'Positive Comment %'
                    },
                    color_continuous_scale='Viridis'
                )
                
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation analysis
                corr = valid_movies['avg_polarity'].corr(valid_movies['revenue'])
                corr_pos = valid_movies['positive_pct'].corr(valid_movies['revenue'])
                corr_neg = valid_movies['negative_pct'].corr(valid_movies['revenue'])
                
                st.subheader("Correlation Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg. Polarity/Revenue", f"{corr:.2f}")
                
                with col2:
                    st.metric("Positive %/Revenue", f"{corr_pos:.2f}")
                
                with col3:
                    st.metric("Negative %/Revenue", f"{corr_neg:.2f}")
                
                # Show additional visualizations
                st.subheader("Additional Insights")
                
                # Positive sentiment % distribution
                fig = px.histogram(
                    valid_movies,
                    x='positive_pct',
                    nbins=10,
                    title="Distribution of Positive Comment Percentage"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Box office by genre
                if 'genres' in valid_movies.columns:
                    # Create genre analysis
                    st.subheader("Genre Analysis")
                    st.info("Coming soon: Detailed genre analysis with sentiment correlation.")
                
            else:
                st.warning("Not enough movies with both box office data and sentiment statistics. Please collect more data.")
        else:
            st.warning("Box office data not available in the dataset. Please make sure to collect movies with box office data.")