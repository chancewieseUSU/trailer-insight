# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
import time
from datetime import datetime
import os
from src.api.youtube import YouTubeClient
from src.api.movie_db import MovieDBClient
from src.preprocessing.clean_text import TextCleaner, clean_text_for_sentiment, clean_text_for_clustering
from src.models.sentiment import SentimentAnalyzer, get_sentiment_distribution, get_top_comments
from src.models.clustering import CommentClusterer
from src.models.summarization import TextRankSummarizer
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
    if 'comments_df' not in st.session_state or st.session_state.comments_df is None:
        st.warning("No data available. Please collect data in the Data Collection page.")
    else:
        # Sidebar controls for sentiment analysis
        st.sidebar.subheader("Sentiment Analysis Options")
        
        # Select movie(s) to analyze
        available_movies = sorted(st.session_state.comments_df['movie'].unique())
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
        try:
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
                        
                        # Ensure clean text is available
                        if 'clean_text' not in sample_df.columns:
                            sample_df['clean_text'] = sample_df['text'].apply(clean_text_for_sentiment)
                        
                        comparison_results = analyzer.compare_sentiment_methods(sample_df['clean_text'])
                        
                        # Show model agreement chart
                        if 'transformer_sentiment' in comparison_results.columns:
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
                                    if 'transformer_rating' in row:
                                        st.markdown(f"- Transformer: {row['transformer_sentiment']} (rating: {row['transformer_rating']})")
                                    st.markdown("---")
                        else:
                            st.error("Transformer model comparison failed. Using TextBlob results only.")
                else:
                    # Analyze sentiment with selected method
                    with st.spinner(f"Analyzing sentiment with {sentiment_method}..."):
                        # Ensure clean text is available
                        if 'clean_text' not in filtered_df.columns:
                            filtered_df['clean_text'] = filtered_df['text'].apply(clean_text_for_sentiment)
                        
                        sentiment_results = analyzer.analyze_sentiment(filtered_df['clean_text'])
                        
                        # Add sentiment results to the dataframe
                        if 'sentiment' in sentiment_results.columns:
                            filtered_df['sentiment'] = sentiment_results['sentiment']
                        
                        # Add additional columns based on method
                        if sentiment_method == "TextBlob" and 'polarity' in sentiment_results.columns:
                            filtered_df['polarity'] = sentiment_results['polarity']
                            if 'subjectivity' in sentiment_results.columns:
                                filtered_df['subjectivity'] = sentiment_results['subjectivity']
                        elif sentiment_method == "Transformer" and 'rating' in sentiment_results.columns:
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
                            
                            try:
                                pie_chart = create_sentiment_distribution_pie(filtered_df, title=title)
                                st.plotly_chart(pie_chart, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating pie chart: {str(e)}")
                                st.dataframe(filtered_df['sentiment'].value_counts())
                            
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
                                
                                try:
                                    # Get sentiment distribution by movie
                                    movie_sentiment = get_sentiment_distribution(filtered_df, group_by='movie')
                                    
                                    # Create comparison chart
                                    comparison_chart = create_sentiment_comparison(movie_sentiment)
                                    st.plotly_chart(comparison_chart, use_container_width=True)
                                    
                                    # Create heatmap
                                    heatmap = create_sentiment_heatmap(movie_sentiment)
                                    st.plotly_chart(heatmap, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating comparison charts: {str(e)}")
                                    st.write("Sentiment distribution by movie:")
                                    st.dataframe(filtered_df.groupby('movie')['sentiment'].value_counts())
                            else:
                                st.info("Select multiple movies in the sidebar to compare sentiment across movies.")
                        
                        with tab3:
                            st.subheader("Top Comments by Sentiment")
                            
                            # Create columns for positive and negative comments
                            pos_col, neg_col = st.columns(2)
                            
                            try:
                                with pos_col:
                                    st.subheader("Top Positive Comments")
                                    top_positive = get_top_comments(filtered_df, 'positive', n=5)
                                    for i, row in top_positive.iterrows():
                                        st.markdown(f"**Comment:** {row['text']}")
                                        if 'polarity' in filtered_df.columns:
                                            pol_value = filtered_df.loc[i, 'polarity'] if i in filtered_df.index else 'N/A'
                                            st.markdown(f"*Polarity: {pol_value if pol_value != 'N/A' else pol_value:.2f}*")
                                        elif 'rating' in filtered_df.columns:
                                            rating_value = filtered_df.loc[i, 'rating'] if i in filtered_df.index else 'N/A'
                                            st.markdown(f"*Rating: {rating_value} stars*")
                                        st.markdown("---")
                                
                                with neg_col:
                                    st.subheader("Top Negative Comments")
                                    top_negative = get_top_comments(filtered_df, 'negative', n=5)
                                    for i, row in top_negative.iterrows():
                                        st.markdown(f"**Comment:** {row['text']}")
                                        if 'polarity' in filtered_df.columns:
                                            pol_value = filtered_df.loc[i, 'polarity'] if i in filtered_df.index else 'N/A'
                                            st.markdown(f"*Polarity: {pol_value if pol_value != 'N/A' else pol_value:.2f}*")
                                        elif 'rating' in filtered_df.columns:
                                            rating_value = filtered_df.loc[i, 'rating'] if i in filtered_df.index else 'N/A'
                                            st.markdown(f"*Rating: {rating_value} stars*")
                                        st.markdown("---")
                                
                                # Show neutral comments if available
                                if 'neutral' in filtered_df['sentiment'].unique():
                                    st.subheader("Top Neutral Comments")
                                    top_neutral = get_top_comments(filtered_df, 'neutral', n=5)
                                    for i, row in top_neutral.iterrows():
                                        st.markdown(f"**Comment:** {row['text']}")
                                        if 'polarity' in filtered_df.columns:
                                            pol_value = filtered_df.loc[i, 'polarity'] if i in filtered_df.index else 'N/A'
                                            st.markdown(f"*Polarity: {pol_value if pol_value != 'N/A' else pol_value:.2f}*")
                                        elif 'rating' in filtered_df.columns:
                                            rating_value = filtered_df.loc[i, 'rating'] if i in filtered_df.index else 'N/A'
                                            st.markdown(f"*Rating: {rating_value} stars*")
                                        st.markdown("---")
                            except Exception as e:
                                st.error(f"Error displaying top comments: {str(e)}")
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            st.info("Try selecting a different sentiment method or check your data.")

elif page == "Comment Clusters":
    st.header("Comment Clusters")
    
    # Check if data is available
    if 'comments_df' not in st.session_state or st.session_state.comments_df is None:
        st.warning("No data available. Please collect data in the Data Collection page.")
    else:
        # Load comments data
        comments_df = st.session_state.comments_df
        
        # Sidebar controls for clustering
        st.sidebar.subheader("Clustering Options")
        
        # Select movie(s) to analyze
        available_movies = sorted(comments_df['movie'].unique())
        selected_movies = st.sidebar.multiselect(
            "Select Movies",
            options=available_movies,
            default=available_movies[0] if len(available_movies) > 0 else None
        )
        
        # Number of clusters
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)
        
        # Filter data by selected movies
        if selected_movies:
            filtered_df = comments_df[comments_df['movie'].isin(selected_movies)]
        else:
            filtered_df = comments_df.copy()
        
        # Clean text for clustering
        # Add progress indicator
        with st.spinner("Clustering comments..."):
            try:
                # Clean and prepare text
                if 'clean_text' not in filtered_df.columns:
                    filtered_df['clean_text'] = filtered_df['text'].apply(clean_text_for_sentiment)
                
                # Create cluster-specific cleaned text
                filtered_df['clean_text_cluster'] = filtered_df['text'].apply(clean_text_for_clustering)
                
                # Initialize and fit clusterer
                clusterer = CommentClusterer(n_clusters=n_clusters)
                clusterer.fit(filtered_df['clean_text_cluster'].fillna(''))
                
                # Add cluster labels to dataframe
                filtered_df['cluster'] = clusterer.predict(filtered_df['clean_text_cluster'].fillna(''))
                
                # Get top terms for each cluster
                top_terms = clusterer.get_top_terms_per_cluster()
                
                # Calculate sentiment distribution by cluster
                cluster_sentiment = filtered_df.groupby('cluster')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Cluster Overview", "Top Comments by Cluster", "Cluster Distribution"])
                
                with tab1:
                    st.subheader("Cluster Themes")
                    
                    # Display top terms and sentiment for each cluster
                    for cluster_id, terms in top_terms.items():
                        # Create columns for terms and sentiment
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Cluster {cluster_id}:** {', '.join(terms)}")
                        
                        with col2:
                            # Calculate counts and percentages
                            cluster_count = (filtered_df['cluster'] == cluster_id).sum()
                            cluster_pct = cluster_count / len(filtered_df) * 100
                            
                            # Get sentiment breakdown
                            if cluster_id in cluster_sentiment.index:
                                pos_pct = cluster_sentiment.loc[cluster_id, 'positive'] * 100 if 'positive' in cluster_sentiment.columns else 0
                                neg_pct = cluster_sentiment.loc[cluster_id, 'negative'] * 100 if 'negative' in cluster_sentiment.columns else 0
                                neu_pct = 100 - pos_pct - neg_pct
                                
                                sentiment_text = f"🟢 {pos_pct:.1f}% | 🟡 {neu_pct:.1f}% | 🔴 {neg_pct:.1f}%"
                            else:
                                sentiment_text = "No sentiment data"
                            
                            st.markdown(f"**Count:** {cluster_count} ({cluster_pct:.1f}%)<br>**Sentiment:** {sentiment_text}", unsafe_allow_html=True)
                        
                        st.markdown("---")
                
                with tab2:
                    st.subheader("Top Comments by Cluster")
                    
                    # Select cluster to view
                    selected_cluster = st.selectbox("Select Cluster", 
                                                   options=range(n_clusters),
                                                   format_func=lambda x: f"Cluster {x}: {', '.join(top_terms[x][:3])}")
                    
                    # Get comments for selected cluster
                    cluster_comments = filtered_df[filtered_df['cluster'] == selected_cluster]
                    
                    # Show comments by sentiment
                    pos_comments = cluster_comments[cluster_comments['sentiment'] == 'positive'].sort_values(by='polarity', ascending=False).head(5) if 'polarity' in cluster_comments.columns else cluster_comments[cluster_comments['sentiment'] == 'positive'].head(5)
                    neg_comments = cluster_comments[cluster_comments['sentiment'] == 'negative'].sort_values(by='polarity', ascending=True).head(5) if 'polarity' in cluster_comments.columns else cluster_comments[cluster_comments['sentiment'] == 'negative'].head(5)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Top Positive Comments**")
                        for _, row in pos_comments.iterrows():
                            st.markdown(f"{row['text']}")
                            st.markdown(f"*Movie: {row['movie']}*")
                            st.markdown("---")
                    
                    with col2:
                        st.markdown("**Top Negative Comments**")
                        for _, row in neg_comments.iterrows():
                            st.markdown(f"{row['text']}")
                            st.markdown(f"*Movie: {row['movie']}*")
                            st.markdown("---")
                
                with tab3:
                    st.subheader("Cluster Distribution")
                    
                    # Create cluster distribution visualization
                    import plotly.express as px
                    
                    # Count comments by cluster
                    cluster_counts = filtered_df['cluster'].value_counts().sort_index().reset_index()
                    cluster_counts.columns = ['Cluster', 'Count']
                    
                    # Add top terms to labels
                    cluster_counts['Label'] = cluster_counts['Cluster'].apply(
                        lambda x: f"Cluster {x}: {', '.join(top_terms[x][:3])}"
                    )
                    
                    # Create bar chart
                    fig = px.bar(
                        cluster_counts,
                        x='Label',
                        y='Count',
                        color='Cluster',
                        title="Comment Distribution by Cluster"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show sentiment distribution by cluster
                    st.subheader("Sentiment Distribution by Cluster")
                    
                    # Create heatmap
                    if not cluster_sentiment.empty:
                        import plotly.graph_objects as go
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=cluster_sentiment.values,
                            x=cluster_sentiment.columns,
                            y=[f"Cluster {c}" for c in cluster_sentiment.index],
                            colorscale=['#e74c3c', '#f1c40f', '#2ecc71'],
                            text=cluster_sentiment.values.round(2),
                            texttemplate="%{text:.0%}",
                            textfont={"size":12},
                        ))
                        
                        fig.update_layout(
                            title="Sentiment Distribution by Cluster",
                            xaxis_title="Sentiment",
                            yaxis_title="Cluster",
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in clustering: {str(e)}")
                st.info("Try selecting different clustering parameters or check your data.")

elif page == "Summaries":
    st.header("Comment Summarization")
    
    # Check if data is available
    if 'comments_df' not in st.session_state or st.session_state.comments_df is None:
        st.warning("No data available. Please collect data in the Data Collection page.")
    else:
        # Load comments data
        comments_df = st.session_state.comments_df
        
        # Sidebar controls
        st.sidebar.subheader("Summarization Options")
        
        # Select movie(s) to analyze
        available_movies = sorted(comments_df['movie'].unique())
        selected_movie = st.sidebar.selectbox(
            "Select Movie",
            options=available_movies,
            index=0 if len(available_movies) > 0 else None
        )
        
        # Select summarization method
        summary_method = st.sidebar.radio(
            "Summarization Method",
            options=["By Sentiment", "By Cluster", "Overall Summary"],
            index=0
        )
        
        # Number of sentences
        n_sentences = st.sidebar.slider("Number of Sentences", min_value=1, max_value=5, value=3)
        
        # Filter data for selected movie
        if selected_movie:
            filtered_df = comments_df[comments_df['movie'] == selected_movie]
        else:
            filtered_df = comments_df.copy()
            st.info("No movie selected. Summarizing all comments.")
        
        # Initialize summarizer
        summarizer = TextRankSummarizer(n_sentences=n_sentences)
        
        # Show progress
        with st.spinner("Generating summaries..."):
            try:
                # Ensure clean text is available
                if 'clean_text' not in filtered_df.columns:
                    filtered_df['clean_text'] = filtered_df['text'].apply(clean_text_for_sentiment)
                
                if summary_method == "By Sentiment":
                    st.subheader(f"Sentiment Summaries for {selected_movie}")
                    
                    # Group by sentiment
                    for sentiment in ['positive', 'negative', 'neutral']:
                        sentiment_comments = filtered_df[filtered_df['sentiment'] == sentiment]
                        
                        if len(sentiment_comments) > 0:
                            # Join all comments for this sentiment
                            text = " ".join(sentiment_comments['clean_text'].fillna(''))
                            
                            # Generate summary
                            if len(text.strip()) > 0:
                                summary = summarizer.summarize(text, n_sentences)
                                
                                # Display summary
                                st.markdown(f"**{sentiment.capitalize()} Comments Summary:**")
                                st.markdown(f"> {summary}")
                                
                                # Add stats
                                st.markdown(f"*Based on {len(sentiment_comments)} {sentiment} comments*")
                            else:
                                st.markdown(f"**{sentiment.capitalize()} Comments Summary:**")
                                st.markdown("*Unable to generate summary due to insufficient text*")
                            
                            st.markdown("---")
                        else:
                            st.markdown(f"**{sentiment.capitalize()} Comments:**")
                            st.markdown("*No comments with this sentiment*")
                            st.markdown("---")
                
                elif summary_method == "By Cluster":
                    # Check if clustering has been done
                    if 'cluster' not in filtered_df.columns:
                        # Perform clustering now
                        st.info("Performing clustering for summarization...")
                        
                        # Clean text for clustering
                        filtered_df['clean_text_cluster'] = filtered_df['text'].apply(clean_text_for_clustering)
                        
                        # Initialize and fit clusterer (use 5 clusters by default)
                        n_clusters = min(5, len(filtered_df) // 10) if len(filtered_df) > 50 else min(3, len(filtered_df) // 3)
                        if n_clusters < 2:
                            n_clusters = 2
                            
                        clusterer = CommentClusterer(n_clusters=n_clusters)
                        clusterer.fit(filtered_df['clean_text_cluster'].fillna(''))
                        
                        # Add cluster labels to dataframe
                        filtered_df['cluster'] = clusterer.predict(filtered_df['clean_text_cluster'].fillna(''))
                        
                        # Get top terms for each cluster
                        top_terms = clusterer.get_top_terms_per_cluster()
                    else:
                        # Get existing cluster terms
                        clusterer = CommentClusterer(n_clusters=filtered_df['cluster'].nunique())
                        clusterer.fit(filtered_df['clean_text'].fillna(''))
                        top_terms = clusterer.get_top_terms_per_cluster()
                    
                    st.subheader(f"Cluster Summaries for {selected_movie}")
                    
                    # Get unique clusters
                    clusters = sorted(filtered_df['cluster'].unique())
                    
                    # Generate summary for each cluster
                    for cluster in clusters:
                        cluster_comments = filtered_df[filtered_df['cluster'] == cluster]
                        
                        if len(cluster_comments) > 0:
                            # Join all comments for this cluster
                            text = " ".join(cluster_comments['clean_text'].fillna(''))
                            
                            # Generate summary
                            if len(text.strip()) > 0:
                                summary = summarizer.summarize(text, n_sentences)
                                
                                # Get cluster terms
                                terms = top_terms.get(cluster, ["Unknown theme"])
                                
                                # Display summary
                                st.markdown(f"**Cluster {cluster} ({', '.join(terms[:3])}):**")
                                st.markdown(f"> {summary}")
                                
                                # Add stats
                                pos_pct = (cluster_comments['sentiment'] == 'positive').mean() * 100
                                neg_pct = (cluster_comments['sentiment'] == 'negative').mean() * 100
                                
                                st.markdown(f"*Based on {len(cluster_comments)} comments ({pos_pct:.1f}% positive, {neg_pct:.1f}% negative)*")
                            else:
                                st.markdown(f"**Cluster {cluster} ({', '.join(terms[:3])}):**")
                                st.markdown("*Unable to generate summary due to insufficient text*")
                                
                            st.markdown("---")
                
                else:  # Overall Summary
                    st.subheader(f"Overall Summary for {selected_movie}")
                    
                    # Join all comments
                    text = " ".join(filtered_df['clean_text'].fillna(''))
                    
                    # Generate summary
                    if len(text.strip()) > 0:
                        summary = summarizer.summarize(text, n_sentences)
                        
                        # Display summary
                        st.markdown(f"> {summary}")
                        
                        # Add stats
                        pos_pct = (filtered_df['sentiment'] == 'positive').mean() * 100
                        neg_pct = (filtered_df['sentiment'] == 'negative').mean() * 100
                        neu_pct = 100 - pos_pct - neg_pct
                        
                        st.markdown(f"*Based on {len(filtered_df)} comments ({pos_pct:.1f}% positive, {neg_pct:.1f}% negative, {neu_pct:.1f}% neutral)*")
                    else:
                        st.warning("Unable to generate summary due to insufficient text.")
            except Exception as e:
                st.error(f"Error generating summaries: {str(e)}")
                st.info("Try selecting a different movie or summarization method.")

elif page == "Box Office Insights":
    st.header("Box Office Prediction Insights")
    
    # Check if data is available
    if 'comments_df' not in st.session_state or st.session_state.comments_df is None or 'movies_df' not in st.session_state or st.session_state.movies_df is None:
        st.warning("No data available. Please collect data in the Data Collection page.")
    else:
        st.info("This section connects comment sentiment to box office performance.")
        
        # Get sentiment stats by movie if not already calculated
        if 'movies_with_stats' not in st.session_state:
            try:
                st.session_state.movies_with_stats = process_sentiment_stats(
                    st.session_state.comments_df, 
                    st.session_state.movies_df
                )
                movies_df = st.session_state.movies_with_stats
            except Exception as e:
                st.error(f"Error processing sentiment stats: {str(e)}")
                movies_df = st.session_state.movies_df
        else:
            movies_df = st.session_state.movies_with_stats
        
        # Add error handling for missing columns
        required_columns = ['revenue', 'title']
        missing_columns = [col for col in required_columns if col not in movies_df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}. Please ensure data is properly processed.")
        else:
            # Check for sentiment stats
            sentiment_columns = ['avg_polarity', 'positive_pct', 'negative_pct']
            missing_sentiment = [col for col in sentiment_columns if col not in movies_df.columns]
            
            if missing_sentiment:
                st.warning(f"Missing sentiment statistics: {', '.join(missing_sentiment)}. Some visualizations may not be available.")
                
                # Try to add basic sentiment if possible
                if 'comments_df' in st.session_state and 'sentiment' in st.session_state.comments_df.columns:
                    st.info("Calculating basic sentiment statistics...")
                    
                    comments_df = st.session_state.comments_df
                    sentiment_stats = []
                    
                    for movie_title in movies_df['title']:
                        movie_comments = comments_df[comments_df['movie'] == movie_title]
                        total_comments = len(movie_comments)
                        
                        if total_comments > 0:
                            positive_count = (movie_comments['sentiment'] == 'positive').sum()
                            negative_count = (movie_comments['sentiment'] == 'negative').sum()
                            
                            positive_pct = positive_count / total_comments * 100
                            negative_pct = negative_count / total_comments * 100
                            
                            if 'polarity' in movie_comments.columns:
                                avg_polarity = movie_comments['polarity'].mean()
                            else:
                                # Estimate polarity from sentiment categories
                                avg_polarity = (positive_count - negative_count) / total_comments
                            
                            sentiment_stats.append({
                                'title': movie_title,
                                'avg_polarity': avg_polarity,
                                'positive_pct': positive_pct,
                                'negative_pct': negative_pct
                            })
                    
                    # Add sentiment stats to movies_df
                    sentiment_df = pd.DataFrame(sentiment_stats)
                    movies_df = pd.merge(movies_df, sentiment_df, on='title', how='left')
            
            # Filter to only movies with both sentiment stats and box office data
            has_box_office = movies_df['revenue'].notna() & (movies_df['revenue'] > 0)
            
            # Check for sentiment columns again after potential additions
            has_sentiment = all(col in movies_df.columns for col in ['avg_polarity'])
            
            valid_movies = movies_df[has_box_office]
            if has_sentiment:
                valid_movies = valid_movies[valid_movies['avg_polarity'].notna()]
            
            if len(valid_movies) > 0:
                # Create tabs for different analyses
                tab1, tab2, tab3 = st.tabs(["Sentiment vs. Revenue", "Correlation Analysis", "Genre Insights"])
                
                with tab1:
                    st.subheader("Sentiment vs. Box Office Revenue")
                    
                    if has_sentiment:
                        # Create scatter plot
                        try:
                            fig = px.scatter(
                                valid_movies, 
                                x='avg_polarity', 
                                y='revenue',
                                text='title',
                                size='comment_count' if 'comment_count' in valid_movies.columns else None,
                                color='positive_pct' if 'positive_pct' in valid_movies.columns else None,
                                hover_data=['positive_pct', 'negative_pct'] if all(col in valid_movies.columns for col in ['positive_pct', 'negative_pct']) else None,
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
                            
                            # Add regression line option
                            if st.checkbox("Show regression line"):
                                import numpy as np
                                from scipy import stats
                                
                                # Calculate regression line
                                x = valid_movies['avg_polarity']
                                y = valid_movies['revenue']
                                
                                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                                
                                # Add regression line to plot
                                x_range = np.linspace(x.min(), x.max(), 100)
                                y_range = slope * x_range + intercept
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_range,
                                        y=y_range,
                                        mode='lines',
                                        name=f'Regression (R²={r_value**2:.2f})',
                                        line=dict(color='red', dash='dash')
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show regression statistics
                                st.markdown(f"""
                                **Regression Statistics:**
                                - Slope: ${slope:,.2f}
                                - Intercept: ${intercept:,.2f}
                                - R-squared: {r_value**2:.4f}
                                - p-value: {p_value:.4f}
                                """)
                        except Exception as e:
                            st.error(f"Error creating scatter plot: {str(e)}")
                    else:
                        # Create basic box office visualization
                        try:
                            fig = px.bar(
                                valid_movies.sort_values('revenue', ascending=False).head(10),
                                x='title',
                                y='revenue',
                                title="Top 10 Movies by Box Office Revenue",
                                labels={'revenue': 'Box Office Revenue ($)', 'title': 'Movie'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.warning("Sentiment data not available for correlation analysis.")
                        except Exception as e:
                            st.error(f"Error creating bar chart: {str(e)}")
                
                with tab2:
                    st.subheader("Correlation Analysis")
                    
                    if has_sentiment and 'positive_pct' in valid_movies.columns and 'negative_pct' in valid_movies.columns:
                        try:
                            # Calculate correlations
                            corr = valid_movies['avg_polarity'].corr(valid_movies['revenue'])
                            corr_pos = valid_movies['positive_pct'].corr(valid_movies['revenue'])
                            corr_neg = valid_movies['negative_pct'].corr(valid_movies['revenue'])
                            
                            # Display correlation metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Avg. Polarity/Revenue", f"{corr:.2f}")
                            
                            with col2:
                                st.metric("Positive %/Revenue", f"{corr_pos:.2f}")
                            
                            with col3:
                                st.metric("Negative %/Revenue", f"{corr_neg:.2f}")
                            
                            # Create correlation heatmap
                            st.subheader("Correlation Matrix")
                            
                            # Select columns for correlation
                            corr_columns = ['revenue', 'avg_polarity', 'positive_pct', 'negative_pct']
                            if 'neutral_pct' in valid_movies.columns:
                                corr_columns.append('neutral_pct')
                            if 'comment_count' in valid_movies.columns:
                                corr_columns.append('comment_count')
                                
                            corr_columns = [col for col in corr_columns if col in valid_movies.columns]
                            
                            # Calculate correlation matrix
                            corr_matrix = valid_movies[corr_columns].corr()
                            
                            # Create heatmap using plotly
                            fig = px.imshow(
                                corr_matrix,
                                text_auto='.2f',
                                color_continuous_scale='RdBu_r',
                                title="Correlation Matrix",
                                labels=dict(x="Variables", y="Variables", color="Correlation")
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Interpretation
                            st.subheader("Interpretation")
                            
                            if abs(corr) > 0.3:
                                if corr > 0:
                                    st.markdown("There appears to be a **positive correlation** between sentiment polarity and box office revenue, suggesting that movies with more positive trailer reactions tend to perform better at the box office.")
                                else:
                                    st.markdown("There appears to be a **negative correlation** between sentiment polarity and box office revenue, which is an interesting finding that could warrant further investigation.")
                            else:
                                st.markdown("There does not appear to be a strong linear correlation between sentiment polarity and box office revenue in this dataset. Other factors may be more influential on performance.")
                        except Exception as e:
                            st.error(f"Error in correlation analysis: {str(e)}")
                    else:
                        st.warning("Sentiment data not available for correlation analysis.")
                
                with tab3:
                    st.subheader("Genre Insights")
                    
                    # Check if genres column exists
                    if 'genres' in valid_movies.columns:
                        try:
                            # Extract genres (comma-separated to list)
                            all_genres = []
                            for genres in valid_movies['genres']:
                                if pd.notna(genres) and genres:
                                    all_genres.extend([g.strip() for g in genres.split(',')])
                            
                            unique_genres = sorted(set(all_genres))
                            
                            if unique_genres:
                                # Create analysis by genre
                                st.markdown("#### Sentiment by Genre")
                                
                                # Select genre to analyze
                                selected_genre = st.selectbox("Select Genre", options=unique_genres)
                                
                                # Filter movies by genre
                                genre_movies = valid_movies[valid_movies['genres'].str.contains(selected_genre, na=False, regex=False)]
                                
                                if len(genre_movies) > 0:
                                    # Show genre statistics
                                    st.markdown(f"**{len(genre_movies)} movies in the {selected_genre} genre**")
                                    
                                    if has_sentiment:
                                        # Create scatter plot for this genre
                                        fig = px.scatter(
                                            genre_movies, 
                                            x='avg_polarity', 
                                            y='revenue',
                                            text='title',
                                            size='comment_count' if 'comment_count' in genre_movies.columns else None,
                                            hover_data=['positive_pct', 'negative_pct'] if all(col in genre_movies.columns for col in ['positive_pct', 'negative_pct']) else None,
                                            title=f"Sentiment vs. Box Office for {selected_genre} Movies",
                                            labels={
                                                'avg_polarity': 'Average Sentiment Polarity',
                                                'revenue': 'Box Office Revenue ($)'
                                            }
                                        )
                                        
                                        fig.update_traces(textposition='top center')
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Calculate genre-specific correlation
                                        genre_corr = genre_movies['avg_polarity'].corr(genre_movies['revenue'])
                                        
                                        st.markdown(f"Correlation for {selected_genre} movies: **{genre_corr:.2f}**")
                                        
                                        # Compare with overall correlation
                                        overall_corr = valid_movies['avg_polarity'].corr(valid_movies['revenue'])
                                        if abs(genre_corr - overall_corr) > 0.1:
                                            if genre_corr > overall_corr:
                                                st.markdown(f"*Sentiment appears to have a stronger relationship with box office performance for {selected_genre} movies compared to the overall dataset.*")
                                            else:
                                                st.markdown(f"*Sentiment appears to have a weaker relationship with box office performance for {selected_genre} movies compared to the overall dataset.*")
                                    else:
                                        # Basic revenue comparison by genre
                                        fig = px.bar(
                                            genre_movies.sort_values('revenue', ascending=False),
                                            x='title',
                                            y='revenue',
                                            title=f"Box Office Revenue for {selected_genre} Movies",
                                            labels={'revenue': 'Box Office Revenue ($)', 'title': 'Movie'}
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning(f"No movies found in the {selected_genre} genre with complete data.")
                            else:
                                st.warning("No genre information available in the dataset.")
                        except Exception as e:
                            st.error(f"Error in genre analysis: {str(e)}")
                    else:
                        st.warning("Genre information not available in the dataset.")
            else:
                st.warning("Not enough movies with both box office data and sentiment statistics. Please collect more data.")