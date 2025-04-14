# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import time
from datetime import datetime

# Import the necessary modules
from src.api.youtube import YouTubeClient
from src.api.movie_db import MovieDBClient
from src.preprocessing.clean_text import TextCleaner, clean_text_for_sentiment
from src.models.sentiment import SentimentAnalyzer, get_sentiment_distribution, get_top_comments
from src.models.clustering import cluster_comments, create_cluster_visualization
from src.models.summarization import summarize_by_sentiment, summarize_by_cluster
from src.visualization.sentiment_viz import (
    create_sentiment_distribution_pie,
    create_sentiment_comparison,
    create_sentiment_heatmap,
    analyze_sentiment_revenue_correlation
)
from src.pipeline import run_integrated_analysis_pipeline

# Import the data collection functions
from data_collection_functions import collect_movie_dataset, process_sentiment_stats

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
    
    st.info("This page collects movie trailer comments and correlates them with box office performance.")
    
    # Current dataset status display
    if 'comments_df' in st.session_state and st.session_state.comments_df is not None:
        st.subheader("Current Dataset Status")
        
        # Display basic stats
        movies_count = len(st.session_state.movies_df) if st.session_state.movies_df is not None else 0
        comments_count = len(st.session_state.comments_df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Movies", movies_count)
        
        with col2:
            st.metric("Comments", comments_count)
            
        with col3:
            avg_comments = comments_count / movies_count if movies_count > 0 else 0
            st.metric("Avg. Comments per Movie", f"{avg_comments:.1f}")
    
    # Main collection form
    with st.form("collection_form"):
        st.subheader("Collection Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_movies = st.number_input("Number of Movies", min_value=10, max_value=100, value=20)
            min_comments = st.number_input("Minimum Comments per Trailer", min_value=50, max_value=500, value=100)
        
        with col2:
            st.markdown("**📊 Data Collection Notes**")
            st.caption("Only movies with box office data will be included in the analysis.")
            st.caption("Previous data will be cleared when starting a new collection.")
        
        submit_button = st.form_submit_button("Start Collection")

    # Process form submission
    if submit_button:
        # Create progress indicators
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        collection_stats = st.empty()
        
        # Start collection
        status_text.text("Starting data collection process...")
        
        try:
            # Setup progress tracking
            collected_movies = []
            
            # Define callback to update progress
            def progress_callback(movie_title, movies_collected, target_movies):
                # Update progress bar
                progress = min(movies_collected / target_movies, 1.0)
                progress_bar.progress(progress)
                
                # Update status text
                status_text.text(f"Collecting data for: {movie_title}")
                
                # Update collection stats
                collection_stats.info(f"**Progress:** {movies_collected} of {target_movies} movies collected ({progress*100:.1f}%)")
                
                # Add to collected movies list
                if movie_title not in collected_movies:
                    collected_movies.append(movie_title)
                
                # Update movies list display
                progress_placeholder.markdown("### Movies Collected")
                progress_placeholder.markdown("\n".join([f"✅ {movie}" for movie in collected_movies]))
            
            # Run collection with progress updates
            with st.spinner("Collecting data..."):
                # Run collection function
                clear_previous_data = True  # Always clear previous data
                comments_df, movies_df = collect_movie_dataset(
                    min_movies=min_movies,
                    min_comments=min_comments,
                    include_box_office=True,  # Always require box office data
                    clear_previous_data=clear_previous_data,
                    save_path='data/processed',
                    progress_callback=progress_callback  # Pass the callback function
                )
                
                # Update progress bar to completion
                progress_bar.progress(100)
                
                # Store data in session state
                if comments_df is not None and not comments_df.empty:
                    st.session_state.comments_df = comments_df
                    
                if movies_df is not None and not movies_df.empty:
                    st.session_state.movies_df = movies_df
                    
                    # Process sentiment stats
                    movies_with_stats = process_sentiment_stats(comments_df, movies_df)
                    st.session_state.movies_with_stats = movies_with_stats
                    
                    # Clear any cached box office analysis to force recalculation
                    if 'box_office_analysis' in st.session_state:
                        del st.session_state.box_office_analysis
                    
                # Show success message
                if comments_df is not None and movies_df is not None and not movies_df.empty:
                    # Run box office analysis immediately
                    status_text.text("Running box office analysis...")
                    with st.spinner("Analyzing box office data..."):
                        # Get data from session state
                        comments_df = st.session_state.comments_df
                        movies_df = st.session_state.movies_df
                        
                        # Run the analysis pipeline
                        from src.visualization.sentiment_viz import analyze_sentiment_revenue_correlation
                        from src.visualization.dashboard_viz import create_dashboard_metrics
                        
                        # Filter movies with revenue data
                        valid_movies = movies_df[movies_df['revenue'] > 0].copy()
                        
                        if len(valid_movies) > 0:
                            # Calculate sentiment metrics per movie
                            movie_sentiment = {}
                            movie_stats = {}
                            
                            # Process sentiment stats for each movie
                            for movie_title in valid_movies['title'].unique():
                                movie_comments = comments_df[comments_df['movie'] == movie_title]
                                
                                if len(movie_comments) == 0:
                                    continue
                                    
                                # Calculate sentiment metrics
                                pos_pct = (movie_comments['sentiment'] == 'positive').mean() * 100
                                neg_pct = (movie_comments['sentiment'] == 'negative').mean() * 100
                                
                                if 'polarity' in movie_comments.columns:
                                    avg_polarity = movie_comments['polarity'].mean()
                                else:
                                    avg_polarity = (pos_pct - neg_pct) / 100
                                
                                # Store metrics
                                movie_sentiment[movie_title] = {
                                    'positive_pct': pos_pct,
                                    'negative_pct': neg_pct,
                                    'avg_polarity': avg_polarity,
                                    'comment_count': len(movie_comments)
                                }
                            
                            # Add sentiment data to movies dataframe
                            for movie in valid_movies.itertuples():
                                title = movie.title
                                if title in movie_sentiment:
                                    valid_movies.loc[valid_movies['title'] == title, 'positive_pct'] = movie_sentiment[title]['positive_pct']
                                    valid_movies.loc[valid_movies['title'] == title, 'negative_pct'] = movie_sentiment[title]['negative_pct']
                                    valid_movies.loc[valid_movies['title'] == title, 'avg_polarity'] = movie_sentiment[title]['avg_polarity']
                            
                            # Calculate correlation
                            correlation_results = analyze_sentiment_revenue_correlation(valid_movies)
                            
                            # Create dashboard metrics
                            dashboard_metrics = create_dashboard_metrics(
                                valid_movies, comments_df, correlation_results
                            )
                            
                            # Store all results in session state with a timestamp to prevent caching issues
                            import time
                            timestamp = time.time()
                            st.session_state.box_office_analysis = {
                                'correlation_results': correlation_results,
                                'dashboard_metrics': dashboard_metrics,
                                'valid_movies': valid_movies,
                                'movie_sentiment': movie_sentiment,
                                'timestamp': timestamp  # Add timestamp to force refresh
                            }
                    
                    status_text.text("Data collection and analysis complete!")
                    st.success(f"Successfully collected data for {len(movies_df)} movies with {len(comments_df)} total comments!")
                else:
                    st.error("Data collection failed or no movies met the criteria. Make sure your API keys are configured correctly.")
        
        except Exception as e:
            st.error(f"An error occurred during data collection: {str(e)}")
    
    # Show download buttons if data is available
    if 'comments_df' in st.session_state and st.session_state.comments_df is not None:
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
        
        # Filter data by selected movies
        if selected_movies:
            filtered_df = st.session_state.comments_df[st.session_state.comments_df['movie'].isin(selected_movies)]
        else:
            filtered_df = st.session_state.comments_df.copy()
        
        # Ensure we have sentiment analysis
        if 'sentiment' not in filtered_df.columns:
            st.info("Running sentiment analysis...")
            # Ensure clean text column exists
            if 'clean_text' not in filtered_df.columns:
                filtered_df['clean_text'] = filtered_df['text'].apply(clean_text_for_sentiment)
            
            # Run sentiment analysis with movie context
            analyzer = SentimentAnalyzer()
            sentiment_results = analyzer.analyze_sentiment(
                filtered_df['clean_text'], 
                movie_names=filtered_df['movie']  # Pass movie names for context
            )
            filtered_df['sentiment'] = sentiment_results['sentiment']
            filtered_df['polarity'] = sentiment_results['polarity']
            
            st.success("Sentiment analysis completed!")
        
        # Create tabs for different visualizations - removed Movie Comparison tab
        tab1, tab2 = st.tabs(["Sentiment Analysis", "Top Comments"])
        
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
            st.subheader("Top Comments by Sentiment")
            
            # Add a note about the sentiment analysis
            st.info("""
            Comments are analyzed using a context-aware sentiment algorithm that understands:
            - Movie-specific terminology and character references
            - Slang like "fire" 🔥, "peak", "goes hard" = positive
            - Genuine criticism vs excited references to characters/scenes
            - Context from frequently mentioned terms in each movie
            """)
            
            # Create columns for positive and negative comments
            pos_col, neg_col = st.columns(2)
            
            with pos_col:
                st.subheader("Top Positive Comments")
                top_positive = get_top_comments(filtered_df, 'positive', n=5)
                for i, row in top_positive.iterrows():
                    st.markdown(f"**Comment:** {row['text']}")
                    if 'polarity' in filtered_df.columns:
                        # Find the correct polarity value
                        original_row = filtered_df[filtered_df['text'] == row['text']]
                        if not original_row.empty:
                            pol_value = original_row['polarity'].values[0]
                            st.markdown(f"*Polarity: {pol_value:.2f}*")
                    st.markdown("---")
            
            with neg_col:
                st.subheader("Top Negative Comments")
                top_negative = get_top_comments(filtered_df, 'negative', n=5)
                for i, row in top_negative.iterrows():
                    st.markdown(f"**Comment:** {row['text']}")
                    if 'polarity' in filtered_df.columns:
                        # Find the correct polarity value
                        original_row = filtered_df[filtered_df['text'] == row['text']]
                        if not original_row.empty:
                            pol_value = original_row['polarity'].values[0]
                            st.markdown(f"*Polarity: {pol_value:.2f}*")
                    st.markdown("---")

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
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=3, max_value=8, value=5)
        
        # Include sentiment in clustering
        include_sentiment = st.sidebar.checkbox("Use Sentiment in Clustering", value=True,
                                             help="Include sentiment as a feature in clustering")
        
        # Filter data by selected movies
        if selected_movies:
            filtered_df = comments_df[comments_df['movie'].isin(selected_movies)]
            
            # Ensure clean text column exists
            if 'clean_text' not in filtered_df.columns:
                filtered_df['clean_text'] = filtered_df['text'].apply(clean_text_for_sentiment)
            
            # Run clustering
            with st.spinner("Analyzing comment themes..."):
                df_with_clusters, cluster_descriptions, cluster_terms, sentiment_by_cluster = cluster_comments(
                    filtered_df, text_column='clean_text', n_clusters=n_clusters, include_sentiment=include_sentiment
                )
            
            # Display theme counts and description
            st.subheader("Comment Themes")
            
            # Introduction text
            st.markdown("""
            The comments are automatically grouped into themes based on content and language patterns.
            Each theme represents a different type of audience reaction or discussion topic.
            """)
            
            # Get cluster visualization data
            cluster_viz = create_cluster_visualization(df_with_clusters, theme_mapping=cluster_descriptions)
            
            # Create bar chart with shortened labels
            fig = px.bar(
                x=cluster_viz['short_labels'],  # Use short labels 
                y=cluster_viz['counts'].values,
                labels={"x": "Theme", "y": "Comment Count"},
                title="Comment Themes Distribution",
                color=cluster_viz['counts'].values,
                color_continuous_scale="Viridis",
                text=cluster_viz['counts'].values
            )
            
            # Update layout
            fig.update_layout(
                xaxis_tickangle=-30,  # Reduced angle for better readability
                showlegend=False,
                height=500
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display theme descriptions - Expanded to show more details
            st.subheader("Theme Descriptions")
            
            for cluster_id, description in cluster_descriptions.items():
                with st.expander(description, expanded=False):
                    # Show theme keywords
                    if cluster_id in cluster_terms:
                        st.markdown(f"**Keywords:** {', '.join(cluster_terms[cluster_id][:7])}")
                    
                    # Show sentiment distribution if available
                    if sentiment_by_cluster is not None and cluster_id in sentiment_by_cluster.index:
                        sentiment_row = sentiment_by_cluster.loc[cluster_id]
                        
                        # Format sentiment distribution
                        sentiment_labels = []
                        for sentiment, value in sentiment_row.items():
                            if not pd.isna(value) and value > 0:
                                sentiment_labels.append(f"{sentiment.title()}: {value*100:.1f}%")
                        
                        if sentiment_labels:
                            st.markdown("**Sentiment:** " + ", ".join(sentiment_labels))
                        
                        # Determine dominant sentiment
                        dominant_sentiment = sentiment_row.idxmax() if not sentiment_row.empty else None
                        if dominant_sentiment:
                            dominant_color = "green" if dominant_sentiment == "positive" else "red" if dominant_sentiment == "negative" else "blue"
                            st.markdown(f"**Dominant Sentiment:** <span style='color:{dominant_color}'>{dominant_sentiment.title()}</span>", unsafe_allow_html=True)
                    
                    # Show sample comments
                    st.markdown("**Sample Comments:**")
                    sample_comments = df_with_clusters[df_with_clusters['cluster'] == cluster_id]['text'].sample(min(3, len(df_with_clusters[df_with_clusters['cluster'] == cluster_id]))).tolist()
                    
                    for i, comment in enumerate(sample_comments):
                        st.markdown(f"{i+1}. {comment}")
            
            # Audience Reaction Summary
            st.subheader("Audience Reaction Summary")
            
            # Count comments in each cluster/theme
            theme_comment_counts = df_with_clusters['cluster'].value_counts()
            total_comments = len(df_with_clusters)
            
            # Calculate percentages and create summary
            summary_data = []
            for cluster_id, count in theme_comment_counts.items():
                percentage = count / total_comments * 100
                
                # Get sentiment if available
                sentiment_info = ""
                if sentiment_by_cluster is not None and cluster_id in sentiment_by_cluster.index:
                    dominant_sentiment = sentiment_by_cluster.loc[cluster_id].idxmax()
                    
                    sentiment_emoji = "✅" if dominant_sentiment == "positive" else "❌" if dominant_sentiment == "negative" else "⚠️"
                    sentiment_info = f"{sentiment_emoji} {dominant_sentiment.title()}"
                
                # Get description without the terms list
                full_description = cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}")
                theme_name = full_description.split(':')[0] if ':' in full_description else full_description
                
                # Add to summary data
                summary_data.append({
                    "Theme": theme_name,
                    "Comments": f"{count} ({percentage:.1f}%)",
                    "Sentiment": sentiment_info
                })
            
            # Create dataframe for display
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Key insights section
            st.subheader("Key Insights")
            
            # Find the dominant theme (most comments)
            dominant_theme_id = theme_comment_counts.idxmax()
            dominant_theme = cluster_descriptions.get(dominant_theme_id, f"Cluster {dominant_theme_id}")
            dominant_theme_name = dominant_theme.split(':')[0] if ':' in dominant_theme else dominant_theme
            dominant_theme_pct = (theme_comment_counts[dominant_theme_id] / total_comments) * 100
            
            # Generate insights
            insights = []
            
            # Insight 1: Dominant theme
            insights.append(f"**{dominant_theme_name}** is the most common theme, representing {dominant_theme_pct:.1f}% of all comments.")
            
            # Insight 2: Sentiment by themes
            if sentiment_by_cluster is not None:
                positive_themes = []
                negative_themes = []
                
                for cluster_id in sentiment_by_cluster.index:
                    row = sentiment_by_cluster.loc[cluster_id]
                    if "positive" in row and row["positive"] > 0.6:  # More than 60% positive
                        theme_name = cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}")
                        theme_name = theme_name.split(':')[0] if ':' in theme_name else theme_name
                        positive_themes.append(theme_name)
                    
                    if "negative" in row and row["negative"] > 0.6:  # More than 60% negative
                        theme_name = cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}")
                        theme_name = theme_name.split(':')[0] if ':' in theme_name else theme_name
                        negative_themes.append(theme_name)
                
                if positive_themes:
                    insights.append(f"**Positive sentiment** is strongest in themes related to {', '.join(positive_themes)}.")
                
                if negative_themes:
                    insights.append(f"**Negative sentiment** is strongest in themes related to {', '.join(negative_themes)}.")
            
            # Display insights as bullet points
            for insight in insights:
                st.markdown(f"• {insight}")
            
            # Additional insights specific to trailer comments
            st.markdown(f"""
            • The distribution of themes suggests how audiences are primarily reacting to the trailer(s).
            • These patterns may indicate what aspects of the movie are driving audience interest.
            • Themes with strong positive sentiment may predict elements that will resonate with viewers.
            """)
        else:
            st.info("Select movies from the sidebar to begin clustering analysis")

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
        
        if selected_movie:
            # Filter comments for selected movie
            movie_comments = comments_df[comments_df['movie'] == selected_movie]
            
            # Ensure clean text column exists
            if 'clean_text' not in movie_comments.columns:
                from src.preprocessing.clean_text import clean_text_for_sentiment
                movie_comments['clean_text'] = movie_comments['text'].apply(clean_text_for_sentiment)
            
            # Ensure sentiment column exists
            if 'sentiment' not in movie_comments.columns:
                from src.models.sentiment import SentimentAnalyzer
                analyzer = SentimentAnalyzer()
                sentiment_results = analyzer.analyze_sentiment(movie_comments['clean_text'])
                movie_comments['sentiment'] = sentiment_results['sentiment']
            
            # Create tabs for different summarization approaches
            tab1, tab2 = st.tabs(["Sentiment Summaries", "Cluster Summaries"])
            
            with tab1:
                st.subheader(f"Sentiment-based Summaries for {selected_movie}")
                
                # Add info about the summarization method
                st.info("""
                These summaries represent the overall themes in comments with similar sentiment.
                The TextRank algorithm identifies the most important sentences that capture the key points.
                """)
                
                # Generate summaries by sentiment
                with st.spinner("Generating sentiment summaries..."):
                    sentiment_summaries = summarize_by_sentiment(
                        movie_comments, 
                        text_column='clean_text', 
                        sentiment_column='sentiment',
                        n_sentences=3  # Fixed at 3 sentences
                    )
                
                # Display sentiment distribution
                sentiment_counts = movie_comments['sentiment'].value_counts()
                total_comments = len(movie_comments)
                
                # Create columns for sentiment stats
                cols = st.columns(len(sentiment_counts))
                
                for i, (sentiment, count) in enumerate(sentiment_counts.items()):
                    with cols[i]:
                        sentiment_color = {
                            'positive': 'green',
                            'negative': 'red',
                            'neutral': 'blue'
                        }.get(sentiment, 'gray')
                        
                        # Display metric without delta
                        st.metric(
                            label=f"{sentiment.title()} Comments",
                            value=count
                        )
                        
                # Display summaries in styled containers
                for sentiment, data in sentiment_summaries.items():
                    # Skip if no summary
                    if "summary" not in data:
                        continue
                        
                    # Select color based on sentiment
                    color = {
                        'positive': '#d4f1d4',  # light green
                        'negative': '#ffebee',  # light red
                        'neutral': '#e3f2fd'    # light blue
                    }.get(sentiment, '#f5f5f5')
                    
                    # Create styled container
                    with st.container(border=True):
                        st.markdown(f"""
                        <h3 style="color: {'green' if sentiment == 'positive' else 'red' if sentiment == 'negative' else 'gray'}">
                            {data['display_name']} Comments Summary ({data['count']} comments)
                        </h3>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style="background-color: {color}; padding: 15px; border-radius: 5px; font-size: 16px; margin-bottom: 20px;">
                            {data['summary']}
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                st.subheader(f"Cluster-based Summaries for {selected_movie}")
                
                # Number of clusters slider for this tab
                n_clusters = st.slider(
                    "Number of Clusters", 
                    min_value=3, 
                    max_value=8, 
                    value=5,
                    help="Adjust the number of thematic clusters to identify in the comments"
                )
                
                # Add info about clustering
                st.info("""
                Comments are clustered into thematic groups based on content similarity.
                Each cluster represents a different type of comment or reaction to the trailer.
                """)
                
                # Check if clustering has been done with the current cluster count
                current_clustering_key = f"clustering_{selected_movie}_{n_clusters}"
                
                if current_clustering_key not in st.session_state:
                    # Run clustering with current parameters
                    with st.spinner("Clustering comments..."):
                        from src.models.clustering import cluster_comments
                        df_with_clusters, cluster_descriptions, cluster_terms, sentiment_by_cluster = cluster_comments(
                            movie_comments, text_column='clean_text', n_clusters=n_clusters, include_sentiment=True
                        )
                        
                        # Store in session state
                        st.session_state[current_clustering_key] = {
                            "df": df_with_clusters,
                            "descriptions": cluster_descriptions,
                            "terms": cluster_terms,
                            "sentiment": sentiment_by_cluster
                        }
                else:
                    # Use cached clustering
                    cached = st.session_state[current_clustering_key]
                    df_with_clusters = cached["df"]
                    cluster_descriptions = cached["descriptions"]
                    cluster_terms = cached["terms"]
                    sentiment_by_cluster = cached["sentiment"]
                
                # Generate summaries by cluster
                with st.spinner("Generating cluster summaries..."):
                    cluster_summaries = summarize_by_cluster(
                        df_with_clusters, 
                        text_column='clean_text', 
                        cluster_column='cluster',
                        sentiment_column='sentiment',
                        n_sentences=3,  # Fixed at 3 sentences
                        cluster_descriptions=cluster_descriptions
                    )
                
                # Display cluster overview
                st.subheader("Theme Overview")
                
                # Create a dataframe for display
                overview_data = []
                for cluster_id, data in cluster_summaries.items():
                    sentiment_info = ""
                    if data['sentiment_stats']:
                        # Get the dominant sentiment
                        dominant = max(data['sentiment_stats'].items(), key=lambda x: x[1]['percentage'])
                        sentiment_info = f"{dominant[0].title()} ({dominant[1]['percentage']:.1f}%)"
                    
                    overview_data.append({
                        "Theme": data['description'],
                        "Comments": data['count'],
                        "Dominant Sentiment": sentiment_info
                    })
                
                overview_df = pd.DataFrame(overview_data)
                st.dataframe(overview_df, use_container_width=True)
                
                # Display summaries in expanders
                for cluster_id in sorted(cluster_summaries.keys()):
                    data = cluster_summaries[cluster_id]
                    
                    with st.expander(f"Theme: {data['description']} ({data['count']} comments)"):
                        # Theme keywords
                        if cluster_id in cluster_terms:
                            st.markdown(f"**Keywords:** {', '.join(cluster_terms[cluster_id][:7])}")
                        
                        # Summary
                        st.markdown("### Summary")
                        st.markdown(f"""
                        <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; font-size: 16px;">
                            {data['summary']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Sample comments
                        if data['sample_comments']:
                            st.markdown("### Sample Comments")
                            for i, comment in enumerate(data['sample_comments']):
                                st.markdown(f"**{i+1}.** {comment}")
        else:
            st.info("Select a movie from the sidebar to generate summaries")

elif page == "Box Office Insights":
    st.header("Box Office Prediction Insights")
    
    # Check if data is available
    if ('comments_df' not in st.session_state or 
        st.session_state.comments_df is None or 
        'movies_df' not in st.session_state or 
        st.session_state.movies_df is None):
        st.warning("No data available. Please collect data in the Data Collection page.")
    else:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Overall Dashboard", "Per-Movie Analysis", "Outlier Analysis"])
        
        # Add sidebar options for analysis
        st.sidebar.subheader("Box Office Analysis Options")
        include_outliers = st.sidebar.checkbox("Include Revenue Outliers", value=False, 
                                            help="Include movies with extreme box office revenue (e.g. blockbusters)")
        
        z_score_threshold = st.sidebar.slider("Outlier Filter Strength", 
                                           min_value=1.5, max_value=3.5, value=2.5, step=0.1,
                                           help="Z-score threshold for filtering outliers (lower = more aggressive filtering)")
        
        # Check if we need to run the analysis
        run_analysis = False
        if 'box_office_analysis' not in st.session_state:
            run_analysis = True
        elif st.button("Refresh Analysis"):
            # Force refresh if button is clicked
            run_analysis = True
            if 'box_office_analysis' in st.session_state:
                del st.session_state.box_office_analysis
        
        # Run the analysis if needed
        if run_analysis:
            # Run the analysis pipeline
            with st.spinner("Analyzing data... This may take a moment."):
                # Import needed functions
                from src.processing.outlier_filtering import filter_box_office_outliers
                from src.visualization.sentiment_viz import analyze_sentiment_revenue_correlation
                from src.visualization.dashboard_viz import create_dashboard_metrics
                
                # Get data from session state
                comments_df = st.session_state.comments_df
                movies_df = st.session_state.movies_df
                
                # Filter movies with revenue data
                all_box_office_movies = movies_df[movies_df['revenue'] > 0].copy()
                
                # Calculate sentiment metrics per movie
                movie_sentiment = {}
                for movie_title in all_box_office_movies['title'].unique():
                    movie_comments = comments_df[comments_df['movie'] == movie_title]
                    
                    if len(movie_comments) == 0:
                        continue
                        
                    # Calculate metrics
                    pos_pct = (movie_comments['sentiment'] == 'positive').mean() * 100
                    neg_pct = (movie_comments['sentiment'] == 'negative').mean() * 100
                    
                    if 'polarity' in movie_comments.columns:
                        avg_polarity = movie_comments['polarity'].mean()
                    else:
                        avg_polarity = (pos_pct - neg_pct) / 100
                    
                    # Store metrics
                    movie_sentiment[movie_title] = {
                        'positive_pct': pos_pct,
                        'negative_pct': neg_pct,
                        'avg_polarity': avg_polarity,
                        'comment_count': len(movie_comments)
                    }
                
                # Add sentiment data to all_box_office_movies
                for movie_title, sentiment_data in movie_sentiment.items():
                    all_box_office_movies.loc[all_box_office_movies['title'] == movie_title, 'positive_pct'] = sentiment_data['positive_pct']
                    all_box_office_movies.loc[all_box_office_movies['title'] == movie_title, 'negative_pct'] = sentiment_data['negative_pct']
                    all_box_office_movies.loc[all_box_office_movies['title'] == movie_title, 'avg_polarity'] = sentiment_data['avg_polarity']
                
                # Apply outlier filtering
                normal_range_movies = filter_box_office_outliers(
                    all_box_office_movies,
                    revenue_column='revenue',
                    budget_column='budget',
                    z_score_threshold=z_score_threshold,
                    min_movies=10
                )
                
                # Calculate outlier stats
                total_movies = len(all_box_office_movies)
                normal_range_count = len(normal_range_movies)
                outlier_count = total_movies - normal_range_count
                outlier_pct = (outlier_count / total_movies * 100) if total_movies > 0 else 0
                
                # Run correlation analysis on both datasets
                all_movies_correlation = analyze_sentiment_revenue_correlation(all_box_office_movies)
                normal_range_correlation = analyze_sentiment_revenue_correlation(normal_range_movies)
                
                # Create dashboard metrics using the appropriate dataset
                if include_outliers:
                    dashboard_metrics = create_dashboard_metrics(all_box_office_movies, comments_df, all_movies_correlation)
                    primary_correlation = all_movies_correlation
                    primary_movies = all_box_office_movies
                else:
                    dashboard_metrics = create_dashboard_metrics(normal_range_movies, comments_df, normal_range_correlation)
                    primary_correlation = normal_range_correlation
                    primary_movies = normal_range_movies
                
                # Store all results in session state with a timestamp
                import time
                timestamp = time.time()
                
                st.session_state.box_office_analysis = {
                    'all_box_office_movies': all_box_office_movies,
                    'normal_range_movies': normal_range_movies,
                    'all_movies_correlation': all_movies_correlation,
                    'normal_range_correlation': normal_range_correlation,
                    'dashboard_metrics': dashboard_metrics,
                    'movie_sentiment': movie_sentiment,
                    'outlier_stats': {
                        'total_box_office_movies': total_movies,
                        'normal_range_count': normal_range_count,
                        'outlier_count': outlier_count,
                        'outlier_pct': outlier_pct,
                        'z_score_threshold': z_score_threshold
                    },
                    'include_outliers': include_outliers,
                    'primary_correlation': primary_correlation,
                    'primary_movies': primary_movies,
                    'timestamp': timestamp
                }
        
        # Retrieve analysis results
        if 'box_office_analysis' in st.session_state:
            analysis = st.session_state.box_office_analysis
            
            # Update the active dataset based on current checkbox
            if include_outliers != analysis.get('include_outliers', False):
                # Need to switch between datasets
                if include_outliers:
                    analysis['primary_correlation'] = analysis['all_movies_correlation']
                    analysis['primary_movies'] = analysis['all_box_office_movies']
                else:
                    analysis['primary_correlation'] = analysis['normal_range_correlation']
                    analysis['primary_movies'] = analysis['normal_range_movies']
                
                # Update the flag
                analysis['include_outliers'] = include_outliers
            
            # Get active dataset
            primary_correlation = analysis['primary_correlation']
            primary_movies = analysis['primary_movies']
            outlier_stats = analysis.get('outlier_stats', {})
            
            with tab1:
                st.subheader("Overall Box Office Insights")
                
                # Show outlier filtering info
                if not include_outliers and outlier_stats.get('outlier_count', 0) > 0:
                    st.info(f"Showing analysis with {outlier_stats.get('outlier_count', 0)} outlier movies filtered out ({outlier_stats.get('outlier_pct', 0):.1f}% of movies). Toggle 'Include Revenue Outliers' in the sidebar to view all movies.")
                elif include_outliers and outlier_stats.get('outlier_count', 0) > 0:
                    st.info(f"Showing analysis with all movies including {outlier_stats.get('outlier_count', 0)} revenue outliers. Toggle 'Include Revenue Outliers' in the sidebar to filter them.")
                
                # Display metrics at the top
                metrics = analysis['dashboard_metrics']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Movies", len(primary_movies))
                
                with col2:
                    st.metric("Total Comments", f"{metrics.get('total_comments', 0):,}")
                
                with col3:
                    st.metric("Positive Comments", f"{metrics.get('positive_pct', 0):.1f}%")
                
                with col4:
                    r_squared = primary_correlation.get("r_squared", 0)
                    st.metric("Sentiment-Revenue R²", f"{r_squared:.2f}")
                
                # Show correlation visualization
                if "scatter_plot" in primary_correlation:
                    st.plotly_chart(
                        primary_correlation["scatter_plot"],
                        use_container_width=True
                    )

                    # Show ROI plot if available
                    if "roi_scatter_plot" in primary_correlation:
                        st.subheader("Sentiment vs. Return on Investment")
                        st.plotly_chart(
                            primary_correlation["roi_scatter_plot"],
                            use_container_width=True
                        )
                
                # Display movie revenue ranking
                valid_movies = primary_movies.sort_values('revenue', ascending=False)
                
                st.subheader("Movie Revenue Ranking")
                
                # Create two columns for the movie revenue chart
                chart_col, table_col = st.columns([3, 2])
                
                with chart_col:
                    # Create revenue bar chart
                    import plotly.express as px
                    
                    # Prepare data (take top 10 movies by revenue)
                    top_movies = valid_movies.head(10).copy()
                    
                    # Format revenue for better display
                    top_movies['revenue_millions'] = top_movies['revenue'] / 1000000
                    
                    # Create bar chart
                    fig = px.bar(
                        top_movies,
                        x='title',
                        y='revenue_millions',
                        color='avg_polarity',
                        color_continuous_scale=px.colors.diverging.RdYlGn,
                        labels={
                            'title': 'Movie',
                            'revenue_millions': 'Revenue ($ millions)',
                            'avg_polarity': 'Comment Sentiment'
                        },
                        title=f"Top 10 Movies by Box Office Revenue (out of {len(valid_movies)} movies)"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with table_col:
                    # Create a table view
                    st.markdown("### Revenue & Sentiment Data")
                    
                    # Create a DataFrame with selected columns
                    table_data = valid_movies.copy()
                    table_data['revenue_millions'] = table_data['revenue'] / 1000000
                    table_data['revenue_millions'] = table_data['revenue_millions'].round(1)
                    table_data['positive_pct'] = table_data['positive_pct'].round(1)
                    
                    # Select and rename columns
                    display_df = table_data[['title', 'revenue_millions', 'positive_pct']].rename(
                        columns={
                            'title': 'Movie',
                            'revenue_millions': 'Revenue ($M)',
                            'positive_pct': 'Positive Comments (%)'
                        }
                    )
                    
                    st.dataframe(display_df, use_container_width=True)
            
            with tab2:
                # Per-Movie Analysis tab content (same as before)
                st.subheader("Per-Movie Analysis")
                
                # Get all movie names from the active dataset
                movie_names = primary_movies['title'].unique()
                
                # Select movie for analysis
                selected_movie = st.selectbox(
                    "Select a movie to analyze",
                    options=movie_names,
                    index=0 if len(movie_names) > 0 else None
                )
                
                if selected_movie:
                    # Get movie data
                    movie_data = primary_movies[primary_movies['title'] == selected_movie].iloc[0]
                    
                    # Create columns for key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Revenue
                        revenue_millions = movie_data['revenue'] / 1000000
                        st.metric("Box Office Revenue", f"${revenue_millions:.1f}M")
                    
                    with col2:
                        # Budget if available
                        if 'budget' in movie_data and movie_data['budget'] > 0:
                            budget_millions = movie_data['budget'] / 1000000
                            st.metric("Production Budget", f"${budget_millions:.1f}M")
                    
                    with col3:
                        # ROI or profit if both revenue and budget are available
                        if 'budget' in movie_data and movie_data['budget'] > 0 and movie_data['revenue'] > 0:
                            profit = movie_data['revenue'] - movie_data['budget']
                            roi = (profit / movie_data['budget']) * 100
                            
                            profit_millions = profit / 1000000
                            
                            if profit > 0:
                                st.metric("Profit", f"${profit_millions:.1f}M (+{roi:.1f}%)")
                            else:
                                st.metric("Loss", f"${profit_millions:.1f}M ({roi:.1f}%)", delta_color="inverse")
                        else:
                            # If budget data isn't available
                            movie_sent = analysis['movie_sentiment'].get(selected_movie, {})
                            st.metric("Positive Comments", f"{movie_sent.get('positive_pct', movie_data.get('positive_pct', 0)):.1f}%")
            
            with tab3:
                # New tab for outlier analysis
                st.subheader("Revenue Outlier Analysis")
                
                # Display outlier stats
                if outlier_stats:
                    st.markdown(f"""
                    ### Revenue Distribution Statistics
                    
                    - **Total movies with box office data**: {outlier_stats.get('total_box_office_movies', 0)}
                    - **Movies within normal revenue range**: {outlier_stats.get('normal_range_count', 0)}
                    - **Outlier movies**: {outlier_stats.get('outlier_count', 0)} ({outlier_stats.get('outlier_pct', 0):.1f}%)
                    - **Z-score threshold used**: {outlier_stats.get('z_score_threshold', 0)}
                    """)
                    
                    # Get all box office movies
                    all_movies = analysis['all_box_office_movies'].copy()
                    normal_range = analysis['normal_range_movies'].copy()
                    
                    # Mark outliers in the full dataset
                    all_movies['is_outlier'] = ~all_movies['title'].isin(normal_range['title'])
                    
                    # Create visualization of revenue distribution
                    import plotly.express as px
                    import numpy as np
                    
                    # Convert to millions for better display
                    all_movies['revenue_millions'] = all_movies['revenue'] / 1000000
                    
                    # Create bar chart of all movies with outliers highlighted
                    fig = px.bar(
                        all_movies.sort_values('revenue', ascending=False),
                        x='title',
                        y='revenue_millions',
                        color='is_outlier',
                        color_discrete_map={True: 'red', False: 'blue'},
                        labels={
                            'title': 'Movie',
                            'revenue_millions': 'Revenue ($ millions)',
                            'is_outlier': 'Revenue Outlier'
                        },
                        title="All Movies by Box Office Revenue (Outliers in Red)"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        template="plotly_white",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display statistics
                    if len(all_movies) > 0 and len(normal_range) > 0:
                        st.subheader("Revenue Statistics")
                        
                        # Calculate statistics
                        all_mean = all_movies['revenue'].mean()
                        all_median = all_movies['revenue'].median()
                        all_std = all_movies['revenue'].std()
                        
                        filtered_mean = normal_range['revenue'].mean()
                        filtered_median = normal_range['revenue'].median()
                        filtered_std = normal_range['revenue'].std()
                        
                        # Display in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### With Outliers")
                            st.metric("Mean Revenue", f"${all_mean/1000000:.1f}M")
                            st.metric("Median Revenue", f"${all_median/1000000:.1f}M")
                            st.metric("Standard Deviation", f"${all_std/1000000:.1f}M")
                            st.metric("Revenue Range", f"${all_movies['revenue'].min()/1000000:.1f}M - ${all_movies['revenue'].max()/1000000:.1f}M")
                        
                        with col2:
                            st.markdown("#### Without Outliers")
                            st.metric("Mean Revenue", f"${filtered_mean/1000000:.1f}M")
                            st.metric("Median Revenue", f"${filtered_median/1000000:.1f}M")
                            st.metric("Standard Deviation", f"${filtered_std/1000000:.1f}M")
                            st.metric("Revenue Range", f"${normal_range['revenue'].min()/1000000:.1f}M - ${normal_range['revenue'].max()/1000000:.1f}M")
                        
                        # Show list of outlier movies
                        if outlier_stats.get('outlier_count', 0) > 0:
                            st.subheader("Outlier Movies")
                            outlier_movies = all_movies[all_movies['is_outlier']].sort_values('revenue', ascending=False)
                            
                            # Create a table with outlier details
                            outlier_table = outlier_movies[['title', 'revenue_millions', 'avg_polarity', 'positive_pct']].copy()
                            outlier_table = outlier_table.rename(columns={
                                'title': 'Movie',
                                'revenue_millions': 'Revenue ($M)',
                                'avg_polarity': 'Sentiment Score',
                                'positive_pct': 'Positive Comments (%)'
                            })
                            
                            st.dataframe(outlier_table.round(2), use_container_width=True)
                            
                            # Compare correlation with and without outliers
                            st.subheader("Impact on Correlation Analysis")
                            
                            # Get correlation values
                            with_outliers_r = all_movies_correlation.get("r_squared", 0)
                            without_outliers_r = normal_range_correlation.get("r_squared", 0)
                            
                            # Calculate the difference
                            diff = without_outliers_r - with_outliers_r
                            diff_pct = (diff / with_outliers_r * 100) if with_outliers_r != 0 else float('inf')
                            
                            # Display comparison
                            corr_col1, corr_col2 = st.columns(2)
                            
                            with corr_col1:
                                st.metric("R² with Outliers", f"{with_outliers_r:.3f}")
                            
                            with corr_col2:
                                # Use delta color to show if filtering improved correlation
                                delta_color = "normal" if diff > 0 else "inverse"
                                st.metric("R² without Outliers", f"{without_outliers_r:.3f}", 
                                          delta=f"{diff_pct:+.1f}%", delta_color=delta_color)
                            
                            # Add explanation
                            if diff > 0:
                                st.success(f"Removing outliers improved the correlation by {diff_pct:.1f}%, suggesting that extreme box office values were distorting the relationship between sentiment and revenue.")
                            elif diff < 0:
                                st.error(f"Removing outliers decreased the correlation by {abs(diff_pct):.1f}%, suggesting that outlier movies actually followed the sentiment-revenue pattern more strongly than typical movies.")
                            else:
                                st.info("Removing outliers had no significant impact on the correlation strength.")
                
                # Insights section with comparison of results
                st.subheader("Key Insights")
                
                # Generate insights based on actual results
                if "correlations" in primary_correlation:
                    corr_value = primary_correlation["correlations"].get('avg_polarity', 0)
                    
                    if abs(corr_value) < 0.3:
                        correlation_text = "weak"
                    elif abs(corr_value) < 0.6:
                        correlation_text = "moderate"
                    else:
                        correlation_text = "strong"
                    
                    st.info(f"""
                    📊 Analysis shows a **{correlation_text} {corr_value:.2f} correlation** between comment sentiment and box office revenue.
                    
                    💰 Movies with more positive trailer comments tend to have {'higher' if corr_value > 0 else 'lower'} box office performance.
                    
                    📈 The sentiment-revenue relationship explains approximately **{primary_correlation.get("r_squared", 0):.1%} of revenue variation** in this dataset.
                    """)
        else:
            # If no analysis has been run yet
            st.info("Collecting data and running analysis. Please wait...")
            # Run the analysis pipeline now
            from src.pipeline import run_integrated_analysis_pipeline
            
            # Get data from session state
            comments_df = st.session_state.comments_df
            movies_df = st.session_state.movies_df
            
            # Run analysis
            with st.spinner("Running analysis pipeline..."):
                results = run_integrated_analysis_pipeline(comments_df, movies_df)
                st.session_state.box_office_analysis = results
                st.experimental_rerun()  # Rerun the app to show results