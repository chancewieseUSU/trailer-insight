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
    
    st.info("This page allows you to collect movie trailer comments and correlate them with box office performance.")
    
    # Main collection form
    with st.form("collection_form"):
        st.subheader("Collection Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_movies = st.number_input("Number of Movies", min_value=10, max_value=100, value=20)
            min_comments = st.number_input("Minimum Comments per Trailer", min_value=50, max_value=500, value=100)
        
        with col2:
            # Box office data is now required, show info text instead of checkbox
            st.markdown("**Box Office Data Collection**")
            st.markdown("ℹ️ Only movies with box office data will be included in the analysis.")
            
            # Add option to clear previous data
            clear_previous = st.checkbox("Clear Previous Data", value=False, 
                                       help="If checked, all previously collected data will be deleted before saving new data.")
        
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
                # Run collection function with required box office data
                comments_df, movies_df = collect_movie_dataset(
                    min_movies=min_movies,
                    min_comments=min_comments,
                    include_box_office=True,  # Always require box office data
                    clear_previous_data=clear_previous,
                    save_path='data/processed'
                )
                
                # Update progress bar
                progress_bar.progress(100)
                
                # Store data in session state
                if comments_df is not None and not comments_df.empty:
                    st.session_state.comments_df = comments_df
                    
                if movies_df is not None and not movies_df.empty:
                    st.session_state.movies_df = movies_df
                    
                    # Process sentiment stats
                    movies_with_stats = process_sentiment_stats(comments_df, movies_df)
                    st.session_state.movies_with_stats = movies_with_stats
                
                # Show success message
                if comments_df is not None and movies_df is not None and not movies_df.empty:
                    st.success(f"Successfully collected data for {len(movies_df)} movies with {len(comments_df)} total comments!")
                else:
                    st.error("Data collection failed or no movies met the criteria. Make sure your API keys are configured correctly.")
        
        except Exception as e:
            st.error(f"An error occurred during data collection: {str(e)}")
            
    # Show data status if available
    if 'comments_df' in st.session_state and st.session_state.comments_df is not None:
        st.subheader("Current Dataset")
        
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
                x=cluster_viz['short_labels'],  # Use short labels here!
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
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Theme Descriptions", "Sentiment by Theme"])
            
            with tab1:
                # Display theme descriptions
                for cluster_id, description in cluster_descriptions.items():
                    with st.expander(description, expanded=False):
                        # Show top terms
                        st.markdown(f"**Top terms:** {', '.join(cluster_terms[cluster_id])}")
                        
                        # Show sample comments
                        st.markdown("**Sample Comments:**")
                        sample_comments = df_with_clusters[df_with_clusters['cluster'] == cluster_id]['text'].sample(min(3, len(df_with_clusters[df_with_clusters['cluster'] == cluster_id]))).tolist()
                        
                        for i, comment in enumerate(sample_comments):
                            st.markdown(f"{i+1}. {comment}")
            
            with tab2:
                # Show sentiment distribution by cluster
                if sentiment_by_cluster is not None:
                    # Create a heatmap for sentiment distribution
                    # Prepare the data
                    sentiment_df = sentiment_by_cluster.reset_index()
                    sentiment_df['Cluster'] = sentiment_df['cluster'].apply(
                        lambda x: cluster_viz['short_labels'][list(cluster_viz['counts'].index).index(x)]
                        if x in cluster_viz['counts'].index else f"Cluster {x}"
                    )
                    
                    # Melt the dataframe for plotting
                    sentiment_plot_data = pd.melt(
                        sentiment_df, 
                        id_vars=['Cluster'], 
                        value_vars=['positive', 'negative', 'neutral'] if all(col in sentiment_df.columns for col in ['positive', 'negative', 'neutral']) else sentiment_df.columns[1:],
                        var_name='Sentiment', 
                        value_name='Proportion'  # Changed from 'Percentage'
                    )
                    
                    # Create the heatmap
                    fig = px.density_heatmap(
                        sentiment_plot_data,
                        x='Sentiment',
                        y='Cluster',
                        z='Proportion',
                        title="Sentiment Distribution by Theme",
                        color_continuous_scale=[
                            [0, '#FFFFFF'],
                            [0.5, '#ABDDA4'],
                            [1, '#2ecc71']
                        ],
                        text_auto=True
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title="Sentiment Category",
                        yaxis_title="Theme",
                        coloraxis_colorbar_title="Proportion"  # Changed from 'Percentage'
                    )
                    
                    # Display the heatmap
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show insight about the dominant sentiment for each cluster
                    st.subheader("Theme Insights")
                    
                    for cluster_id in sentiment_by_cluster.index:
                        row = sentiment_by_cluster.loc[cluster_id]
                        dominant_sentiment = row.idxmax()
                        
                        # Use emoji based on sentiment
                        if dominant_sentiment == 'positive':
                            emoji = "✅"
                            color = "green"
                        elif dominant_sentiment == 'negative':
                            emoji = "❌"
                            color = "red"
                        else:
                            emoji = "⚠️"
                            color = "orange"
                        
                        st.markdown(
                            f"{emoji} **{cluster_descriptions[cluster_id]}** - "
                            f"<span style='color:{color}'>{dominant_sentiment.title()}</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("Sentiment data not available for clusters.")
                    
            # Generate themes summary
            st.subheader("Audience Reaction Summary")
            
            # Count comments in each cluster/theme
            theme_comment_counts = df_with_clusters['cluster'].value_counts()
            total_comments = len(df_with_clusters)
            
            # Top themes (>10% of comments)
            main_themes = [(cluster_id, count) for cluster_id, count in theme_comment_counts.items() 
                         if count / total_comments >= 0.1]
            
            if main_themes:
                theme_insights = []
                
                for cluster_id, count in main_themes:
                    percentage = count / total_comments * 100
                    
                    # Get sentiment if available
                    sentiment_info = ""
                    if sentiment_by_cluster is not None and cluster_id in sentiment_by_cluster.index:
                        row = sentiment_by_cluster.loc[cluster_id]
                        dominant_sentiment = row.idxmax()
                        sentiment_info = f" with {dominant_sentiment} sentiment"
                    
                    # Get the theme name but without the terms list
                    theme_full = cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}")
                    theme_name = theme_full.split(':')[0] if ':' in theme_full else theme_full
                    
                    theme_insights.append(
                        f"- **{theme_name}** represents {percentage:.1f}% of audience comments{sentiment_info}."
                    )
                
                st.markdown("\n".join(theme_insights))
            else:
                st.info("No major themes found. Try adjusting the number of clusters.")
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
                movie_comments['clean_text'] = movie_comments['text'].apply(clean_text_for_sentiment)
            
            # Ensure sentiment column exists
            if 'sentiment' not in movie_comments.columns:
                analyzer = SentimentAnalyzer()
                sentiment_results = analyzer.analyze_sentiment(movie_comments['clean_text'])
                movie_comments['sentiment'] = sentiment_results['sentiment']
            
            # Create tabs for different summarization approaches
            tab1, tab2 = st.tabs(["Sentiment Summaries", "Cluster Summaries"])
            
            with tab1:
                st.subheader(f"Sentiment-based Summaries for {selected_movie}")
                
                # Generate summaries by sentiment
                with st.spinner("Generating sentiment summaries..."):
                    sentiment_summaries = summarize_by_sentiment(
                        movie_comments, text_column='clean_text', sentiment_column='sentiment'
                    )
                
                # Display summaries
                if 'positive' in sentiment_summaries:
                    st.subheader("Positive Comments Summary")
                    st.write(sentiment_summaries['positive'])
                
                if 'negative' in sentiment_summaries:
                    st.subheader("Negative Comments Summary")
                    st.write(sentiment_summaries['negative'])
                
                if 'neutral' in sentiment_summaries:
                    st.subheader("Neutral Comments Summary")
                    st.write(sentiment_summaries['neutral'])
            
            with tab2:
                st.subheader(f"Cluster-based Summaries for {selected_movie}")
                
                # Check if clustering has been done
                if 'cluster' not in movie_comments.columns:
                    # Run clustering
                    with st.spinner("Clustering comments..."):
                        df_with_clusters, cluster_descriptions, cluster_terms, _ = cluster_comments(
                            movie_comments, text_column='clean_text', n_clusters=5
                        )
                else:
                    df_with_clusters = movie_comments
                    
                    # Generate or retrieve cluster descriptions
                    if 'cluster_descriptions' in st.session_state:
                        cluster_descriptions = st.session_state.cluster_descriptions
                    else:
                        # Simple default descriptions
                        cluster_descriptions = {i: f"Cluster {i}" for i in df_with_clusters['cluster'].unique()}
                
                # Generate summaries by cluster
                with st.spinner("Generating cluster summaries..."):
                    cluster_summaries = summarize_by_cluster(
                        df_with_clusters, text_column='clean_text', 
                        cluster_column='cluster', cluster_descriptions=cluster_descriptions
                    )
                
                # Display summaries
                for cluster_id, summary in cluster_summaries.items():
                    with st.expander(f"Cluster {cluster_id} Summary"):
                        st.write(summary)
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
        # Run the integrated analysis pipeline
        with st.spinner("Analyzing data... This may take a moment."):
            # Get data from session state
            comments_df = st.session_state.comments_df
            movies_df = st.session_state.movies_df
            
            # Run the analysis pipeline
            pipeline_results = run_integrated_analysis_pipeline(comments_df, movies_df)
            
            # Store updated dataframes in session state
            st.session_state.comments_df = pipeline_results['comments_df']
            st.session_state.movies_df = pipeline_results['movies_df']
            
            # Store results in session state for reuse
            st.session_state.pipeline_results = pipeline_results
        
        # Layout the dashboard
        st.subheader("TrailerInsight Dashboard")
        
        # Display metrics at the top
        metrics = pipeline_results['dashboard_metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", metrics.get("total_movies", 0))
        
        with col2:
            st.metric("Total Comments", f"{metrics.get('total_comments', 0):,}")
        
        with col3:
            st.metric("Positive Comments", f"{metrics.get('positive_pct', 0):.1f}%")
        
        with col4:
            r_squared = metrics.get("sentiment_correlation", 0)
            st.metric("Sentiment-Revenue R²", f"{r_squared:.2f}")
        
        # Create tabs for visualizations and insights
        tab1, tab2 = st.tabs(["Sentiment-Revenue Analysis", "Cluster Analysis"])
        
        with tab1:
            st.subheader("Sentiment Impact on Box Office Performance")
            
            # Show correlation visualization
            if "correlation_results" in pipeline_results and "scatter_plot" in pipeline_results["correlation_results"]:
                st.plotly_chart(
                    pipeline_results["correlation_results"]["scatter_plot"],
                    use_container_width=True
                )
                
                # Show correlation values
                if "correlations" in pipeline_results["correlation_results"]:
                    correlations = pipeline_results["correlation_results"]["correlations"]
                    st.subheader("Sentiment-Revenue Correlations")
                    
                    corr_df = pd.DataFrame({
                        "Metric": list(correlations.keys()),
                        "Correlation": list(correlations.values())
                    })
                    st.dataframe(corr_df)
            else:
                st.info("No correlation analysis available. Make sure movies have revenue data.")
        
        with tab2:
            st.subheader("Cluster Analysis")
            
            # Show cluster distributions
            if "cluster_descriptions" in pipeline_results:
                # Display cluster descriptions
                st.subheader("Cluster Themes")
                
                for cluster_id, description in pipeline_results["cluster_descriptions"].items():
                    st.write(f"**Cluster {cluster_id}:** {description}")
                
                # Display cluster summaries
                if "cluster_summaries" in pipeline_results:
                    st.subheader("Cluster Summaries")
                    
                    for cluster_id, summary in pipeline_results["cluster_summaries"].items():
                        with st.expander(f"Cluster {cluster_id} Summary"):
                            st.write(summary)
            else:
                st.info("No cluster analysis available.")