# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Import the necessary modules
from src.preprocessing.clean_text import clean_text_for_sentiment
from src.models.sentiment import SentimentAnalyzer, get_top_comments
from src.models.clustering import cluster_comments, create_cluster_visualization
from src.models.summarization import summarize_by_sentiment, summarize_by_cluster
from src.visualization.sentiment_viz import create_sentiment_distribution_pie, analyze_sentiment_revenue_correlation
from src.visualization.dashboard_viz import create_dashboard_metrics
from data_collection_functions import collect_movie_dataset, process_sentiment_stats

# Set page configuration
st.set_page_config(
    page_title="TrailerInsight",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state for data persistence
if 'comments_df' not in st.session_state:
    data_path = os.path.join('data', 'processed', 'comments_processed.csv')
    if os.path.exists(data_path):
        st.session_state.comments_df = pd.read_csv(data_path)
    else:
        st.session_state.comments_df = None

if 'movies_df' not in st.session_state:
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
                comments_df, movies_df = collect_movie_dataset(
                    min_movies=min_movies,
                    min_comments=min_comments,
                    save_path='data/processed',
                    progress_callback=progress_callback
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
                    
                    # Clear any cached box office analysis
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
                        
                        # Filter movies with revenue data
                        valid_movies = movies_df[movies_df['revenue'] > 0].copy()
                        
                        if len(valid_movies) > 0:
                            # Calculate sentiment metrics per movie
                            movie_sentiment = {}
                            
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
                            
                            # Store all results in session state with a timestamp
                            import time
                            timestamp = time.time()
                            st.session_state.box_office_analysis = {
                                'correlation_results': correlation_results,
                                'dashboard_metrics': dashboard_metrics,
                                'valid_movies': valid_movies,
                                'movie_sentiment': movie_sentiment,
                                'timestamp': timestamp
                            }
                    
                    status_text.text("Data collection and analysis complete!")
                    st.success(f"Successfully collected data for {len(movies_df)} movies with {len(comments_df)} total comments!")
                else:
                    st.error("Data collection failed or no movies met the criteria.")
        
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
                movie_names=filtered_df['movie']
            )
            filtered_df['sentiment'] = sentiment_results['sentiment']
            filtered_df['polarity'] = sentiment_results['polarity']
            
            st.success("Sentiment analysis completed!")
        
        # Create tabs for different visualizations
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
                x=cluster_viz['short_labels'],
                y=cluster_viz['counts'].values,
                labels={"x": "Theme", "y": "Comment Count"},
                title="Comment Themes Distribution",
                color=cluster_viz['counts'].values,
                color_continuous_scale="Viridis",
                text=cluster_viz['counts'].values
            )
            
            # Update layout
            fig.update_layout(
                xaxis_tickangle=-30,
                showlegend=False,
                height=500
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display theme descriptions
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
                    if "positive" in row and row["positive"] > 0.6:
                        theme_name = cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}")
                        theme_name = theme_name.split(':')[0] if ':' in theme_name else theme_name
                        positive_themes.append(theme_name)
                    
                    if "negative" in row and row["negative"] > 0.6:
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
                        n_sentences=3
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
                        'positive': '#d4f1d4',
                        'negative': '#ffebee',
                        'neutral': '#e3f2fd'
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
                        n_sentences=3,
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
                        
                        # Sample comments - Check if 'sample_comments' exists before trying to access it
                        # Fix for the KeyError
                        if 'sample_comments' in data and data['sample_comments']:
                            st.markdown("### Sample Comments")
                            for i, comment in enumerate(data['sample_comments']):
                                st.markdown(f"**{i+1}.** {comment}")
                        else:
                            # Get sample comments from the cluster
                            sample_comments = df_with_clusters[df_with_clusters['cluster'] == cluster_id]['text'].sample(
                                min(3, len(df_with_clusters[df_with_clusters['cluster'] == cluster_id]))
                            ).tolist()
                            
                            if sample_comments:
                                st.markdown("### Sample Comments")
                                for i, comment in enumerate(sample_comments):
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
        tab1, tab2 = st.tabs(["Prediction Analysis", "Per-Movie Details"])
        
        # Get data from session state
        comments_df = st.session_state.comments_df
        movies_df = st.session_state.movies_df
        
        # Filter movies with revenue data
        valid_movies = movies_df[movies_df['revenue'] > 0].copy()
        
        # Calculate basic metrics per movie
        movie_metrics = {}
        
        for movie_title in valid_movies['title'].unique():
            movie_comments = comments_df[comments_df['movie'] == movie_title]
            
            if len(movie_comments) == 0:
                continue
                
            # Calculate like metrics if available
            if 'likes' in movie_comments.columns:
                total_likes = movie_comments['likes'].sum()
                
                # Calculate comment-to-like ratio
                comment_count = len(movie_comments)
                comment_like_ratio = comment_count / total_likes if total_likes > 0 else 0
            else:
                total_likes = 0
                comment_like_ratio = 0
                
            # Calculate sentiment metrics
            if 'sentiment' in movie_comments.columns:
                pos_pct = (movie_comments['sentiment'] == 'positive').mean() * 100
                
                # Calculate sentiment consistency
                # If polarity values are available, use standard deviation of polarity
                if 'polarity' in movie_comments.columns:
                    sentiment_consistency = 1 - movie_comments['polarity'].std()  # Higher value means more consistent
                else:
                    # Estimate consistency from sentiment categories
                    neutral_pct = (movie_comments['sentiment'] == 'neutral').mean() * 100
                    neg_pct = (movie_comments['sentiment'] == 'negative').mean() * 100
                    sentiment_consistency = 1 - (abs(pos_pct - 50) + abs(neg_pct - 25) + abs(neutral_pct - 25)) / 100
            else:
                pos_pct = 0
                sentiment_consistency = 0
                
            # Store metrics
            movie_metrics[movie_title] = {
                'like_count': total_likes,
                'positive_pct': pos_pct,
                'comment_like_ratio': comment_like_ratio,
                'sentiment_consistency': sentiment_consistency
            }
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(movie_metrics, orient='index').reset_index()
        metrics_df.rename(columns={'index': 'title'}, inplace=True)
        
        # Merge with valid_movies
        prediction_df = pd.merge(valid_movies[['title', 'revenue']], metrics_df, on='title', how='inner')
        
        # Create predictive score
        prediction_df['prediction_score'] = (
            0.3 * prediction_df['positive_pct'] / 100 +
            0.3 * (1 - prediction_df['comment_like_ratio'] / prediction_df['comment_like_ratio'].max() if prediction_df['comment_like_ratio'].max() > 0 else 0) +  # Inverse since lower ratio is better
            0.2 * prediction_df['sentiment_consistency'] + 
            0.2 * prediction_df['like_count'] / prediction_df['like_count'].max() if prediction_df['like_count'].max() > 0 else 0
        )
        
        # Calculate correlations with revenue
        correlations = {}
        metrics_to_check = [
            ('positive_pct', 'Positive Sentiment %'),
            ('comment_like_ratio', 'Comment-to-Like Ratio'),
            ('sentiment_consistency', 'Sentiment Consistency'),
            ('prediction_score', 'Prediction Score')
        ]
        
        for col, name in metrics_to_check:
            if col in prediction_df.columns:
                correlations[name] = prediction_df[col].corr(prediction_df['revenue'])
        
        # Sort correlations by absolute value
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        with tab1:
            st.subheader("Box Office Prediction Analysis")
            
            # Top metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Movies Analyzed", len(prediction_df))
            
            with col2:
                top_predictor = sorted_correlations[0][0]
                top_corr = sorted_correlations[0][1]
                st.metric("Best Predictor", top_predictor)
            
            with col3:
                st.metric("Correlation Strength", f"{top_corr:.2f}")
            
            # Explanation of metrics
            st.subheader("Understanding the Metrics")
            
            st.markdown("""
            | Metric | Description | Relationship to Box Office |
            | ------ | ----------- | -------------------------- |
            | **Positive Sentiment %** | Percentage of comments classified as positive | More positive reception might drive ticket sales |
            | **Comment-to-Like Ratio** | Number of comments relative to likes | Lower ratios (fewer comments per like) may indicate less controversy |
            | **Sentiment Consistency** | How uniform sentiment is across comments | Higher consistency may indicate clearer audience reception |
            | **Prediction Score** | Combined score using several metrics | Holistic measure of audience enthusiasm |
            """)
            
            # Correlation chart
            st.subheader("Metric Correlations with Box Office Revenue")
            
            corr_data = pd.DataFrame({
                'Metric': [name for name, _ in sorted_correlations],
                'Correlation': [corr for _, corr in sorted_correlations]
            })
            
            fig = px.bar(
                corr_data,
                x='Metric',
                y='Correlation',
                color='Correlation',
                color_continuous_scale='RdBu_r',
                title="Which Metrics Best Predict Box Office Revenue?",
                labels={'Correlation': 'Correlation Coefficient'}
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create scatter plot for prediction score
            st.subheader("Prediction Score Analysis")
            
            prediction_fig = px.scatter(
                prediction_df,
                x='prediction_score',
                y='revenue',
                hover_data=['title'],
                labels={
                    'prediction_score': 'Prediction Score',
                    'revenue': 'Box Office Revenue ($)',
                    'title': 'Movie'
                },
                title=f"Prediction Score vs. Box Office Revenue"
            )
            
            # Calculate R²
            pred_r_squared = correlations['Prediction Score'] ** 2
            
            # Add annotation
            prediction_fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Correlation: {correlations['Prediction Score']:.2f} (R²: {pred_r_squared:.2f})",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            # Add simple trendline
            x_values = prediction_df['prediction_score']
            y_values = prediction_df['revenue']
            if len(x_values) > 1:
                x_min, x_max = min(x_values), max(x_values)
                
                # Simple linear fit (using averages to approximate)
                x_mean = sum(x_values) / len(x_values)
                y_mean = sum(y_values) / len(y_values)
                
                # Calculate slope (covariance / variance)
                numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
                denominator = sum((x - x_mean) ** 2 for x in x_values)
                
                if denominator != 0:
                    slope = numerator / denominator
                    intercept = y_mean - slope * x_mean
                    
                    y_min = slope * x_min + intercept
                    y_max = slope * x_max + intercept
                    
                    # Add line
                    prediction_fig.add_shape(
                        type="line",
                        x0=x_min,
                        y0=y_min,
                        x1=x_max,
                        y1=y_max,
                        line=dict(color="Red", width=2, dash="dash")
                    )
            
            st.plotly_chart(prediction_fig, use_container_width=True)
            
            # Explanation of prediction score
            st.info("""
            ### Prediction Score

            Our combined prediction score weights multiple factors to create a single predictive metric:
            
            - **30%** Positive sentiment percentage (genuine audience enthusiasm)
            - **30%** Inverse comment-to-like ratio (fewer comments per like indicates less controversy)
            - **20%** Sentiment consistency (uniform audience reactions)
            - **20%** Like count (overall engagement level)
            
            This combined metric has shown the strongest correlation with box office success.
            """)
            
            # Prediction data table
            st.subheader("Revenue Prediction Data")
            
            # Prepare data for display
            display_df = prediction_df.copy()
            display_df['revenue_millions'] = display_df['revenue'] / 1000000
            
            # Select and format columns
            table_cols = [
                'title', 'revenue_millions', 'prediction_score', 
                'positive_pct', 'like_count', 
                'comment_like_ratio', 'sentiment_consistency'
            ]
            
            display_cols = {
                'title': 'Movie',
                'revenue_millions': 'Revenue ($M)',
                'prediction_score': 'Prediction Score',
                'positive_pct': 'Positive Sentiment (%)',
                'like_count': 'Likes',
                'comment_like_ratio': 'Comment-to-Like Ratio',
                'sentiment_consistency': 'Sentiment Consistency'
            }
            
            # Format and round numbers
            final_df = display_df[table_cols].copy()
            final_df['revenue_millions'] = final_df['revenue_millions'].round(1)
            final_df['prediction_score'] = final_df['prediction_score'].round(2)
            final_df['positive_pct'] = final_df['positive_pct'].round(1)
            final_df['comment_like_ratio'] = final_df['comment_like_ratio'].round(4)
            final_df['sentiment_consistency'] = final_df['sentiment_consistency'].round(2)
            
            # Rename columns
            final_df.columns = [display_cols.get(col, col) for col in final_df.columns]
            
            # Sort by revenue
            final_df = final_df.sort_values('Revenue ($M)', ascending=False)
            
            # Display table
            st.dataframe(final_df, use_container_width=True)
        
        with tab2:
            st.subheader("Per-Movie Analysis")
            
            # Select movie for analysis
            movie_list = prediction_df['title'].tolist()
            selected_movie = st.selectbox(
                "Select a movie to analyze",
                options=movie_list,
                index=0 if movie_list else None
            )
            
            if selected_movie:
                # Get movie data
                movie_data = valid_movies[valid_movies['title'] == selected_movie].iloc[0]
                movie_metrics = prediction_df[prediction_df['title'] == selected_movie].iloc[0]
                movie_comments = comments_df[comments_df['movie'] == selected_movie]
                
                # Movie header
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"## {movie_data['title']}")
                    
                    # Description if available
                    if 'overview' in movie_data and not pd.isna(movie_data['overview']):
                        st.markdown(f"**Description:** {movie_data['overview']}")
                    
                    # Genre if available
                    if 'genres' in movie_data and not pd.isna(movie_data['genres']):
                        st.markdown(f"**Genres:** {movie_data['genres']}")
                
                with col2:
                    # Movie poster if available
                    if 'poster_path' in movie_data and not pd.isna(movie_data['poster_path']):
                        poster_url = f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}"
                        st.image(poster_url, width=150)
                
                # Key metrics
                st.subheader("Box Office & Prediction Metrics")
                
                # Display in 3 columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    revenue_millions = movie_metrics['revenue'] / 1000000
                    st.metric("Box Office Revenue", f"${revenue_millions:.1f}M")
                
                with col2:
                    st.metric("Prediction Score", f"{movie_metrics['prediction_score']:.2f}")
                
                with col3:
                    relative_prediction = (movie_metrics['prediction_score'] / prediction_df['prediction_score'].mean() - 1) * 100
                    st.metric("Vs. Average", f"{relative_prediction:.1f}%", delta_color="normal" if relative_prediction >= 0 else "inverse")
                
                # Key metrics
                st.subheader("Key Metrics")
                
                metrics_cols = st.columns(4)
                
                with metrics_cols[0]:
                    st.metric("Like Count", f"{int(movie_metrics['like_count']):,}")
                
                with metrics_cols[1]:
                    st.metric("Positive Sentiment", f"{movie_metrics['positive_pct']:.1f}%")
                
                with metrics_cols[2]:
                    st.metric("Comment-to-Like Ratio", f"{movie_metrics['comment_like_ratio']:.4f}")
                
                with metrics_cols[3]:
                    st.metric("Sentiment Consistency", f"{movie_metrics['sentiment_consistency']:.2f}")
                
                # Sentiment breakdown
                st.subheader("Audience Sentiment")
                
                # Create sentiment chart
                sentiment_counts = movie_comments['sentiment'].value_counts()
                
                fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    title=f"Sentiment Distribution for {selected_movie}",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'neutral': '#f1c40f',
                        'negative': '#e74c3c'
                    }
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Top comments
                st.subheader("Top Audience Comments")
                
                # Get top comments by likes if available
                if 'likes' in movie_comments.columns:
                    top_comments = movie_comments.nlargest(5, 'likes')
                    
                    for i, row in top_comments.iterrows():
                        with st.container(border=True):
                            st.markdown(f"**Comment:** {row['text']}")
                            
                            # Show metrics in columns
                            metrics_cols = st.columns(2)
                            with metrics_cols[0]:
                                st.metric("Likes", f"{int(row['likes']):,}")
                            
                            with metrics_cols[1]:
                                st.metric("Sentiment", row['sentiment'].capitalize())
                else:
                    st.info("Like data not available for comments")