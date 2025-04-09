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
import shutil
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
    
    st.info("This page allows you to collect movie trailer comments and correlate them with box office performance.")
    
    # Main collection form
    with st.form("collection_form"):
        st.subheader("Collection Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_movies = st.number_input("Minimum Number of Movies", min_value=10, max_value=100, value=50)
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
                
                # Update progress bar as data is collected
                for i in range(100):
                    # This simulates the progress
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                # Store data in session state
                if comments_df is not None and not comments_df.empty:
                    st.session_state.comments_df = comments_df
                    
                if movies_df is not None and not movies_df.empty:
                    st.session_state.movies_df = movies_df
                    
                    # Process sentiment stats
                    movies_with_stats = process_sentiment_stats(comments_df, movies_df)
                    st.session_state.movies_with_stats = movies_with_stats

                # Add data integration validation
                if comments_df is not None and movies_df is not None:
                    # Integrate datasets with validation
                    from data_collection_functions import integrate_datasets, verify_data_integration
                    
                    with st.spinner("Integrating and validating data..."):
                        # Integrate datasets
                        integrated_movies, filtered_comments = integrate_datasets(
                            comments_df, 
                            movies_df,
                            min_comments=min_comments // 2  # Use half the min_comments as threshold
                        )
                        
                        # Verify integration
                        integration_valid = verify_data_integration(integrated_movies, filtered_comments)
                        
                        if integration_valid:
                            # Update session state with integrated data
                            st.session_state.movies_df = integrated_movies
                            st.session_state.comments_df = filtered_comments
                            st.session_state.movies_with_stats = integrated_movies  # Already has stats
                            
                            st.success(f"Successfully integrated data for {len(integrated_movies)} movies with {len(filtered_comments)} total comments!")
                        else:
                            st.warning("Data integration completed with warnings. Some movies may have insufficient data.")
                            
                            # Still update session state but with a warning
                            st.session_state.movies_df = integrated_movies
                            st.session_state.comments_df = filtered_comments
                            st.session_state.movies_with_stats = integrated_movies
                
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
        
        # Hide sentiment methods behind Advanced Options
        sentiment_method = "TextBlob"  # Default method
        
        show_advanced = st.sidebar.checkbox("Show Advanced Options", value=False)
        if show_advanced:
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
        # Rest of the Comment Clusters code...
        # (This is a placeholder to keep the artifact size manageable)
        
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
        
        # Clustering logic...
        # (Rest of clustering functionality would go here)
        st.info("Select movies from the sidebar to begin clustering analysis")

elif page == "Summaries":
    st.header("Comment Summarization")
    
    # Check if data is available
    if 'comments_df' not in st.session_state or st.session_state.comments_df is None:
        st.warning("No data available. Please collect data in the Data Collection page.")
    else:
        # Rest of the Summaries code...
        # (This is a placeholder to keep the artifact size manageable)
        
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
        
        # Summarization logic...
        # (Rest of summarization functionality would go here)
        st.info("Select a movie from the sidebar to generate summaries")

elif page == "Box Office Insights":
    st.header("Box Office Prediction Insights")
    
    # Check if data is available and properly integrated
    if ('comments_df' not in st.session_state or 
        st.session_state.comments_df is None or 
        'movies_df' not in st.session_state or 
        st.session_state.movies_df is None):
        st.warning("No data available. Please collect data in the Data Collection page.")
    else:
        st.info("This section connects comment sentiment to box office performance.")
        
        # Use the integrated movies dataframe with sentiment stats
        if 'movies_with_stats' in st.session_state:
            movies_df = st.session_state.movies_with_stats
        else:
            movies_df = st.session_state.movies_df
            
            # Check if we need to calculate sentiment stats
            if 'avg_sentiment' not in movies_df.columns:
                st.warning("Sentiment statistics not found. Re-calculating now...")
                
                try:
                    from data_collection_functions import calculate_movie_sentiment_stats
                    movies_df = calculate_movie_sentiment_stats(
                        st.session_state.comments_df, 
                        movies_df
                    )
                    st.session_state.movies_with_stats = movies_df
                except Exception as e:
                    st.error(f"Error calculating sentiment stats: {str(e)}")
        
        # Add a simple data quality check
        valid_for_analysis = (
            'revenue' in movies_df.columns and 
            'avg_sentiment' in movies_df.columns and
            (movies_df['revenue'] > 0).any() and
            len(movies_df) >= 5  # Need at least 5 movies for meaningful analysis
        )
        
        if not valid_for_analysis:
            st.error("Insufficient data for box office analysis. Please collect more movie data with box office information.")
        else:
            # Data validation metrics
            total_movies = len(movies_df)
            movies_with_box_office = (movies_df['revenue'] > 0).sum()
            movies_with_sentiment = (~movies_df['avg_sentiment'].isna()).sum()
            
            # Show data quality metrics
            quality_col1, quality_col2, quality_col3 = st.columns(3)
            with quality_col1:
                st.metric("Total Movies", total_movies)
            with quality_col2:
                st.metric("With Box Office Data", movies_with_box_office)
            with quality_col3:
                st.metric("With Sentiment Data", movies_with_sentiment)

            # Filter to only movies with both sentiment stats and box office data
            has_box_office = movies_df['revenue'].notna() & (movies_df['revenue'] > 0)

            # Check for sentiment columns again after potential additions
            has_sentiment = all(col in movies_df.columns for col in ['avg_polarity'])

            valid_movies = movies_df[has_box_office]
            if has_sentiment:
                valid_movies = valid_movies[valid_movies['avg_polarity'].notna()]
                
            # Proceed with analysis if we have enough valid data
            if movies_with_box_office >= 5 and movies_with_sentiment >= 5:
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