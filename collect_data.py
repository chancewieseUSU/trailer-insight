# collect_data.py
import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from src.api.youtube import YouTubeClient
from src.api.movie_db import MovieDBClient
from src.preprocessing.clean_text import TextCleaner, clean_text_for_sentiment
from src.models.sentiment import SentimentAnalyzer

# Import the collection function
from data_collection_functions import collect_movie_dataset, process_sentiment_stats

# Set page configuration
st.set_page_config(
    page_title="TrailerInsight: Data Collection",
    page_icon="🎬",
    layout="wide"
)

# Title and introduction
st.title("TrailerInsight: Automated Data Collection")
st.markdown("This utility will automatically collect data for movies with trailers and box office information.")

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
        # Override the collection function to update Streamlit progress
        def collect_with_progress():
            # Initialize API clients
            youtube_client = YouTubeClient()
            movie_db_client = MovieDBClient()
            
            # Initialize storage for data
            all_comments = []
            all_movies = []
            
            # Initialize counters and progress tracking
            movies_processed = 0
            movies_collected = 0
            page = 1
            max_pages = 20  # Limit the maximum number of pages to search
            
            status_text.text(f"Starting data collection for {min_movies} movies with {min_comments}+ comments each...")
            
            # Create directory if it doesn't exist
            os.makedirs(save_directory, exist_ok=True)
            
            # Continue until we have enough movies or reach max pages
            while movies_collected < min_movies and page <= max_pages:
                status_text.text(f"Fetching popular movies page {page}...")
                
                # Get popular movies from TMDB
                popular_movies = movie_db_client.get_popular_movies(page=page, max_results=20)
                
                if not popular_movies:
                    status_text.text("No more movies available from API.")
                    break
                
                # Process each movie
                for i, movie in enumerate(popular_movies):
                    movie_title = movie['title']
                    movies_processed += 1
                    
                    status_text.text(f"[{movies_processed}] Processing: {movie_title} ({movie.get('release_date', 'N/A')[:4] if movie.get('release_date') else 'N/A'})")
                    
                    # Get TMDB movie details including revenue info
                    tmdb_details = movie_db_client.get_movie_by_id_tmdb(movie['id'])
                    
                    # Check if movie has revenue/budget data from TMDB
                    has_box_office = False
                    if tmdb_details and require_box_office:
                        revenue = tmdb_details.get('revenue', 0)
                        budget = tmdb_details.get('budget', 0)
                        has_box_office = revenue > 0
                        
                        if not has_box_office:
                            status_text.text(f"  - No box office data available on TMDB, trying OMDB...")
                            # Try OMDB as fallback for box office data
                            omdb_details = movie_db_client.get_movie_details_omdb(movie_title)
                            if omdb_details and omdb_details.get('Response') == 'True':
                                box_office = omdb_details.get('BoxOffice', 'N/A')
                                if box_office != 'N/A':
                                    has_box_office = True
                                    status_text.text(f"  - Found box office data on OMDB: {box_office}")
                                else:
                                    status_text.text(f"  - No box office data available on OMDB either.")
                        else:
                            status_text.text(f"  - Found box office data on TMDB: ${revenue:,}")
                    
                    # Skip movies without box office data if required
                    if require_box_office and not has_box_office:
                        status_text.text(f"  - Skipping (no box office data)")
                        continue
                    
                    # Get trailer ID
                    trailer_id = movie_db_client.get_movie_trailer(movie['id'])
                    if not trailer_id:
                        status_text.text(f"  - Skipping (no trailer available)")
                        continue
                    
                    # Get trailer comments
                    status_text.text(f"  - Found trailer (ID: {trailer_id}), fetching comments...")
                    comments = youtube_client.get_video_comments(trailer_id, max_results=300)
                    
                    if not comments:
                        status_text.text(f"  - Skipping (no comments available)")
                        continue
                    
                    comment_count = len(comments)
                    status_text.text(f"  - Found {comment_count} comments")
                    
                    # Skip if not enough comments
                    if comment_count < min_comments:
                        status_text.text(f"  - Skipping (insufficient comments, need {min_comments})")
                        continue
                    
                    # Process comments
                    comments_df = youtube_client.comments_to_dataframe(comments)
                    comments_df['movie'] = movie_title
                    comments_df['movie_id'] = movie['id']
                    
                    # Clean comments
                    status_text.text(f"  - Cleaning and processing comments...")
                    text_cleaner = TextCleaner()
                    comments_df['clean_text'] = comments_df['text'].apply(clean_text_for_sentiment)
                    
                    # Add sentiment analysis
                    analyzer = SentimentAnalyzer(method='textblob')
                    sentiment_results = analyzer.analyze_sentiment(
                        comments_df['clean_text'],
                        movie_names=[movie_title] * len(comments_df)
                    )
                    comments_df['sentiment'] = sentiment_results['sentiment']
                    comments_df['polarity'] = sentiment_results['polarity']
                    comments_df['subjectivity'] = sentiment_results['subjectivity']
                    
                    # Add to collection
                    all_comments.append(comments_df)
                    
                    # Prepare movie metadata
                    movie_data = {
                        'title': movie_title,
                        'tmdb_id': movie['id'],
                        'release_date': movie.get('release_date'),
                        'trailer_id': trailer_id,
                        'poster_path': movie.get('poster_path'),
                        'overview': movie.get('overview'),
                        'comment_count': comment_count
                    }
                    
                    # Add TMDB details
                    if tmdb_details:
                        movie_data.update({
                            'runtime': tmdb_details.get('runtime'),
                            'genres': ', '.join([g['name'] for g in tmdb_details.get('genres', [])]),
                            'production_companies': ', '.join([c['name'] for c in tmdb_details.get('production_companies', [])]),
                            'tmdb_vote_average': tmdb_details.get('vote_average'),
                            'tmdb_vote_count': tmdb_details.get('vote_count'),
                            'budget': tmdb_details.get('budget'),
                            'revenue': tmdb_details.get('revenue'),
                            'popularity': tmdb_details.get('popularity')
                        })
                    
                    # Add OMDB details (as supplement)
                    omdb_details = movie_db_client.get_movie_details_omdb(movie_title)
                    if omdb_details and omdb_details.get('Response') == 'True':
                        movie_data.update({
                            'box_office': omdb_details.get('BoxOffice'),
                            'imdb_rating': omdb_details.get('imdbRating'),
                            'imdb_votes': omdb_details.get('imdbVotes'),
                            'imdb_id': omdb_details.get('imdbID'),
                            'metacritic': omdb_details.get('Metascore'),
                            'rated': omdb_details.get('Rated'),
                            'awards': omdb_details.get('Awards'),
                            'director': omdb_details.get('Director'),
                            'actors': omdb_details.get('Actors')
                        })
                        
                        # Try to extract box office revenue from OMDB if not available from TMDB
                        if movie_data.get('revenue', 0) == 0 and omdb_details.get('BoxOffice'):
                            try:
                                # Remove currency symbols and commas
                                box_office_str = omdb_details.get('BoxOffice', '$0').replace('$', '').replace(',', '')
                                box_office_value = int(box_office_str)
                                movie_data['revenue'] = box_office_value
                            except (ValueError, TypeError):
                                pass
                    
                    # Add to movie collection
                    all_movies.append(movie_data)
                    movies_collected += 1
                    
                    # Update progress bar
                    progress_bar.progress(min(movies_collected / min_movies, 1.0))
                    
                    status_text.text(f"  ✓ Successfully collected data ({movies_collected}/{min_movies})")
                    
                    # Save data incrementally to prevent data loss
                    if movies_collected % 10 == 0 or movies_collected >= min_movies:
                        # Save collected data so far
                        tmp_comments_df = pd.concat(all_comments, ignore_index=True) if all_comments else pd.DataFrame()
                        tmp_movies_df = pd.DataFrame(all_movies) if all_movies else pd.DataFrame()
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                        
                        # Save the temporary results
                        tmp_comments_df.to_csv(f"{save_directory}/comments_processed_partial_{timestamp}.csv", index=False)
                        tmp_movies_df.to_csv(f"{save_directory}/movies_partial_{timestamp}.csv", index=False)
                        
                        status_text.text(f"Interim data saved: {movies_collected} movies collected so far")
                    
                    # Break if we have collected enough movies
                    if movies_collected >= min_movies:
                        break
                    
                    # Sleep to avoid hitting API rate limits
                    time.sleep(2)
                
                # Move to next page if we haven't collected enough movies
                if movies_collected < min_movies:
                    page += 1
                    time.sleep(5)  # Longer pause between pages
            
            # Combine all collected data
            if all_comments and all_movies:
                comments_df = pd.concat(all_comments, ignore_index=True)
                movies_df = pd.DataFrame(all_movies)
                
                # Save final dataset
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                
                # Save with timestamp
                comments_df.to_csv(f"{save_directory}/comments_processed_{timestamp}.csv", index=False)
                movies_df.to_csv(f"{save_directory}/movies_{timestamp}.csv", index=False)
                
                # Save without timestamp (canonical version for the app)
                comments_df.to_csv(f"{save_directory}/comments_processed.csv", index=False)
                movies_df.to_csv(f"{save_directory}/movies.csv", index=False)
                
                # Store in session state for display
                st.session_state.comments_df = comments_df
                st.session_state.movies_df = movies_df
                
                # Process sentiment stats
                movies_with_stats = process_sentiment_stats(comments_df, movies_df)
                st.session_state.movies_with_stats = movies_with_stats
                
                # Save stats
                movies_with_stats.to_csv(f"{save_directory}/movies_with_stats_{timestamp}.csv", index=False)
                movies_with_stats.to_csv(f"{save_directory}/movies_with_stats.csv", index=False)
                
                # Clear any cached box office analysis
                if 'box_office_analysis' in st.session_state:
                    del st.session_state.box_office_analysis
                
                status_text.text(f"Data collection complete: {movies_collected} movies with {len(comments_df)} total comments")
                
                return comments_df, movies_df, movies_with_stats
            else:
                status_text.text("No data collected. Please check API keys and connectivity.")
                return None, None, None
        
        # Run collection
        comments_df, movies_df, movies_with_stats = collect_with_progress()
        
        # Complete progress bar
        progress_bar.progress(1.0)
        
        if comments_df is not None and movies_df is not None:
            # Show success message and data preview
            st.success(f"Successfully collected data for {len(movies_df)} movies with {len(comments_df)} total comments!")
            
            # Stats tabs
            tab1, tab2, tab3 = st.tabs(["Movie Stats", "Comment Stats", "Data Preview"])
            
            with tab1:
                st.subheader("Movie Statistics")
                # Display movie stats
                st.write(f"Total movies collected: {len(movies_df)}")
                
                # Box office stats
                box_office_count = sum(1 for m in movies_df['revenue'] if m and m > 0)
                st.write(f"Movies with box office data: {box_office_count} ({box_office_count/len(movies_df):.1%})")
                
                # Genre distribution
                genres = []
                for g in movies_df['genres']:
                    if pd.notna(g):
                        genres.extend([genre.strip() for genre in g.split(',')])
                
                genre_counts = pd.Series(genres).value_counts()
                st.bar_chart(genre_counts)
            
            with tab2:
                st.subheader("Comment Statistics")
                # Display comment stats
                st.write(f"Total comments collected: {len(comments_df)}")
                
                # Comment distribution by movie
                comment_counts = comments_df['movie'].value_counts()
                st.write(f"Average comments per movie: {comment_counts.mean():.1f}")
                st.bar_chart(comment_counts)
                
                # Sentiment distribution
                sentiment_counts = comments_df['sentiment'].value_counts()
                st.write("Overall Sentiment Distribution")
                st.write(sentiment_counts)
                
                # Calculate overall sentiment percentages
                sentiment_pct = sentiment_counts / len(comments_df) * 100
                st.write(f"Positive: {sentiment_pct.get('positive', 0):.1f}%")
                st.write(f"Negative: {sentiment_pct.get('negative', 0):.1f}%")
                st.write(f"Neutral: {sentiment_pct.get('neutral', 0):.1f}%")
            
            with tab3:
                st.subheader("Data Preview")
                # Display data previews
                st.write("Movies DataFrame (first 10 rows)")
                st.dataframe(movies_df.head(10))
                
                st.write("Comments DataFrame (first 10 rows)")
                st.dataframe(comments_df.head(10))
                
            # Try to run box office analysis immediately
            status_text = st.empty()
            status_text.text("Running box office analysis...")
            with st.spinner("Analyzing box office data..."):
                # Run the analysis pipeline to precompute box office insights
                from src.visualization.sentiment_viz import analyze_sentiment_revenue_correlation
                from src.visualization.dashboard_viz import create_dashboard_metrics
                
                # Filter movies with revenue data
                valid_movies = movies_df[movies_df['revenue'] > 0].copy()
                
                if len(valid_movies) > 0:
                    # Add sentiment data from comments
                    for movie_title in valid_movies['title'].unique():
                        movie_comments = comments_df[comments_df['movie'] == movie_title]
                        
                        if len(movie_comments) > 0:
                            # Calculate metrics
                            pos_pct = (movie_comments['sentiment'] == 'positive').mean() * 100
                            neg_pct = (movie_comments['sentiment'] == 'negative').mean() * 100
                            
                            # Add to movies dataframe
                            valid_movies.loc[valid_movies['title'] == movie_title, 'positive_pct'] = pos_pct
                            valid_movies.loc[valid_movies['title'] == movie_title, 'negative_pct'] = neg_pct
                            
                            # Add polarity if available
                            if 'polarity' in movie_comments.columns:
                                avg_polarity = movie_comments['polarity'].mean()
                                valid_movies.loc[valid_movies['title'] == movie_title, 'avg_polarity'] = avg_polarity
                    
                    # Calculate correlation
                    correlation_results = analyze_sentiment_revenue_correlation(valid_movies)
                    
                    # Create dashboard metrics
                    dashboard_metrics = create_dashboard_metrics(valid_movies, comments_df, correlation_results)
                    
                    # Store in session state with timestamp to prevent caching issues
                    import time
                    timestamp = time.time()
                    st.session_state.box_office_analysis = {
                        'correlation_results': correlation_results,
                        'dashboard_metrics': dashboard_metrics,
                        'valid_movies': valid_movies,
                        'timestamp': timestamp  # Add timestamp to force refresh
                    }
                    
                    status_text.text("Box office analysis complete!")
        else:
            st.error("Data collection failed or no movies met the criteria.")
    
    except Exception as e:
        st.error(f"An error occurred during data collection: {str(e)}")
        raise e

# If data exists in session state, show download buttons
if 'comments_df' in st.session_state and 'movies_df' in st.session_state:
    st.subheader("Download Collected Data")
    
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