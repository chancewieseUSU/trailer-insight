# collect_data.py
import streamlit as st
import pandas as pd
import os
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
        min_movies = st.number_input("Minimum Number of Movies", min_value=10, max_value=100, value=20)
        min_comments = st.number_input("Minimum Comments per Trailer", min_value=50, max_value=500, value=100)
    
    with col2:
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
        # Define progress callback
        def progress_callback(movie_title, movies_collected, target_movies):
            progress = min(movies_collected / target_movies, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {movie_title} ({movies_collected}/{target_movies})")
        
        # Run collection
        comments_df, movies_df = collect_movie_dataset(
            min_movies=min_movies,
            min_comments=min_comments,
            save_path=save_directory,
            progress_callback=progress_callback
        )
        
        # Complete progress bar
        progress_bar.progress(1.0)
        
        if comments_df is not None and movies_df is not None:
            # Process sentiment stats
            movies_with_stats = process_sentiment_stats(comments_df, movies_df)
            
            # Show success message
            st.success(f"Successfully collected data for {len(movies_df)} movies with {len(comments_df)} total comments!")
            
            # Stats tabs
            tab1, tab2, tab3 = st.tabs(["Movie Stats", "Comment Stats", "Data Preview"])
            
            with tab1:
                st.subheader("Movie Statistics")
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
                st.write(f"Total comments collected: {len(comments_df)}")
                
                # Comment distribution by movie
                comment_counts = comments_df['movie'].value_counts()
                st.write(f"Average comments per movie: {comment_counts.mean():.1f}")
                st.bar_chart(comment_counts)
                
                # Sentiment distribution
                sentiment_counts = comments_df['sentiment'].value_counts()
                st.write("Overall Sentiment Distribution")
                
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
        else:
            st.error("Data collection failed or no movies met the criteria.")
    
    except Exception as e:
        st.error(f"An error occurred during data collection: {str(e)}")

# If data exists in files, show download buttons
data_path = os.path.join(save_directory, 'comments_processed.csv')
if os.path.exists(data_path):
    st.subheader("Download Collected Data")
    
    comments_df = pd.read_csv(data_path)
    movies_path = os.path.join(save_directory, 'movies.csv')
    movies_with_stats_path = os.path.join(save_directory, 'movies_with_stats.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        comments_csv = comments_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Comments Data (CSV)",
            comments_csv,
            "trailer_comments.csv",
            "text/csv",
            key="download-comments"
        )
    
    with col2:
        if os.path.exists(movies_with_stats_path):
            movies_df = pd.read_csv(movies_with_stats_path)
        elif os.path.exists(movies_path):
            movies_df = pd.read_csv(movies_path)
        else:
            movies_df = None
            
        if movies_df is not None:
            movies_csv = movies_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Movies Data (CSV)",
                movies_csv,
                "movie_data.csv",
                "text/csv",
                key="download-movies"
            )