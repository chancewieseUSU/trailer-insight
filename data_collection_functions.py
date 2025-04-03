# data_collection_functions.py
import os
import pandas as pd
import time
from datetime import datetime
from src.preprocessing.clean_text import TextCleaner, clean_text_for_sentiment
from src.models.sentiment import SentimentAnalyzer
from src.api.youtube import YouTubeClient
from src.api.movie_db import MovieDBClient

def collect_movie_dataset(min_movies=50, min_comments=100, include_box_office=True, save_path='data/processed'):
    """
    Automatically collects a dataset of movies with trailers and box office data.
    
    Parameters:
    min_movies (int): Minimum number of movies to collect
    min_comments (int): Minimum number of comments required per trailer
    include_box_office (bool): Whether to filter for movies with box office data
    save_path (str): Directory to save the collected data
    
    Returns:
    tuple: (comments_df, movies_df) - DataFrames containing the collected data
    """
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
    
    print(f"Starting data collection for {min_movies} movies with {min_comments}+ comments each...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Continue until we have enough movies or reach max pages
    while movies_collected < min_movies and page <= max_pages:
        print(f"\nFetching popular movies page {page}...")
        
        # Get popular movies from TMDB
        popular_movies = movie_db_client.get_popular_movies(page=page, max_results=20)
        
        if not popular_movies:
            print("No more movies available from API.")
            break
        
        # Process each movie
        for i, movie in enumerate(popular_movies):
            movie_title = movie['title']
            movies_processed += 1
            
            print(f"[{movies_processed}] Processing: {movie_title} ({movie.get('release_date', 'N/A')[:4] if movie.get('release_date') else 'N/A'})")
            
            # Get TMDB movie details including revenue info
            tmdb_details = movie_db_client.get_movie_by_id_tmdb(movie['id'])
            
            # Check if movie has revenue/budget data from TMDB
            has_box_office = False
            if tmdb_details and include_box_office:
                revenue = tmdb_details.get('revenue', 0)
                budget = tmdb_details.get('budget', 0)
                has_box_office = revenue > 0
                
                if not has_box_office:
                    print(f"  - No box office data available on TMDB, trying OMDB...")
                    # Try OMDB as fallback for box office data
                    omdb_details = movie_db_client.get_movie_details_omdb(movie_title)
                    if omdb_details and omdb_details.get('Response') == 'True':
                        box_office = omdb_details.get('BoxOffice', 'N/A')
                        if box_office != 'N/A':
                            has_box_office = True
                            print(f"  - Found box office data on OMDB: {box_office}")
                        else:
                            print(f"  - No box office data available on OMDB either.")
                else:
                    print(f"  - Found box office data on TMDB: ${revenue:,}")
            
            # Skip movies without box office data if required
            if include_box_office and not has_box_office:
                print(f"  - Skipping (no box office data)")
                continue
            
            # Get trailer ID
            trailer_id = movie_db_client.get_movie_trailer(movie['id'])
            if not trailer_id:
                print(f"  - Skipping (no trailer available)")
                continue
            
            # Get trailer comments
            print(f"  - Found trailer (ID: {trailer_id}), fetching comments...")
            comments = youtube_client.get_video_comments(trailer_id, max_results=300)
            
            if not comments:
                print(f"  - Skipping (no comments available)")
                continue
            
            comment_count = len(comments)
            print(f"  - Found {comment_count} comments")
            
            # Skip if not enough comments
            if comment_count < min_comments:
                print(f"  - Skipping (insufficient comments, need {min_comments})")
                continue
            
            # Process comments
            comments_df = youtube_client.comments_to_dataframe(comments)
            comments_df['movie'] = movie_title
            comments_df['movie_id'] = movie['id']
            
            # Clean comments
            print(f"  - Cleaning and processing comments...")
            text_cleaner = TextCleaner()
            comments_df['clean_text'] = comments_df['text'].apply(clean_text_for_sentiment)
            
            # Add sentiment analysis
            analyzer = SentimentAnalyzer(method='textblob')
            sentiment_results = analyzer.analyze_sentiment(comments_df['clean_text'])
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
            
            # Add to movie collection
            all_movies.append(movie_data)
            movies_collected += 1
            
            print(f"  ✓ Successfully collected data ({movies_collected}/{min_movies})")
            
            # Save data incrementally to prevent data loss
            if movies_collected % 10 == 0 or movies_collected >= min_movies:
                # Save collected data so far
                tmp_comments_df = pd.concat(all_comments, ignore_index=True) if all_comments else pd.DataFrame()
                tmp_movies_df = pd.DataFrame(all_movies) if all_movies else pd.DataFrame()
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                
                # Save the temporary results
                tmp_comments_df.to_csv(f"{save_path}/comments_processed_partial_{timestamp}.csv", index=False)
                tmp_movies_df.to_csv(f"{save_path}/movies_partial_{timestamp}.csv", index=False)
                
                print(f"\n--- Interim data saved: {movies_collected} movies collected so far ---\n")
            
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
        comments_df.to_csv(f"{save_path}/comments_processed_{timestamp}.csv", index=False)
        movies_df.to_csv(f"{save_path}/movies_{timestamp}.csv", index=False)
        
        print(f"\nData collection complete: {movies_collected} movies with {len(comments_df)} total comments")
        
        # Generate some basic statistics
        movies_with_box_office = sum(1 for m in all_movies if 
                                   (m.get('revenue', 0) > 0) or 
                                   (m.get('box_office', 'N/A') != 'N/A'))
        
        avg_comments = comments_df.groupby('movie')['comment_id'].count().mean()
        
        print(f"Movies with box office data: {movies_with_box_office} ({movies_with_box_office/movies_collected:.1%})")
        print(f"Average comments per movie: {avg_comments:.1f}")
        
        # Return the dataframes
        return comments_df, movies_df
    else:
        print("No data collected. Please check API keys and connectivity.")
        return None, None

def process_sentiment_stats(comments_df, movies_df):
    """
    Process sentiment statistics and add them to the movies dataframe.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data
    movies_df (DataFrame): DataFrame with movie data
    
    Returns:
    DataFrame: Updated movies dataframe with sentiment stats
    """
    if comments_df is None or movies_df is None:
        return None
    
    # Group comments by movie and calculate sentiment stats
    sentiment_stats = []
    
    for movie_title in movies_df['title']:
        movie_comments = comments_df[comments_df['movie'] == movie_title]
        
        # Calculate sentiment statistics
        total_comments = len(movie_comments)
        if total_comments == 0:
            continue
            
        positive_count = (movie_comments['sentiment'] == 'positive').sum()
        negative_count = (movie_comments['sentiment'] == 'negative').sum()
        neutral_count = total_comments - positive_count - negative_count
        
        positive_pct = positive_count / total_comments * 100
        negative_pct = negative_count / total_comments * 100
        neutral_pct = neutral_count / total_comments * 100
        
        avg_polarity = movie_comments['polarity'].mean()
        
        # Create stats dictionary
        stats = {
            'title': movie_title,
            'comment_count': total_comments,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_pct': positive_pct,
            'negative_pct': negative_pct,
            'neutral_pct': neutral_pct,
            'avg_polarity': avg_polarity
        }
        
        sentiment_stats.append(stats)
    
    # Convert to DataFrame
    sentiment_df = pd.DataFrame(sentiment_stats)
    
    # Merge with movies_df
    movies_with_stats = pd.merge(movies_df, sentiment_df, on='title', how='left')
    
    return movies_with_stats