# data_collection_functions.py
import os
import pandas as pd
import time
from datetime import datetime
from src.preprocessing.clean_text import TextCleaner, clean_text_for_sentiment
from src.models.sentiment import SentimentAnalyzer
from src.api.youtube import YouTubeClient
from src.api.movie_db import MovieDBClient

def collect_movie_dataset(min_movies=50, min_comments=100, include_box_office=True, clear_previous_data=False, save_path='data/processed', progress_callback=None):
    """
    Automatically collects a dataset of movies with trailers and box office data.
    
    Parameters:
    min_movies (int): Minimum number of movies to collect
    min_comments (int): Minimum number of comments required per trailer
    include_box_office (bool): Whether to filter for movies with box office data (default: True)
    clear_previous_data (bool): Whether to clear previous data files before saving new data
    save_path (str): Directory to save the collected data
    progress_callback (function): Optional callback function to report progress
    
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
    
    # Clear previous data if requested
    if clear_previous_data:
        for file in os.listdir(save_path):
            if file.endswith('.csv'):
                os.remove(os.path.join(save_path, file))
                print(f"Removed previous file: {file}")
    
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
                            # Try to convert box office string to numeric
                            try:
                                # Remove currency symbols and commas
                                box_office_clean = box_office.replace('$', '').replace(',', '')
                                box_office_value = int(box_office_clean)
                                
                                # Update TMDB details with OMDB box office
                                if tmdb_details:
                                    tmdb_details['revenue'] = box_office_value
                                
                                has_box_office = True
                                print(f"  - Found box office data on OMDB: {box_office}")
                            except (ValueError, AttributeError) as e:
                                print(f"  - Could not parse box office value '{box_office}': {e}")
                        else:
                            print(f"  - No box office data available on OMDB either.")
                else:
                    print(f"  - Found box office data on TMDB: ${revenue:,}")
            
            # Skip movies without box office data if required
            if include_box_office and not has_box_office:
                print(f"  - Skipping (no box office data)")
                continue
            
            # Get movie release date
            release_date = None
            if 'release_date' in movie and movie['release_date']:
                try:
                    release_date = datetime.strptime(movie['release_date'], '%Y-%m-%d')
                    print(f"  - Movie release date: {release_date.strftime('%Y-%m-%d')}")
                except (ValueError, TypeError) as e:
                    print(f"  - Could not parse release date: {e}")
                    release_date = None
            else:
                print(f"  - No release date available")
            
            # Get trailer ID
            trailer_id = movie_db_client.get_movie_trailer(movie['id'])
            if not trailer_id:
                print(f"  - Skipping (no trailer available)")
                continue
            
            # Get trailer comments
            print(f"  - Found trailer (ID: {trailer_id}), fetching comments...")
            comments = youtube_client.get_video_comments(
                trailer_id, 
                max_results=300,
                end_date=release_date  # Only get comments before release date
            )
            
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
            analyzer = SentimentAnalyzer()
            sentiment_results = analyzer.analyze_sentiment(
                comments_df['clean_text'],
                movie_names=comments_df['movie']  # Pass movie names for context
            )
            comments_df['sentiment'] = sentiment_results['sentiment']
            comments_df['polarity'] = sentiment_results['polarity']
            
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
            
            # Call progress callback if provided
            if progress_callback and callable(progress_callback):
                progress_callback(movie_title, movies_collected, min_movies)
            
            print(f"  ✓ Successfully collected data ({movies_collected}/{min_movies})")
            
            # Save data incrementally to prevent data loss
            if movies_collected % 10 == 0 or movies_collected >= min_movies:
                # Save collected data so far
                tmp_comments_df = pd.concat(all_comments, ignore_index=True) if all_comments else pd.DataFrame()
                tmp_movies_df = pd.DataFrame(all_movies) if all_movies else pd.DataFrame()
                
                # Save the temporary results
                tmp_comments_df.to_csv(f"{save_path}/comments_processed_partial.csv", index=False)
                tmp_movies_df.to_csv(f"{save_path}/movies_partial.csv", index=False)
                
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
        comments_df.to_csv(f"{save_path}/comments_processed.csv", index=False)
        movies_df.to_csv(f"{save_path}/movies.csv", index=False)
        
        print(f"\nData collection complete: {movies_collected} movies with {len(comments_df)} total comments")
        
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
    
    # Ensure there's data with revenue for box office analysis
    has_revenue = False
    if 'revenue' in movies_df.columns:
        has_revenue = (movies_df['revenue'] > 0).any()
    
    if not has_revenue:
        print("Warning: No movies have revenue data, box office insights will be limited")
    
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
        
        # Calculate average polarity if available
        if 'polarity' in movie_comments.columns:
            avg_polarity = movie_comments['polarity'].mean()
        else:
            # Estimate polarity from sentiment categories
            avg_polarity = (positive_count - negative_count) / total_comments
        
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
    
    # Ensure box office data is properly formatted
    if 'revenue' in movies_with_stats.columns:
        # Make sure revenue is numeric
        movies_with_stats['revenue'] = pd.to_numeric(movies_with_stats['revenue'], errors='coerce')
        
        # For movies with box_office string from OMDB but no revenue, try to extract revenue
        if 'box_office' in movies_with_stats.columns:
            for idx, row in movies_with_stats.iterrows():
                if (pd.isna(row['revenue']) or row['revenue'] == 0) and isinstance(row.get('box_office'), str):
                    try:
                        # Remove currency symbols and commas
                        box_office_str = row['box_office'].replace('$', '').replace(',', '')
                        box_office_value = int(box_office_str)
                        movies_with_stats.loc[idx, 'revenue'] = box_office_value
                    except (ValueError, TypeError, AttributeError):
                        pass
    
    # Fill NA values for movies without sentiment data
    for col in ['comment_count', 'positive_count', 'negative_count', 'neutral_count']:
        if col in movies_with_stats.columns:
            movies_with_stats[col] = movies_with_stats[col].fillna(0).astype(int)
    
    for col in ['positive_pct', 'negative_pct', 'neutral_pct', 'avg_polarity']:
        if col in movies_with_stats.columns:
            movies_with_stats[col] = movies_with_stats[col].fillna(0)
    
    return movies_with_stats

def integrate_datasets(comments_df, movies_df, min_comments=50):
    """
    Integrate comment data with movie box office data, filtering out movies with insufficient data.
    
    Parameters:
    comments_df (DataFrame): DataFrame containing comment data
    movies_df (DataFrame): DataFrame containing movie data with box office information
    min_comments (int): Minimum number of comments required for a movie to be included
    
    Returns:
    tuple: (integrated_movies_df, filtered_comments_df) - DataFrames with integrated data
    """
    import pandas as pd
    import numpy as np
    
    # Ensure we have the required columns
    if 'movie' not in comments_df.columns:
        raise ValueError("Comments DataFrame must have a 'movie' column")
    
    if 'title' not in movies_df.columns or 'revenue' not in movies_df.columns:
        raise ValueError("Movies DataFrame must have 'title' and 'revenue' columns")
    
    # Count comments per movie
    comment_counts = comments_df['movie'].value_counts().reset_index()
    comment_counts.columns = ['title', 'comment_count']
    
    # Merge comment counts with movie data
    integrated_df = pd.merge(movies_df, comment_counts, on='title', how='left')
    
    # Fill missing comment counts with 0
    integrated_df['comment_count'] = integrated_df['comment_count'].fillna(0).astype(int)
    
    # Filter movies with sufficient comments and revenue data
    valid_movies = integrated_df[
        (integrated_df['comment_count'] >= min_comments) & 
        (integrated_df['revenue'].notna()) & 
        (integrated_df['revenue'] > 0)
    ]
    
    # Calculate what percentage of movies passed the filter
    total_movies = len(integrated_df)
    valid_count = len(valid_movies)
    retention_rate = (valid_count / total_movies) * 100 if total_movies > 0 else 0
    
    print(f"Data integration summary:")
    print(f"- Total movies: {total_movies}")
    print(f"- Movies with sufficient data: {valid_count} ({retention_rate:.1f}%)")
    print(f"- Movies filtered out: {total_movies - valid_count}")
    
    # If no movies pass the filter, return the original datasets with a warning
    if valid_count == 0:
        print("WARNING: No movies meet the criteria for minimum comments and revenue data.")
        return movies_df, comments_df
    
    # Filter comments to only include valid movies
    valid_titles = valid_movies['title'].tolist()
    filtered_comments = comments_df[comments_df['movie'].isin(valid_titles)]
    
    # Add a validation check for the final datasets
    if len(filtered_comments) == 0:
        print("WARNING: No comments remain after filtering.")
        return valid_movies, filtered_comments
        
    # Calculate sentiment statistics per movie
    sentiment_stats = calculate_movie_sentiment_stats(filtered_comments, valid_movies)
    
    # Return the integrated dataframes
    return sentiment_stats, filtered_comments

def calculate_movie_sentiment_stats(comments_df, movies_df):
    """
    Calculate sentiment statistics for each movie and add to the movies dataframe.
    
    Parameters:
    comments_df (DataFrame): DataFrame with comment data including sentiment
    movies_df (DataFrame): DataFrame with movie data
    
    Returns:
    DataFrame: Updated movies dataframe with sentiment statistics
    """
    # Check for required columns
    required_columns = ['movie', 'sentiment']
    if not all(col in comments_df.columns for col in required_columns):
        print("WARNING: Comments dataframe missing required columns for sentiment analysis.")
        if 'sentiment' not in comments_df.columns and 'polarity' in comments_df.columns:
            # Use polarity as sentiment if available
            comments_df['sentiment'] = comments_df['polarity']
        else:
            # Return the original dataframe if we can't calculate stats
            return movies_df
    
    # Group by movie and calculate sentiment stats
    sentiment_stats = []
    
    for movie_title in movies_df['title']:
        movie_comments = comments_df[comments_df['movie'] == movie_title]
        total_comments = len(movie_comments)
        
        if total_comments == 0:
            continue
            
        # Calculate sentiment metrics
        if 'sentiment_category' in movie_comments.columns:
            # If we have categorical sentiment
            positive_count = (movie_comments['sentiment_category'] == 'positive').sum()
            negative_count = (movie_comments['sentiment_category'] == 'negative').sum()
        elif 'sentiment' in movie_comments.columns and pd.api.types.is_numeric_dtype(movie_comments['sentiment']):
            # If we have numerical sentiment
            positive_count = (movie_comments['sentiment'] > 0).sum()
            negative_count = (movie_comments['sentiment'] < 0).sum()
        else:
            # Use polarity if available
            if 'polarity' in movie_comments.columns:
                positive_count = (movie_comments['polarity'] > 0).sum()
                negative_count = (movie_comments['polarity'] < 0).sum()
            else:
                # Skip if we can't calculate
                continue
        
        neutral_count = total_comments - positive_count - negative_count
        
        # Calculate percentages
        positive_pct = positive_count / total_comments * 100
        negative_pct = negative_count / total_comments * 100
        neutral_pct = neutral_count / total_comments * 100
        
        # Calculate average sentiment
        if 'sentiment' in movie_comments.columns and pd.api.types.is_numeric_dtype(movie_comments['sentiment']):
            avg_sentiment = movie_comments['sentiment'].mean()
        elif 'polarity' in movie_comments.columns:
            avg_sentiment = movie_comments['polarity'].mean()
        else:
            avg_sentiment = (positive_count - negative_count) / total_comments
        
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
            'avg_sentiment': avg_sentiment
        }
        
        sentiment_stats.append(stats)
    
    # Convert to DataFrame
    sentiment_df = pd.DataFrame(sentiment_stats)
    
    # Merge with movies_df, keeping all movies even if they don't have sentiment stats
    if len(sentiment_df) > 0:
        movies_with_stats = pd.merge(movies_df, sentiment_df, on='title', how='left')
        
        # Fill NaN values for movies without sentiment data
        for col in ['comment_count', 'positive_count', 'negative_count', 'neutral_count']:
            if col in movies_with_stats.columns:
                movies_with_stats[col] = movies_with_stats[col].fillna(0).astype(int)
                
        for col in ['positive_pct', 'negative_pct', 'neutral_pct', 'avg_sentiment']:
            if col in movies_with_stats.columns:
                movies_with_stats[col] = movies_with_stats[col].fillna(0)
                
        return movies_with_stats
    else:
        print("WARNING: Could not calculate sentiment statistics for any movies.")
        return movies_df

def verify_data_integration(movies_df, comments_df):
    """
    Verify that data integration was successful by running basic checks.
    
    Parameters:
    movies_df (DataFrame): Integrated movies DataFrame
    comments_df (DataFrame): Filtered comments DataFrame
    
    Returns:
    bool: True if verification passes, False otherwise
    """
    verification_passed = True
    
    # Check 1: All movies have revenue data
    movies_without_revenue = movies_df[movies_df['revenue'].isna() | (movies_df['revenue'] <= 0)]
    if len(movies_without_revenue) > 0:
        print(f"WARNING: {len(movies_without_revenue)} movies don't have valid revenue data.")
        verification_passed = False
    
    # Check 2: All movies have comment data
    movies_without_comments = movies_df[movies_df['comment_count'] <= 0]
    if len(movies_without_comments) > 0:
        print(f"WARNING: {len(movies_without_comments)} movies don't have any comments.")
        verification_passed = False
    
    # Check 3: All comments reference valid movies
    if 'movie' in comments_df.columns:
        valid_titles = set(movies_df['title'])
        invalid_comments = comments_df[~comments_df['movie'].isin(valid_titles)]
        if len(invalid_comments) > 0:
            print(f"WARNING: {len(invalid_comments)} comments reference movies not in the movies dataset.")
            verification_passed = False
    
    # Check 4: Sentiment data is available for analysis
    sentiment_cols = ['avg_sentiment', 'positive_pct', 'negative_pct']
    missing_sentiment = [col for col in sentiment_cols if col not in movies_df.columns]
    if missing_sentiment:
        print(f"WARNING: Missing sentiment columns: {', '.join(missing_sentiment)}")
        verification_passed = False
    
    # If all checks pass
    if verification_passed:
        print("✓ Data integration verification PASSED")
        print(f"- {len(movies_df)} movies with valid data")
        print(f"- {len(comments_df)} comments across these movies")
        print(f"- Average {comments_df.shape[0] / movies_df.shape[0]:.1f} comments per movie")
    
    return verification_passed