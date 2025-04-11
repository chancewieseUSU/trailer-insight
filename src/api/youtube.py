# src/api/youtube.py
import os
import pandas as pd
import config
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta

class YouTubeClient:
    """Class for interacting with the YouTube Data API."""
    
    def __init__(self, api_key=None):
        """
        Initialize the YouTube client.
        
        Parameters:
        api_key (str): YouTube Data API key
        """
        self.api_key = api_key or config.YOUTUBE_API_KEY
        if not self.api_key:
            print("Warning: No API key provided. Set YOUTUBE_API_KEY environment variable or provide in constructor.")
        
        self.youtube = None
        if self.api_key:
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
                print("YouTube API client initialized successfully")
            except Exception as e:
                print(f"Error initializing YouTube API client: {e}")
    
    def get_video_id(self, url):
        """
        Extract video ID from YouTube URL.
        
        Parameters:
        url (str): YouTube video URL
        
        Returns:
        str: YouTube video ID
        """
        if not url:
            return None
            
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        elif 'youtube.com/watch' in url:
            import urllib.parse
            query = urllib.parse.urlparse(url).query
            params = urllib.parse.parse_qs(query)
            return params.get('v', [''])[0]
        elif 'youtube.com/embed/' in url:
            return url.split('youtube.com/embed/')[-1].split('?')[0]
        else:
            return url  # Assume it's already a video ID
    
    def get_video_details(self, video_id):
        """
        Get details about a YouTube video.
        
        Parameters:
        video_id (str): YouTube video ID
        
        Returns:
        dict: Video details
        """
        if not self.youtube:
            raise ValueError("YouTube API client not initialized")
        
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()
            
            if response['items']:
                return response['items'][0]
            else:
                return None
        
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            return None
    
    def get_video_comments(self, video_id, max_results=100, published_after=None, end_date=None):
        """
        Get comments for a YouTube video.
        
        Parameters:
        video_id (str): YouTube video ID
        max_results (int): Maximum number of comments to retrieve
        published_after (str): ISO format datetime string to filter comments by date
        end_date (datetime): Only include comments before this date (e.g., movie release date)
        
        Returns:
        list: List of comment data
        """
        if not self.youtube:
            raise ValueError("YouTube API client not initialized")
        
        try:
            comments = []
            next_page_token = None
            
            # Create request parameters
            params = {
                'part': 'snippet',
                'videoId': video_id,
                'maxResults': min(100, max_results - len(comments)),
                'textFormat': 'plainText',
                'order': 'relevance'  # Can be 'time' or 'relevance'
            }
            
            if next_page_token:
                params['pageToken'] = next_page_token
                
            if published_after:
                params['publishedAfter'] = published_after
            
            while len(comments) < max_results:
                # Make API request
                response = self.youtube.commentThreads().list(**params).execute()
                
                # Extract comments
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    
                    # Filter by end date if provided
                    if end_date:
                        try:
                            # Convert comment date to datetime object without timezone info
                            comment_date_str = comment['publishedAt']
                            # Remove timezone info to make it naive like the release date
                            if 'Z' in comment_date_str:
                                comment_date_str = comment_date_str.replace('Z', '')
                            if '+' in comment_date_str:
                                comment_date_str = comment_date_str.split('+')[0]
                            
                            comment_date = datetime.fromisoformat(comment_date_str)
                            
                            # Ensure end_date is also naive
                            end_date_naive = end_date
                            if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                                # Strip timezone info if present
                                end_date_naive = datetime(
                                    end_date.year, end_date.month, end_date.day, 
                                    end_date.hour, end_date.minute, end_date.second
                                )
                                
                            # Skip comments after the release date
                            if comment_date > end_date_naive:
                                continue
                        except Exception as e:
                            # If there's an error parsing dates, log it and include the comment anyway
                            print(f"Error comparing dates: {e}")
                    
                    # Add comment to collection
                    comments.append({
                        'comment_id': item['id'],
                        'text': comment['textDisplay'],
                        'author': comment['authorDisplayName'],
                        'likes': comment['likeCount'],
                        'published_at': comment['publishedAt'],
                        'updated_at': comment['updatedAt']
                    })
                
                # Check if there are more comments
                next_page_token = response.get('nextPageToken')
                if not next_page_token or len(comments) >= max_results:
                    break
                
                # Update page token for next request
                params['pageToken'] = next_page_token
            
            # Print a summary of the date filtering
            if end_date and comments:
                try:
                    # Convert to naive datetime objects for comparison
                    comment_dates = []
                    for c in comments:
                        date_str = c['published_at']
                        if 'Z' in date_str:
                            date_str = date_str.replace('Z', '')
                        if '+' in date_str:
                            date_str = date_str.split('+')[0]
                        comment_dates.append(datetime.fromisoformat(date_str))
                    
                    min_date = min(comment_dates)
                    max_date = max(comment_dates)
                    
                    print(f"Filtered comments from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                    print(f"Movie release date: {end_date.strftime('%Y-%m-%d')}")
                    print(f"Retrieved {len(comments)} comments before the release date")
                except Exception as e:
                    print(f"Error while printing date summary: {e}")
            
            return comments
        
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            return []
    
    def search_videos(self, query, max_results=10, published_after=None, video_category=None):
        """
        Search for videos on YouTube.
        
        Parameters:
        query (str): Search query
        max_results (int): Maximum number of results to retrieve
        published_after (str): ISO format datetime string to filter videos by date
        video_category (str): Video category ID
        
        Returns:
        list: List of video data
        """
        if not self.youtube:
            raise ValueError("YouTube API client not initialized")
        
        try:
            # Create request parameters
            params = {
                'part': 'snippet',
                'q': query,
                'maxResults': min(50, max_results),
                'type': 'video',
                'order': 'relevance'  # Can be 'date', 'rating', 'relevance', 'title', 'viewCount'
            }
            
            if published_after:
                params['publishedAfter'] = published_after
                
            if video_category:
                params['videoCategoryId'] = video_category
            
            # Make API request
            response = self.youtube.search().list(**params).execute()
            
            # Extract video data
            videos = []
            for item in response['items']:
                video_id = item['id']['videoId']
                snippet = item['snippet']
                
                videos.append({
                    'video_id': video_id,
                    'title': snippet['title'],
                    'description': snippet['description'],
                    'published_at': snippet['publishedAt'],
                    'channel_id': snippet['channelId'],
                    'channel_title': snippet['channelTitle'],
                    'thumbnail_url': snippet['thumbnails']['high']['url']
                })
            
            return videos
        
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            return []
    
    def search_movie_trailers(self, movie_title, year=None, max_results=5):
        """
        Search for movie trailers on YouTube.
        
        Parameters:
        movie_title (str): Movie title
        year (int): Movie release year
        max_results (int): Maximum number of results to retrieve
        
        Returns:
        list: List of trailer video data
        """
        # Construct search query
        query = f"{movie_title} official trailer"
        if year:
            query += f" {year}"
        
        # Search for videos
        videos = self.search_videos(query, max_results=max_results)
        
        # Filter for likely trailers
        trailers = []
        for video in videos:
            title = video['title'].lower()
            description = video['description'].lower()
            
            # Check if this looks like a trailer
            is_trailer = (
                ('trailer' in title or 'teaser' in title) and
                (movie_title.lower() in title or movie_title.lower() in description)
            )
            
            if is_trailer:
                trailers.append(video)
        
        return trailers
    
    def comments_to_dataframe(self, comments):
        """
        Convert comments to a pandas DataFrame.
        
        Parameters:
        comments (list): List of comment data
        
        Returns:
        DataFrame: Comments as a DataFrame
        """
        df = pd.DataFrame(comments)
        
        # Convert date columns to datetime
        if 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'])
        
        if 'updated_at' in df.columns:
            df['updated_at'] = pd.to_datetime(df['updated_at'])
            
        return df
    
    def get_comments_by_date_range(self, video_id, start_date, end_date=None, max_results=100):
        """
        Get comments for a YouTube video within a date range.
        
        Parameters:
        video_id (str): YouTube video ID
        start_date (str): Start date in ISO format (YYYY-MM-DD)
        end_date (str): End date in ISO format (YYYY-MM-DD), defaults to today
        max_results (int): Maximum number of comments to retrieve
        
        Returns:
        list: List of comment data
        """
        # Convert dates to ISO format
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Get all comments
        all_comments = self.get_video_comments(video_id, max_results=max_results * 2)  # Get more to allow for filtering
        
        # Convert to DataFrame for easier filtering
        df = self.comments_to_dataframe(all_comments)
        
        # Filter by date range
        if 'published_at' in df.columns:
            mask = (df['published_at'] >= pd.to_datetime(start_date)) & (df['published_at'] <= pd.to_datetime(end_date))
            df = df[mask]
        
        # Limit to requested max_results
        df = df.head(max_results)
        
        # Convert back to list of dictionaries
        return df.to_dict('records')