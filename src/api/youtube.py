# src/api/youtube.py
import pandas as pd
import config
from googleapiclient.discovery import build
from datetime import datetime

class YouTubeClient:
    """Class for interacting with the YouTube Data API."""
    
    def __init__(self, api_key=None):
        """Initialize the YouTube client."""
        self.api_key = api_key or config.YOUTUBE_API_KEY
        
        self.youtube = None
        if self.api_key:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
    
    def get_video_id(self, url):
        """Extract video ID from YouTube URL."""
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
        """Get details about a YouTube video."""
        if not self.youtube:
            return None
        
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()
            
            if response['items']:
                return response['items'][0]
            else:
                return None
        
        except Exception:
            return None
    
    def get_video_comments(self, video_id, max_results=100, end_date=None):
        """Get comments for a YouTube video."""
        if not self.youtube:
            return []
        
        try:
            comments = []
            next_page_token = None
            
            while len(comments) < max_results:
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
                
                # Make API request
                response = self.youtube.commentThreads().list(**params).execute()
                
                # Extract comments
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    
                    # Filter by end date if provided
                    if end_date:
                        try:
                            comment_date_str = comment['publishedAt'].replace('Z', '')
                            comment_date = datetime.fromisoformat(comment_date_str)
                            
                            if comment_date > end_date:
                                continue
                        except:
                            pass
                    
                    # Add comment to collection
                    comments.append({
                        'comment_id': item['id'],
                        'text': comment['textDisplay'],
                        'author': comment['authorDisplayName'],
                        'likes': comment['likeCount'],
                        'published_at': comment['publishedAt']
                    })
                
                # Check if there are more comments
                next_page_token = response.get('nextPageToken')
                if not next_page_token or len(comments) >= max_results:
                    break
            
            return comments
        
        except Exception as e:
            print(f"Error fetching comments: {e}")
            return []
    
    def search_videos(self, query, max_results=10):
        """Search for videos on YouTube."""
        if not self.youtube:
            return []
        
        try:
            # Create request parameters
            params = {
                'part': 'snippet',
                'q': query,
                'maxResults': min(50, max_results),
                'type': 'video',
                'order': 'relevance'
            }
            
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
                    'channel_title': snippet['channelTitle'],
                    'thumbnail_url': snippet['thumbnails']['high']['url']
                })
            
            return videos
        
        except Exception:
            return []
    
    def search_movie_trailers(self, movie_title, year=None, max_results=5):
        """Search for movie trailers on YouTube."""
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
        """Convert comments to a pandas DataFrame."""
        df = pd.DataFrame(comments)
        
        # Convert date columns to datetime
        if 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'])
            
        return df