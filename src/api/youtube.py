import os
import pandas as pd
import config
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
    
    def get_video_id(self, url):
        """
        Extract video ID from YouTube URL.
        
        Parameters:
        url (str): YouTube video URL
        
        Returns:
        str: YouTube video ID
        """
        if 'youtu.be' in url:
            return url.split('/')[-1]
        elif 'youtube.com/watch' in url:
            import urllib.parse
            query = urllib.parse.urlparse(url).query
            params = urllib.parse.parse_qs(query)
            return params.get('v', [''])[0]
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
                part='snippet,statistics',
                id=video_id
            ).execute()
            
            if response['items']:
                return response['items'][0]
            else:
                return None
        
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            return None
    
    def get_video_comments(self, video_id, max_results=100):
        """
        Get comments for a YouTube video.
        
        Parameters:
        video_id (str): YouTube video ID
        max_results (int): Maximum number of comments to retrieve
        
        Returns:
        list: List of comment data
        """
        if not self.youtube:
            raise ValueError("YouTube API client not initialized")
        
        try:
            comments = []
            next_page_token = None
            
            while len(comments) < max_results:
                # Make API request
                response = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token
                ).execute()
                
                # Extract comments
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
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
        
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            return []
    
    def comments_to_dataframe(self, comments):
        """
        Convert comments to a pandas DataFrame.
        
        Parameters:
        comments (list): List of comment data
        
        Returns:
        DataFrame: Comments as a DataFrame
        """
        return pd.DataFrame(comments)