# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
OMDB_API_KEY = os.getenv('OMDB_API_KEY')
TMDB_KEY = os.getenv('TMDB_KEY')

# Other configuration variables
MAX_COMMENTS = 200
DEFAULT_CLUSTERS = 5