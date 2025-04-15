# src/api/movie_db.py
import requests
import config

class MovieDBClient:
    """Client for movie database APIs (TMDB and OMDB)."""
    
    def __init__(self):
        self.tmdb_key = config.TMDB_KEY
        self.omdb_key = config.OMDB_API_KEY
    
    def search_movies(self, query, year=None, max_results=10):
        """Search for movies on TMDB."""
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": self.tmdb_key,
            "query": query,
            "include_adult": False,
            "language": "en-US",
            "page": 1
        }
        
        if year:
            params["year"] = year
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            # Sort by popularity
            results.sort(key=lambda x: x.get("popularity", 0), reverse=True)
            
            return results[:max_results]
        
        return []
    
    def get_popular_movies(self, page=1, max_results=10):
        """Get currently popular movies from TMDB."""
        url = f"https://api.themoviedb.org/3/movie/popular"
        params = {
            "api_key": self.tmdb_key,
            "language": "en-US",
            "page": page
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            return results[:max_results]
        
        return []
    
    def get_upcoming_movies(self, page=1, max_results=10):
        """Get upcoming movies from TMDB."""
        url = f"https://api.themoviedb.org/3/movie/upcoming"
        params = {
            "api_key": self.tmdb_key,
            "language": "en-US",
            "page": page
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            return results[:max_results]
        
        return []
    
    def get_movie_details_tmdb(self, movie_title):
        """Get movie details from TMDB."""
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": self.tmdb_key,
            "query": movie_title
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data["results"]:
                movie_id = data["results"][0]["id"]
                return self.get_movie_by_id_tmdb(movie_id)
        
        return None
    
    def get_movie_by_id_tmdb(self, movie_id):
        """Get detailed movie information by ID from TMDB."""
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            "api_key": self.tmdb_key,
            "append_to_response": "credits,keywords,videos"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        
        return None
    
    def get_movie_trailer(self, movie_id):
        """Get YouTube trailer ID for a movie from TMDB."""
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        params = {
            "api_key": self.tmdb_key,
            "language": "en-US"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            videos = data.get("results", [])
            
            # First look for official trailers
            official_trailers = [v for v in videos if v.get("type") == "Trailer" and 
                               v.get("site") == "YouTube"]
            
            if official_trailers:
                return official_trailers[0].get("key")
            
            # Then look for teasers
            teasers = [v for v in videos if v.get("type") == "Teaser" and 
                     v.get("site") == "YouTube"]
            
            if teasers:
                return teasers[0].get("key")
        
        return None
    
    def get_movie_details_omdb(self, movie_title):
        """Get movie details from OMDB."""
        url = "http://www.omdbapi.com/"
        params = {
            "apikey": self.omdb_key,
            "t": movie_title,
            "type": "movie",
            "plot": "full"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True":
                return data
        
        return None