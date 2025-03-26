import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import config  # Add this import
from src.api.youtube import YouTubeClient
from src.api.movie_db import MovieDBClient

# Initialize API clients
youtube_client = YouTubeClient()
movie_db_client = MovieDBClient()

# Set page configuration
st.set_page_config(
    page_title="TrailerInsight",
    page_icon="🎬",
    layout="wide"
)

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
    
    tab1, tab2 = st.tabs(["YouTube API", "Upload Data"])
    
    with tab1:
        st.subheader("Collect comments from YouTube")
        trailer_url = st.text_input("YouTube Trailer URL")
        movie_title = st.text_input("Movie Title")
        
        if st.button("Collect Comments"):
            st.info("This will connect to YouTube API and collect comments (not implemented yet)")
    
    with tab2:
        st.subheader("Upload existing data")
        comments_file = st.file_uploader("Upload comments CSV", type=["csv"])
        metadata_file = st.file_uploader("Upload movie metadata CSV", type=["csv"])
        
        if comments_file and metadata_file:
            st.success("Files uploaded successfully!")

elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    
    # Placeholder for sentiment visualization
    st.info("This section will show sentiment distribution for selected movies")
    
    # Example visualization with dummy data
    sentiment_data = pd.DataFrame({
        'Movie': ['Movie A', 'Movie A', 'Movie A', 'Movie B', 'Movie B', 'Movie B'],
        'Sentiment': ['Positive', 'Neutral', 'Negative', 'Positive', 'Neutral', 'Negative'],
        'Count': [65, 20, 15, 45, 30, 25]
    })
    
    fig = px.bar(sentiment_data, x='Movie', y='Count', color='Sentiment', 
                barmode='group', color_discrete_sequence=['#2ecc71', '#f1c40f', '#e74c3c'])
    st.plotly_chart(fig, use_container_width=True)

elif page == "Comment Clusters":
    st.header("Comment Clusters")
    st.info("This section will show thematic clusters of comments")
    
    # Example clusters with dummy data
    st.subheader("Cluster Distribution")
    cluster_data = pd.DataFrame({
        'Cluster': ['Actor Mentions', 'Story Elements', 'Visual Effects', 'Comparisons', 'Release Anticipation', 'Music'],
        'Count': [120, 85, 65, 50, 40, 30]
    })
    
    fig = px.pie(cluster_data, values='Count', names='Cluster', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Summaries":
    st.header("Comment Summarization")
    st.info("This section will display auto-generated summaries of key themes")
    
    # Example summaries
    st.subheader("Key Themes by Sentiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Positive Sentiment")
        st.markdown("""
        - **Visual Effects**: Audiences are impressed by the visual effects, particularly the realistic CGI.
        - **Acting Performance**: Lead actor's performance is receiving significant praise.
        - **Music Score**: Original soundtrack is highlighted as enhancing emotional scenes.
        """)
    
    with col2:
        st.markdown("#### Negative Sentiment")
        st.markdown("""
        - **Plot Concerns**: Some viewers express worries about plot similarities to previous films.
        - **Pacing Issues**: Several comments mention concerns about the trailer showing too many action scenes.
        - **Character Development**: Questions about depth of supporting characters.
        """)

elif page == "Box Office Insights":
    st.header("Box Office Prediction Insights")
    st.info("This section will connect comment analysis to box office potential")
    
    # Example visualization
    st.subheader("Sentiment vs. Box Office Performance")
    
    # Create dummy data
    np.random.seed(42)
    movies = ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E', 
              'Movie F', 'Movie G', 'Movie H', 'Movie I', 'Movie J']
    positive_sentiment = np.random.uniform(50, 95, 10)
    box_office = positive_sentiment * np.random.uniform(0.8, 1.2, 10) * 10
    
    df = pd.DataFrame({
        'Movie': movies,
        'Positive Sentiment %': positive_sentiment,
        'Box Office (millions $)': box_office
    })
    
    fig = px.scatter(df, x='Positive Sentiment %', y='Box Office (millions $)', 
                    text='Movie', size='Box Office (millions $)',
                    color='Positive Sentiment %', color_continuous_scale='Viridis')
    fig.update_traces(textposition='top center')
    
    st.plotly_chart(fig, use_container_width=True)