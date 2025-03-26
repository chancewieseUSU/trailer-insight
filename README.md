# TrailerInsight

An interactive dashboard for analyzing YouTube trailer comments and predicting box office performance.

## Overview

TrailerInsight is a Streamlit-based web application that implements various text mining methods to analyze audience reactions to movie trailers on YouTube. The system processes comments, performs sentiment analysis, identifies thematic patterns through clustering, and generates concise summaries of major themes.

## Features

- Import and clean YouTube trailer comments
- Categorize comments by sentiment
- Identify thematic patterns through document clustering
- Generate summaries of key audience reactions
- Connect text insights to potential box office performance indicators

## Project Structure

- `app.py`: Main Streamlit application
- `data/`: Data directory for raw and processed datasets
- `src/`: Source code including preprocessing, models, and visualization components

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Methods Used

- NLP Pipeline (Preprocessing & Normalization)
- Sentiment Analysis
- Document Clustering
- Document Summarization

## Future Work

- Integration with real-time YouTube API data
- Enhanced predictive models for box office performance
- Expanded genre-specific analysis