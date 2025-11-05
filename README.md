# Book Recommendation System - Jupyter Notebook

This Jupyter notebook implements a content-based book recommendation system.

## Features
- Content-based recommendations using book titles, authors, and publishers
- Memory-efficient implementation using K-Nearest Neighbors
- Interactive recommendation system
- Data exploration and visualization
- Search functionality

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Launch Jupyter: `jupyter notebook`
3. Open `book_recommendation_system.ipynb`
4. Run all cells

## Dataset
The system uses the Book-Crossing dataset with:
- Books data: Title, Author, ISBN, Publisher, Year
- Ratings data: User ratings for books

## Key Sections
1. Data Loading and Exploration
2. Data Cleaning and Preprocessing
3. Content-Based Recommender Implementation
4. Model Testing and Evaluation
5. Interactive Recommendations
6. Analysis and Visualization

## Model Details
- Uses TF-IDF for text vectorization
- Cosine similarity for content matching
- K-Nearest Neighbors for efficient similarity search
- Sample-based approach for memory efficiency