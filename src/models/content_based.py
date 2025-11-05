# src/models/content_based.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self, books_df, sample_size=5000):
        """
        Memory-efficient content-based recommender
        """
        self.books_df = books_df
        self.sample_size = min(sample_size, len(books_df))
        self.tfidf = None
        self.tfidf_matrix = None
        self.knn_model = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data with memory efficiency"""
        print(f"Preparing data with {self.sample_size} books...")
        
        # Sample the data to reduce size
        if len(self.books_df) > self.sample_size:
            books_sample = self.books_df.sample(n=self.sample_size, random_state=42)
        else:
            books_sample = self.books_df.copy()
        
        # Fill missing values
        books_sample['Book-Title'] = books_sample['Book-Title'].fillna('')
        books_sample['Book-Author'] = books_sample['Book-Author'].fillna('')
        books_sample['Publisher'] = books_sample['Publisher'].fillna('')
        
        # Combine features
        books_sample['content'] = (
            books_sample['Book-Title'] + " " +
            books_sample['Book-Author'] + " " +
            books_sample['Publisher']
        )
        
        # Create TF-IDF matrix with limited features
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=1000,  # Reduced features
            ngram_range=(1, 1)  # Only unigrams
        )
        
        self.tfidf_matrix = self.tfidf.fit_transform(books_sample['content'])
        
        # Use KNN for efficient similarity search
        self.knn_model = NearestNeighbors(
            n_neighbors=11,  # 10 neighbors + itself
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(self.tfidf_matrix)
        
        # Store the sample for lookup
        self.books_sample = books_sample.reset_index(drop=True)
        
        print("Data preparation completed!")
    
    def get_recommendations(self, book_title, n_recommendations=10):
        """Get recommendations using KNN for efficiency"""
        try:
            # Find the book in our sample
            book_match = self.books_sample[
                self.books_sample['Book-Title'].str.contains(book_title, case=False, na=False)
            ]
            
            if len(book_match) == 0:
                # Try fuzzy matching with first few words
                search_words = book_title.lower().split()[:3]
                for book_idx, row in self.books_sample.iterrows():
                    title_words = str(row['Book-Title']).lower().split()
                    if any(word in title_words for word in search_words):
                        book_match = pd.DataFrame([row])
                        break
            
            if len(book_match) == 0:
                return pd.DataFrame()
            
            book_idx = book_match.index[0]
            
            # Find nearest neighbors
            distances, indices = self.knn_model.kneighbors(
                self.tfidf_matrix[book_idx], 
                n_neighbors=n_recommendations + 1
            )
            
            # Remove the first result (itself)
            similar_indices = indices[0][1:]
            
            recommendations = self.books_sample.iloc[similar_indices]
            return recommendations[['ISBN', 'Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']]
            
        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            return pd.DataFrame()
    
    def get_recommendations_by_isbn(self, isbn, n_recommendations=10):
        """Get recommendations using ISBN"""
        try:
            book_match = self.books_sample[self.books_sample['ISBN'] == isbn]
            if len(book_match) == 0:
                return pd.DataFrame()
            
            book_title = book_match['Book-Title'].iloc[0]
            return self.get_recommendations(book_title, n_recommendations)
            
        except Exception as e:
            print(f"Error in get_recommendations_by_isbn: {e}")
            return pd.DataFrame()
    
    def search_books(self, query, n_results=10):
        """Search books by title, author, or publisher"""
        try:
            query = query.lower()
            matches = self.books_sample[
                self.books_sample['Book-Title'].str.lower().str.contains(query, na=False) |
                self.books_sample['Book-Author'].str.lower().str.contains(query, na=False) |
                self.books_sample['Publisher'].str.lower().str.contains(query, na=False)
            ]
            return matches.head(n_results)
        except Exception as e:
            print(f"Error in search_books: {e}")
            return pd.DataFrame()