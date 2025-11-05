# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

# Import content-based model
from src.models.content_based import ContentBasedRecommender

# Set page config
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Book Recommendation System")
st.markdown("Discover books similar to your favorites!")

# Function to get book cover image URL
def get_book_cover(isbn, title, author, size='M'):
    """Generate book cover image URL"""
    if pd.notna(isbn) and isbn != '' and str(isbn) != 'nan':
        return f"https://covers.openlibrary.org/b/isbn/{isbn}-{size}.jpg"
    # Create a placeholder with first letter of title
    first_letter = title[0] if title else 'B'
    return f"https://via.placeholder.com/150x200/4F46E5/FFFFFF?text={first_letter}"

# Load data with better error handling
@st.cache_data
def load_data():
    try:
        # Load with specific dtypes to handle mixed types
        ratings = pd.read_csv(
            'data/ratings.csv', 
            encoding='latin-1', 
            sep=';', 
            on_bad_lines='skip',
            dtype={'User-ID': 'str', 'ISBN': 'str', 'Book-Rating': 'float64'}
        )
        books = pd.read_csv(
            'data/books.csv', 
            encoding='latin-1', 
            sep=';', 
            on_bad_lines='skip',
            dtype={'ISBN': 'str', 'Book-Title': 'str', 'Book-Author': 'str', 
                   'Year-Of-Publication': 'str', 'Publisher': 'str'}
        )
        
        # Clean the data
        books = books.dropna(subset=['Book-Title'])  # Remove books without titles
        books = books[books['Book-Title'].str.strip() != '']  # Remove empty titles
        
        st.success(f"✅ Loaded {len(books)} books and {len(ratings)} ratings")
        return ratings, books
        
    except Exception as e:
        st.error(f"Could not load data files: {str(e)}")
        return None, None

# Load data
ratings, books = load_data()

if ratings is not None and books is not None:
    # Initialize content-based recommender with progress
    @st.cache_resource
    def load_recommender():
        with st.spinner("Building recommendation engine... This may take a moment."):
            return ContentBasedRecommender(books, sample_size=3000)  # Reduced sample size
    
    try:
        recommender = load_recommender()
        st.success("✅ Content-based recommendation model loaded!")
        
        # Recommendation methods
        st.sidebar.header("🔍 Find Similar Books")
        
        method = st.sidebar.radio(
            "How would you like to find books?",
            ["Search by Book Title", "Browse Popular Books", "Search by Keyword"]
        )
        
        if method == "Search by Book Title":
            st.sidebar.subheader("📖 Search by Title")
            
            # Get a sample of book titles for dropdown
            sample_titles = books['Book-Title'].dropna().unique()
            if len(sample_titles) > 1000:
                sample_titles = np.random.choice(sample_titles, 1000, replace=False)
            
            selected_title = st.sidebar.selectbox(
                "Select a Book:",
                sorted(sample_titles),
                help="Choose a book to find similar recommendations"
            )
            
            num_recommendations = st.sidebar.slider(
                "Number of recommendations:",
                min_value=1,
                max_value=10,
                value=5
            )
            
            if st.sidebar.button("Find Similar Books", type="primary"):
                with st.spinner("Finding similar books..."):
                    recommendations = recommender.get_recommendations(
                        selected_title, 
                        num_recommendations
                    )
                    
                    if not recommendations.empty:
                        st.success(f"🎉 Found {len(recommendations)} books similar to '{selected_title}'")
                        
                        # Display recommendations in a grid
                        st.subheader("📚 Recommended Books")
                        
                        # Create columns for grid layout
                        cols = st.columns(2)
                        
                        for i, (idx, row) in enumerate(recommendations.iterrows()):
                            with cols[i % 2]:
                                with st.container():
                                    st.markdown("---")
                                    
                                    # Image and text columns
                                    col_img, col_text = st.columns([1, 2])
                                    
                                    with col_img:
                                        cover_url = get_book_cover(
                                            row['ISBN'],
                                            row['Book-Title'],
                                            row['Book-Author']
                                        )
                                        try:
                                            st.image(cover_url, width=100, caption=f"#{i+1}")
                                        except:
                                            st.image(
                                                "https://via.placeholder.com/100x150/6B7280/FFFFFF?text=Cover",
                                                width=100,
                                                caption=f"#{i+1}"
                                            )
                                    
                                    with col_text:
                                        st.write(f"**{row['Book-Title']}**")
                                        st.write(f"*by {row['Book-Author']}*")
                                        if 'Year-Of-Publication' in row and pd.notna(row['Year-Of-Publication']):
                                            try:
                                                st.write(f"**Year:** {int(float(row['Year-Of-Publication']))}")
                                            except:
                                                st.write(f"**Year:** {row['Year-Of-Publication']}")
                                        if 'Publisher' in row and pd.notna(row['Publisher']):
                                            st.write(f"**Publisher:** {row['Publisher']}")
                                        st.write(f"**ISBN:** `{row['ISBN']}`")
                    
                    else:
                        st.warning(f"❌ No recommendations found for '{selected_title}'. Try a different book.")
        
        elif method == "Browse Popular Books":
            st.sidebar.subheader("🔥 Popular Books")
            
            # Calculate popular books based on rating counts
            if ratings is not None:
                rating_counts = ratings['ISBN'].value_counts()
                books_with_ratings = books.copy()
                books_with_ratings['rating_count'] = books_with_ratings['ISBN'].map(rating_counts).fillna(0)
                popular_books = books_with_ratings[books_with_ratings['rating_count'] > 0].sort_values('rating_count', ascending=False)
            else:
                popular_books = books.head(20)  # Fallback to first 20 books
            
            num_popular = st.sidebar.slider(
                "Number of books to show:",
                min_value=5,
                max_value=20,
                value=10
            )
            
            st.subheader("🔥 Popular Books")
            
            # Display popular books
            popular_to_show = popular_books.head(num_popular)
            cols = st.columns(2)
            
            for i, (idx, row) in enumerate(popular_to_show.iterrows()):
                with cols[i % 2]:
                    with st.container():
                        st.markdown("---")
                        
                        col_img, col_text = st.columns([1, 2])
                        
                        with col_img:
                            cover_url = get_book_cover(
                                row['ISBN'],
                                row['Book-Title'],
                                row['Book-Author']
                            )
                            try:
                                st.image(cover_url, width=100, caption=f"#{i+1}")
                            except:
                                st.image(
                                    "https://via.placeholder.com/100x150/6B7280/FFFFFF?text=Cover",
                                    width=100,
                                    caption=f"#{i+1}"
                                )
                        
                        with col_text:
                            st.write(f"**{row['Book-Title']}**")
                            st.write(f"*by {row['Book-Author']}*")
                            if 'rating_count' in row:
                                st.write(f"**Ratings:** {int(row['rating_count'])}")
                            if 'Year-Of-Publication' in row and pd.notna(row['Year-Of-Publication']):
                                try:
                                    st.write(f"**Year:** {int(float(row['Year-Of-Publication']))}")
                                except:
                                    st.write(f"**Year:** {row['Year-Of-Publication']}")
                            st.write(f"**ISBN:** `{row['ISBN']}`")
        
        else:  # Search by Keyword
            st.sidebar.subheader("🔍 Search by Keyword")
            
            search_query = st.sidebar.text_input(
                "Enter book title, author, or publisher:",
                placeholder="e.g., Harry Potter or Stephen King"
            )
            
            num_results = st.sidebar.slider(
                "Number of results:",
                min_value=5,
                max_value=20,
                value=10
            )
            
            if st.sidebar.button("Search Books", type="primary"):
                if search_query:
                    with st.spinner("Searching books..."):
                        search_results = recommender.search_books(search_query, num_results)
                        
                        if not search_results.empty:
                            st.success(f"🔍 Found {len(search_results)} books matching '{search_query}'")
                            
                            # Display search results
                            st.subheader("📚 Search Results")
                            
                            cols = st.columns(2)
                            for i, (idx, row) in enumerate(search_results.iterrows()):
                                with cols[i % 2]:
                                    with st.container():
                                        st.markdown("---")
                                        
                                        col_img, col_text = st.columns([1, 2])
                                        
                                        with col_img:
                                            cover_url = get_book_cover(
                                                row['ISBN'],
                                                row['Book-Title'],
                                                row['Book-Author']
                                            )
                                            try:
                                                st.image(cover_url, width=100)
                                            except:
                                                st.image(
                                                    "https://via.placeholder.com/100x150/6B7280/FFFFFF?text=Cover",
                                                    width=100
                                                )
                                        
                                        with col_text:
                                            st.write(f"**{row['Book-Title']}**")
                                            st.write(f"*by {row['Book-Author']}*")
                                            if 'Year-Of-Publication' in row and pd.notna(row['Year-Of-Publication']):
                                                try:
                                                    st.write(f"**Year:** {int(float(row['Year-Of-Publication']))}")
                                                except:
                                                    st.write(f"**Year:** {row['Year-Of-Publication']}")
                                            if 'Publisher' in row and pd.notna(row['Publisher']):
                                                st.write(f"**Publisher:** {row['Publisher']}")
                                            st.write(f"**ISBN:** `{row['ISBN']}`")
                                            
                                            # Add button to get recommendations for this book
                                            if st.button(f"Find Similar", key=f"similar_{i}"):
                                                st.session_state.selected_book = row['Book-Title']
                        else:
                            st.warning(f"❌ No books found matching '{search_query}'")
        
        # Display dataset stats
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Dataset Info")
        st.sidebar.write(f"Total Books: {len(books):,}")
        if ratings is not None:
            st.sidebar.write(f"Total Ratings: {len(ratings):,}")
            st.sidebar.write(f"Total Users: {ratings['User-ID'].nunique():,}")
        
    except Exception as e:
        st.error(f"❌ Error loading recommender: {str(e)}")
        st.info("💡 Try reducing the sample size in the code if memory issues persist.")

else:
    st.error("Please make sure your data files are in the correct location.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Content-Based Book Recommendation System")