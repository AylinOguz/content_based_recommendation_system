
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)
df.head()
df["overview"].head()


#################################
# 1. Creating The TF-IDF Matrix
#################################

tfidf = TfidfVectorizer(stop_words="english")
# tfidf object created


df['overview'] = df['overview'].fillna('')
# Some rows in the 'overview' column were NaN. We filled NaNs with an empty string.

tfidf_matrix = tfidf.fit_transform(df['overview'])
# We used the fit_transform method of the tfidf object, specifying which variable to use.

tfidf_matrix.shape
# (45466, 75827)


#################################
# 2. Creating the Cosine Similarity Matrix
#################################

# Cosine similarity is calculated based on words in movie descriptions.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
# (45466, 45466)
cosine_sim[1]

#################################
# 3. Making Recommendations Based on Similarity
#################################

# To make recommendations, we need to know the movie titles.
# We create a pandas Series with movie titles as the index.
indices = pd.Series(df.index, index=df['title'])

# Some movies have duplicate titles (e.g., "Cinderella" has multiple movies).
indices.index.value_counts()

indices["Cinderella"]
# Returns multiple indices for "Cinderella"

# Keep only the last occurrence for duplicate titles:
indices = indices[~indices.index.duplicated(keep='last')]

indices["Cinderella"]
# Last index for "Cinderella"

indices["Sherlock Holmes"]
movie_index = indices["Sherlock Holmes"]

# Access similarity values of "Sherlock Holmes" with other movies
cosine_sim[movie_index]
# numpy array with similarity scores

# cosine_sim is a numpy array without index information.
# By providing the movie index, we access the similarity scores for that movie.

indeks = df[df["title"] == 'Inception'].index.tolist()[0]
cosine_sim[indeks]

cosine_df = pd.DataFrame(cosine_sim[indeks], columns=["score"])

recomendation_index= cosine_df.sort_values(by="score", ascending=False)[1:10].index.tolist()
df["title"].iloc[recomendation_index]

# Create a DataFrame of cosine similarity scores for "Sherlock Holmes"
similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["Sherlock Holmes_score"])

# Sort by score and get the top 11 indices (excluding 0, which is the movie itself)
movie_indices = similarity_scores.sort_values("Sherlock Holmes_score", ascending=False)[1:11].index

# Get the recommended movie titles
df['title'].iloc[movie_indices]

#################################
# 4. Preparing a Reusable Script
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # Create indices
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # Get the index of the movie
    movie_index = indices[title]
    # Compute similarity scores
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # Get top 10 movies excluding the movie itself
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)
content_based_recommender("The Matrix", cosine_sim, df)
content_based_recommender("The Godfather", cosine_sim, df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

# These operations are not repeated every time.
# For example, for the 100 most-watched movies, the above steps are done once,
# and for each movie, 5 recommendations are generated.
# The IDs of these movies and their recommendations are stored in a SQL table.

# Example format:
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
