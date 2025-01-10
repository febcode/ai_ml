# Import libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Create a simple dataset
data = {
    "User1": [5, 4, 1, 0],
    "User2": [4, 0, 0, 3],
    "User3": [1, 1, 0, 5],
    "User4": [0, 3, 5, 4]
}
movies = ["MovieA", "MovieB", "MovieC", "MovieD"]
df = pd.DataFrame(data, index=movies)

# Calculate similarity between movies
similarity = cosine_similarity(df)
similarity_df = pd.DataFrame(similarity, index=movies, columns=movies)

# Recommend similar movies for "MovieA"
print("Movies similar to 'MovieA':")
print(similarity_df["MovieA"].sort_values(ascending=False))
