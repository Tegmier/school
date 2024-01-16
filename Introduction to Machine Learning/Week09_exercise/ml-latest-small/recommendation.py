import matplotlib.pyplot as pyplot
import numpy as numpy
import pandas as pd

rattings = pd.read_csv('D:/code/school/Introduction to Machine Learning/Week09_exercise/ml-latest-small/ratings.csv')
movies = pd.read_csv('D:/code/school/Introduction to Machine Learning/Week09_exercise/ml-latest-small/movies.csv')

data = pd.merge(rattings, movies, on = 'movieId')

ratings_average = pd.DataFrame(data.groupby('title')['rating'].mean())
ratings_average['ratings_count'] = pd.DataFrame(data.groupby('title')['rating'].count())

ratings_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# print(ratings_average.sort_values('ratings_count', ascending=False).head(10))
# print(ratings_matrix.head(10))

## find user rating for a Movie
# favorite_movie_ratings = ratings_matrix['Aladdin (1992)']
favorite_movie_ratings = ratings_matrix['Matrix, The (1999)']

#print(favorite_movie_ratings.head(10))

##Finding similar movies
similar_movies = ratings_matrix.corrwith(favorite_movie_ratings)
# print(similar_movies.head(10))

##Remove empty values
correlation = pd.DataFrame(similar_movies, columns=['Correlation'])
correlation.dropna(inplace=True)

# print(correlation.sort_values('Correlation', ascending=False).head(10))
## Add Rating counts
correlation = correlation.join(ratings_average['ratings_count'])
# print(correlation.sort_values('Correlation', ascending=False).head(10))

##See the recommendations
recommendation = correlation[correlation['ratings_count']>100].sort_values('Correlation', ascending=False)
# print(recommendation.head(10))

##Confirm the recommendation Quality
recommendation = recommendation.merge(movies, on='title')
print(recommendation.head(10))

# print(ratings_average.sort_values(by = ['ratings_count', 'rating'], ascending = False))