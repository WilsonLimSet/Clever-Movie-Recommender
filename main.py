# Wilson Lim Setiawan - Clever Movie Recommender
from flask import Flask, redirect, render_template, request, session, url_for
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os
import sqlite3 as sl
from PIL import Image
import io
import base64
from flask import send_file

app = Flask(__name__)
#First Thing User will see (homepage)
@app.route("/")
def home():
    im = Image.open("oscars.jpg")
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('user.html',img_data=encoded_img_data.decode('utf-8'))

#Get Reccommendations
@app.route("/get_recs", methods=["POST", "GET"])
def get_recs():
    if request.method == "POST":
        title = request.form['title']
        recs = reccomend(title)
        return render_template('recommendations.html',title = title,recs = recs)
    else:
        return "1"

#Get Rankings
@app.route("/rank", methods=["POST", "GET"])
def rank():
    if request.method == "POST":
        #in case no genre is selected, wont crash
        try:
            genree = request.form['genre']
            return render_template('rankings.html', genree=genree,)
        except:
            im = Image.open("oscars.jpg")
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            return render_template('user.html', img_data=encoded_img_data.decode('utf-8'))
    else:
        return "1"

#homepage
@app.route("/homepage", methods=["POST", "GET"])
def homepage():
    im = Image.open("oscars.jpg")
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('user.html', img_data=encoded_img_data.decode('utf-8'))

#Plotting the top 10 movies by genre
@app.route('/plot<params>')
def plot(params):
    # Connecting to our Movie Database
    conn = sl.connect('moviedatabase.db')
    curs = conn.cursor()

    # SQL queries to fetch the data and convert them to Dataframes
    stmt1 = "SELECT * FROM movies"
    stmt2 = "SELECT * FROM ratings"
    moviedata = curs.execute(stmt1)
    movie = pd.DataFrame(data=moviedata, columns=['movieID', 'title', 'genres'])
    movie = movie.iloc[1:, :]
    ratingsdata = curs.execute(stmt2)
    ratings = pd.DataFrame(data=ratingsdata, columns=['userId', 'movieID', 'rating', 'timestamp'])
    ratings.drop(columns='timestamp', inplace=True)
    ratings = ratings.iloc[1:, :]

    # Merging the ratings and movie table into one DF
    mergedf = pd.merge(movie, ratings, on='movieID')
    # One-Hot-Encoding Method for genres
    masterdf = pd.concat([mergedf.drop('genres', axis=1), mergedf.genres.str.get_dummies(sep='|')], axis=1)
    # Dropping movies with less than 8 reviews (so random movies won't be ranked)
    masterdf['count'] = masterdf['title'].map(masterdf['title'].value_counts())
    masterdf.drop(masterdf[(masterdf['count'] < 8)].index, inplace=True)

    # Getting a column for average rating
    masterdf["rating"] = pd.to_numeric(masterdf["rating"], downcast="float")
    x = masterdf.groupby('movieID').rating.mean()
    masterdf = pd.merge(masterdf, x, how='outer', on='movieID')
    # drop not useful columns
    masterdf = masterdf.drop(columns=['rating_x', 'userId', 'count'], axis=1)
    masterdf = masterdf.drop_duplicates()

    # genre specific DF for visualisation
    genredf = masterdf.loc[masterdf[params] == 1]
    genredf = genredf.sort_values(['rating_y'], ascending=[False])
    genredf = genredf.head(10)

    fig = plt.figure(figsize=[14, 7])

    # Creating a bar chart from the DataFrame df
    plt.bar(genredf.rating_y, genredf.title, width=0.5, linewidth=3)
    plt.grid()

    plt.title('Top 10 Movies', fontsize=15)
    plt.xlabel('Rating', fontsize=15)
    plt.ylabel('Movies', fontsize=15)
    plt.tight_layout()
    plt.show()

    img_bytes = io.BytesIO()
    fig.savefig(img_bytes)
    img_bytes.seek(0)
    return send_file(img_bytes, mimetype='image/png')

#Reccomend 10 movies by title using KNN
@app.route('/reccomend<params>')
def reccomend(params):
    conn = sl.connect('moviedatabase.db')
    curs = conn.cursor()
    # SQL queries to fetch the data and convert them to Dataframes
    stmt1 = "SELECT * FROM movies"
    stmt2 = "SELECT * FROM ratings"
    moviedata = curs.execute(stmt1)
    movie = pd.DataFrame(data=moviedata, columns=['movieId', 'title', 'genres'])
    movie = movie.iloc[1:, :]
    movie.drop(columns='genres', inplace=True)

    ratingsdata = curs.execute(stmt2)
    ratings = pd.DataFrame(data=ratingsdata, columns=['userId', 'movieId', 'rating', 'timestamp'])
    ratings = ratings.iloc[1:, :]
    ratings.drop(columns='timestamp', inplace=True)

    data = pd.merge(movie, ratings)
    data = data.iloc[:50000, :]
    user_movie_table = data.pivot_table(index="title", columns="userId", values="rating").fillna(0)
    print(user_movie_table.head())

    query_index = user_movie_table.index.get_loc(params)
    movie_features_df_matrix = csr_matrix(user_movie_table.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(movie_features_df_matrix)

    distances, indices = model_knn.kneighbors(user_movie_table.
                                              iloc[query_index, :].values.reshape(1, -1),
                                              n_neighbors=6)

    s = ''
    for i in range(0, len(distances.flatten())):

        if i == 0:
            s += ('Recommendations for {0}:\n'.format(user_movie_table.index[query_index]))
            s+= "          "

        else:
            s+=('{0}: {1}, with distance of {2}.'.format(i, user_movie_table.index[indices.flatten()[i]],
                                                       distances.flatten()[i]))
            s += "          "
    return s

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)



