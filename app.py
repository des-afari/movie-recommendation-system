import streamlit as st
import pickle

def load_data():
    with open('recommender.pkl', 'rb') as file:
        data = pickle.load(file)
        return data

def get_recommendations(movie_title, model, similarity_df):
    ix = model.kneighbors(similarity_df[movie_title].to_numpy().reshape(1, -1), return_distance=False)
    closest_titles = similarity_df.columns[ix[0]].tolist()
    return closest_titles

data = load_data()

df = data['df']
similarity_df = data['similarity_df']
model = data['model']
titles = df['title']

st.write('# Movie Recommendation System')
title = st.selectbox('title', titles)
submit = st.button('get recommendations')

if submit:
    movies = get_recommendations(title, model, similarity_df)
    movies.pop(0)
    st.write(movies)