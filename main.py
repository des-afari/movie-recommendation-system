import pickle
import pandas as pd
from fastapi import FastAPI
from schema import MovieSchema

app = FastAPI()

@app.get('/')
async def root():
    return {"message": "connection established"}


@app.get('/titles', status_code=200)
async def get_titles():
   with open('recommender.pkl', 'rb') as file:
    data = pickle.load(file)

    df = data['df']

    return {"titles": df.title}


@app.post('/recommendations', status_code=201)
async def get_movies(schema: MovieSchema):
   with open('recommender.pkl', 'rb') as file:
    data = pickle.load(file)

    df = data['df']
    similarity_df = data['similarity_df']
    model = data['model']

    def get_recommendations(movie_title, similarity_df, df):
        ix = model.kneighbors(similarity_df[movie_title].to_numpy().reshape(1, -1), return_distance=False)
        closest = similarity_df.columns[ix].flatten()
        recommended_movies = pd.DataFrame(closest, columns=['title']).merge(df)

        return recommended_movies.to_dict(orient='records')
   
    movies = get_recommendations(schema.title, similarity_df, df)

    return movies


