#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import BytesIO
from IPython.display import Image
from googletrans import Translator, LANGUAGES
from IPython.display import HTML
import ast


# In[2]:


df_final = pd.read_csv('base_complete_finale.csv', sep = ',', low_memory=False)


# In[5]:


#df_final['genres'] = df_final['genres'].str.split(',')

df_final['actor'] = df_final['actor'].apply(ast.literal_eval)
df_final['actress'] = df_final['actress'].apply(ast.literal_eval)
df_final['nb_actor']= df_final['actor'].apply(len)
df_final['nb_actress']= df_final['actress'].apply(len)



translator = Translator()
# Clé API Tmdb : 
api_key = '2318d49dc7b23307a950da1ce4326e24'



def onglet0(): 
        
    st.markdown("<h1 style='text-align: center;'>CINETFLIX A L'AIR DU DIGITAL</h1>", unsafe_allow_html=True)    

    st.write('')
    st.write('')


    
    st.markdown("<h3>Le cinéma Cinetflix passe le cap du digital !!!</h3>", unsafe_allow_html=True) 
    
    st.markdown("<h5>Situé dans la Creuse, Cinetflix franchit un nouveau cap technologique.</h5>", unsafe_allow_html=True) 
    
    st.markdown("<h5>Mandaté par Cinetflix, CineHackers vous propose une application de recommandation de films ainsi que quelques indicateurs sur la base de données mise à disposition.</h5>", unsafe_allow_html=True) 
    st.write('')
    st.write('')

    old_film = "https://drive.google.com/uc?export=view&id=1UofoYQzuXI_-XA01VxtnDkrNUGi0IYBr"
    st.image(old_film) 
    


def onglet1():
   

    
    st.markdown("<h1 style='text-align: center;'>Informations sur les films</h1>", unsafe_allow_html=True)
    


    total_films = len(df_final)

    fig = go.Figure()

    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Nombre total de films: {:,}".format(total_films),
        showarrow=False,
        font=dict(size=35, color="black", family="Arial, sans-serif"),
        align="center",
        bordercolor="black",
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        opacity=0.8
    )

    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        paper_bgcolor='rgba(0,0,0,0)',  
        plot_bgcolor='rgba(0,0,0,0)', 
        height=100,  
    )
  
    st.plotly_chart(fig)

    
    
    st.subheader('Répartition des films par note')

    figure = make_subplots(
    rows=1, cols=1)



    # Répartition des films par note
    figure.add_trace(
        go.Histogram(
            x=df_final['averageRating'],
            marker=dict(color='Cadetblue', line=dict(color='black', width=1)),
            name='Note',
            hoverinfo='x+y',
            hovertemplate='Note moyenne: %{x}<br>Nombre de films: %{y}'
        ),
        row=1, col=1
    )


    figure.update_xaxes(title_text="Note moyenne", row=1, col=1)
    figure.update_yaxes(title_text="Nombre de films", row=1, col=1, color = "white")

    
    figure.update_layout(height=500, yaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=1))
  


    st.plotly_chart(figure)

    
    st.subheader('Evolution de la parité H/F')
    
    figure2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Évolution du nb d'acteurs.ices au fil des années",
                        "Évolution du nb d'acteurs.ices par genre")
    )

    # Pour le premier graphique evolution de la parité H/F par année
    df_grouped = df_final.groupby('startYear').agg({'nb_actor': 'mean', 'nb_actress': 'mean'}).reset_index()

    figure2.add_trace(
        go.Scatter(
        x=df_grouped['startYear'],
        y=df_grouped['nb_actor'],
        mode='lines',
        name='Acteurs',
        line=dict(color='lightblue'),  # Choisissez la couleur que vous préférez
        hoverinfo='x+y',
        hovertemplate='Année: %{x}<br>Nombre d\'acteurs: %{y}'
    ),
        row=1, col=1
    )
    figure2.add_trace(
        go.Scatter(
        x=df_grouped['startYear'],
        y=df_grouped['nb_actress'],
        mode='lines',
        name='Actrices',
        line=dict(color='orange'),  # Choisissez la couleur que vous préférez
        hoverinfo='x+y',
        hovertemplate='Année: %{x}<br>Nombre d\'actrices: %{y}'
    ),
        row=1, col=1
    )


    # Pour le premier graphique evolution de la parité H/F par genre
    df_exploded = df_final.assign(genres=df_final['genres'].apply(lambda x: eval(x))).explode('genres')
    df_grouped2 = df_exploded.groupby('genres').agg({'nb_actor': 'mean', 'nb_actress': 'mean'}).reset_index()

    figure2.add_trace(
        go.Scatter(
        x=df_grouped2['genres'],
        y=df_grouped2['nb_actor'],
        mode='lines',
        name='Acteurs',
        line=dict(color='lightblue'),  # Choisissez la couleur que vous préférez
        hoverinfo='x+y',
        hovertemplate='Genre: %{x}<br>Nombre d\'acteurs: %{y}'
    ),
        row=2, col=1
    )
    figure2.add_trace(
        go.Scatter(
        x=df_grouped2['genres'],
        y=df_grouped2['nb_actress'],
        mode='lines',
        name='Actrices',
        line=dict(color='orange'),  # Choisissez la couleur que vous préférez
        hoverinfo='x+y',
        hovertemplate='Genre: %{x}<br>Nombre d\'actrices: %{y}'
    ),
        row=2, col=1
    )


    # Mise à jour des axes et des lignes de marqueur pour chaque trace
    figure2.update_xaxes(title_text="Année", row=1, col=1)
    figure2.update_xaxes(title_text="Genre", row=1, col=2)

    figure2.update_yaxes(title_text="Nombre d'acteurs.ices",showgrid=True, gridcolor='lightgrey', gridwidth=1, row=1, col=1)
    figure2.update_yaxes(title_text="Nombre d'acteurs.ices",showgrid=True, gridcolor='lightgrey', gridwidth=1, row=2, col=1)

    # Mise à jour du layout si nécessaire, par exemple pour ajuster la hauteur totale de la figure
    figure2.update_layout(height=800)
    
    st.plotly_chart(figure2)

    
    
    st.subheader('Analyse des votants')
    
    
    df_grouped_3 = df_exploded.groupby('genres').agg({'numVotes': 'mean'}).reset_index()

    figure3 = make_subplots(
        rows=2, cols=1,subplot_titles=("Moyenne de votants par genre",
                        "Moyenne de votants par Année")
        )



    # Répartition des films par note
    figure3.add_trace(
        go.Bar(
            x=df_grouped_3['genres'],
            y = df_grouped_3['numVotes'],
            marker=dict(color='alice blue', line=dict(color='black', width=1)),
            name='Nb votes moyen',
            hoverinfo='x+y',
            hovertemplate='Genre: %{x}<br>Nombre de votes moyen: %{y}'
        ),
        row=1, col=1
    )
    
    df_grouped4 = df_final.groupby('startYear').agg({'numVotes': 'mean'}).reset_index()

    figure3.add_trace(
        go.Scatter(
        x=df_grouped4['startYear'],
        y=df_grouped4['numVotes'],
        mode='lines',
        fill='tozeroy',
        name='Nb votes moyen',
        line=dict(color='aqua'),  
        hoverinfo='x+y',
        hovertemplate='Année: %{x}<br>Nombre moyen de votants: %{y}'
    ),
        row=2, col=1
    )



    figure3.update_xaxes(title_text="Genres", row=1, col=1, categoryorder='total descending')
    figure3.update_yaxes(title_text="Nombre votes moyen", showgrid=True, gridcolor='lightgrey', gridwidth=1, row=1, col=1, color = "white")
    figure3.update_xaxes(title_text="Année", row=2, col=1)
    figure3.update_yaxes(title_text="Nombre votes moyen", showgrid=True, gridcolor='lightgrey', gridwidth=1, row=2, col=1)
    
    figure3.update_layout(height=800)
    
    st.plotly_chart(figure3)
    
    
    


    
    
    

def onglet2():

    st.markdown("<h1 style='text-align: center;'>Système de recommandations</h1>", unsafe_allow_html=True)

    

    titles = [""] + list(df_final["title"])
    actors = [""] + list(df_final["actors"].apply(lambda x: eval(x)).explode().unique())
    movie_user = st.selectbox("Sélectionnez le nom du film",titles)
    actors_user = st.selectbox("Sélectionnez le nom d'un acteur",actors)
    nb_options = list(range(1, 11))  # Options from 1 to 10
    nb = st.selectbox("Nombre de recommandations", nb_options, index=4)  

    col1, col2 = st.columns(2)

    if col1.button("Titre recommandations"):
        recommendationstitre(movie_user, nb)

    elif col2.button("Acteur recommandations"):
        recommendationsacteurs(actors_user, nb)
        
        
def recommendationsacteurs(actors_user, nb):
  
    tfidf2 = TfidfVectorizer()
    count_matrix2 = tfidf2.fit_transform(df_final['actors'])

    X2 = count_matrix2
    NN2 = NearestNeighbors(metric='cosine', n_neighbors=nb + 1)
    NN2.fit(X2)
    if actors_user =="":
        st.write("Vous n'avez pas saisie d'acteur, veuillez saisir un acteur")
    else :
        try:
            film_index2 = df_final[df_final['actors'].str.contains(actors_user)].index[0]
            dist2, indices2 = NN2.kneighbors(count_matrix2[film_index2:film_index2 + 1])

            indices2 = indices2.flatten()

            recommended_films2 = df_final.iloc[indices2[1:]]

            recommended_films2['distance'] = dist2.flatten()[1:]

            recommended_films2['overview_french'] = recommended_films2['overview'].apply(translate_text)



            num_cols = 2  
            num_rows = (len(recommended_films2) + 1) // num_cols  
            cols = st.columns(num_cols) 

            film_count2 = 0

            for index, row in recommended_films2.iterrows():
                if film_count2 % 2 == 0:
                    cols = st.columns(2)  
                film_count2 += 1
                with cols[film_count2 % 2]:
                    st.markdown(f"<h2 style='font-size:24px'><strong>{row['title']}</strong></h2>", unsafe_allow_html=True)
                    st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}", width=300)
                    st.write("Synopsis : ")
                    st.write(row['overview_french'])
                    url_videos = f"https://api.themoviedb.org/3/movie/{row['tconst']}/videos?api_key={api_key}"
                    response_videos = requests.get(url_videos)

                    if response_videos.status_code == 200:
                        videos_data = response_videos.json()

                        for video in videos_data['results']:
                            # Vérifier si la vidéo est une bande-annonce sur YouTube
                            if video['site'] == "YouTube" and "Trailer" in video['type']:
                                youtube_id = video['key']
                                url = f"https://www.youtube.com/watch?v={youtube_id}"
                                st.write(f"Bande-annonce YouTube : {url}")

                                # Utiliser st.video pour afficher la vidéo dans Streamlit
                                st.video(url)
                                break  # Arrêter après avoir trouvé la première bande-annonce
                    else:
                        st.error("Erreur lors de la requête API pour les vidéos")


            if film_count2 % 2 != 0:
                cols[film_count2 % 2].empty()



        except IndexError:
            st.write("Film non trouvé. Veuillez entrer un titre de film valide.")


def recommendationstitre(movie_user, nb):
    df_final['combined_features'] = (df_final['genres'].apply(lambda x: ', '.join(map(str, x))) + ' ' +
                                     df_final['title'] + ' ' + df_final['director'] + ' ' +
                                     df_final['actors']) #+ ' ' + df_final['overview_lemma'])

    tfidf = TfidfVectorizer()
    count_matrix = tfidf.fit_transform(df_final['combined_features'])

    X = count_matrix
    NN = NearestNeighbors(metric='cosine', n_neighbors=nb + 1)
    NN.fit(X)
    
    if movie_user =="":
        st.write("Vous n'avez pas saisie de film, veuillez saisir un film")
    else :
        try:
            film_index = df_final[df_final['title'] == movie_user].index[0]
            dist, indices = NN.kneighbors(count_matrix[film_index:film_index + 1])

            indices = indices.flatten()

            recommended_films = df_final.iloc[indices[1:]]

            recommended_films['distance'] = dist.flatten()[1:]

            recommended_films['overview_french'] = recommended_films['overview'].apply(translate_text)



            num_cols = 2  
            num_rows = (len(recommended_films) + 1) // num_cols  
            cols = st.columns(num_cols) 

            film_count = 0

            for index, row in recommended_films.iterrows():
                if film_count % 2 == 0:
                    cols = st.columns(2)  
                film_count += 1
                with cols[film_count % 2]:
                    st.markdown(f"<h2 style='font-size:24px'><strong>{row['title']}</strong></h2>", unsafe_allow_html=True)
                    st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}", width=300)
                    st.write("Synopsis : ")
                    st.write(row['overview_french'])
                    url_videos = f"https://api.themoviedb.org/3/movie/{row['tconst']}/videos?api_key={api_key}"
                    response_videos = requests.get(url_videos)

                    if response_videos.status_code == 200:
                        videos_data = response_videos.json()

                        for video in videos_data['results']:
                            # Vérifier si la vidéo est une bande-annonce sur YouTube
                            if video['site'] == "YouTube" and "Trailer" in video['type']:
                                youtube_id = video['key']
                                url = f"https://www.youtube.com/watch?v={youtube_id}"
                                st.write(f"Bande-annonce YouTube : {url}")

                                # Utiliser st.video pour afficher la vidéo dans Streamlit
                                st.video(url)
                                break  # Arrêter après avoir trouvé la première bande-annonce
                    else:
                        st.error("Erreur lors de la requête API pour les vidéos")


            if film_count % 2 != 0:
                cols[film_count % 2].empty()



        except IndexError:
            st.write("Film non trouvé. Veuillez entrer un titre de film valide.")


def translate_text(text):
    if pd.isna(text):
        return text
    try:
        return translator.translate(text, src='en', dest='fr').text
    except Exception as e:
        print(f"Erreur lors de la traduction: {e}")
        return text    
    
     
    
    
def main():
    url2 = 'https://drive.google.com/uc?export=view&id=1e57cwl_8Pv_733qiClKvLpF_QVA2FvZP'
    st.sidebar.image(url2)

    st.sidebar.title("Navigation")
    onglet_selectionne = st.sidebar.selectbox("Sélectionner un onglet", ["CINETFLIX A L'AIR DU DIGITAL","Informations sur les films", "Système de recommandations"])
 


    if onglet_selectionne == "CINETFLIX A L'AIR DU DIGITAL":
        onglet0()
    elif onglet_selectionne == "Informations sur les films":
        onglet1()
    elif onglet_selectionne == "Système de recommandations":
        onglet2()

if __name__ == "__main__":
    main()




