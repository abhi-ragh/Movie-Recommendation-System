print("Loading... Wait Patiently")

import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import subprocess


credits_df = pd.read_csv("credits.csv") 
movies_df = pd.read_csv("movies.csv")

movies_df = movies_df.merge(credits_df, on='title') #merge credits and movies together
movies_df = movies_df[['movie_id','title','overview','genres','keywords','cast','crew']] #use only the columns we will use

movies_df.dropna(inplace=True) #drop null values


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies_df["genres"] = movies_df["genres"].apply(convert)  
movies_df['keywords'] = movies_df['keywords'].apply(convert)

def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
        return L

movies_df['cast'] = movies_df['cast'].apply(convert3)

movies_df.dropna(inplace=True) #drop null again coz cast a bitch

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies_df['crew'] = movies_df['crew'].apply(fetch_director)

movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())

#remove spaces from data 
movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ",'')for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ",'')for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ",'')for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ",'')for i in x])
 
movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']

new_df = movies_df[['movie_id','title','tags']]
#new_df['tags'] = new_df['tags'].apply(lambda x:' '.join(x))
#new_df['tags'] = new_df['tags'].apply(lambda X:X.lower())

new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())



cv = CountVectorizer(max_features=5000,stop_words='english')

cv.fit_transform(new_df['tags']).toarray().shape
vectors = cv.fit_transform(new_df['tags']).toarray()

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

#new_df['tags'] = new_df['tags'].apply(stem)
new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)),reverse=True,key= lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# Clear the terminal screen
subprocess.call('clear', shell=True)

print('''  __  __            _        _____                                                   _       _   _                _____           _                 
 |  \/  |          (_)      |  __ \                                                 | |     | | (_)              / ____|         | |                
 | \  / | _____   ___  ___  | |__) |___  ___ ___  _ __ ___  _ __ ___   ___ _ __   __| | __ _| |_ _  ___  _ __   | (___  _   _ ___| |_ ___ _ __ ___  
 | |\/| |/ _ \ \ / / |/ _ \ |  _  // _ \/ __/ _ \| '_ ` _ \| '_ ` _ \ / _ \ '_ \ / _` |/ _` | __| |/ _ \| '_ \   \___ \| | | / __| __/ _ \ '_ ` _ \ 
 | |  | | (_) \ V /| |  __/ | | \ \  __/ (_| (_) | | | | | | | | | | |  __/ | | | (_| | (_| | |_| | (_) | | | |  ____) | |_| \__ \ ||  __/ | | | | |
 |_|  |_|\___/ \_/ |_|\___| |_|  \_\___|\___\___/|_| |_| |_|_| |_| |_|\___|_| |_|\__,_|\__,_|\__|_|\___/|_| |_| |_____/ \__, |___/\__\___|_| |_| |_|
                                                                                                                         __/ |                      
                                                                                                                        |___/                       ''')

while(True):
    try:
        movie_input = input("\nEnter Movie : ")
        print("\nRecommendations Based On Your Preference:\n")
        recommend(movie_input)
    except:
        print("Movie Not Found\n")

    c = input("Continue(Y/N):\n")
    if c=='N' or c=='n':
        break

print("Thank You")
subprocess.call('clear', shell=True)