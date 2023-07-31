import sys
import os
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


def get_genre(df):
    model = pickle.load(open('resources/classifier.pkl', 'rb'))
    encoder = pickle.load(open('resources/lable_encoder.pkl', 'rb'))

    in_feat = ['energy', 'loudness', 'speechiness', 'acousticness',	'instrumentalness', 
            'liveness',	'valence', 'tempo', 'duration_ms']
    feats = df[in_feat]

    genres =  model.predict(feats)
    return encoder.inverse_transform(genres)

def add_temp_genres(df):
    df_temp = pd.read_csv('genres_v2.csv')
    genres = df_temp['genre'].unique()
    for genre in genres:
        df[genre] = 0
        df[genre] = df[genre].astype('uint8')
    return df

def get_clusters(df):
    to_drop = ['Unnamed: 0', 'title', "song_name", 'analysis_url', 'uri',
            'track_href', 'type', 'id', 'genre']
    drop_feats_available = []
    for feat in to_drop:
        if feat in df.columns:
            drop_feats_available.append(feat)

    df.drop(drop_feats_available, axis=1, inplace=True)

    scaler = pickle.load(open('resources/scaler.pkl', 'rb'))
    df_scaled = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

    fs = pickle.load(open('resources/feature_selector.pkl', 'rb'))
    df_main = pd.DataFrame(fs.transform(df_scaled))
    
    for column in df_main:
        if df_main[column].dtype == 'float64':
            df_main[column] = pd.to_numeric(df_main[column], downcast='float')
    
    genres = get_genre(df_scaled)
    df_gen = add_temp_genres(df_main)
    for index, genre in enumerate(genres):
        df_gen.loc[index, genre] = 1

    model = pickle.load(open('resources/model.pkl', 'rb'))
    return model.predict(df_gen)

def extract_fav_clusters(clusters):
    dict_clusters = {}
    for cluster in clusters:
        if cluster in dict_clusters:
            dict_clusters[cluster] += 1
        else:
            dict_clusters[cluster] = 1
    
    dict_clusters = sorted(dict_clusters, key=lambda item: dict_clusters[item], reverse=True)

    res = []
    for cluster in dict_clusters:
        res.append(cluster)

    rep_idx = 0
    while len(res) < 5:
        res.append(res[rep_idx])
        rep_idx += 1
    
    return res[:5]


def make_recommandations(clusters):
    fav_clusters = extract_fav_clusters(clusters)
    
    df_clustered = pd.read_csv('resources/musics_clustered.csv')
    
    mixs = [df_clustered[df_clustered['cluster'] == index].sample(n=5) for index in fav_clusters]

    for index, df in enumerate(mixs):
        df.drop('cluster', inplace=True, axis=1)
        df.to_csv(f'output/mix_{index + 1}.csv', index=False)

    df_comp = pd.concat([mixs[0], mixs[1], mixs[2], mixs[3], mixs[4]], axis=0)
    df_comp.to_csv(f'output/mix_comp.csv', index=False)


def main(args) -> None:
    """ Main function to be called when the script is run from the command line. 
    This function will recommend songs based on the user's input and save the
    playlist to a csv file.
    
    Parameters
    ----------
    args: list 
        list of arguments from the command line
    Returns
    -------
    None
    """
    arg_list = args[1:]
    if len(arg_list) == 0:
        print("Usage: python3 musicRecommender.py <csv file>")
        sys.exit()
    else:
        file_name = arg_list[0]
        if not os.path.isfile(file_name):
            print("File does not exist")
            sys.exit()
        else:
            userPreferences = pd.read_csv(file_name)

    print("Determining genres and clusters of input dataset (it may takes 1 or 2 mins to complete.) ...")
    clusters = get_clusters(userPreferences)

    print("Finding top 5 clusters for the user and save recommandations ...")
    make_recommandations(clusters)

    print("Recommandations saved successfuly.")

    # 1. Use your train model to make recommendations for the user.
    # 2. Output the recommendations as 5 different playlists with
    #    the top 5 songs in each playlist. (5 playlists x 5 songs)
    # 2.1. Musics in a single playlist should be from the same cluster.
    # 2.2. Save playlists to a csv file.
    # 3. Output another single playlist recommendation with all top songs from all clusters.



if __name__ == "__main__":
    # get arguments from command line
    args = sys.argv
    main(args)