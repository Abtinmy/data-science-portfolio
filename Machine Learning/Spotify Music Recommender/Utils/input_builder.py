import requests
from secrets import *
import base64
import json
import pandas as pd


aut_url = 'https://accounts.spotify.com/api/token'

def get_access_token(client_id, client_secret):
    msg = f"{client_id}:{client_secret}"
    msg_bytes = msg.encode('ascii')
    base64_bytes = base64.b64encode(msg_bytes)
    base64_msg = base64_bytes.decode('ascii')

    aut_header = {
        'Authorization': 'Basic ' + base64_msg
    }

    aut_data = {
        'grant_type': 'client_credentials'
    }

    res = requests.post(aut_url, headers=aut_header, data=aut_data)
    res_obj = res.json()

    return res_obj['access_token']

def get_playlist_tracks(token, playlist_id):
    endpoint = f'https://api.spotify.com/v1/playlists/{playlist_id}'
    get_header = {
        'Authorization': 'Bearer ' + token
    }

    res = requests.get(endpoint, headers=get_header)

    return res.json()

def get_genres(token, ids):

    endpoint = f'https://api.spotify.com/v1/tracks'
    get_header = {
        'Authorization': 'Bearer ' + token
    }

    tracks = []
    for id in ids[0:5]:
        print(id)
        params = (('id', str(id)), )

        res = requests.get(endpoint, headers=get_header, params=params)
        tracks.append(res.json())
        print(tracks)

    # genres = []
    # for track in tracks['tracks']:
    #     genres.append(track['artists'][0]['genres'])
    return tracks


def main():
    token = get_access_token(client_id, client_secret)
    print(token)

    # token = 'BQBySX81asqfaeb99iGiE_IfoKq16Sp_UH4_bbVg3zewUsFXcxM5fjrjdSaWREIxEYO5dBvtKfM2KT8Sfbk'

    # playlist_id = '3xh8euz9SF1lGHNgqq6bcX?si=b5dcaaf9255d47aa'
    # tracks = get_playlist_tracks(token, playlist_id)

    df = pd.read_csv('data/input/input_tracks.csv')
    genres = get_genres(token, df['id'])
    print(genres)

if __name__ == '__main__':
    main()