#%%
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Replace with your credentials
client_id = 'f969f3a8f66a4d83854fcc79400cce2f'
client_secret = '6fd983d956454f92b82dea20cfa31334'

# Authenticate
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#%%
# Function to search for tracks by year
def search_tracks_by_year(year, limit=50, offset=0):
    query = f'year:{year}'
    results = sp.search(q=query, type='track', limit=limit, offset=offset)
    return results['tracks']['items']

# Function to collect tracks
def collect_tracks(start_year, end_year, max_tracks=50000):
    tracks = []
    current_year = start_year

    while current_year >= end_year and len(tracks) < max_tracks:
        offset = 0
        while True:
            try:
                results = search_tracks_by_year(current_year, limit=50, offset=offset)
                if not results:
                    break
                tracks.extend(results)
                offset += 50
                if len(tracks) >= max_tracks:
                    break
                time.sleep(1)  # Respect rate limits
            except Exception as e:
                print(f"Error occurred: {e}")
                break
        current_year -= 1

    return tracks[:max_tracks]

#%%
# Collect tracks from 2020 to 2024
start_year = 2024
end_year = 2000  # Adjust this to the earliest year you want to include
max_tracks = 50000

# tracks = collect_tracks(start_year, end_year, max_tracks)

genres = ['rock', 'pop', 'ballad', 'electronic', 'jazz', 'latin'
          'classical', 'indie', 'country', 'r&b', 'hip-hop', 'lofi']


def search_tracks_by_year_and_genre(year, genre, limit=50, offset=0):
    query = f'year:{year} genre:{genre}'
    results = sp.search(q=query, type='track', limit=limit, offset=offset)
    return results['tracks']['items']

# Collect tracks
tracks = []
for year in range(2000, 2025):  # From 2000 to 2024
    for genre in genres:
        offset = 0
        while True:
            try:
                results = search_tracks_by_year_and_genre(year, genre, limit=50, offset=offset)
                if not results:
                    break
                tracks.extend(results)
                offset += 50
                if len(tracks) >= 50000:  # Stop if you reach 50,000 tracks
                    break
                time.sleep(1)  # Respect rate limits
            except Exception as e:
                print(f"Error occurred: {e}")
                break
        if len(tracks) >= 50000:
            break
    if len(tracks) >= 50000:
        break

print(f"Collected {len(tracks)} tracks.")