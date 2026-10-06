import zipfile
import json
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os
 

def parse_dataset(zip_path="data/spotify_train_dataset.zip", output_dir="processed_data"):
    '''
    Función para procesar el dataset de entrenamiento. Lee los archivos JSON dentro del ZIP, 
    construye una matriz sparse de interacción entre playlists y tracks, y guarda la 
    matriz junto con los mapeos de playlists y tracks a índices numéricos.
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    # Listas para construir la matriz sparse
    rows, cols, values = [], [], []
    
    # Mapeos para convertir URIs de Spotify y pids a índices numéricos
    track_to_id = {} # Mapeo Track URI -> Índice
    id_to_track = [] # Lista de Track URIs por índice
    playlist_to_id = {} # Mapeo Playlist ID -> Índice
    id_to_playlist = [] # Lista de Playlist IDs por índice
 
    print(f"Abriendo {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zipf:
        # Nos quedamos con los archivos JSON dentro del ZIP
        json_files = [f for f in zipf.namelist() if f.endswith(".json")]
        
        # Se procesa cada archivo JSON
        for file_name in json_files:
            with zipf.open(file_name) as f:
                data = json.loads(f.read())
                
                # Acceder al array de playlists dentro del JSON
                playlists = data.get("playlists", [])
                
                # Para cada playlist en el archivo
                for pl in playlists:
                    pid = pl["pid"]
                    
                    # Registrar Playlist ID
                    if pid not in playlist_to_id:
                        # El pid ya es un entero, pero se hace igualmente el mapeo porque sino
                        # los índices podrían no ser consecutivos y provocar filas vacías en la matriz
                        playlist_to_id[pid] = len(id_to_playlist)
                        id_to_playlist.append(pid)
                    
                    # ID numérico de la playlist
                    current_pl_idx = playlist_to_id[pid]
                    
                    # Para cada canción en la playlist
                    for track in pl["tracks"]:
                        t_uri = track["track_uri"]
                        
                        # Registrar Track URI
                        if t_uri not in track_to_id:
                            track_to_id[t_uri] = len(id_to_track)
                            id_to_track.append(t_uri)
                        
                        # ID numérico de la canción
                        current_track_idx = track_to_id[t_uri]
                        
                        # Guardar la interacción (Playlist, Track) -> 1
                        rows.append(current_pl_idx)
                        cols.append(current_track_idx)
                        values.append(1)
            
            print(f"Procesado: {file_name}")
 
    # Crear la matriz CSR
    print("Construyendo matriz dispersa...")
    matrix = csr_matrix((values, (rows, cols)),
                        shape=(len(id_to_playlist), len(id_to_track)),
                        dtype=np.int8)
 
    # Guardamos la matriz dispersa
    print(f"Guardando archivos en {output_dir}...")
    save_npz(f"{output_dir}/train_matrix.npz", matrix)
    
    # Guardamos el mapeo de tracks a IDs y viceversa
    with open(f"{output_dir}/track_map.txt", "w") as f:
        # Guardamos solo las listas de URIs, el índice se infiere por la posición en la lista
        for track_uri in id_to_track:
            f.write(f"{track_uri}\n")
    
    # Guardamos el mapeo de playlists a IDs y viceversa
    with open(f"{output_dir}/playlist_map.txt", "w") as f:
        # Guardamos solo las listas de PIDs, el índice se infiere por la posición en la lista
        for pid in id_to_playlist:
            f.write(f"{pid}\n")
 
    print("Procesamiento completado.")
 



def parse_test_input(test_input_path="data/spotify_test_playlists/test_input_playlists.json", 
                     track_map_path="processed_data/track_map.txt", 
                     output_dir="processed_data"):
    '''
    Función para procesar el archivo de test. Lee el JSON de test, construye una matriz 
    sparse de interacción entre las playlists de test y los tracks (usando el mismo 
    mapeo de tracks que el dataset de entrenamiento), y guarda la matriz junto con el 
    mapeo de playlists de test a índices numéricos.
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Cargar el mapeo de tracks para mantener la consistencia con el dataset de entrenamiento
    with open(track_map_path, "r") as f:
        track_map = [line.strip() for line in f.readlines()]
    track_to_id = {uri: i for i, uri in enumerate(track_map)}
    
    # 2. Leer JSON de test
    with open(test_input_path, "r") as f:
        test_data = json.load(f)
    
    rows, cols, values = [], [], []
    test_playlist_to_id = {} # Nuevo mapeo para PIDs de test
    id_to_test_playlist = [] # Para reconstruir el PID original luego
    
    for pl in test_data["playlists"]:
        pid = pl["pid"]
        
        # Mapeamos los PID de test a índice consecutivo (0, 1, 2...)
        if pid not in test_playlist_to_id:
            test_playlist_to_id[pid] = len(id_to_test_playlist)
            id_to_test_playlist.append(pid)
        
        current_row_idx = test_playlist_to_id[pid]
        
        for track in pl["tracks"]:
            t_uri = track["track_uri"]
            if t_uri in track_to_id:
                track_idx = track_to_id[t_uri]
                rows.append(current_row_idx) 
                cols.append(track_idx)
                values.append(1)

    # 3. Crear matriz CSR con los datos de test
    num_test_playlists = len(id_to_test_playlist)
    test_matrix = csr_matrix((values, (rows, cols)),
                             shape=(num_test_playlists, len(track_map)),
                             dtype=np.int8)
    
    # 4. Guardar matriz y el mapeo de playlists de test
    save_npz(f"{output_dir}/test_matrix.npz", test_matrix)
    
    with open(f"{output_dir}/test_playlist_map.txt", "w") as f:
        for pid in id_to_test_playlist:
            f.write(f"{pid}\n")
            
    print(f"Archivo de test convertido a matriz sparse y guardado en {output_dir}/test_matrix.npz")
    print(f"Dimensiones de la matriz de test: {test_matrix.shape}")


if __name__ == "__main__":
    parse_dataset("data/spotify_train_dataset.zip")
    parse_test_input()