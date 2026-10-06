from os import environ
environ["OPENBLAS_NUM_THREADS"] = "8" # Número de hilos (ajustar según CPU)

import os
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import json
import gzip
from scipy.sparse import vstack
from tqdm import tqdm
from scipy.sparse.linalg import svds
import tensorflow as tf

### RECOMENDADOR 0: Basado en popularidad global de tracks (top 500 más populares sin repetir)
def recommender_top500(matrix_path="processed_data/train_matrix.npz",
                        track_map_path="processed_data/track_map.txt",
                        test_playlists_path="data/spotify_test_playlists/test_input_playlists.json"):
    
    # 1. Cargar la matriz sparse de playlists y tracks
    matrix = load_npz(matrix_path)

    # 2. Cargar los mapeos de tracks
    with open(track_map_path, "r") as f:
        id_to_track = [line.strip() for line in f]

    # 3. Calcular popularidad de cada track (número de playlists en las que aparece)
    print("Calculando popularidad de tracks...")
    track_popularity = np.array(matrix.sum(axis=0)).flatten()

    # 4. Obtener los índices de los tracks más populares (lista de índices ordenados por popularidad)
    top_tracks_indices = np.argsort(track_popularity)[::-1]

    # # Comprobación: Imprimir los tracks más populares
    top_k = 10
    print(f"\nTop {top_k} tracks más populares:")

    for idx in top_tracks_indices[:top_k]:
        track_uri = id_to_track[idx]
        popularity = track_popularity[idx]
        print(f"Índice: {idx} - Track: {track_uri} - Popularidad: {popularity}")

    # 5. Cargar las playlists de test para recomendar
    with open(test_playlists_path, "r") as f:
        test_data = json.load(f)

    # 6. Generar recomendaciones para cada playlist de test basándonos en los tracks más populares
    # Nos aseguramos de no recomendar tracks que ya estén en la playlist
    print("\nGenerando recomendaciones (500 populares)...")

    recommendations = {}
    for pl in test_data["playlists"]:
        pid = pl["pid"]
        
        # Obtener URIs de canciones que ya están en la playlist (para no repetirlas)
        seed_tracks = set(t["track_uri"] for t in pl["tracks"])
        
        recs = []
        for idx in top_tracks_indices:
            track_uri = id_to_track[idx]
            
            # Solo añadir si no estaba ya en la playlist
            if track_uri not in seed_tracks:
                recs.append(track_uri)
            
            # Parar cuando tengamos 500 recomendaciones
            if len(recs) == 500:
                break
        
        recommendations[pid] = recs

    print("Recomendaciones generadas para todas las playlists de test.")

    return recommendations

### FUNCIÓN AUXILIAR: Calcular las canciones más populares
def calculate_popular_tracks(train_matrix):
    print("Calculando popularidad de tracks como fallback...")
    
    track_popularity = np.array(train_matrix.sum(axis=0)).flatten()
    top_tracks_indices = np.argsort(track_popularity)[::-1]
    
    return top_tracks_indices


### RECOMENDADOR 2: Filtrado colaborativo basado en vecinos precalculados (User-Based o Item-Based)
def recommender_collaborative(mode='user', 
                              train_matrix_path="processed_data/train_matrix.npz",
                              test_matrix_path="processed_data/test_matrix.npz",
                              track_map_path="processed_data/track_map.txt",
                              test_playlists_path="data/spotify_test_playlists/test_input_playlists.json",
                              k=150):
    
    # 1. Cargar datos
    R_train = load_npz(train_matrix_path) # Matriz de train
    
    with open(track_map_path, "r") as f:
        id_to_track = [line.strip() for line in f] # Mapeo de índices a URIs de tracks
    
    with open(test_playlists_path, "r") as f:
        test_data = json.load(f) # Playlists de test

    knn_data = np.load(f"processed_data/{mode}_knn_data.npz") # Vecinos precalculados (índices y similitudes)
    
    # Cortamos a los K vecinos más cercanos
    neigh_idx = knn_data['indices'][:, :k]
    neigh_sim = knn_data['similarities'][:, :k]
    knn_data.close()
    
    # 2. Calculamos popularidad para fallback en caso de playlists sin vecinos o sin tracks
    top_tracks_indices = calculate_popular_tracks(R_train)
    top_popular_idx = top_tracks_indices[:600]

    recommendations = {}

    if mode == 'user':
        # 3. Construimos la matriz sparse de similitudes entre playlists de test y sus playlists vecinas
        print(f"Construyendo matriz S y calculando scores (USER-BASED con k={k})...")

        # Arrays para construir la matriz S de similitudes entre playlists y sus playlists vecinas
        rows = np.repeat(np.arange(neigh_idx.shape[0]), neigh_idx.shape[1] )
        cols = neigh_idx.flatten()
        data = neigh_sim.flatten()
        
        # Filtramos aquellos vecinos con similitud > 0 (por si acaso no hay k vecinos no ocupar con ceros)
        mask = data > 0

        # Matriz S (playlists de test x playlists de train) con las similitudes
        S = csr_matrix((data[mask], (rows[mask], cols[mask])), shape=(neigh_idx.shape[0], R_train.shape[0]))
        
        # 4. Calcular los ratings para cada canción en cada playlist de test
        # - La multiplicación con CSR solo calcula los ratings para las canciones en las que no hay ceros (reducción de catálogo)
        ratings = S.dot(R_train) 


    elif mode == 'item':
        print(f"Construyendo matriz S y calculando scores (ITEM-BASED con k={k})...")
        R_test = load_npz(test_matrix_path)
        
        # 3. Construimos la matriz sparse de similitudes entre canciones y sus canciones vecinas
        indptr = np.arange(0, neigh_idx.shape[0] * neigh_idx.shape[1] + 1, neigh_idx.shape[1], dtype=np.int32)
        indices = neigh_idx.ravel()
        data = neigh_sim.ravel()

        # Matriz S_item (canciones x canciones)
        S_item = csr_matrix((data, indices, indptr), shape=(neigh_idx.shape[0], neigh_idx.shape[0]))
        S_item.eliminate_zeros()
        
        # 4. Calcular los ratings
        # Multiplicamos las playlists de test por las similitudes de las canciones
        ratings = R_test.dot(S_item)

    else:
        print("Modo no reconocido. Usa 'user' o 'item'.")
    
    # 5. Generar recomendaciones para cada playlist de test
    print("Generando recomendaciones finales...")
    for i, pl in enumerate(test_data["playlists"]):
        pid = pl["pid"]
        seed_uris = set(t["track_uri"] for t in pl["tracks"])

        # Obtenemos la fila de ratings para esta playlist de test          
        ratings_row = ratings.getrow(i)
        
        # Vector de índices de canciones (columnas de R_Train con rating > 0 para esta playlist de test)
        song_idx = ratings_row.indices
        # Rating calculado de cada canción
        rating_song = ratings_row.data
        # Ordenamos por rating calculado
        top_local = np.argsort(-rating_song)[:500 + len(seed_uris)]
        
        recs = []
        for idx in top_local:
            uri = id_to_track[song_idx[idx]]
            if uri not in seed_uris:
                recs.append(uri)
            if len(recs) == 500: 
                break

        # Si no se llega a 500 recomendaciones, rellenamos con los más populares
        if len(recs) < 500:
            for idx in top_popular_idx:
                uri = id_to_track[idx]
                if uri not in seed_uris and uri not in recs:
                    recs.append(uri)
                if len(recs) == 500: 
                    break
        
        recommendations[pid] = recs

    return recommendations


### RECOMENDADOR 3: PureSVD

def recommender_puresvd(train_matrix_path="processed_data/train_matrix.npz",
                        test_matrix_path="processed_data/test_matrix.npz",
                        track_map_path="processed_data/track_map.txt",
                        test_playlists_path="data/spotify_test_playlists/test_input_playlists.json",
                        mode=1,
                        k=50):
    
    # 1. Cargar datos
    train_matrix = load_npz(train_matrix_path).astype(np.float32)
    test_matrix = load_npz(test_matrix_path).astype(np.float32)

    with open(track_map_path, "r") as f:
        id_to_track = [line.strip() for line in f] # Mapeo de índices a URIs de tracks
        
    with open(test_playlists_path, "r") as f:
        test_data = json.load(f) # Playlists de test

    pids = [pl["pid"] for pl in test_data["playlists"]]

    recommendations = {}

    if mode == 1:

        # 2. Calcular SVD de la matriz combinada
        print(f"Calculando SVD sobre la matriz concatenada (k={k})...")

        # Concatenamos para proyectar a todos los usuarios en el mismo espacio latente
        combined_matrix = vstack([train_matrix, test_matrix])
        U, Sigma, Vt = svds(combined_matrix, k=k)

        # Extraemos solo las filas correspondientes a los usuarios de test
        U_test = U[train_matrix.shape[0]:, :]

        # Multiplicamos por Sigma antes del bucle
        U_test_scaled = U_test * Sigma

    else:

        print(f"Calculando SVD sobre la matriz de entrenamiento (k={k})...")

        _, Sigma, Vt = svds(train_matrix, k=k)

        # Proyectar las playlists de test en el espacio latente
        # Para calcular la proyección, se multiplica por la inversa de Sigma y, posteriormente,
        # para obtener los ratings, se multiplica de nuevo por Sigma.
        # Se optimiza el código evitando las multiplicaciones
        U_test_scaled = test_matrix.dot(Vt.T)

    # 3. Calcular los ratings por bloques
    chunk_size = 50
    
    print(f"Calculando scores por bloques de {chunk_size}...")
    
    with tqdm(total=U_test_scaled.shape[0], desc="Generando recomendaciones") as pbar:
        
        for start_idx in range(0, U_test_scaled.shape[0], chunk_size):
            
            # Cortamos a los usuarios del bloque actual
            end_idx = min(start_idx + chunk_size, U_test_scaled.shape[0])
            chunk = U_test_scaled[start_idx:end_idx]

            # Obtenemos los scores
            chunk_ratings = chunk.dot(Vt)

            # Para cada playlist, establecemos a -inf los tracks que ya están en la playlist
            test_chunk = test_matrix[start_idx:end_idx]
            rows, cols = test_chunk.nonzero()
            chunk_ratings[rows, cols] = -np.inf

            top_500_unsorted_idx = np.argpartition(chunk_ratings, -500, axis=1)[:, -500:]
            top_500_values = np.take_along_axis(chunk_ratings, top_500_unsorted_idx, axis=1)
            
            # Ordenamos localmente los 500 candidatos
            local_sort_idx = np.argsort(-top_500_values, axis=1)
            top_indices_block = np.take_along_axis(top_500_unsorted_idx, local_sort_idx, axis=1)
            
            del chunk_ratings
            
            # 5. Generar recomendaciones para cada playlist de este bloque
            for i in range(chunk.shape[0]):

                global_idx = start_idx + i 
                pid = pids[global_idx]
                
                # Obtenemos el vector de índices ordenados para esta playlist
                top_local = top_indices_block[i]
                
                recs = [id_to_track[idx] for idx in top_local]
                    
                recommendations[pid] = recs
                
            pbar.update(chunk.shape[0])

    return recommendations


### RECOMENDADOR 4: SLIM Y FISM

def recommender_it3(test_matrix_trimmed_path, track_map_path, test_playlists_path, weights_path):
    
    trained_S = np.load(weights_path)

    # La recoemndacion se hace multiplicando la matriz de test por la matriz de pesos entrenada
    test_matrix = load_npz(test_matrix_trimmed_path).astype(np.float32)
    scores = test_matrix.dot(trained_S)

    # Obtenemos las canciones que ya están en cada playlist de test
    rows, cols = test_matrix.nonzero()
    # Establecemos sus scores a menos infinito para que queden al final al ordenar
    scores[rows, cols] = -np.inf

    # Cargamos los mapeos de tracks y playlists
    with open(track_map_path, "r") as f:
        id_to_track = [line.strip() for line in f]
    
    # Cargamos las playlists de test para recomendar
    with open(test_playlists_path, "r") as f:
        test_data = json.load(f)
    
    pids = [pl["pid"] for pl in test_data["playlists"]]

    # Cargar la matriz de entrenamiento para calcular la popularidad
    train_matrix_path = "processed_data/train_matrix_trimmed.npz"
    train_matrix = load_npz(train_matrix_path).astype(np.float32)
    
    # Calcular la popularidad de cada track
    print("Calculando popularidad para desempate...")
    track_popularity = np.array(train_matrix.sum(axis=0)).flatten()

    recommendations = {}
    
    # Definimos la estructura del array que usaremos para ordenar por score y, en caso de empate, por popularidad
    conjunto = [('score', np.float32), ('pop', np.int32)]
    
    print("Generando recomendaciones finales (usando Arrays Estructurados)...")
    
    for i, pid in enumerate(pids):
        # Obtenemos el vector de scores para esta playlist de test
        rating_song = scores[i]
        
        # Creamos el array con la estructura definida
        score_pop = np.zeros(len(rating_song), dtype=conjunto)
        
        # Rellenamos el array con los scores y la popularidad
        score_pop['score'] = -rating_song
        score_pop['pop'] = -track_popularity
        
        # Hacemos el argsort indicándo que queremos ordenar por score y, en caso de empate, por popularidad
        top_local = np.argsort(score_pop, order=['score', 'pop'])[:500]
        
        # Mapeamos los índices a URIs
        recs = [id_to_track[idx] for idx in top_local]
        
        recommendations[pid] = recs

    return recommendations


### FUNCIÓN PARA ESCRIBIR EL ARCHIVO DE SUBMISSION A PARTIR DE LAS RECOMENDACIONES GENERADAS
def write_submission_file(recommendations, output_path):
    print(f"\nEscribiendo archivo de submission en {output_path}...")

    if not os.path.exists("submissions"):
        os.makedirs("submissions")

    with gzip.open(output_path, "wt") as f:
        # Escribimos la cabecera
        f.write("team_info,AndresAngelRicky,angel.vilarino.garcia@udc.es\n\n")
        # Escribimos las recomendaciones para cada playlist
        for pid, recs in recommendations.items():
            f.write(f"{pid}, " + ", ".join(recs) + "\n")
        print(f"Archivo {output_path} generado correctamente.")

    return output_path
    

if __name__ == "__main__":
    # Rutas a los archivos que se usarán para el sistema recomendador
    matrix_path = "processed_data/train_matrix.npz"
    track_map_path = "processed_data/track_map.txt"
    test_playlists_path = "data/spotify_test_playlists/test_input_playlists.json"

    # ITERACIÓN 0: Recomendador basado en popularidad

    # Obtenemos las recomendaciones
    recommendations_top500 = recommender_top500(matrix_path, track_map_path, test_playlists_path)

    # Escribimos el archivo de submission
    output_path = write_submission_file(recommendations_top500, "submissions/submission0.csv.gz")

    # ITERACIÓN 1: Filtrado colaborativo (Neighborhood-based recommendation)

    # USER-BASED
    recommendations_collab_user = recommender_collaborative(mode='user',
                                                    train_matrix_path=matrix_path,
                                                    test_matrix_path="processed_data/test_matrix.npz",
                                                    track_map_path=track_map_path,
                                                    test_playlists_path=test_playlists_path,
                                                    k=10)
    
    output_path_collab_user = write_submission_file(recommendations_collab_user, "submissions/submission_collab_user.csv.gz")

    # ITEM-BASED
    recommendations_collab_item = recommender_collaborative(mode='item',
                                                    train_matrix_path=matrix_path,
                                                    test_matrix_path="processed_data/test_matrix.npz",
                                                    track_map_path=track_map_path,
                                                    test_playlists_path=test_playlists_path,
                                                    k=10)

    output_path_collab_item = write_submission_file(recommendations_collab_item, "submissions/submission_collab_item.csv.gz")

    # ITERACIÓN 2: PureSVD

    # VERSIÓN 1
    recommendations_puresvd = recommender_puresvd(train_matrix_path=matrix_path,
                                                    test_matrix_path="processed_data/test_matrix.npz",
                                                    track_map_path=track_map_path,
                                                    test_playlists_path=test_playlists_path,
                                                    mode=1,
                                                    k=50)

    output_path_puresvd = write_submission_file(recommendations_puresvd, "submissions/submission_puresvd_v1.csv.gz")

    # VERSIÓN 2
    recommendations_puresvd = recommender_puresvd(train_matrix_path=matrix_path,
                                                    test_matrix_path="processed_data/test_matrix.npz",
                                                    track_map_path=track_map_path,
                                                    test_playlists_path=test_playlists_path,
                                                    mode=2,
                                                    k=50)

    output_path_puresvd = write_submission_file(recommendations_puresvd, "submissions/submission_puresvd_v2.csv.gz")
