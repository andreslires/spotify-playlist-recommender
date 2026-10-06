import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Funciones worker

def process_user_chunk(args):
    test_chunk, k, train_m_T = args
    
    # Similitud entre playlists del chunk y todas las playlists de entrenamiento
    similarities_chunk = test_chunk.dot(train_m_T).tocsr()
    
    n_rows = similarities_chunk.shape[0]
    indices_chunk = np.zeros((n_rows, k), dtype=np.int32)
    sims_chunk = np.zeros((n_rows, k), dtype=np.float32)

    for i in range(n_rows):
        r_slice = slice(similarities_chunk.indptr[i], similarities_chunk.indptr[i+1])
        d, idx = similarities_chunk.data[r_slice], similarities_chunk.indices[r_slice]
        
        if d.size > 0:
            order = np.argsort(-d)[:k]
            indices_chunk[i, :len(order)] = idx[order]
            sims_chunk[i, :len(order)] = d[order]
            
    return indices_chunk, sims_chunk

def process_item_chunk(args):
    start, item_chunk, k, item_m_T = args
    
    # Similitud entre tracks del chunk y todas las tracks
    similarities_chunk = item_chunk.dot(item_m_T).tocsr()
    
    n_rows = similarities_chunk.shape[0]
    indices_chunk = np.zeros((n_rows, k), dtype=np.int32)
    sims_chunk = np.zeros((n_rows, k), dtype=np.float32)

    for i in range(n_rows):
        r_slice = slice(similarities_chunk.indptr[i], similarities_chunk.indptr[i+1])
        d, idx = similarities_chunk.data[r_slice], similarities_chunk.indices[r_slice]
        
        if d.size > 0:
            # Eliminamos la auto-similitud (la canción con ella misma es 1.0)
            filter_idx = idx != start + i
            d_filtered, idx_filtered = d[filter_idx], idx[filter_idx]
            
            # Ordenar los mejores K
            order = np.argsort(-d_filtered)[:k]
            
            indices_chunk[i, :len(order)] = idx_filtered[order]
            sims_chunk[i, :len(order)] = d_filtered[order]
            
    return indices_chunk, sims_chunk


# Funciones de cálculo

def calculate_user_neighbours(train_matrix_path="processed_data/train_matrix.npz",
                              test_matrix_path="processed_data/test_matrix.npz",
                              output_dir="processed_data",
                              k=150):
    print("\nIniciando User-Based...")
    train_m = load_npz(train_matrix_path).astype(np.float32)
    test_m = load_npz(test_matrix_path).astype(np.float32)
    
    # Normalizar playlists (filas)
    train_m = normalize(train_m, norm='l2', axis=1)
    test_m = normalize(test_m, norm='l2', axis=1)
    train_m_T = train_m.T.tocsr()

    num_playlists_test = test_m.shape[0]
    chunk_size = 500 

    chunks = [(test_m[s:min(s + chunk_size, num_playlists_test)], k, train_m_T) 
              for s in range(0, num_playlists_test, chunk_size)]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Pasamos los chunks ya cortados
        results = list(tqdm(executor.map(process_user_chunk, chunks), total=len(chunks), desc="Calculando User-Based"))

    indices_final = np.vstack([r[0] for r in results])
    sims_final = np.vstack([r[1] for r in results])

    np.savez_compressed(f"{output_dir}/user_knn_data.npz", 
                        indices=indices_final,
                        similarities=sims_final)
    print("User-Based completado.")


def calculate_item_neighbours(train_matrix_path="processed_data/train_matrix.npz",
                              output_dir="processed_data",
                              k=150):
    print("\nIniciando Item-Based...")
    train_m = load_npz(train_matrix_path).astype(np.float32)
    
    # Transponemos para que las filas sean canciones
    item_m = train_m.T.tocsr()
    # Normalizar canciones (filas de la transpuesta)
    item_m = normalize(item_m, norm='l2', axis=1)

    # Precalculamos la transpuesta para el producto punto
    item_m_T = item_m.T.tocsr()

    num_items = item_m.shape[0]
    chunk_size = 500

    chunks = [(s, item_m[s:min(s + chunk_size, num_items)], k, item_m_T) 
              for s in range(0, num_items, chunk_size)]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_item_chunk, chunks), total=len(chunks), desc="Calculando Item-Based"))

    indices_final = np.vstack([r[0] for r in results])
    sims_final = np.vstack([r[1] for r in results])

    np.savez_compressed(f"{output_dir}/item_knn_data.npz", 
                        indices=indices_final, 
                        similarities=sims_final)
    print("Item-Based completado.")



if __name__ == "__main__":
    # Creamos directorio si no existe
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")
        
    # Ejecutamos ambos
    calculate_user_neighbours()
    calculate_item_neighbours()