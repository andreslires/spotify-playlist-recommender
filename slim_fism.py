import os
import numpy as np
from scipy.sparse import load_npz, csr_matrix, save_npz
import json
import tensorflow as tf

# ---------------------------------- Funciones para leer y procesar el nuevo dataset ----------------------------------

### Parsear el nuevo dataset
def parse_dataset_trimmed(json_file="data/trimmed_dataset/train_trimmed.json", output_dir="processed_data"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    # Listas para construir la matriz sparse
    rows, cols, values = [], [], []
    
    # Mapeos para convertir URIs de Spotify y pids a índices numéricos
    track_to_id = {} # Mapeo Track URI -> Índice
    id_to_track = [] # Lista de Track URIs por índice
    playlist_to_id = {} # Mapeo Playlist ID -> Índice
    id_to_playlist = [] # Lista de Playlist IDs por índice
 
    with open(json_file, "r") as f:
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
    
    print(f"Procesado: {json_file}")
 
    # Crear la matriz CSR
    print("Construyendo matriz dispersa...")
    matrix = csr_matrix((values, (rows, cols)),
                        shape=(len(id_to_playlist), len(id_to_track)),
                        dtype=np.int8)

    # Guardamos la matriz dispersa
    print(f"Guardando archivos en {output_dir}...")
    save_npz(f"{output_dir}/train_matrix_trimmed.npz", matrix)
    
    # Guardamos el mapeo de tracks a IDs y viceversa
    with open(f"{output_dir}/track_map_trimmed.txt", "w") as f:
        for track_uri in id_to_track:
            f.write(f"{track_uri}\n")
    
    # Guardamos el mapeo de playlists a IDs y viceversa
    with open(f"{output_dir}/playlist_map_trimmed.txt", "w") as f:
        for pid in id_to_playlist:
            f.write(f"{pid}\n")
 
    print("Procesamiento completado.")


def parse_test_input_trimmed(test_input_path="data/trimmed_dataset/test_input_trimmed.json", 
                     track_map_path="processed_data/track_map_trimmed.txt", 
                     output_dir="processed_data"):

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
    save_npz(f"{output_dir}/test_trimmed_matrix.npz", test_matrix)
    
    with open(f"{output_dir}/test_playlist_map_trimmed.txt", "w") as f:
        for pid in id_to_test_playlist:
            f.write(f"{pid}\n")
            
    print(f"Archivo de test convertido a matriz sparse y guardado en {output_dir}/test_trimmed_matrix.npz")
    print(f"Dimensiones de la matriz de test: {test_matrix.shape}")


# ---------------------------------- Funciones modelo SLIM -----------------------------------

### Hiperparámetros para SLIM
LEARNING_RATE_SLIM = 0.0001
EPOCHS_SLIM = 100
LAMBDA_SLIM = 0.1  # Coeficiente de regularización L2
BETA_SLIM = 0.1  # Coeficiente de regularización L1

### FUNCIÓN AUXILIAR: Convertir matriz sparse a tensor de TensorFlow
def convert_sparse_matrix_to_sparse_tensor(sparse_matrix):
    coo = sparse_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col)).T
    values = coo.data
    shape = coo.shape
    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

### FUNCIÓN AUXILIAR: Inicializar matriz W aleatoriamente con la diagonal a cero
def initialize_matrix(rows, cols):
    S = tf.random.uniform((rows, cols), minval=0, maxval=0.001, dtype=tf.float32)
    S = tf.linalg.set_diag(S, tf.zeros(rows))

    S = tf.Variable(S, trainable=True)
    return S

### FUNCIÓN AUXILIAR: Compute loss
def compute_loss(X, X_dense, S):
    XS = tf.sparse.sparse_dense_matmul(X, S)

    loss = tf.reduce_sum(tf.square(XS - X_dense))

    # L1 Regularization
    reg_L1 = LAMBDA_SLIM * tf.reduce_sum(tf.abs(S))
    # L2 Regularization
    reg_L2 = BETA_SLIM * tf.reduce_sum(tf.square(S))

    return loss + reg_L2 + reg_L1

### FUNCIÓN AUXILIAR: Compute gradients
def compute_gradients(loss_fn, X, X_dense, S):
    with tf.GradientTape() as tape:
        loss = loss_fn(X, X_dense, S)
    gradients = tape.gradient(loss, [S])
    return gradients, loss

### FUNCIÓN AUXILIAR: Update parameters
def update_parameters(optimizer, variables, gradients):
    optimizer.apply_gradients(zip(gradients, variables))


### FUNCIÓN PRINCIPAL: Entrenar modelo SLIM
def train_slim(train_matrix_path="processed_data/train_matrix_trimmed.npz"):
    # Cargar matriz de entrenamiento
    X = load_npz(train_matrix_path).tocsr().astype(np.float32)

    # Convertir matriz sparse a tensor de TensorFlow
    X_tensor = convert_sparse_matrix_to_sparse_tensor(X)

    # Para el cálculo de la pérdida, necesitamos la matriz densa (definimos X_dense aquí para evitar convertirla en cada iteración)
    X_dense = tf.sparse.to_dense(X_tensor)

    # Inicializar matriz S aleatoriamente con la diagonal a cero
    items = X.shape[1]
    S = initialize_matrix(rows=items, cols=items)

    # Definir optimizador
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SLIM)

    # Training loop
    for epoch in range(EPOCHS_SLIM):
        gradients, loss = compute_gradients(compute_loss, X_tensor, X_dense, S)
        update_parameters(optimizer, [S], gradients)

        # Asegurarnos de que la diagonal de S se mantenga a cero después de cada actualización
        S.assign(tf.linalg.set_diag(S, tf.zeros(items)))  

        if (epoch + 1) % 10 == 0:
            print("Epoch:", epoch + 1, "Loss:", loss.numpy())

    # Obtener matriz de pesos entrenada
    trained_S = S.numpy()

    # Garantizar que la diagonal sea exactamente cero
    np.fill_diagonal(trained_S, 0)

    # Guardar la matriz de pesos entrenada
    np.save("processed_data/trained_slim_weights.npy", trained_S)

    return trained_S


# ---------------------------------- Funciones modelo FISM -----------------------------------

### Hiperparámetros para FISM
LATENT_DIM = 64  # Dimensión del espacio latente
LEARNING_RATE_FISM = 0.001
EPOCHS_FISM = 100
LAMBDA_FISM = 0.1
BETA_FISM = 0.1

### FUNCIÓN AUXILIAR: Inicializar matrices P y Q
def initialize_fism_matrices(items, latent_dim):
    # Inicializamos P y Q con valores pequeños para estabilidad
    P = tf.Variable(tf.random.uniform((items, latent_dim), minval=0, maxval=0.001), trainable=True)
    Q = tf.Variable(tf.random.uniform((items, latent_dim), minval=0, maxval=0.001), trainable=True)
    return P, Q

### FUNCIÓN AUXILIAR: Compute FISM loss
def compute_fism_loss(X, X_dense, P, Q):
    # S = P * Q.T
    S_approx = tf.matmul(P, Q, transpose_b=True)
    
    # Restricción obligatoria: diagonal cero
    S_diag_zero = tf.linalg.set_diag(S_approx, tf.zeros(P.shape[0]))
    
    # Predicción: X * (PQ.T)
    XS = tf.sparse.sparse_dense_matmul(X, S_diag_zero)
    # Error de reconstrucción
    reconstruction_loss = tf.reduce_sum(tf.square(XS - X_dense))

    # Regularización L1 y L2 sobre P y Q
    reg_L1 = LAMBDA_FISM * (tf.reduce_sum(tf.abs(P)) + tf.reduce_sum(tf.abs(Q)))
    reg_L2 = BETA_FISM * (tf.reduce_sum(tf.square(P)) + tf.reduce_sum(tf.square(Q)))

    return reconstruction_loss + reg_L2 + reg_L1


### FUNCIÓN PRINCIPAL: Entrenar modelo FISM
def train_fism(train_matrix_path="processed_data/train_matrix_trimmed.npz"):
    # Cargar matriz de entrenamiento
    X = load_npz(train_matrix_path).tocsr().astype(np.float32)
    # Convertir matriz sparse a tensor de TensorFlow
    X_tensor = convert_sparse_matrix_to_sparse_tensor(X)
    # Convertir matriz sparse a densa para el cálculo de la pérdida
    X_dense = tf.sparse.to_dense(X_tensor)

    # Inicializar matrices P y Q
    items = X.shape[1]
    P, Q = initialize_fism_matrices(items, LATENT_DIM)

    # Definir optimizador
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_FISM)

    print("Iniciando entrenamiento FISM...")
    for epoch in range(EPOCHS_FISM):
        with tf.GradientTape() as tape:
            loss = compute_fism_loss(X_tensor, X_dense, P, Q)
        
        gradients = tape.gradient(loss, [P, Q])
        optimizer.apply_gradients(zip(gradients, [P, Q]))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1} - Loss: {loss.numpy()}")

    # Obtener la matriz de pesos entrenada S = P * Q.T
    trained_S = tf.matmul(P, Q, transpose_b=True).numpy()

    np.fill_diagonal(trained_S, 0) # Garantizar diagonal a 0

    # Guardar la matriz de pesos entrenada
    np.save("processed_data/trained_fism_weights.npy", trained_S)
    
    return trained_S


if __name__ == "__main__":
    
    if not os.path.exists("processed_data/train_matrix_trimmed.npz"):
        parse_dataset_trimmed()

    if not os.path.exists("processed_data/test_trimmed_matrix.npz"):
        parse_test_input_trimmed()

    # Entrenar SLIM
    print("Entrenando modelo SLIM...")
    train_slim()

    # Entrenar FISM
    print("Entrenando modelo FISM...")
    train_fism()

           



