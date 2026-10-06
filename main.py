from neighbours import calculate_item_neighbours, calculate_user_neighbours
from recommender import recommender_collaborative, recommender_it3, recommender_top500, write_submission_file, recommender_puresvd
from evaluator import evaluate
from dataset_parser import parse_dataset, parse_test_input
import os
from slim_fism import parse_dataset_trimmed, parse_test_input_trimmed, train_slim, train_fism
import numpy as np

# Rutas a los archivos que se usarán para el sistema recomendador
matrix_path = "processed_data/train_matrix.npz"
track_map_path = "processed_data/track_map.txt"

test_playlists_path = "data/spotify_test_playlists/test_input_playlists.json"

ground_truth_path = "data/spotify_test_playlists/test_eval_playlists.json"


# ------------------------------------------------
# Iteración 0: Recomendador basado en popularidad
# ------------------------------------------------

def iteration_0():
    '''
    En esta primera iteración, implementamos un sistema de recomendación basado en la popularidad de los tracks. 
    Para esto, contamos cuántas veces aparece cada track en el dataset de entrenamiento y recomendamos los 500 
    tracks más populares para cada playlist de test.
    '''

    # Obtenemos las recomendaciones
    recommendations = recommender_top500(matrix_path, track_map_path, test_playlists_path)

    # Escribimos el archivo de submission
    output_path = write_submission_file(recommendations, "submissions/submission0.csv.gz")

    # Evaluamos los resultados
    evaluate(output_path, ground_truth_path, "results/evaluation_results0.txt")


# ----------------------------------------------------------------------
# Iteración 1: Filtrado colaborativo (Neighborhood-based recommendation)
# ----------------------------------------------------------------------

def iteration_1():
    '''
    En esta iteración, implementamos un sistema de recomendación basado en filtrado colaborativo. 
    Para esto, calculamos la similitud entre playlists (user-based) o entre tracks (item-based) y 
    recomendamos los tracks más similares a los que ya están en la playlist de test.
    '''

    # Comprobamos si el dataset de test ya ha sido procesado, si no lo procesamos
    if not os.path.exists("processed_data/test_matrix.npz"):
        print("Procesando dataset de test...")
        parse_test_input(test_input_path=test_playlists_path, output_dir="processed_data")

    # Preguntamos al usuario qué modo de filtrado colaborativo quiere ejecutar
    print("Selecciona el modo de filtrado colaborativo:")
    print("1: User-Based")
    print("2: Item-Based")
    mode = input("Introduce el número del modo: ")
    print()

    if mode in ["1", "2"]:

        try:
            k = int(input("Introduce el número de vecinos a usar (k < 150): "))
        except ValueError:
            print("Entrada no válida. Usando k=10 por defecto.")
            k = 10
        print()

        if k <= 0 or k>150:
            print("Número de vecinos no válido. Usando k=10 por defecto.")
            k = 10

        if mode == "1":
            rec_mode = 'user'
            sub_file = "submissions/submission_collab_user.csv.gz"
            eval_file = "results/evaluation_results_collab_user.txt"
            
            # Comprobamos/calculamos vecinos de usuarios
            if not os.path.exists("processed_data/user_knn_data.npz"):
                print("Calculando vecinos de usuarios...")
                calculate_user_neighbours(train_matrix_path=matrix_path, 
                                        test_matrix_path="processed_data/test_matrix.npz", 
                                        output_dir="processed_data", 
                                        k=150)

        elif mode == "2":
            rec_mode = 'item'
            sub_file = "submissions/submission_collab_item.csv.gz"
            eval_file = "results/evaluation_results_collab_item.txt"
            
            # Comprobamos/calculamos vecinos de items
            if not os.path.exists("processed_data/item_knn_data.npz"):
                print("Calculando vecinos de items...")
                calculate_item_neighbours(train_matrix_path=matrix_path, 
                                        output_dir="processed_data", 
                                        k=150)

        # Obtenemos las recomendaciones, guardamos y evaluamos una sola vez
        recommendations = recommender_collaborative(mode=rec_mode,
                                                    train_matrix_path=matrix_path,
                                                    test_matrix_path="processed_data/test_matrix.npz",
                                                    track_map_path=track_map_path,
                                                    test_playlists_path=test_playlists_path,
                                                    k=k)
        
        output_path = write_submission_file(recommendations, sub_file)
        evaluate(output_path, ground_truth_path, eval_file)

    else:
        print("Modo no válido.")


# ------------------------------------------------------
# Iteración 2: Recomendador basado en PureSVD
# ------------------------------------------------------

def iteration_2():
    '''
    En esta iteración, implementamos un sistema de recomendación basado en PureSVD. 
    Para esto, descomponemos la matriz de entrenamiento usando SVD y luego multiplicamos la matriz de test por las componentes principales para obtener las recomendaciones.
    '''

    print("Selecciona un modo para PureSVD:")
    print("1: Versión 1: SVD con la matriz concatenada (train + test)")
    print("2: Versión 2: SVD con la matriz de entrenamiento y proyección de la matriz de test")
    mode = input("Introduce el número del modo: ")
    print()

    if mode not in ["1", "2"]:
        print("Modo no válido. Usando modo 1 por defecto.")
        mode = "1"

    # Obtenemos las recomendaciones
    recommendations = recommender_puresvd(train_matrix_path=matrix_path,
                                            test_matrix_path="processed_data/test_matrix.npz",
                                            track_map_path=track_map_path,
                                            test_playlists_path=test_playlists_path,
                                            mode=int(mode),
                                            k=50)

    # Escribimos el archivo de submission
    output_path = write_submission_file(recommendations, f"submissions/submission_puresvd_v{mode}.csv.gz")
    evaluate(output_path, ground_truth_path, f"results/evaluation_results_puresvd_v{mode}.txt")


# --------------------------------------------------------------
# Iteración 3: Recomendador basado en SLIM
# --------------------------------------------------------------

def iteration_3():
    print("Selecciona el modelo para la Iteración 3:")
    print("1: SLIM")
    print("2: FISM")
    mode = input("Introduce el número del modelo: ")

    if mode == "1":
        model_name = "slim"
        weights_path = "processed_data/trained_slim_weights.npy"
    elif mode == "2":
        model_name = "fism"
        weights_path = "processed_data/trained_fism_weights.npy"
    else:
        print("Opción no válida.")
        return

    # Comprobamos si los datasets recortados ya han sido procesados, si no los procesamos
    if not os.path.exists("processed_data/train_matrix_trimmed.npz"):
        print("Procesando dataset de entrenamiento recortado...")
        parse_dataset_trimmed()

    if not os.path.exists("processed_data/test_trimmed_matrix.npz"):
        print("Procesando dataset de test recortado...")
        parse_test_input_trimmed()

    # Comprobamos si ya se han entrenado los modelos, si no los entrenamos
    if not os.path.exists(weights_path):
        print(f"Entrenando modelo {model_name.upper()}...")
        if mode == "1":
            trained_S = train_slim()
        else:
            trained_S = train_fism()
        np.save(weights_path, trained_S)

    track_map_trimmed = "processed_data/track_map_trimmed.txt"
    test_json_trimmed = "data/trimmed_dataset/test_input_trimmed.json"
    gt_json_trimmed = "data/trimmed_dataset/test_eval_trimmed.json"

    recommendations = recommender_it3(
        test_matrix_trimmed_path="processed_data/test_trimmed_matrix.npz",
        track_map_path=track_map_trimmed,
        test_playlists_path=test_json_trimmed,
        weights_path=weights_path
    )
    
    output_path = write_submission_file(recommendations, f"submissions/submission_{model_name}.csv.gz")
    evaluate(output_path, gt_json_trimmed, f"results/evaluation_results_{model_name}.txt")



# --------------------------------------------------------------
# Sección para ejecutar las iteraciones
# --------------------------------------------------------------

if __name__ == "__main__":
    # Comprobamos si el dataset de entrenamiento ya ha sido procesado, si no lo procesamos
    if not os.path.exists(matrix_path) or not os.path.exists(track_map_path):
        print("Procesando dataset de entrenamiento...")
        parse_dataset(zip_path="data/spotify_train_dataset.zip", output_dir="processed_data")

    print("Selecciona la iteración a ejecutar:")
    print("0: Recomendador basado en popularidad")
    print("1: Filtrado colaborativo (Neighborhood-based recommendation)")
    print("2: Recomendador basado en PureSVD")
    print("3: Recomendador basado en SLIM o FISM")
    iteration = input("Introduce el número de la iteración: ")
    print()

    if iteration == "0":
        iteration_0()
    elif iteration == "1":
        iteration_1()
    elif iteration == "2":
        iteration_2()
    elif iteration == "3":
        iteration_3()
    else:
        print("Iteración no válida.")