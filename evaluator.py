import json
import gzip
import os
import numpy as np
from metrics import r_precision, ndcg, recommended_song_clicks

def load_submission(file_path):
    """Lee el archivo comprimido de recomendaciones."""
    recommendations = {}
    # Abrimos el archivo gzip
    with gzip.open(file_path, "rt") as f:
        # Para cada línea
        for line in f:
            # Ignoramos la línea de encabezado y las líneas vacías
            if line.startswith("team_info") or not line.strip():
                continue
            # Dividimos la línea por comas y extraemos el PID y las canciones recomendadas
            parts = [p.strip() for p in line.split(",")]
            pid = int(parts[0])
            tracks = parts[1:]
            recommendations[pid] = tracks
    return recommendations

def evaluate(submission_path, ground_truth_path, output_path="evaluation_results.txt"):
    # 1. Cargar las recomendaciones del equipo
    print(f"\nCargando recomendaciones de {submission_path}...")
    recs = load_submission(submission_path)

    # 2. Cargar la ground truth de evaluación
    print(f"Cargando ground truth de {ground_truth_path}...")
    with open(ground_truth_path, "r") as f:
        gt_data = json.load(f)

    # Listas para almacenar los resultados de cada playlist
    r_precisions = []
    ndcgs = []
    clicks = []

    # 3. Calcular métricas para cada playlist de evaluación
    print("\nEvaluando resultados...")
    for pl in gt_data["playlists"]:
        pid = pl["pid"]
        
        # Extraemos el ground truth de canciones para esta playlist
        ground_truth = [t["track_uri"] for t in pl["tracks"]]
        
        if pid in recs:
            recommended = recs[pid]
            # Usamos las funciones de metrics.py
            r_precisions.append(r_precision(recommended, ground_truth))
            ndcgs.append(ndcg(recommended, ground_truth))
            clicks.append(recommended_song_clicks(recommended, ground_truth))
        else:
            print(f"Aviso: La playlist {pid} no se encuentra en el archivo de submission.")

    # 4. Mostrar promedios finales
    print("\n" + "-" * 50)
    print(f"RESULTADOS FINALES ({len(r_precisions)} playlists):")
    print("-" * 50)
    print(f"R-Precision promedio: {np.mean(r_precisions):.4f}")
    print(f"NDCG promedio:        {np.mean(ndcgs):.4f}")
    print(f"Clicks promedio:      {np.mean(clicks):.2f}")
    print("-" * 50)

    # Guardar resultados en un archivo de texto
    if not os.path.exists("results"):
        os.makedirs("results")
    with open(output_path, "w") as f:
        f.write("RESULTADOS FINALES:\n")
        f.write(f"R-Precision promedio: {np.mean(r_precisions):.4f}\n")
        f.write(f"NDCG promedio:        {np.mean(ndcgs):.4f}\n")
        f.write(f"Clicks promedio:      {np.mean(clicks):.2f}\n")


if __name__ == "__main__":
    # Ruta al archivo de ground truth
    ground_truth_path = "data/spotify_test_playlists/test_eval_playlists.json"
    
    # Archivo de submission iteración 0
    submission_path = "submissions/submission0.csv.gz"

    # Ejecutar la evaluación
    evaluate(submission_path, ground_truth_path, "results/evaluation_results0.txt")

    # Archivo de submission iteración 1 (User-Based)
    submission_path_collab = "submissions/submission_collab_user.csv.gz"
    evaluate(submission_path_collab, ground_truth_path, "results/evaluation_results_collab_user.txt")

    # Archivo de submission iteración 1 (Item-Based)
    submission_path_collab_item = "submissions/submission_collab_item.csv.gz"
    evaluate(submission_path_collab_item, ground_truth_path, "results/evaluation_results_collab_item.txt")

    # Archivo de submission iteración 2 (PureSVD)
    submission_path_puresvd = "submissions/submission_puresvd.csv.gz"
    evaluate(submission_path_puresvd, ground_truth_path, "results/evaluation_results_puresvd.txt")
