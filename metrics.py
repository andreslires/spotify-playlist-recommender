import math

def r_precision(recommended, ground_truth):
    '''
    R-precision is the number of retrieved relevant tracks divided by the number of 
    known relevant tracks (i.e., the number of withheld tracks)
    
    Args:
        recommended (list): List of recommended tracks (track URIs).
        ground_truth (list): List of relevant tracks (track URIs). 
    '''
    # Convertir a set para mejorar la eficiencia de búsqueda
    gt_set = set(ground_truth)
    # Número de relevantes conocidos (R)
    R = len(gt_set)
    
    if R == 0: 
        return 0
    
    hits = [t for t in recommended[:R] if t in gt_set]

    return len(hits) / R


def ndcg(recommended, ground_truth):
    '''
    Calcula el NDCG (Normalized Discounted Cumulative Gain) para una lista de recomendaciones
    dada una lista de canciones relevantes (ground truth).
    El NDCG se define como el DCG (Discounted Cumulative Gain) dividido por el IDCG (Ideal DCG).
    El DCG se calcula sumando 1/log2(i+2) para cada canción relevante encontrada en la lista de
    recomendaciones, donde i es la posición de la canción en la lista de recomendaciones
    (comenzando en 0).

    Args:
        recommended (list): Lista de canciones recomendadas (track URIs).
        ground_truth (list): Lista de canciones relevantes (track URIs).
    '''
    # Convertir a set para mejorar la eficiencia de búsqueda
    gt_set = set(ground_truth)
    
    if not gt_set:
        return 0
    
    # Calcular DCG (i + 2 porque los índices de python comienzan en 0)
    dcg = sum([1.0 / math.log2(i + 2) for i, t in enumerate(recommended) if t in gt_set])
    # Calcular IDCG (DCG ideal)
    idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(gt_set), len(recommended)))])
    
    if idcg == 0:
        return 0
    
    return dcg / idcg


def recommended_song_clicks(recommended, ground_truth):
    '''
    Calcula el número de clicks necesarios para que el usuario encuentre una canción relevante
    en la lista de recomendaciones. Se asume que el usuario hace click en las canciones de 10
    en 10 (es decir, revisa las primeras 10 canciones, luego las siguientes 10, etc.). Si el usuario
    no encuentra ninguna canción relevante, se devuelve 51 (fuera del rango de recomendaciones).

    Args:
        recommended (list): Lista de canciones recomendadas (track URIs).
        ground_truth (list): Lista de canciones relevantes (track URIs).
    '''
    # Convertir a set para mejorar la eficiencia de búsqueda
    gt_set = set(ground_truth)

    # Si no hay relevantes, se devuelve 51 por defecto
    if not gt_set:
        return 51

    for i, track in enumerate(recommended): 
        if track in gt_set:
            # Operación floor para contar los clicks de 10 en 10
            return i // 10

    # Si no se encuentra ningún track relevante, se devuelve 51        
    return 51
