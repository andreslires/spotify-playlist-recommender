# Spotify Playlist Recommender System

A high-performance recommendation engine designed to tackle the "Million Playlist Dataset Challenge". This project implements various collaborative filtering and matrix factorization techniques to provide accurate song recommendations for playlist continuation.

## Core Features

This system provides multiple recommendation strategies:

*   **Popularity-Based:** A non-personalized approach recommending globally trending tracks.
*   **Collaborative Filtering:** User-based and Item-based recommendations using K-Nearest Neighbors (KNN).
*   **PureSVD:** Matrix factorization approach projecting playlists into a latent space.
*   **SLIM & FISM (Optimization-Based):** Advanced models using L1/L2 regularized optimization to learn optimal recommendation matrices and latent factors via gradient descent.

## Technology Stack

*   **Language:** Python 3
*   **Core Libraries:** `numpy`, `scipy`, `tensorflow` (for optimization), `tqdm`, `scikit-learn`

## Usage

### Prerequisites
Ensure you have the required libraries installed:
```bash
pip install numpy scipy tqdm sklearn tensorflow
```

### Running the System
The `main.py` script serves as the main entry point:
```bash
python main.py
```
This will launch an interactive menu allowing you to choose between the different recommendation models.

## Credits

This project was developed in collaboration with:
- [Ángel Vilariño García](https://github.com/angelvilarino)
- [Ricardo Martin Buba Sopko](https://github.com/ricardobuba)

## Technical Details

The project utilizes several advanced recommendation models:

*   **SLIM & FISM (`slim_fism.py`):** These models utilize optimization with L1/L2 regularization to learn recommendation matrices. We implemented gradient descent to learn the coefficient matrix $S$ (for SLIM) or latent factor matrices $P$ and $Q$ (for FISM).
*   **Design Considerations:** To handle the computational cost of the Million Playlist Dataset, we employed a *trimmed* version of the dataset. For inference, we implemented a fall-back mechanism to global popularity for cold-start cases, ensuring robust recommendations even when the rating estimate is zero.

## Project Structure

To ensure the system functions correctly, organize the files as follows:

```text
project_root/
├── main.py
├── slim_fism.py
├── recommender.py
├── neighbours.py
├── dataset_parser.py
├── metrics.py
├── evaluator.py
├── data/
│   ├── trimmed_dataset/  # Contains training and test subsets
│   └── ...
├── processed_data/       # Generated during execution
├── results/              # Evaluation output
└── submissions/          # Generated submission files
```

## Running the System

1.  **First Run:** Execute `python main.py`. If data has not been processed or models have not been trained, the system will handle these steps automatically.
2.  **Re-training:** To re-train SLIM or FISM with different hyperparameters, execute `slim_fism.py` directly to generate new model matrices, which `main.py` will then use.

---
*Note: This repository is a refined version of a university project, focused on demonstrating the implemented recommendation algorithms and optimization techniques.*

