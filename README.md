# VAE Models for Reconstruct Dataset

## Overview

This repository contains Python code implementing Variational Autoencoders (VAEs) using TensorFlow/Keras to reconstruct user-item interaction matrices from recommendation datasets. The primary goal is to learn latent representations of user preferences and potentially generate augmented datasets by adding plausible interactions based on the VAE's reconstructions.

Implementations are provided for two distinct datasets:
1.  Indonesia Tourism Destination Dataset
2.  MovieLens 100K Dataset

## Key Functionality

* **Data Preprocessing:** Loads datasets, filters users/items based on interaction counts, remaps IDs, and splits data into training, validation, and test sets. Implicit feedback (user interacted with item) is used.
* **VAE Implementation:** Defines a custom VAE model using Keras subclassing (`tf.keras.models.Model`), including encoder, decoder, reparameterization trick, and a combined loss function (Binary Cross-Entropy reconstruction loss + KL divergence with annealing).
* **Hyperparameter Tuning:** Uses `sklearn.model_selection.ParameterGrid` to perform a grid search over specified hyperparameters (latent dimension size, annealing cap, learning rate, batch size, optimizer) to find the best model based on NDCG@K on the validation set.
* **Model Training:** Trains the final VAE model on the combined training and validation data using the best hyperparameters found during tuning. Includes callbacks like `EarlyStopping` and `ReduceLROnPlateau`.
* **Evaluation:** Evaluates the trained model on the test set using standard recommendation metrics: Precision@K, Recall@K, and NDCG@K for various values of K.
* **Recommendation Generation:** Generates Top-N (specifically Top-10) item recommendations for each user based on the VAE's predicted interaction scores for items the user hasn't interacted with before.
* **Dataset Augmentation/Reconstruction:**
    * Outputs the original user-item interaction matrix.
    * Outputs the full reconstructed matrix (containing probability scores from the VAE).
    * Generates several *augmented* binary interaction matrices by adding the highest-scoring *new* interactions from the VAE reconstruction until a target total number of interactions (N) is reached.
    * Calculates and reports the sparsity of the original and augmented matrices.

## Datasets Used

1.  **Indonesia Tourism Destination Dataset:**
    * `tourism_rating.csv`: Contains user ratings for tourism destinations. Used for interaction data.
    * `tourism_with_id.csv`: Contains metadata about the destinations (used indirectly for context if needed, but primarily IDs are mapped from the rating file).
2.  **MovieLens 100K Dataset:**
    * Downloaded automatically within the notebook (`ml-100k.zip`).
    * `u.data`: Contains user ratings for movies. Used for interaction data.
    * `u.item`: Contains movie titles (used for generating human-readable recommendations).

## Notebooks

* **`VAE_Indonesia_Tourism_Destination.ipynb`**: Implements the entire pipeline (preprocessing, tuning, training, evaluation, recommendation, matrix generation) specifically for the Indonesia Tourism dataset.
* **`VAE_MovieLens100K_Dataset.ipynb`**: Implements the same pipeline for the MovieLens 100K dataset, including the data download and extraction steps.

## Technologies & Libraries Used

* **Python 3**
* **TensorFlow / Keras:** For building and training the VAE model.
* **Pandas:** For data loading and manipulation.
* **NumPy:** For numerical operations and array handling.
* **SciPy:** Specifically for creating and handling sparse matrices (`scipy.sparse`).
* **Scikit-learn:** For hyperparameter grid generation (`sklearn.model_selection.ParameterGrid`).
* **Matplotlib:** For plotting the training curves.
* **Requests:** For downloading the MovieLens dataset.
* **Zipfile:** For extracting the MovieLens dataset.
* **OS:** For directory/path management.
* **IO:** For handling byte streams during download.
* **Warnings:** For filtering specific warning types.
