"""
Utility functions for deep learning-based globular cluster analysis.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, Any, List
import warnings


def prepare_stellar_data(
    cluster_data: pd.DataFrame,
    features: List[str] = ['ra', 'dec', 'pmra', 'pmdec', 'parallax'],
    magnitude_features: Optional[List[str]] = None,
    handle_missing: str = 'drop',
    outlier_removal: bool = True,
    outlier_threshold: float = 3.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare stellar data for deep learning clustering.
    
    Parameters:
    -----------
    cluster_data : pd.DataFrame
        Raw stellar data from GRASP
    features : List[str]
        Base astrometric features to use
    magnitude_features : List[str], optional
        Additional magnitude/color features
    handle_missing : str
        How to handle missing values ('drop', 'interpolate', 'median')
    outlier_removal : bool
        Whether to remove outliers
    outlier_threshold : float
        Z-score threshold for outlier removal
        
    Returns:
    --------
    prepared_data : np.ndarray
        Cleaned and preprocessed data
    feature_names : List[str]
        Names of the features in order
    """
    
    # Combine all features
    all_features = features.copy()
    if magnitude_features:
        all_features.extend(magnitude_features)
    
    # Select available features
    available_features = [f for f in all_features if f in cluster_data.columns]
    
    if len(available_features) == 0:
        raise ValueError("No specified features found in the data")
    
    # Extract feature data
    data = cluster_data[available_features].copy()
    
    # Handle missing values
    if handle_missing == 'drop':
        data = data.dropna()
    elif handle_missing == 'interpolate':
        data = data.interpolate()
    elif handle_missing == 'median':
        data = data.fillna(data.median())
    else:
        raise ValueError("handle_missing must be 'drop', 'interpolate', or 'median'")
    
    # Remove outliers if requested
    if outlier_removal:
        # Calculate Z-scores
        z_scores = np.abs((data - data.mean()) / data.std())
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        data = data[outlier_mask]
        
        print(f"Removed {(~outlier_mask).sum()} outliers ({(~outlier_mask).mean()*100:.1f}%)")
    
    print(f"Final dataset shape: {data.shape}")
    print(f"Features used: {available_features}")
    
    return data.values, available_features


def create_color_features(
    cluster_data: pd.DataFrame,
    magnitude_columns: List[str]
) -> pd.DataFrame:
    """
    Create color indices from magnitude data.
    
    Parameters:
    -----------
    cluster_data : pd.DataFrame
        Stellar data with magnitude columns
    magnitude_columns : List[str]
        Names of magnitude columns (e.g., ['G', 'BP', 'RP'])
        
    Returns:
    --------
    pd.DataFrame
        Data with added color features
    """
    data_with_colors = cluster_data.copy()
    
    # Common color indices for Gaia data
    if 'G' in magnitude_columns and 'BP' in magnitude_columns:
        data_with_colors['BP_G'] = data_with_colors['BP'] - data_with_colors['G']
    
    if 'G' in magnitude_columns and 'RP' in magnitude_columns:
        data_with_colors['G_RP'] = data_with_colors['G'] - data_with_colors['RP']
    
    if 'BP' in magnitude_columns and 'RP' in magnitude_columns:
        data_with_colors['BP_RP'] = data_with_colors['BP'] - data_with_colors['RP']
    
    return data_with_colors


def evaluate_clustering(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    data: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of clustering results.
    
    Parameters:
    -----------
    true_labels : np.ndarray
        Ground truth cluster labels
    predicted_labels : np.ndarray
        Predicted cluster labels
    data : np.ndarray, optional
        Original data for silhouette score
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    
    metrics = {}
    
    # External evaluation metrics (require true labels)
    if true_labels is not None:
        metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, predicted_labels)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, predicted_labels)
        
        # Calculate accuracy (best permutation of labels)
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(true_labels, predicted_labels)
        row_ind, col_ind = linear_sum_assignment(-cm)
        accuracy = cm[row_ind, col_ind].sum() / cm.sum()
        metrics['clustering_accuracy'] = accuracy
    
    # Internal evaluation metrics
    if data is not None:
        try:
            metrics['silhouette_score'] = silhouette_score(data, predicted_labels)
        except ValueError:
            metrics['silhouette_score'] = np.nan
    
    # Cluster statistics
    unique_labels = np.unique(predicted_labels)
    metrics['n_clusters'] = len(unique_labels)
    
    cluster_sizes = [np.sum(predicted_labels == label) for label in unique_labels]
    metrics['min_cluster_size'] = min(cluster_sizes)
    metrics['max_cluster_size'] = max(cluster_sizes)
    metrics['mean_cluster_size'] = np.mean(cluster_sizes)
    metrics['cluster_size_std'] = np.std(cluster_sizes)
    
    if verbose:
        print("Clustering Evaluation Results:")
        print("=" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    return metrics


def plot_cluster_comparison(
    data: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None
):
    """
    Create comparison plots between true and predicted clusters.
    
    Parameters:
    -----------
    data : np.ndarray
        Original stellar data
    true_labels : np.ndarray
        Ground truth labels
    predicted_labels : np.ndarray
        Predicted labels
    feature_names : List[str]
        Names of features
    save_path : str, optional
        Path to save the plot
    """
    
    # Reduce dimensionality for visualization
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
    else:
        data_2d = data
        x_label = feature_names[0] if len(feature_names) > 0 else "Feature 1"
        y_label = feature_names[1] if len(feature_names) > 1 else "Feature 2"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True clusters
    scatter1 = ax1.scatter(
        data_2d[:, 0], 
        data_2d[:, 1], 
        c=true_labels, 
        cmap='viridis',
        alpha=0.7,
        s=20
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title('True Clusters')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='True Cluster')
    
    # Predicted clusters
    scatter2 = ax2.scatter(
        data_2d[:, 0], 
        data_2d[:, 1], 
        c=predicted_labels, 
        cmap='viridis',
        alpha=0.7,
        s=20
    )
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_title('Predicted Clusters')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Predicted Cluster')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cluster_properties(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot distributions of stellar properties by cluster.
    
    Parameters:
    -----------
    data : np.ndarray
        Stellar data
    labels : np.ndarray
        Cluster labels
    feature_names : List[str]
        Names of features
    save_path : str, optional
        Path to save the plot
    """
    
    n_features = len(feature_names)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_rows == 1:
        axes = axes if n_features > 1 else [axes]
    else:
        axes = axes.flatten()
    
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        for j, label in enumerate(unique_labels):
            mask = labels == label
            ax.hist(
                data[mask, i], 
                alpha=0.7, 
                label=f'Cluster {label}',
                color=colors[j],
                bins=30
            )
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {feature_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_cmd_plot(
    cluster_data: pd.DataFrame,
    labels: np.ndarray,
    magnitude_col: str = 'G',
    color_col: str = 'BP_RP',
    save_path: Optional[str] = None
):
    """
    Create a Color-Magnitude Diagram (CMD) with cluster assignments.
    
    Parameters:
    -----------
    cluster_data : pd.DataFrame
        Stellar data with magnitudes
    labels : np.ndarray
        Cluster labels
    magnitude_col : str
        Column name for absolute magnitude
    color_col : str
        Column name for color index
    save_path : str, optional
        Path to save the plot
    """
    
    if magnitude_col not in cluster_data.columns or color_col not in cluster_data.columns:
        print(f"Warning: Required columns {magnitude_col} or {color_col} not found")
        return
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            cluster_data.loc[mask, color_col],
            cluster_data.loc[mask, magnitude_col],
            alpha=0.7,
            label=f'Cluster {label}',
            color=colors[i],
            s=20
        )
    
    plt.xlabel(f'{color_col} (Color)')
    plt.ylabel(f'{magnitude_col} (Magnitude)')
    plt.title('Color-Magnitude Diagram by Cluster')
    plt.gca().invert_yaxis()  # Brighter stars at top
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_cluster_statistics(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Calculate statistical properties for each cluster.
    
    Parameters:
    -----------
    data : np.ndarray
        Stellar data
    labels : np.ndarray
        Cluster labels
    feature_names : List[str]
        Names of features
        
    Returns:
    --------
    pd.DataFrame
        Statistical summary by cluster
    """
    
    unique_labels = np.unique(labels)
    stats_list = []
    
    for label in unique_labels:
        mask = labels == label
        cluster_data = data[mask]
        
        stats = {
            'cluster': label,
            'n_stars': len(cluster_data),
            'percentage': len(cluster_data) / len(data) * 100
        }
        
        # Add statistics for each feature
        for i, feature_name in enumerate(feature_names):
            feature_data = cluster_data[:, i]
            stats.update({
                f'{feature_name}_mean': np.mean(feature_data),
                f'{feature_name}_std': np.std(feature_data),
                f'{feature_name}_median': np.median(feature_data),
                f'{feature_name}_min': np.min(feature_data),
                f'{feature_name}_max': np.max(feature_data)
            })
        
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


def save_clustering_results(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    output_path: str,
    original_indices: Optional[np.ndarray] = None
):
    """
    Save clustering results to CSV file.
    
    Parameters:
    -----------
    data : np.ndarray
        Original stellar data
    labels : np.ndarray
        Cluster assignments
    feature_names : List[str]
        Names of features
    output_path : str
        Path to save results
    original_indices : np.ndarray, optional
        Original row indices if data was filtered
    """
    
    # Create results DataFrame
    results_df = pd.DataFrame(data, columns=feature_names)
    results_df['cluster'] = labels
    
    if original_indices is not None:
        results_df['original_index'] = original_indices
    
    # Add cluster statistics
    cluster_stats = calculate_cluster_statistics(data, labels, feature_names)
    
    # Save main results
    results_df.to_csv(output_path, index=False)
    
    # Save cluster statistics
    stats_path = output_path.replace('.csv', '_cluster_stats.csv')
    cluster_stats.to_csv(stats_path, index=False)
    
    print(f"Results saved to {output_path}")
    print(f"Cluster statistics saved to {stats_path}")


def gpu_memory_usage():
    """Check GPU memory usage."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU Device: {torch.cuda.get_device_name(device)}")
        print(f"Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    else:
        print("CUDA not available")


def optimize_hyperparameters(
    data: np.ndarray,
    n_clusters_range: List[int] = [2, 3, 4, 5, 6],
    latent_dims: List[int] = [4, 8, 16],
    hidden_dims_options: List[List[int]] = [[32, 16], [64, 32, 16], [128, 64, 32]],
    n_trials: int = 3
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using grid search.
    
    Parameters:
    -----------
    data : np.ndarray
        Training data
    n_clusters_range : List[int]
        Range of cluster numbers to try
    latent_dims : List[int]
        Latent dimensions to try
    hidden_dims_options : List[List[int]]
        Hidden layer configurations to try
    n_trials : int
        Number of trials per configuration
        
    Returns:
    --------
    Dict[str, Any]
        Best hyperparameters and performance
    """
    
    from itertools import product
    from .clustering import DeepClusteringModel
    
    best_score = -np.inf
    best_params = None
    
    # Create parameter combinations
    param_combinations = list(product(
        n_clusters_range,
        latent_dims,
        hidden_dims_options
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for n_clusters, latent_dim, hidden_dims in param_combinations:
        scores = []
        
        for trial in range(n_trials):
            try:
                # Create and train model
                model = DeepClusteringModel(
                    input_dim=data.shape[1],
                    latent_dim=latent_dim,
                    n_clusters=n_clusters,
                    hidden_dims=hidden_dims
                )
                
                # Prepare data
                train_loader, _ = model.prepare_data(data, batch_size=256)
                
                # Train (reduced epochs for hyperparameter search)
                model.pretrain_autoencoder(train_loader, epochs=20)
                model.train_deep_clustering(train_loader, epochs=10)
                
                # Evaluate using silhouette score
                _, cluster_probs = model.predict_clusters(data)
                cluster_labels = np.argmax(cluster_probs, axis=1)
                
                score = silhouette_score(data, cluster_labels)
                scores.append(score)
                
            except Exception as e:
                print(f"Error with params {n_clusters}, {latent_dim}, {hidden_dims}: {e}")
                scores.append(-1)
        
        avg_score = np.mean(scores)
        
        print(f"n_clusters={n_clusters}, latent_dim={latent_dim}, "
              f"hidden_dims={hidden_dims}: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = {
                'n_clusters': n_clusters,
                'latent_dim': latent_dim,
                'hidden_dims': hidden_dims,
                'score': avg_score
            }
    
    print(f"\nBest parameters: {best_params}")
    return best_params
