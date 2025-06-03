"""
Advanced clustering optimizations for globular cluster analysis.

This module provides enhanced clustering techniques specifically designed
for stellar populations and globular cluster physics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class PhysicsInformedClustering:
    """
    Physics-informed clustering that incorporates stellar evolution constraints.
    """
    
    def __init__(
        self,
        stellar_evolution_model: Optional[str] = 'simple',
        metallicity_constraint: bool = True,
        age_constraint: bool = True
    ):
        """
        Initialize physics-informed clustering.
        
        Parameters:
        -----------
        stellar_evolution_model : str
            Type of stellar evolution model to use
        metallicity_constraint : bool
            Whether to apply metallicity constraints
        age_constraint : bool
            Whether to apply age constraints
        """
        self.stellar_evolution_model = stellar_evolution_model
        self.metallicity_constraint = metallicity_constraint
        self.age_constraint = age_constraint
    
    def compute_stellar_physics_loss(
        self, 
        features: torch.Tensor,
        cluster_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics-based loss for stellar populations.
        
        Parameters:
        -----------
        features : torch.Tensor
            Stellar features [ra, dec, pmra, pmdec, parallax, ...]
        cluster_assignments : torch.Tensor
            Soft cluster assignments
            
        Returns:
        --------
        torch.Tensor
            Physics-informed loss
        """
        # Extract proper motions and positions
        ra, dec = features[:, 0], features[:, 1]
        pmra, pmdec = features[:, 2], features[:, 3]
        
        # Compute physics-based constraints
        physics_loss = 0.0
        
        # 1. Spatial coherence: members should be spatially clustered
        for k in range(cluster_assignments.shape[1]):
            weights = cluster_assignments[:, k]
            
            # Weighted spatial variance
            weighted_ra_mean = torch.sum(weights * ra) / torch.sum(weights)
            weighted_dec_mean = torch.sum(weights * dec) / torch.sum(weights)
            
            spatial_var = torch.sum(weights * ((ra - weighted_ra_mean)**2 + (dec - weighted_dec_mean)**2))
            physics_loss += spatial_var * 0.1
        
        # 2. Kinematic coherence: similar proper motions within clusters
        for k in range(cluster_assignments.shape[1]):
            weights = cluster_assignments[:, k]
            
            # Weighted kinematic variance
            weighted_pmra_mean = torch.sum(weights * pmra) / torch.sum(weights)
            weighted_pmdec_mean = torch.sum(weights * pmdec) / torch.sum(weights)
            
            kinematic_var = torch.sum(weights * ((pmra - weighted_pmra_mean)**2 + (pmdec - weighted_pmdec_mean)**2))
            physics_loss += kinematic_var * 0.2
        
        return physics_loss


class AdaptiveClusteringLoss(nn.Module):
    """
    Adaptive loss function that adjusts weights based on clustering quality.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha  # Reconstruction loss weight
        self.beta = beta    # Clustering loss weight
        self.physics_informed = PhysicsInformedClustering()
        
    def forward(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        cluster_probs: torch.Tensor,
        target_distribution: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute adaptive clustering loss.
        
        Returns:
        --------
        loss : torch.Tensor
            Total loss
        loss_components : Dict[str, float]
            Individual loss components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, original)
        
        # Clustering loss (KL divergence)
        clustering_loss = F.kl_div(
            torch.log(cluster_probs + 1e-8),
            target_distribution,
            reduction='batchmean'
        )
        
        # Physics-informed loss
        physics_loss = self.physics_informed.compute_stellar_physics_loss(
            features, cluster_probs
        )
        
        # Adaptive weighting based on training progress
        total_loss = (
            self.alpha * recon_loss + 
            self.beta * clustering_loss + 
            0.1 * physics_loss
        )
        
        loss_components = {
            'reconstruction': recon_loss.item(),
            'clustering': clustering_loss.item(),
            'physics': physics_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization for deep clustering.
    """
    
    def __init__(self, search_space: Dict):
        """
        Initialize hyperparameter optimizer.
        
        Parameters:
        -----------
        search_space : Dict
            Dictionary defining the search space for hyperparameters
        """
        self.search_space = search_space
        self.best_params = None
        self.best_score = -np.inf
        self.history = []
    
    def objective_function(
        self,
        params: Dict,
        data: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Objective function to optimize (silhouette score).
        
        Parameters:
        -----------
        params : Dict
            Hyperparameters to evaluate
        data : np.ndarray
            Training data
        true_labels : np.ndarray, optional
            True labels for evaluation
            
        Returns:
        --------
        float
            Objective score (higher is better)
        """
        try:
            from grasp.deep_learning.clustering import DeepClusteringModel
            
            # Create model with current parameters
            model = DeepClusteringModel(
                input_dim=data.shape[1],
                latent_dim=params['latent_dim'],
                n_clusters=params['n_clusters'],
                hidden_dims=params['hidden_dims'],
                learning_rate=params['learning_rate']
            )
            
            # Prepare data
            train_loader, val_loader = model.prepare_data(
                data, 
                batch_size=params['batch_size']
            )
            
            # Quick training for evaluation
            model.pretrain_autoencoder(
                train_loader, val_loader, 
                epochs=params['pretrain_epochs']
            )
            
            model.train_clustering(
                train_loader, 
                epochs=params['clustering_epochs']
            )
            
            # Get predictions and evaluate
            predictions = model.predict(data)
            
            # Use silhouette score as primary metric
            score = silhouette_score(data, predictions)
            
            # Add penalty for too many/few clusters
            n_predicted_clusters = len(np.unique(predictions))
            if n_predicted_clusters < 2:
                score -= 1.0
            elif n_predicted_clusters > params['n_clusters'] * 2:
                score -= 0.5
            
            return score
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return -1.0
    
    def random_search(
        self,
        data: np.ndarray,
        n_trials: int = 20,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Perform random search for hyperparameter optimization.
        
        Parameters:
        -----------
        data : np.ndarray
            Training data
        n_trials : int
            Number of trials to run
        true_labels : np.ndarray, optional
            True labels for evaluation
            
        Returns:
        --------
        Dict
            Best hyperparameters found
        """
        print(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        for trial in range(n_trials):
            # Sample parameters from search space
            params = {}
            for key, value_range in self.search_space.items():
                if isinstance(value_range, list):
                    params[key] = np.random.choice(value_range)
                elif isinstance(value_range, tuple) and len(value_range) == 2:
                    if isinstance(value_range[0], int):
                        params[key] = np.random.randint(value_range[0], value_range[1])
                    else:
                        params[key] = np.random.uniform(value_range[0], value_range[1])
            
            # Evaluate parameters
            score = self.objective_function(params, data, true_labels)
            
            self.history.append({
                'trial': trial,
                'params': params.copy(),
                'score': score
            })
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                
            print(f"Trial {trial+1}/{n_trials}: Score = {score:.3f} (Best: {self.best_score:.3f})")
        
        print(f"\nOptimization complete!")
        print(f"Best score: {self.best_score:.3f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params


def optimize_clustering_hyperparameters(
    data: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    n_trials: int = 10
) -> Dict:
    """
    Convenience function for hyperparameter optimization.
    
    Parameters:
    -----------
    data : np.ndarray
        Training data
    true_labels : np.ndarray, optional
        True labels for evaluation
    n_trials : int
        Number of optimization trials
        
    Returns:
    --------
    Dict
        Optimized hyperparameters
    """
    # Define search space
    search_space = {
        'latent_dim': [8, 16, 32, 64],
        'n_clusters': [2, 3, 4, 5],
        'hidden_dims': [
            [64, 32],
            [128, 64, 32],
            [256, 128, 64],
            [128, 64, 32, 16]
        ],
        'learning_rate': (1e-4, 1e-2),
        'batch_size': [64, 128, 256],
        'pretrain_epochs': [10, 20, 30],
        'clustering_epochs': [10, 15, 20]
    }
    
    optimizer = HyperparameterOptimizer(search_space)
    best_params = optimizer.random_search(data, n_trials, true_labels)
    
    return best_params, optimizer.history


class UncertaintyQuantification:
    """
    Uncertainty quantification for clustering predictions.
    """
    
    def __init__(self, n_bootstrap: int = 100):
        """
        Initialize uncertainty quantification.
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap samples for uncertainty estimation
        """
        self.n_bootstrap = n_bootstrap
    
    def bootstrap_clustering(
        self,
        model,
        data: np.ndarray,
        bootstrap_fraction: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform bootstrap clustering for uncertainty estimation.
        
        Parameters:
        -----------
        model : DeepClusteringModel
            Trained clustering model
        data : np.ndarray
            Input data
        bootstrap_fraction : float
            Fraction of data to use in each bootstrap sample
            
        Returns:
        --------
        predictions_array : np.ndarray
            Array of predictions from bootstrap samples
        uncertainty : np.ndarray
            Uncertainty estimate for each prediction
        """
        n_samples = len(data)
        bootstrap_size = int(n_samples * bootstrap_fraction)
        predictions_array = np.zeros((self.n_bootstrap, n_samples))
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, bootstrap_size, replace=True)
            bootstrap_data = data[indices]
            
            # Get predictions for bootstrap sample
            bootstrap_predictions = model.predict(bootstrap_data)
            
            # Map back to full dataset (assign -1 to missing samples)
            full_predictions = np.full(n_samples, -1)
            full_predictions[indices] = bootstrap_predictions
            
            predictions_array[i] = full_predictions
        
        # Calculate uncertainty as prediction variance
        uncertainty = np.var(predictions_array, axis=0)
        
        return predictions_array, uncertainty
    
    def prediction_confidence(
        self,
        model,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction confidence based on cluster probabilities.
        
        Parameters:
        -----------
        model : DeepClusteringModel
            Trained clustering model
        data : np.ndarray
            Input data
            
        Returns:
        --------
        predictions : np.ndarray
            Cluster predictions
        confidence : np.ndarray
            Confidence scores for each prediction
        """
        # Get cluster probabilities
        cluster_probs = model.predict_proba(data)
        
        # Predictions are argmax of probabilities
        predictions = np.argmax(cluster_probs, axis=1)
        
        # Confidence is the maximum probability
        confidence = np.max(cluster_probs, axis=1)
        
        return predictions, confidence
