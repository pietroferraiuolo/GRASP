"""
Deep clustering implementation for globular cluster analysis.

This module provides the main clustering interface and training procedures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import Tuple, Optional, Dict, Any, List
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .models import StellarAutoencoder, DeepEmbeddedClustering, VariationalStellarAutoencoder


class ClusteringDataset(Dataset):
    """PyTorch Dataset for stellar clustering data."""
    
    def __init__(
        self, 
        data: np.ndarray, 
        labels: Optional[np.ndarray] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize clustering dataset.
        
        Parameters:
        -----------
        data : np.ndarray
            Stellar features [n_stars, n_features]
        labels : np.ndarray, optional
            True cluster labels (if available)
        transform : callable, optional
            Data transformation function
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {'data': self.data[idx]}
        
        if self.transform:
            sample['data'] = self.transform(sample['data'])
            
        if self.labels is not None:
            sample['label'] = self.labels[idx]
            
        return sample


class DeepClusteringModel:
    """
    Main interface for deep learning-based clustering of globular cluster data.
    
    Supports multiple clustering approaches:
    - Autoencoder + K-means
    - Deep Embedded Clustering (DEC)
    - Variational Autoencoder clustering
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 8,
        n_clusters: int = 3,
        hidden_dims: List[int] = [64, 32, 16],
        learning_rate: float = 1e-3,
        device: str = 'auto'
    ):
        """
        Initialize Deep Clustering Model.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        latent_dim : int
            Dimension of latent space
        n_clusters : int
            Number of clusters to identify
        hidden_dims : List[int]
            Hidden layer dimensions
        learning_rate : float
            Learning rate for optimization
        device : str
            Device to use ('cpu', 'cuda', or 'auto')
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize models - save initialization parameters
        self.hidden_dims = hidden_dims
        self.autoencoder = StellarAutoencoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim
        ).to(self.device)
        
        self.dec_model = None  # Will be initialized after pretraining
        self.scaler = StandardScaler()
        
        # Training history
        self.history = {
            'pretrain_loss': [],
            'clustering_loss': [],
            'reconstruction_loss': [],
            'cluster_loss': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(
        self, 
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        batch_size: int = 256,
        normalize: bool = True
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data for training.
        
        Parameters:
        -----------
        data : np.ndarray
            Stellar features
        labels : np.ndarray, optional
            True labels (for evaluation)
        batch_size : int
            Batch size for training
        normalize : bool
            Whether to normalize the data
            
        Returns:
        --------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader (if labels provided)
        """
        if normalize:
            data = self.scaler.fit_transform(data)
        
        # Create dataset
        dataset = ClusteringDataset(data, labels)
        
        # Split into train/validation if labels are provided
        if labels is not None:
            # Simple 80/20 split
            n_train = int(0.8 * len(dataset))
            n_val = len(dataset) - n_train
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_val]
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False
            )
            
            return train_loader, val_loader
        else:
            train_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True
            )
            return train_loader, None
    
    def pretrain_autoencoder(
        self, 
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ):
        """
        Pretrain the autoencoder for feature learning.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data
        val_loader : DataLoader, optional
            Validation data
        epochs : int
            Number of training epochs
        early_stopping_patience : int
            Early stopping patience
        """
        self.logger.info("Starting autoencoder pretraining...")
        
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), 
            lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.autoencoder.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                data = batch['data'].to(self.device)
                
                optimizer.zero_grad()
                reconstruction, _ = self.autoencoder(data)
                loss = F.mse_loss(reconstruction, data)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['pretrain_loss'].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                self.autoencoder.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        data = batch['data'].to(self.device)
                        reconstruction, _ = self.autoencoder(data)
                        loss = F.mse_loss(reconstruction, data)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.autoencoder.state_dict(), 'best_autoencoder.pth')
                else:
                    patience_counter += 1
                
                self.logger.info(
                    f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}"
                )
                
                if patience_counter >= early_stopping_patience:
                    self.logger.info("Early stopping triggered!")
                    break
            else:
                self.logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}")
        
        # Load best model if validation was used
        if val_loader is not None:
            self.autoencoder.load_state_dict(torch.load('best_autoencoder.pth'))
        
        self.logger.info("Autoencoder pretraining completed!")
    
    def initialize_clusters(self, data_loader: DataLoader) -> np.ndarray:
        """
        Initialize cluster centers using K-means on latent representations.
        
        Parameters:
        -----------
        data_loader : DataLoader
            Data for cluster initialization
            
        Returns:
        --------
        np.ndarray
            Initial cluster centers
        """
        self.logger.info("Initializing cluster centers...")
        
        self.autoencoder.eval()
        latent_representations = []
        
        with torch.no_grad():
            for batch in data_loader:
                data = batch['data'].to(self.device)
                _, latent = self.autoencoder(data)
                latent_representations.append(latent.cpu().numpy())
        
        latent_representations = np.vstack(latent_representations)
        
        # Use K-means to initialize cluster centers
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(latent_representations)
        
        return kmeans.cluster_centers_
    
    def train_deep_clustering(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        update_interval: int = 10,
        tol: float = 1e-3
    ):
        """
        Train Deep Embedded Clustering (DEC) model.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data
        val_loader : DataLoader, optional
            Validation data
        epochs : int
            Number of training epochs
        update_interval : int
            Interval for updating target distribution
        tol : float
            Tolerance for early stopping
        """
        self.logger.info("Starting deep clustering training...")
        
        # Initialize DEC model
        cluster_centers = self.initialize_clusters(train_loader)
        self.dec_model = DeepEmbeddedClustering(
            self.autoencoder, 
            self.n_clusters
        ).to(self.device)
        
        # Initialize cluster centers
        self.dec_model.cluster_centers.data = torch.FloatTensor(
            cluster_centers
        ).to(self.device)
        
        # Optimizer for clustering phase
        optimizer = torch.optim.Adam(
            self.dec_model.parameters(), 
            lr=self.learning_rate * 0.1  # Lower learning rate for fine-tuning
        )
        
        # Training loop
        for epoch in range(epochs):
            self.dec_model.train()
            total_loss = 0.0
            
            # Update target distribution every update_interval epochs
            if epoch % update_interval == 0:
                target_distribution = self._update_target_distribution(train_loader)
            
            batch_idx = 0
            for batch in tqdm(train_loader, desc=f"Clustering Epoch {epoch+1}/{epochs}"):
                data = batch['data'].to(self.device)
                
                optimizer.zero_grad()
                
                reconstruction, latent, q = self.dec_model(data)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(reconstruction, data)
                
                # Clustering loss (KL divergence)
                p = target_distribution[batch_idx:batch_idx+len(data)]
                cluster_loss = F.kl_div(
                    q.log(), 
                    torch.FloatTensor(p).to(self.device), 
                    reduction='batchmean'
                )
                
                # Combined loss
                loss = recon_loss + cluster_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_idx += len(data)
            
            avg_loss = total_loss / len(train_loader)
            self.history['clustering_loss'].append(avg_loss)
            
            self.logger.info(f"Clustering Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        self.logger.info("Deep clustering training completed!")
    
    def _update_target_distribution(self, data_loader: DataLoader) -> np.ndarray:
        """Update target distribution for clustering loss."""
        self.dec_model.eval()
        q_values = []
        
        with torch.no_grad():
            for batch in data_loader:
                data = batch['data'].to(self.device)
                _, _, q = self.dec_model(data)
                q_values.append(q.cpu().numpy())
        
        q_values = np.vstack(q_values)
        p = self.dec_model.target_distribution(torch.FloatTensor(q_values))
        
        return p.numpy()
    
    def predict_clusters(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster assignments for new data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input stellar data
            
        Returns:
        --------
        cluster_labels : np.ndarray
            Hard cluster assignments
        cluster_probs : np.ndarray
            Soft cluster probabilities
        """
        # Normalize data
        data_normalized = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_normalized).to(self.device)
        
        self.dec_model.eval()
        with torch.no_grad():
            _, _, q = self.dec_model(data_tensor)
            cluster_probs = q.cpu().numpy()
            cluster_labels = np.argmax(cluster_probs, axis=1)
        
        return cluster_labels, cluster_probs
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Get cluster assignment probabilities for input data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input stellar data
            
        Returns:
        --------
        np.ndarray
            Cluster assignment probabilities (n_samples, n_clusters)
        """
        if self.dec_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Normalize data
        data_normalized = self.scaler.transform(data)
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(data_normalized).to(self.device)
        
        # Get cluster probabilities
        self.dec_model.eval()
        with torch.no_grad():
            # Get latent representation
            latent_repr = self.autoencoder.encode(data_tensor)
            
            # Get cluster probabilities from DEC model
            cluster_probs = self.dec_model.cluster_layer(latent_repr)
            cluster_probs = cluster_probs.cpu().numpy()
        
        return cluster_probs

    def get_latent_representation(self, data: np.ndarray) -> np.ndarray:
        """
        Get latent representations for input data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input stellar data
            
        Returns:
        --------
        np.ndarray
            Latent representations
        """
        data_normalized = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_normalized).to(self.device)
        
        self.autoencoder.eval()
        with torch.no_grad():
            latent = self.autoencoder.encode(data_tensor)
            
        return latent.cpu().numpy()
    
    def evaluate_clustering(
        self, 
        true_labels: np.ndarray, 
        predicted_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate clustering performance.
        
        Parameters:
        -----------
        true_labels : np.ndarray
            Ground truth cluster labels
        predicted_labels : np.ndarray
            Predicted cluster labels
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        
        return {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        # Prepare safe data for saving (convert numpy arrays to lists)
        save_data = {
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'dec_model_state_dict': self.dec_model.state_dict() if self.dec_model else None,
            'scaler_params': {
                'mean_': self.scaler.mean_.tolist(),
                'scale_': self.scaler.scale_.tolist(),
                'var_': self.scaler.var_.tolist(),
                'n_features_in_': self.scaler.n_features_in_,
                'feature_names_in_': getattr(self.scaler, 'feature_names_in_', None)
            },
            'hyperparameters': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'n_clusters': self.n_clusters,
                'learning_rate': self.learning_rate,
                'hidden_dims': self.hidden_dims
            },
            'history': self.history
        }
        
        torch.save(save_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        # Use weights_only=False explicitly for backward compatibility
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.input_dim = hyperparams['input_dim']
        self.latent_dim = hyperparams['latent_dim']
        self.n_clusters = hyperparams['n_clusters']
        self.learning_rate = hyperparams['learning_rate']
        self.hidden_dims = hyperparams.get('hidden_dims', [64, 32, 16])  # Default if not saved
        
        # Recreate models with correct architecture
        self.autoencoder = StellarAutoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim
        ).to(self.device)
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        
        if checkpoint['dec_model_state_dict'] is not None:
            self.dec_model = DeepEmbeddedClustering(
                self.autoencoder,
                self.n_clusters
            ).to(self.device)
            self.dec_model.load_state_dict(checkpoint['dec_model_state_dict'])
        
        # Recreate scaler from saved parameters (convert lists back to numpy arrays)
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        self.scaler = StandardScaler()
        scaler_params = checkpoint['scaler_params']
        self.scaler.mean_ = np.array(scaler_params['mean_'])
        self.scaler.scale_ = np.array(scaler_params['scale_'])
        self.scaler.var_ = np.array(scaler_params['var_'])
        self.scaler.n_features_in_ = scaler_params['n_features_in_']
        if scaler_params['feature_names_in_'] is not None:
            self.scaler.feature_names_in_ = scaler_params['feature_names_in_']
        
        # Load history
        self.history = checkpoint['history']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Pretraining loss
        if self.history['pretrain_loss']:
            axes[0, 0].plot(self.history['pretrain_loss'])
            axes[0, 0].set_title('Pretraining Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE Loss')
        
        # Clustering loss
        if self.history['clustering_loss']:
            axes[0, 1].plot(self.history['clustering_loss'])
            axes[0, 1].set_title('Clustering Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Combined Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_latent_space(
        self, 
        data: np.ndarray, 
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize the latent space representation.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        labels : np.ndarray, optional
            Cluster labels for coloring
        save_path : str, optional
            Path to save the plot
        """
        latent_repr = self.get_latent_representation(data)
        
        # Use PCA for visualization if latent_dim > 2
        if self.latent_dim > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_repr)
        else:
            latent_2d = latent_repr
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(
                latent_2d[:, 0], 
                latent_2d[:, 1], 
                c=labels, 
                cmap='viridis',
                alpha=0.7
            )
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(
                latent_2d[:, 0], 
                latent_2d[:, 1], 
                alpha=0.7
            )
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Representation')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
