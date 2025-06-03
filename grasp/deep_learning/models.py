"""
Neural Network models for globular cluster analysis.

This module contains PyTorch models for unsupervised clustering of stellar data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class StellarAutoencoder(nn.Module):
    """
    Autoencoder for stellar data dimensionality reduction and feature learning.
    
    Designed for globular cluster data with features:
    [ra, dec, pmra, pmdec, parallax, magnitudes, colors]
    """
    
    def __init__(
        self, 
        input_dim: int = 5,
        hidden_dims: list = [64, 32, 16],
        latent_dim: int = 8,
        dropout_rate: float = 0.1,
        batch_norm: bool = True
    ):
        """
        Initialize the Stellar Autoencoder.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (default: 5 for ra, dec, pmra, pmdec, parallax)
        hidden_dims : list
            List of hidden layer dimensions for encoder
        latent_dim : int
            Dimension of the latent representation
        dropout_rate : float
            Dropout rate for regularization
        batch_norm : bool
            Whether to use batch normalization
        """
        super(StellarAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to original space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Returns:
        --------
        reconstruction : torch.Tensor
            Reconstructed input
        latent : torch.Tensor
            Latent representation
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class DeepEmbeddedClustering(nn.Module):
    """
    Deep Embedded Clustering (DEC) for stellar data.
    
    Combines autoencoder feature learning with clustering optimization.
    """
    
    def __init__(
        self,
        autoencoder: StellarAutoencoder,
        n_clusters: int,
        alpha: float = 1.0
    ):
        """
        Initialize Deep Embedded Clustering model.
        
        Parameters:
        -----------
        autoencoder : StellarAutoencoder
            Pre-trained autoencoder for feature extraction
        n_clusters : int
            Number of clusters
        alpha : float
            Degrees of freedom for t-distribution (clustering parameter)
        """
        super(DeepEmbeddedClustering, self).__init__()
        
        self.autoencoder = autoencoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Cluster centers in latent space
        self.cluster_centers = nn.Parameter(
            torch.zeros(n_clusters, autoencoder.latent_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through DEC model.
        
        Returns:
        --------
        reconstruction : torch.Tensor
            Reconstructed input
        latent : torch.Tensor
            Latent representation
        cluster_probs : torch.Tensor
            Cluster assignment probabilities
        """
        reconstruction, latent = self.autoencoder(x)
        cluster_probs = self._soft_assignment(latent)
        
        return reconstruction, latent, cluster_probs
    
    def _soft_assignment(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments using t-distribution.
        
        Parameters:
        -----------
        latent : torch.Tensor
            Latent representations [batch_size, latent_dim]
            
        Returns:
        --------
        torch.Tensor
            Soft cluster assignments [batch_size, n_clusters]
        """
        # Compute squared distances to cluster centers
        distances = torch.sum((latent.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        
        # Apply t-distribution
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return q
    
    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution for clustering loss.
        
        Parameters:
        -----------
        q : torch.Tensor
            Soft cluster assignments
            
        Returns:
        --------
        torch.Tensor
            Target distribution for KL divergence loss
        """
        weight = q ** 2 / torch.sum(q, dim=0)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p


class VariationalStellarAutoencoder(nn.Module):
    """
    Variational Autoencoder for probabilistic stellar clustering.
    
    Provides uncertainty quantification in latent representations.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list = [64, 32, 16],
        latent_dim: int = 8,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Variational Autoencoder.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dims : list
            Hidden layer dimensions
        latent_dim : int
            Latent space dimension
        dropout_rate : float
            Dropout rate
        """
        super(VariationalStellarAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent variable to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Returns:
        --------
        reconstruction : torch.Tensor
            Reconstructed input
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


class StellarTransformer(nn.Module):
    """
    Transformer-based model for stellar sequence modeling.
    
    Useful for analyzing stellar populations and evolutionary sequences.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize Stellar Transformer.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features per star
        d_model : int
            Model dimension
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        dim_feedforward : int
            Feedforward network dimension
        dropout : float
            Dropout rate
        """
        super(StellarTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding for stellar sequences
        self.pos_encoder = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Stellar Transformer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input stellar data [batch_size, seq_len, input_dim]
        mask : torch.Tensor, optional
            Attention mask for variable length sequences
            
        Returns:
        --------
        torch.Tensor
            Transformed stellar representations
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:seq_len].unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Project back to original dimension
        x = self.output_projection(x)
        
        return x
