"""
Deep Learning module for GRASP - Globular cluster analysis with neural networks.

This module provides advanced deep learning capabilities for unsupervised 
clustering of stellar data, specifically designed for globular cluster analysis.

Features:
- Autoencoder-based feature learning
- Deep Embedded Clustering (DEC)
- Variational Autoencoders for probabilistic clustering
- Physics-informed clustering constraints
- GPU acceleration with PyTorch
- Hyperparameter optimization
- Uncertainty quantification
- Model persistence and interpretability
"""

from .clustering import DeepClusteringModel, ClusteringDataset
from .models import StellarAutoencoder, DeepEmbeddedClustering
from .utils import prepare_stellar_data, evaluate_clustering

# Import advanced features if available
try:
    from .advanced import (
        PhysicsInformedClustering,
        HyperparameterOptimizer,
        UncertaintyQuantification,
        optimize_clustering_hyperparameters
    )
    _ADVANCED_FEATURES = True
except ImportError:
    _ADVANCED_FEATURES = False

__all__ = [
    'DeepClusteringModel',
    'ClusteringDataset', 
    'StellarAutoencoder',
    'DeepEmbeddedClustering',
    'prepare_stellar_data',
    'evaluate_clustering'
]

# Add advanced features to __all__ if available
if _ADVANCED_FEATURES:
    __all__.extend([
        'PhysicsInformedClustering',
        'HyperparameterOptimizer', 
        'UncertaintyQuantification',
        'optimize_clustering_hyperparameters'
    ])
