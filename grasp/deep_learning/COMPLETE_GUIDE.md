# Deep Learning Clustering for Globular Clusters - Complete Guide

## Overview

This comprehensive deep learning module provides state-of-the-art neural network-based clustering for globular cluster analysis using PyTorch with GPU acceleration. The system combines traditional astronomical analysis with modern machine learning techniques to identify stellar populations and understand cluster structure.

## 🚀 Quick Start

```python
from grasp.deep_learning.clustering import DeepClusteringModel
from grasp.deep_learning.utils import generate_synthetic_cluster_data

# Generate sample data
data, labels = generate_synthetic_cluster_data(n_samples=1000, n_clusters=3)

# Initialize model
model = DeepClusteringModel(
    input_dim=5,
    latent_dim=16,
    n_clusters=3,
    learning_rate=1e-3
)

# Train the model
train_loader, val_loader = model.prepare_data(data, batch_size=128)
model.pretrain_autoencoder(train_loader, val_loader, epochs=50)
model.train_clustering(train_loader, epochs=20)

# Get predictions
predictions = model.predict(data)
```

## 🧠 Core Architecture

### 1. Stellar Autoencoder
```python
StellarAutoencoder(
    input_dim=5,           # [ra, dec, pmra, pmdec, parallax]
    hidden_dims=[128, 64, 32],  # Progressive dimensionality reduction
    latent_dim=16,         # Compressed representation
    dropout_rate=0.1,      # Regularization
    batch_norm=True        # Normalization
)
```

**Features:**
- Progressive dimensionality reduction with skip connections
- Batch normalization for stable training
- Dropout for regularization
- Xavier weight initialization

### 2. Deep Embedded Clustering (DEC)
```python
DeepEmbeddedClustering(
    autoencoder,           # Pre-trained autoencoder
    n_clusters=3,          # Number of clusters to identify
    alpha=1.0              # Student's t-distribution parameter
)
```

**Features:**
- Soft cluster assignments using Student's t-distribution
- KL divergence loss for cluster optimization
- Joint feature learning and clustering
- Iterative target distribution updating

### 3. Variational Autoencoder (VAE)
```python
VariationalStellarAutoencoder(
    input_dim=5,
    latent_dim=16,
    hidden_dims=[128, 64, 32]
)
```

**Features:**
- Probabilistic latent representations
- Uncertainty quantification
- Regularized latent space
- Smooth interpolation capabilities

## 🔧 Advanced Features

### 1. Physics-Informed Clustering

Incorporates astronomical constraints into the clustering process:

```python
from grasp.deep_learning.advanced import PhysicsInformedClustering

physics_clustering = PhysicsInformedClustering(
    stellar_evolution_model='simple',
    metallicity_constraint=True,
    age_constraint=True
)
```

**Constraints:**
- **Spatial Coherence**: Cluster members should be spatially concentrated
- **Kinematic Coherence**: Similar proper motions within clusters
- **Distance Consistency**: Members at similar parallax distances
- **Stellar Evolution**: Age-metallicity relationships

### 2. Hyperparameter Optimization

Automated optimization using random search:

```python
from grasp.deep_learning.advanced import optimize_clustering_hyperparameters

best_params, history = optimize_clustering_hyperparameters(
    data, true_labels, n_trials=20
)

# Search space includes:
# - latent_dim: [8, 16, 32, 64]
# - n_clusters: [2, 3, 4, 5]
# - hidden_dims: Various architectures
# - learning_rate: (1e-4, 1e-2)
# - batch_size: [64, 128, 256]
```

### 3. Uncertainty Quantification

Estimate prediction confidence:

```python
from grasp.deep_learning.advanced import UncertaintyQuantification

uq = UncertaintyQuantification(n_bootstrap=100)

# Prediction confidence
predictions, confidence = uq.prediction_confidence(model, data)

# Bootstrap uncertainty estimation
pred_array, uncertainty = uq.bootstrap_clustering(model, data)
```

## 📊 Model Training Pipeline

### Stage 1: Autoencoder Pretraining
```python
model.pretrain_autoencoder(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    early_stopping_patience=10
)
```

**Purpose:**
- Learn meaningful feature representations
- Compress 5D stellar data to lower-dimensional latent space
- Initialize encoder for clustering stage

### Stage 2: Deep Clustering Training
```python
model.train_clustering(
    train_loader=train_loader,
    epochs=20,
    update_interval=5
)
```

**Process:**
1. Initialize cluster centers using K-means on latent features
2. Compute soft cluster assignments using Student's t-distribution
3. Generate target distribution for better cluster separation
4. Optimize KL divergence between assignments and targets
5. Update target distribution periodically

### Stage 3: Fine-tuning (Optional)
```python
# Custom loss function combining reconstruction and clustering
adaptive_loss = AdaptiveClusteringLoss(alpha=1.0, beta=1.0)
```

## 🎯 Evaluation Metrics

### Clustering Quality
- **Adjusted Rand Index (ARI)**: Measures similarity to true clusters
- **Normalized Mutual Information (NMI)**: Information-theoretic measure
- **Silhouette Score**: Internal cluster validation

### Astronomical Metrics
- **Spatial Compactness**: Cluster spatial concentration
- **Kinematic Coherence**: Proper motion consistency
- **Distance Homogeneity**: Parallax distribution within clusters

### Model Quality
- **Reconstruction Error**: Autoencoder quality
- **Latent Space Structure**: t-SNE/UMAP visualization
- **Prediction Confidence**: Uncertainty quantification

## 🌟 Real-World Usage Examples

### Example 1: NGC 6121 (M4) Analysis
```python
import pandas as pd
from grasp.gaia.query import GaiaQuery

# Query Gaia data for M4
gaia_query = GaiaQuery()
m4_data = gaia_query.cone_search(
    ra=245.8967,  # M4 coordinates
    dec=-26.5256,
    radius=0.5,   # 30 arcminutes
    dr='dr3'
)

# Prepare features for clustering
features = np.column_stack([
    m4_data['ra'],
    m4_data['dec'], 
    m4_data['pmra'],
    m4_data['pmdec'],
    m4_data['parallax']
])

# Apply clustering
model = DeepClusteringModel(input_dim=5, latent_dim=16, n_clusters=3)
train_loader, _ = model.prepare_data(features)
model.pretrain_autoencoder(train_loader, epochs=100)
model.train_clustering(train_loader, epochs=30)

predictions = model.predict(features)
```

### Example 2: Multi-Cluster Comparison
```python
cluster_names = ['NGC_6121', 'NGC_6656', 'NGC_104']
results = {}

for cluster in cluster_names:
    # Load cluster data
    data = load_cluster_data(cluster)
    
    # Optimize hyperparameters
    best_params, _ = optimize_clustering_hyperparameters(data, n_trials=10)
    
    # Train optimized model
    model = DeepClusteringModel(**best_params)
    model.fit(data)
    
    # Evaluate
    predictions = model.predict(data)
    metrics = evaluate_clustering(data, predictions)
    
    results[cluster] = {
        'model': model,
        'predictions': predictions,
        'metrics': metrics
    }
```

## 📈 Performance Optimization

### GPU Acceleration
```python
# Automatic GPU detection
model = DeepClusteringModel(device='auto')  # Uses GPU if available

# Manual GPU selection
model = DeepClusteringModel(device='cuda:0')  # Specific GPU

# Monitor GPU memory
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Memory Management
```python
# Gradient checkpointing for large models
model.autoencoder.use_checkpointing = True

# Mixed precision training
model.use_amp = True

# Batch size optimization
optimal_batch_size = model.find_optimal_batch_size(data)
```

### Distributed Training
```python
# Multi-GPU training (future feature)
model = DeepClusteringModel(
    input_dim=5,
    latent_dim=16,
    n_clusters=3,
    distributed=True,
    world_size=4  # Number of GPUs
)
```

## 🔬 Scientific Applications

### 1. Stellar Population Analysis
- **Main Sequence Stars**: Core cluster members
- **Binary Systems**: Kinematically distinct pairs
- **Field Contamination**: Background/foreground stars
- **Stellar Evolution**: Different evolutionary phases

### 2. Globular Cluster Structure
- **Core vs Halo**: Radial population gradients
- **Rotation Signatures**: Systematic proper motion patterns  
- **Tidal Effects**: Extended stellar halos
- **Multiple Populations**: Chemical abundance variations

### 3. Galactic Archaeology
- **Formation History**: Age-metallicity relationships
- **Accretion Events**: Disrupted cluster remnants
- **Chemical Evolution**: Abundance pattern analysis
- **Orbital Dynamics**: Cluster orbit reconstruction

## 🛠️ Troubleshooting

### Common Issues

**1. Poor Clustering Performance**
```python
# Try different architectures
model = DeepClusteringModel(
    hidden_dims=[256, 128, 64, 32],  # Deeper network
    latent_dim=32,                   # Larger latent space
    learning_rate=5e-4               # Lower learning rate
)

# More training epochs
model.pretrain_autoencoder(epochs=100)
model.train_clustering(epochs=50)
```

**2. Memory Issues**
```python
# Reduce batch size
train_loader, _ = model.prepare_data(data, batch_size=64)

# Use gradient accumulation
model.gradient_accumulation_steps = 4
```

**3. Convergence Problems**
```python
# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

# Early stopping
model.pretrain_autoencoder(early_stopping_patience=10)
```

### Performance Tuning

**Data Preprocessing**
```python
# Feature scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # More robust to outliers

# Outlier removal
from sklearn.ensemble import IsolationForest
outlier_detector = IsolationForest(contamination=0.1)
```

**Model Architecture**
```python
# Residual connections for deeper networks
class ResidualStellarAutoencoder(StellarAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_residual = True

# Attention mechanisms
class AttentionStellarAutoencoder(StellarAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = MultiheadAttention(embed_dim=64, num_heads=8)
```

## 📚 References

1. **Deep Embedded Clustering**: Xie, J., Girshick, R., & Farhadi, A. (2016)
2. **Variational Autoencoders**: Kingma, D. P., & Welling, M. (2013)
3. **Globular Cluster Dynamics**: Harris, W. E. (2010)
4. **Gaia Data Processing**: Gaia Collaboration (2023)
5. **Stellar Evolution**: Dotter, A. et al. (2008)

## 🤝 Contributing

Contributions are welcome! Please see the main GRASP repository for contribution guidelines.

### Development Setup
```bash
# Clone repository
git clone https://github.com/username/GRASP.git
cd GRASP

# Install development dependencies
pip install -r requirements-deep-learning.txt
pip install -e .

# Run tests
python test_deep_learning.py
```

### Adding New Models
```python
# Implement in grasp/deep_learning/models.py
class NewModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Implementation

# Add to __init__.py
from .models import NewModel
__all__.append('NewModel')
```

## 📞 Support

For questions and support:
- GitHub Issues: [GRASP Repository](https://github.com/username/GRASP)
- Documentation: [Read the Docs](https://grasp.readthedocs.io)
- Email: support@grasp-project.org

---

**Citation**: If you use this deep learning module in your research, please cite:
```bibtex
@software{grasp_deep_learning,
  title={GRASP Deep Learning: Neural Network Clustering for Globular Clusters},
  author={GRASP Development Team},
  year={2025},
  url={https://github.com/username/GRASP}
}
```
