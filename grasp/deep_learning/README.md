# Deep Learning Module for GRASP

This module provides PyTorch-based deep neural networks for unsupervised clustering analysis of globular cluster data with GPU acceleration support.

## Overview

The deep learning module extends GRASP's clustering capabilities with modern neural network approaches:

- **Autoencoder-based feature learning** for stellar data dimensionality reduction
- **Deep Embedded Clustering (DEC)** for joint representation learning and clustering  
- **Variational Autoencoders** for probabilistic clustering with uncertainty quantification
- **GPU acceleration** with CUDA support for large-scale datasets
- **Integration** with existing GRASP framework and data structures

## Architecture

```
Input Features → Encoder → Latent Space → Decoder → Reconstruction
     ↓              ↓           ↓
[ra, dec, pmra,   [64,32,16]   [8D]  ← Clustering happens here
 pmdec, parallax]
```

### Key Components

1. **StellarAutoencoder**: Neural network for learning compressed representations of stellar data
2. **DeepEmbeddedClustering**: Joint optimization of feature learning and clustering  
3. **DeepClusteringModel**: Main interface combining training and evaluation
4. **Utility functions**: Data preparation, evaluation metrics, and visualization tools

## Installation

### Requirements

```bash
# Install PyTorch (CPU version)
pip install torch torchvision

# For GPU support (CUDA 11.6 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu116

# Install additional requirements
pip install -r requirements-deep-learning.txt
```

### Verify Installation

```bash
cd /home/pietrof/git/GRASP
python test_deep_learning.py
```

## Usage Examples

### Basic Usage

```python
import numpy as np
from grasp.deep_learning import DeepClusteringModel, prepare_stellar_data

# Load your globular cluster data
stellar_data, feature_names = prepare_stellar_data(
    cluster_data,
    features=['ra', 'dec', 'pmra', 'pmdec', 'parallax'],
    outlier_removal=True
)

# Initialize model
model = DeepClusteringModel(
    input_dim=5,
    latent_dim=8,
    n_clusters=3,
    device='auto'  # Automatically use GPU if available
)

# Prepare data
train_loader, val_loader = model.prepare_data(stellar_data, batch_size=256)

# Train the model
model.pretrain_autoencoder(train_loader, val_loader, epochs=50)
model.train_deep_clustering(train_loader, epochs=30)

# Get cluster predictions
predicted_labels, cluster_probs = model.predict_clusters(stellar_data)
```

### Advanced Features

```python
# Use color features for better clustering
from grasp.deep_learning.utils import create_color_features

cluster_data_with_colors = create_color_features(
    cluster_data, 
    magnitude_columns=['G', 'BP', 'RP']
)

# Comprehensive evaluation
from grasp.deep_learning.utils import evaluate_clustering, plot_cluster_comparison

metrics = evaluate_clustering(
    true_labels=ground_truth,
    predicted_labels=predicted_labels,
    data=stellar_data
)

# Visualize results
model.plot_latent_space(stellar_data, labels=predicted_labels)
plot_cluster_comparison(stellar_data, ground_truth, predicted_labels, feature_names)
```

## Complete Example

See `examples/deep_learning_clustering_example.ipynb` for a comprehensive tutorial covering:

- Data preparation and feature engineering
- Model architecture and hyperparameter selection
- Training with GPU acceleration
- Evaluation and comparison with traditional methods
- Visualization and astrophysical interpretation
- Model persistence and deployment

## Performance Considerations

### GPU Acceleration

The module automatically detects and uses GPU acceleration when available:

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")

# Monitor GPU memory usage
from grasp.deep_learning.utils import gpu_memory_usage
gpu_memory_usage()
```

### Memory Optimization

For large datasets (>100k stars):

```python
# Use smaller batch sizes
train_loader, _ = model.prepare_data(data, batch_size=128)

# Reduce model complexity
model = DeepClusteringModel(
    input_dim=5,
    latent_dim=4,
    hidden_dims=[32, 16]  # Smaller architecture
)
```

### Hyperparameter Optimization

```python
from grasp.deep_learning.utils import optimize_hyperparameters

best_params = optimize_hyperparameters(
    data=stellar_data,
    n_clusters_range=[2, 3, 4, 5],
    latent_dims=[4, 8, 16],
    hidden_dims_options=[[32, 16], [64, 32, 16]]
)
```

## Comparison with Traditional Methods

| Method | Pros | Cons | Best Use Case |
|--------|------|------|---------------|
| **Deep Learning** | Non-linear features, GPU scaling, complex patterns | Requires tuning, less interpretable | Large datasets, complex populations |
| **Gaussian Mixture** | Fast, interpretable, probabilistic | Linear assumptions, fixed features | Well-separated populations |
| **K-Means** | Very fast, simple | Spherical clusters only | Quick exploration |

## Model Architecture Details

### Autoencoder Design

- **Encoder**: Progressive dimensionality reduction (5D → 64 → 32 → 16 → 8D)
- **Decoder**: Mirror architecture for reconstruction
- **Activation**: ReLU with batch normalization and dropout
- **Loss**: Mean squared error for reconstruction

### Clustering Optimization

- **Method**: Deep Embedded Clustering (DEC)
- **Distance**: t-SNE-like soft assignment
- **Loss**: KL divergence between current and target distributions
- **Optimization**: Joint training of representation and clustering

## Integration with GRASP

The deep learning module seamlessly integrates with GRASP:

```python
from grasp.core import Cluster
from grasp.deep_learning import DeepClusteringModel

# Load cluster using GRASP
cluster = Cluster("NGC6121")  # M4 globular cluster

# Apply deep learning clustering
stellar_data, features = prepare_stellar_data(cluster.data)
model = DeepClusteringModel(input_dim=len(features), n_clusters=3)

# Train and predict
# ... training code ...
labels, probs = model.predict_clusters(stellar_data)

# Add results back to cluster data
cluster.data['dl_cluster'] = labels
cluster.data['dl_probability'] = probs.max(axis=1)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or model complexity
2. **Slow training**: Ensure GPU is being used and data is properly batched
3. **Poor clustering**: Try different hyperparameters or more preprocessing
4. **Import errors**: Ensure PyTorch is properly installed

### Performance Tips

- Use mixed precision training for faster GPU computation
- Implement gradient clipping for stable training
- Use learning rate scheduling for better convergence
- Apply data augmentation for more robust features

## Future Extensions

The framework is designed to be extensible:

- **Variational Autoencoders** for uncertainty quantification
- **Graph Neural Networks** for stellar proximity relationships  
- **Transformer architectures** for stellar sequence modeling
- **Multi-modal learning** combining different data types
- **Physics-informed networks** incorporating stellar evolution models

## API Reference

### Classes

- `DeepClusteringModel`: Main clustering interface
- `StellarAutoencoder`: Neural network for feature learning
- `DeepEmbeddedClustering`: Joint clustering optimization
- `ClusteringDataset`: PyTorch dataset for stellar data

### Functions

- `prepare_stellar_data()`: Data preprocessing and cleaning
- `evaluate_clustering()`: Comprehensive clustering evaluation
- `create_color_features()`: Photometric color index calculation
- `optimize_hyperparameters()`: Automated parameter tuning

See the module docstrings for detailed API documentation.

## Contributing

Contributions are welcome! Areas for improvement:

- Additional neural network architectures
- Better hyperparameter optimization
- Integration with more astronomical data formats
- Improved visualization tools
- Performance optimizations

## Citation

If you use this deep learning module in your research, please cite:

```bibtex
@software{grasp_deep_learning,
  title={Deep Learning Module for GRASP: Neural Network-based Clustering of Globular Clusters},
  author={GRASP Development Team},
  year={2024},
  url={https://github.com/yourusername/GRASP}
}
```
