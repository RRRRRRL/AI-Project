# Hyperparameter Optimization and Model Improvements

## Summary

This document details the optimizations made to improve the LSTM trajectory forecasting model for Hong Kong airspace. The changes focus on better model architecture, training dynamics, and hyperparameter selection.

## Key Optimizations

### 1. Feature Normalization (src/dataset.py)

**Problem**: Raw features have different scales (velocities in m/s, speeds, trigonometric values), which can slow convergence and hurt performance.

**Solution**: Implemented z-score normalization:
- Compute mean and std from training data only (prevents data leakage)
- Apply same statistics to validation and test sets
- Normalize both input features (X) and target offsets (Y)

**Impact**: Faster convergence, more stable training, better gradient flow

**Usage**: Add `--normalize` flag to training and evaluation

### 2. Model Architecture Improvements (src/model.py)

#### Layer Normalization
- Added to LSTM output and all MLP layers
- Stabilizes training by normalizing activations
- Reduces internal covariate shift

#### Deeper MLP Head
- Changed from 2-layer to 3-layer MLP
- Architecture: hidden_size → hidden_size → hidden_size/2 → output
- Better expressiveness for complex trajectory patterns

#### Increased Model Capacity
- Hidden size: 128 → 256 (100% increase)
- Number of layers: 2 → 3 (50% increase)
- Total parameters: ~230K → ~1.4M

**Usage**: Add `--use_layer_norm` flag to enable layer normalization

### 3. Learning Rate Schedule (src/train.py)

**Problem**: Fixed learning rate can overshoot optimal solution or converge too slowly.

**Solution**: Implemented warmup + cosine annealing:
- Linear warmup for first 5 epochs (0 → lr)
- Cosine annealing for remaining epochs (lr → 0.5*lr)
- Smooth learning rate decay

**Benefits**:
- Stable training at beginning (warmup prevents large updates)
- Better final convergence (annealing finds local minima)

### 4. Regularization Improvements

#### Weight Decay
- Added L2 regularization: weight_decay = 1e-5
- Prevents overfitting by penalizing large weights

#### Increased Dropout
- Dropout: 0.2 → 0.3
- Applied between all MLP layers
- Better generalization on unseen trajectories

### 5. Training Dynamics

#### Batch Size
- 128 → 256 (100% increase)
- Better gradient estimates
- More efficient GPU utilization
- Faster training per epoch

#### Early Stopping
- Patience: 5 → 10 epochs
- Configurable minimum improvement threshold (default: 0.01m ADE)
- More patient convergence allows finding better solutions

#### Extended Training
- Epochs: 20 → 50
- With early stopping, typically converges in 20-30 epochs
- Allows more exploration of parameter space

### 6. Configuration & Reproducibility

#### Model Config Saving
- Save model architecture to `config.json`
- Enables reproducible evaluation
- No need to remember training hyperparameters

#### Normalization Stats Saving
- Save training statistics to `norm_stats.npz`
- Ensures consistent preprocessing at inference
- Prevents train/test distribution mismatch

## Hyperparameter Comparison

| Parameter | Original | Optimized | Reasoning |
|-----------|----------|-----------|-----------|
| Hidden Size | 128 | 256 | Better capacity for complex patterns |
| Num Layers | 2 | 3 | Deeper representations |
| Dropout | 0.2 | 0.3 | Better regularization |
| Batch Size | 128 | 256 | Better gradient estimates, faster training |
| Epochs | 20 | 50 | More training time with early stopping |
| Learning Rate | 1e-3 (fixed) | 1e-3 (scheduled) | Better convergence |
| Weight Decay | 0.0 | 1e-5 | L2 regularization |
| Patience | 5 | 10 | More patient convergence |
| Warmup Epochs | 0 | 5 | Stable training start |

## Expected Improvements

Based on standard deep learning practices, these optimizations typically provide:

1. **Faster Convergence**: 20-30% reduction in epochs needed
2. **Better Final Performance**: 10-20% improvement in ADE/FDE metrics
3. **More Stable Training**: Reduced variance in validation metrics
4. **Better Generalization**: Improved test set performance

## Resource Considerations

### Memory Usage
- Model size: ~5.6MB (256 hidden, 3 layers)
- Training batch memory: ~50MB (batch_size=256, seq_len=40)
- **Total**: ~200-300MB GPU memory (easily fits on any modern GPU)

### Training Time
- Per epoch: ~10-20 seconds (1000 samples, batch_size=256, GPU)
- Total training: ~5-10 minutes (with early stopping)
- **Recommendation**: Use GPU if available, CPU is acceptable for <10K samples

### Inference Speed
- Single trajectory: <1ms
- Batch of 256: ~10-20ms
- **Real-time capable**: Yes, suitable for online trajectory prediction

## Best Practices Implemented

### Data Splitting
- **Flight-level splitting** prevents data leakage
- 70/15/15 (train/val/test) is standard for sequence modeling
- Validation set used for hyperparameter tuning and early stopping
- Test set only used for final evaluation

### Training Workflow
1. Compute normalization statistics from training set only
2. Apply to all splits (train/val/test)
3. Train with learning rate warmup
4. Monitor validation ADE for early stopping
5. Save best checkpoint based on validation performance
6. Evaluate on test set with saved statistics

### Evaluation Workflow
1. Load saved model configuration
2. Load saved normalization statistics
3. Create test dataset with same preprocessing
4. Load model checkpoint
5. Compute metrics and visualizations

## Usage Examples

### Training with Optimized Hyperparameters
```bash
python src/train.py \
  --data_npz data/processed/hkg_seq.npz \
  --epochs 50 \
  --batch_size 256 \
  --hidden 256 \
  --layers 3 \
  --lr 1e-3 \
  --dropout 0.3 \
  --weight_decay 1e-5 \
  --patience 10 \
  --warmup_epochs 5 \
  --normalize \
  --use_layer_norm \
  --output_dir outputs/optimized
```

### Evaluation with Saved Configuration
```bash
python src/evaluate.py \
  --data_npz data/processed/hkg_seq.npz \
  --model_dir outputs/optimized/best \
  --normalize \
  --output_dir outputs/optimized
```

## Validation Results

Tested on synthetic dataset (200 flights, 987 sequences):
- Training converges smoothly with warmup + cosine schedule
- Validation metrics improve steadily
- Early stopping triggered appropriately
- Configuration and normalization stats saved correctly
- Evaluation loads config and reproduces results

## Future Improvements

Potential further optimizations:
1. **Mixed precision training**: FP16 for 2x speedup
2. **Attention mechanisms**: For longer sequences
3. **Teacher forcing schedule**: Gradually reduce teacher forcing ratio
4. **Multi-task learning**: Predict both trajectory and uncertainty
5. **Data augmentation**: Temporal jittering, noise injection
6. **Ensemble methods**: Train multiple models and average predictions

## References

- Learning rate scheduling: Smith, L.N. (2017). "Cyclical Learning Rates for Training Neural Networks"
- Layer normalization: Ba, J.L., et al. (2016). "Layer Normalization"
- Adam with weight decay: Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization"
