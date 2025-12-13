# LSTM Trajectory Forecasting for Hong Kong Airspace (Colab-first)

This project forecasts aircraft trajectories around Hong Kong (VHHH) using a deep LSTM model. It predicts the next K steps of position offsets (east, north, up in meters) from the last H observed steps. It starts with a synthetic dataset for instant Colab training and can be swapped with real OpenSky data later.

Why this scope
- Meets deep learning requirement (multi-layer LSTM with nonlinearities).
- Clear evaluation with ADE/FDE and visualization.
- Hong Kong focus via region filtering and origin set near VHHH.

Quickstart (Google Colab)
1) Install dependencies
```
!pip install -q torch==2.* numpy pandas matplotlib seaborn tqdm scikit-learn pyproj pymap3d pyyaml
```

2) Create folders
```
!mkdir -p data/raw data/processed outputs configs src notebooks
```

3) Save the files in `src/`, `configs/`, and `notebooks/` from this repo (clone or copy in Colab).

4) Generate a synthetic dataset (immediate training) OR bring your real OpenSky CSV
```
!python src/simulate_data.py --num_flights 600 --min_len 160 --max_len 260 --out_csv data/raw/hkg_synth.csv
```

5) Prepare sequences
```
!python src/prepare_sequences.py --input_csv data/raw/hkg_synth.csv --output_npz data/processed/hkg_seq.npz --input_len 40 --pred_len 20 --resample_sec 30 --region "113.5,115.5,21.5,23.0"
```

6) Train
```
!python src/train.py --data_npz data/processed/hkg_seq.npz --epochs 50 --batch_size 256 --hidden 256 --layers 3 --lr 1e-3 --dropout 0.3 --weight_decay 1e-5 --patience 10 --warmup_epochs 5 --normalize --use_layer_norm --output_dir outputs/run1
```

7) Evaluate and visualize
```
!python src/evaluate.py --data_npz data/processed/hkg_seq.npz --model_dir outputs/run1/best --output_dir outputs/run1 --normalize
```

Files
- src/simulate_data.py — synthetic flight generator centered near VHHH
- src/prepare_sequences.py — preprocessing, ENU conversion, windowing
- src/dataset.py — PyTorch dataset for (X history, Y future offsets)
- src/model.py — multi-layer LSTM with multi-horizon head
- src/train.py — training loop with early stopping on validation ADE
- src/evaluate.py — ADE/FDE metrics, CV baseline, plots
- src/geo.py — ENU/lat-lon utility wrappers
- configs/hkg_small.yaml — example config
- notebooks/colab_setup.md — copy-paste cells for Colab

Real OpenSky data (optional later)
- Export CSV with columns at least: flight_id (or icao24 + a per-flight segment id), timestamp (unix seconds), lat, lon, alt (meters).
- Filter to HK region: lon ∈ [113.5, 115.5], lat ∈ [21.5, 23.0], and alt > 0.
- Use the same `prepare_sequences.py` command with your CSV path.

Metrics
- ADE (average displacement error, meters)
- FDE (final displacement error, meters)
- Plots: loss curves, sample trajectory overlays, error histograms

Extensions (after base done)
- Add wind features (u/v) from a small GFS/ERA5 sample.
- Retrieval augmentation: prepend stats from k similar past segments.
- Small planning demo: greedy rollouts or A* on a synthetic cost map.

## Best Practices

### Training, Validation, and Testing Splits
- **70/15/15 split**: 70% training, 15% validation, 15% testing
- Splits are done at the **flight level** to prevent data leakage (no sequences from the same flight appear in multiple splits)
- Validation set is used for early stopping and hyperparameter tuning
- Test set is held out for final evaluation only

### Hyperparameter Optimization
The following hyperparameters have been optimized for this task:
- **Hidden size**: 256 (increased from 128) for better model capacity
- **Number of layers**: 3 (increased from 2) for deeper representations
- **Dropout**: 0.3 (increased from 0.2) for better regularization
- **Batch size**: 256 (increased from 128) for more stable gradient estimates and better GPU utilization
- **Learning rate**: 1e-3 with cosine annealing schedule after warmup
- **Weight decay**: 1e-5 for L2 regularization
- **Epochs**: 50 (increased from 20) with early stopping (patience=10)
- **Warmup epochs**: 5 for stable training initialization

### Model Architecture Improvements
- **Layer Normalization**: Added to LSTM output and MLP layers for training stability
- **Deeper MLP head**: 3-layer MLP with intermediate layer for better expressiveness
- **Feature Normalization**: Input features and target offsets are normalized using training set statistics

### Training Improvements
- **Learning rate warmup**: Linear warmup for 5 epochs followed by cosine annealing
- **Early stopping**: Monitors validation ADE with patience of 10 epochs
- **Gradient clipping**: Set to 1.0 to prevent exploding gradients
- **Optimized criterion**: SmoothL1Loss for robustness to outliers

### Resource Considerations
- Batch size of 256 balances memory usage and training speed
- Sequence length (input=40, pred=20) chosen for 20-30 minute prediction horizons
- Model has ~1-2M parameters, suitable for CPU or single GPU training
- Training typically converges in 20-30 epochs with early stopping