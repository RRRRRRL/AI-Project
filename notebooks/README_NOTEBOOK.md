# LSTM Trajectory Prediction - End-to-End Notebook

## Overview

The `lstm_trajectory_prediction_end_to_end.ipynb` notebook provides a complete, beginner-friendly walkthrough of aircraft trajectory prediction using the LSTMMultiHorizon model with OpenSky data.

## What's Included

This notebook demonstrates:

1. **Data Fetching** - Using `opensky_to_model_csv.py` to extract OpenSky API data (or synthetic data generation)
2. **Feature Preprocessing** - Using `prepare_sequences.py` to create sequences with 8 features:
   - Velocity components (vx, vy, vz)
   - Horizontal speed
   - Track angle encoding (cos_track, sin_track)
   - Time-of-day encoding (sin_tod, cos_tod)
3. **Model Training** - Training a multi-layer LSTM with:
   - 2-layer LSTM architecture
   - Smooth L1 Loss (Huber loss)
   - AdamW optimizer
   - Early stopping on validation ADE
4. **Evaluation** - Performance assessment with:
   - ADE (Average Displacement Error)
   - FDE (Final Displacement Error)
   - Baseline comparison (constant velocity model)
5. **Visualization** - Multiple plots:
   - Training curves
   - Prediction vs ground truth overlays
   - Error distributions and growth
   - 3D trajectory visualizations

## How to Run

### Option 1: Local/Jupyter

```bash
# Install dependencies
pip install torch numpy pandas matplotlib seaborn tqdm scikit-learn pyproj pymap3d pyyaml requests

# Start Jupyter
jupyter notebook

# Open the notebook and run all cells
```

### Option 2: Google Colab

1. Upload the notebook to Google Colab
2. Uncomment the first cell to install dependencies
3. Run all cells sequentially

## Features

- **Beginner-Friendly**: Extensive markdown explanations for each step
- **Reproducible**: Uses synthetic data by default (no API credentials required)
- **Comprehensive**: Covers the complete ML pipeline from data to deployment-ready model
- **Well-Documented**: Inline comments and outputs explained
- **Saves Everything**: All outputs (models, plots, metrics) saved to `outputs/notebook_run/`

## Output Files

After running the notebook, you'll find:

- `data/raw/hkg_data.csv` - Raw trajectory data
- `data/processed/hkg_seq.npz` - Preprocessed sequences (X, Y arrays)
- `outputs/notebook_run/best/model.pt` - Best trained model weights
- `outputs/notebook_run/training_curves.png` - Loss and metric curves
- `outputs/notebook_run/test_metrics.txt` - Final test metrics
- `outputs/notebook_run/predictions_vs_groundtruth.png` - Sample predictions
- `outputs/notebook_run/error_analysis.png` - Error distributions
- `outputs/notebook_run/3d_trajectories.png` - 3D trajectory plots

## Customization

### Using Real OpenSky Data

To use real OpenSky Network data instead of synthetic data:

1. Uncomment and modify the OpenSky API cell (Cell #6)
2. Provide your OpenSky credentials if needed
3. Adjust the time range and bounding box parameters

### Hyperparameters

You can easily modify hyperparameters in the notebook:

- `INPUT_LEN` (H) - History window length (default: 40)
- `PRED_LEN` (K) - Prediction horizon (default: 20)
- `RESAMPLE_SEC` - Time resolution (default: 30 seconds)
- `HIDDEN_SIZE` - LSTM hidden dimension (default: 128)
- `NUM_LAYERS` - Number of LSTM layers (default: 2)
- `DROPOUT` - Dropout rate (default: 0.2)
- `LEARNING_RATE` - Learning rate (default: 1e-3)
- `BATCH_SIZE` - Batch size (default: 128)
- `EPOCHS` - Maximum epochs (default: 20)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib, Seaborn
- pyproj, pymap3d (for coordinate transformations)
- tqdm (for progress bars)

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE`
- Reduce `HIDDEN_SIZE`
- Use CPU instead: `device = torch.device('cpu')`

### Long Training Time
- Reduce `EPOCHS`
- Use synthetic data with fewer flights
- Enable GPU acceleration if available

### Import Errors
- Ensure all dependencies are installed
- Make sure `sys.path` includes the `src/` directory
- Check that all files from the repository are present

## Next Steps

After completing this notebook, you can:

1. **Add Weather Features**: Incorporate wind data (u/v components)
2. **Real-Time Inference**: Deploy the model as an API endpoint
3. **Hyperparameter Tuning**: Experiment with architecture variations
4. **Ensemble Methods**: Combine predictions from multiple models
5. **Advanced Metrics**: Add trajectory-specific metrics (cross-track error, along-track error)

## Support

For issues or questions:
- Check the main repository README
- Review the source code in `src/` directory
- Ensure all dependencies are correctly installed

---

**Note**: This notebook is designed to be self-contained and educational. All code is explained with markdown cells and inline comments.
