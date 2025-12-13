# LSTM Trajectory Forecasting for Hong Kong Airspace (Colab-first)

This project forecasts aircraft trajectories around Hong Kong (VHHH) using a deep LSTM model. It predicts the next K steps of position offsets (east, north, up in meters) from the last H observed steps. It starts with a synthetic dataset for instant Colab training and can be swapped with real OpenSky data later.

## 🎯 Quick Start: Pre-Built Deliverables

**NEW**: Complete, self-contained deliverables are now available for immediate use:

1. **📓 Jupyter Notebook**: [`LSTM_Trajectory_Prediction_Pipeline.ipynb`](LSTM_Trajectory_Prediction_Pipeline.ipynb)
   - Full pipeline from data generation to evaluation
   - Run directly in Google Colab or local Jupyter
   - No external files needed - completely self-contained

2. **📄 Technical Report**: [`LSTM_Trajectory_Prediction_Report.md`](LSTM_Trajectory_Prediction_Report.md)
   - 3.5-page comprehensive report
   - Ready for PDF conversion
   - Includes all methodology and results

3. **🎥 Video Script**: [`Video_Presentation_Script.md`](Video_Presentation_Script.md)
   - 3-minute presentation script
   - Complete with narration and visual guidance
   - 8 slides, perfectly timed

📚 **See [`DELIVERABLES_README.md`](DELIVERABLES_README.md) for detailed usage instructions.**

---

## Original Project Structure

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
!python src/train.py --data_npz data/processed/hkg_seq.npz --epochs 20 --batch_size 128 --hidden 128 --layers 2 --lr 1e-3 --dropout 0.2 --output_dir outputs/run1
```

7) Evaluate and visualize
```
!python src/evaluate.py --data_npz data/processed/hkg_seq.npz --model_dir outputs/run1/best --output_dir outputs/run1
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