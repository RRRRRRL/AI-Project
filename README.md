# AI-Project

## Project Overview
This repository implements an AI pipeline encompassing data preprocessing, model training, and evaluation for flight data analysis. Each stage of the pipeline is well-documented below.

---

## Workflow

1. **Data Preprocessing**
    - `opensky_to_model_csv.py`: A script to extract raw data from OpenSky Network and convert it into a CSV suitable for modeling.
      - **Usage**: `python opensky_to_model_csv.py --input opensky_raw.json --output model_data.csv`
      - **Inputs**: OpenSky Network JSON files.
      - **Outputs**: A CSV summarizing flight data ready for processing.
    - `prepare_sequences_weather.py`: This tool combines weather data with flight sequences using the previously processed model_data.csv.
      - **Usage**: `python prepare_sequences_weather.py --flights model_data.csv --weather weather_data.json --output sequences_weather.csv`
      - **Inputs**: - Flights (`model_data.csv`) and Weather data (`weather_data.json`).
      - **Outputs**: Enriched CSV ready for training.

2. **Model Training**
    - Leverage the prepared sequences to train predictive models. Specify model configurations in `train_model.py`.
      - **Usage Example**: `python train_model.py --config config.yaml`

3. **Model Evaluation**
    - Evaluate trained models using test datasets and assess their performance against benchmarks.
      - Use scripts like `evaluate_model.py` for these tasks.
      - **Example**: `python evaluate_model.py --model best_model.pth --test_data test_data.csv`

6) Train
```
!python src/train.py --data_npz data/processed/hkg_seq.npz --epochs 50 --batch_size 256 --hidden 256 --layers 3 --lr 1e-3 --dropout 0.3 --weight_decay 1e-5 --patience 10 --warmup_epochs 5 --normalize --use_layer_norm --output_dir outputs/run1
```

7) Evaluate and visualize
```
!python src/evaluate.py --data_npz data/processed/hkg_seq.npz --model_dir outputs/run1/best --output_dir outputs/run1 --normalize
```

### Prerequisites
Ensure you have the correct versions of required dependencies outlined in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

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
