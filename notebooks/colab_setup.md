# Colab Setup Cells (copy into a new Colab notebook)

## 1) Install dependencies
```python
!pip install -q torch==2.* numpy pandas matplotlib seaborn tqdm scikit-learn pyproj pymap3d pyyaml
```

## 2) Create folders
```python
import os
for d in ["data/raw","data/processed","outputs","configs","src","notebooks"]:
    os.makedirs(d, exist_ok=True)
print("Folders ready.")
```

## 3) Create files (copy code blocks from the repo into Colab)
- Create and run code cells with the contents of:
  - `src/geo.py`
  - `src/simulate_data.py`
  - `src/prepare_sequences.py`
  - `src/dataset.py`
  - `src/model.py`
  - `src/train.py`
  - `src/evaluate.py`

Tip: Use
```python
from google.colab import files
```
to upload local .py files if you downloaded them.

## 4) Generate synthetic Hong Kong dataset
```python
!python src/simulate_data.py --num_flights 600 --min_len 160 --max_len 260 --out_csv data/raw/hkg_synth.csv
```

## 5) Prepare sequences (history H=40, horizon K=20)
```python
!python src/prepare_sequences.py --input_csv data/raw/hkg_synth.csv --output_npz data/processed/hkg_seq.npz --input_len 40 --pred_len 20 --resample_sec 30 --region "113.5,115.5,21.5,23.0"
```

## 6) Train LSTM
```python
!python src/train.py --data_npz data/processed/hkg_seq.npz --epochs 20 --batch_size 128 --hidden 128 --layers 2 --lr 1e-3 --dropout 0.2 --output_dir outputs/run1
```

## 7) Evaluate and visualize
```python
!python src/evaluate.py --data_npz data/processed/hkg_seq.npz --model_dir outputs/run1/best --output_dir outputs/run1
```

## 8) Swap in real OpenSky data later
- Upload your CSV to `data/raw/opensky_hkg.csv` with columns: `flight_id,timestamp,lat,lon,alt`
- Then run step 5 with `--input_csv data/raw/opensky_hkg.csv`.