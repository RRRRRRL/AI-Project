### Google Colab Usage Guide
1. **Setup Instructions**:
   - Clone the repository to your Colab environment:
     ```bash
     !git clone https://github.com/RRRRRRL/AI-Project.git
     ```
   - Navigate to the project directory:
     ```bash
     %cd AI-Project
     ```
   - Install required dependencies:
     ```bash
     !pip install -r requirements.txt
     ```
   - Open the Jupyter notebooks:
     - Run the notebook for end-to-end trajectory prediction (`notebooks/lstm_trajectory_prediction_end_to_end.ipynb`).

2. **Notebook Instructions**:
   - The notebook covers data fetching, preprocessing, training, and evaluation using synthetic data or OpenSky API.
   - To use OpenSky API data, replace the `generate_synthetic_data` function with a call to the API in the notebook's data preprocessing section.

3. **Troubleshooting**:
   - If CUDA OOM errors occur, reduce the batch size in the training settings.
   - For missing dependencies, manually install them or update the `requirements.txt`.

### OpenSky Historical Data Ingestion
The repository now includes `opensky_historical_ingestion.py` for processing historical OpenSky flight data:

1. **Download OpenSky Data**:
   - Visit https://opensky-network.org/datasets/states/
   - Download historical state vectors in parquet format

2. **Process Historical Data**:
   ```bash
   python opensky_historical_ingestion.py --input /path/to/opensky_data.parquet --output data/processed/flights.csv
   ```
   
   The script will:
   - Load and parse the parquet file
   - Handle various column name variations
   - Drop rows with missing essential values
   - Convert timestamps to unix format
   - Filter invalid coordinates and altitudes
   - Save to CSV format compatible with `prepare_sequences.py`

3. **Prepare Sequences for Training**:
   ```bash
   python src/prepare_sequences.py --input_csv data/processed/flights.csv --output_npz data/sequences.npz
   ```

### Recent Updates
The repository now includes:
- `Pull Request #1`: Optimizations for LSTM architecture, hyperparameters, and regularization.
- `Pull Request #2`: A comprehensive Jupyter notebook for demonstrating the trajectory prediction pipeline.
- `Pull Request #3`: Fully self-contained deliverables including a production-ready notebook, a technical report, and a presentation script.