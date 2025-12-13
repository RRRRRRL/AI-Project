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
   - Both notebooks (`LSTM_Trajectory_Prediction_Pipeline.ipynb` and `notebooks/lstm_trajectory_prediction_end_to_end.ipynb`) now support OpenSky historical data.
   - **Option A (Default)**: Use synthetic data for quick experimentation (no external dependencies).
   - **Option B**: Use real OpenSky historical data by following the instructions below.
   - To switch data sources, simply uncomment the desired option in the notebook's data loading section.

3. **Troubleshooting**:
   - If CUDA OOM errors occur, reduce the batch size in the training settings.
   - For missing dependencies, manually install them or update the `requirements.txt`.

### OpenSky Historical Data Ingestion
The repository includes integrated support for OpenSky historical flight data in both Jupyter notebooks:

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

3. **Use in Notebooks**:
   - Open either `LSTM_Trajectory_Prediction_Pipeline.ipynb` or `notebooks/lstm_trajectory_prediction_end_to_end.ipynb`
   - Uncomment the OpenSky data loading section (Option A or B depending on notebook)
   - Comment out the synthetic data generation code
   - Run the notebook normally - all preprocessing and training steps work identically

4. **Prepare Sequences for Training** (if using command-line tools):
   ```bash
   python src/prepare_sequences.py --input_csv data/processed/flights.csv --output_npz data/sequences.npz
   ```

### Recent Updates
The repository now includes:
- `Pull Request #1`: Optimizations for LSTM architecture, hyperparameters, and regularization.
- `Pull Request #2`: A comprehensive Jupyter notebook for demonstrating the trajectory prediction pipeline.
- `Pull Request #3`: Fully self-contained deliverables including a production-ready notebook, a technical report, and a presentation script.