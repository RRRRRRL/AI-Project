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
     - Or use `LSTM_Trajectory_Prediction_Pipeline.ipynb` for the ATFMTraj dataset integration.

2. **Notebook Instructions**:
   - The `LSTM_Trajectory_Prediction_Pipeline.ipynb` notebook now supports the **ATFMTraj dataset** from Hugging Face by default.
   - The `notebooks/lstm_trajectory_prediction_end_to_end.ipynb` notebook supports OpenSky historical data and synthetic data.
   - To switch data sources in the end-to-end notebook, simply uncomment the desired option in the notebook's data loading section.

3. **Troubleshooting**:
   - If CUDA OOM errors occur, reduce the batch size in the training settings.
   - For missing dependencies, manually install them or update the `requirements.txt`.

### ATFMTraj Dataset Integration
The `LSTM_Trajectory_Prediction_Pipeline.ipynb` notebook now integrates with the **ATFMTraj (Air Traffic Flow Management Trajectories)** dataset from Hugging Face:

1. **Automatic Loading**:
   - The dataset is loaded directly from Hugging Face using the `datasets` library
   - No manual download or preprocessing required
   - Dataset is automatically converted to pandas DataFrame for processing

2. **Dataset Schema**:
   - **latitude**: Aircraft latitude position (degrees)
   - **longitude**: Aircraft longitude position (degrees)
   - **altitude**: Aircraft altitude (meters)
   - **flight_id**: Unique identifier for each flight
   - **timestamp**: Unix timestamp for each position update

3. **Usage**:
   - Simply run the notebook cells in order
   - The pipeline handles all data loading, preprocessing, and model training automatically
   - Column names are automatically mapped (`latitude/longitude/altitude` → internal processing)

### OpenSky Historical Data Ingestion
The repository also includes support for OpenSky historical flight data:

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
   - Open `notebooks/lstm_trajectory_prediction_end_to_end.ipynb` (supports OpenSky and synthetic data)
   - Uncomment the OpenSky data loading section (Option A or B depending on notebook)
   - Comment out the synthetic data generation code
   - Run the notebook normally - all preprocessing and training steps work identically

4. **Prepare Sequences for Training** (if using command-line tools):
   ```bash
   python src/prepare_sequences.py --input_csv data/processed/flights.csv --output_npz data/sequences.npz
   ```
   
   Note: The `prepare_sequences.py` script now supports both old (`lat`, `lon`, `alt`) and new (`latitude`, `longitude`, `altitude`) column naming conventions for backward compatibility.

### Recent Updates
The repository now includes:
- `Pull Request #1`: Optimizations for LSTM architecture, hyperparameters, and regularization.
- `Pull Request #2`: A comprehensive Jupyter notebook for demonstrating the trajectory prediction pipeline.
- `Pull Request #3`: Fully self-contained deliverables including a production-ready notebook, a technical report, and a presentation script.