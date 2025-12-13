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

---

## Getting Started

### Prerequisites
Ensure you have the correct versions of required dependencies outlined in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

For further contributions or details, refer to the [Issues](https://github.com/RRRRRRL/AI-Project/issues) tab to report bugs, discuss improvements, and track enhancements.