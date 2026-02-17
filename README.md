# Ising Model ML Project

Link for presentation: https://prezi.com/view/O9FTr9iUchsZddY8nORg/?referral_token=Dk3mf5lnB3FN

This project implements a Machine Learning pipeline to classify Ising Model spin configurations as "Ordered" (Ferromagnetic) or "Disordered" (Paramagnetic). It mimics a real-world scenario with data corruption ("Sensor Errors") and compares two modeling approaches:
1. **Physics-Based**: Using features like Magnetization and Energy.
2. **Raw Data**: Using Deep Learning / Random Forest on raw spin matrices.

## Project Structure

- `data_generator.py`: Generates synthetic Ising Model data using Metropolis-Hastings and injects noise (NaNs, outliers).
- `data_cleaning.py`: Cleans the corrupted data (imputation, outlier handling).
- `feature_engineering.py`: Extracts features or flattens data.
- `models.py`: Contains the `ModelTrainer` class for various classifiers.
- `analysis.py`: Functions for plotting Confusion Matrices and Accuracy vs. Temperature curves.
- `main_pipeline.py`: Orchestrates the entire experiment.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Models

Run the main pipeline to generate data, train models, and create analysis plots:

```bash
python main_pipeline.py
```

This will:
- Generate or load the Ising Model dataset (`ising_data.npz`)
- Train both Physics-based and Raw Data models
- Save trained models (`model_physics.pkl`, `model_raw.pkl`)
- Generate analysis plots

### Running the Streamlit App

After training the models, launch the interactive web app:

```bash
streamlit run app.py
```

The app features:
- **Temperature Slider**: Control T from 0.1 to 5.0
- **Lattice Generation**: Generate 16×16 spin configurations
- **Heatmap Visualization**: See the spin states in real-time
- **Physics Metrics**: Display Magnetization and Energy
- **ML Prediction**: Classify as Ordered/Disordered using the trained model

## Results

The script will generate the following plots:
- `cm_physics.png`: Confusion Matrix for the Physics-based model.
- `acc_temp_physics.png`: Accuracy vs. Temperature for the Physics-based model.
- `cm_raw.png`: Confusion Matrix for the Raw Data model.
- `acc_temp_raw.png`: Accuracy vs. Temperature for the Raw Data model.

## Interpretation

- **Physics Model** (Logistic Regression): Fast and interpretable. Accuracy drops near Critical Temperature ($T \approx 2.27$).
- **Raw Data Model** (Random Forest): Learns spatial correlations without explicit physics knowledge. Should perform comparably.
