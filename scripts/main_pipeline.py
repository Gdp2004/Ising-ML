import numpy as np
import os
from sklearn.model_selection import train_test_split
from data_generator import IsingSimulation, corrupt_data, save_dataset, load_dataset
from data_cleaning import DataCleaner
from feature_engineering import FeatureExtractor
from models import ModelTrainer
from analysis import plot_confusion_matrix, plot_accuracy_vs_temp

def main():
    print("=== STARTING ISING MODEL ML PIPELINE ===")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    REPORT_DIR = os.path.join(BASE_DIR, "reports")
    
    DATA_FILE = os.path.join(DATA_DIR, "ising_data.npz")
    
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 1. Load or Generate Data
    if os.path.exists(DATA_FILE):
        print(f"\nFound existing data file {DATA_FILE}.")
        regenerate = input("Vuoi rigenerare il dataset? (s/n): ").strip().lower()
        
        if regenerate == 's':
            print("Rigenerazione del dataset in corso...")
            sim = IsingSimulation(L=16)
            X_raw, y, temps = sim.generate_dataset(n_low=1000, n_high=1000, n_crit=500)
            save_dataset(X_raw, y, temps, DATA_FILE)
        else:
            print("Caricamento del dataset esistente...")
            X_raw, y, temps = load_dataset(DATA_FILE)
    else:
        print("No existing data found. Generating new dataset...")
        sim = IsingSimulation(L=16)
        X_raw, y, temps = sim.generate_dataset(n_low=1000, n_high=1000, n_crit=500)
        save_dataset(X_raw, y, temps, DATA_FILE)
    
    # 2. Corrupt Data
    X_corrupted = corrupt_data(X_raw, missing_prob=0.05, outlier_prob=0.02)
    
    # 3. Clean Data
    cleaner = DataCleaner()
    X_clean = cleaner.clean(X_corrupted)
    
    # 4. Feature Engineering
    extractor = FeatureExtractor()
    
    # Approach A: Physics-Based (Magnetization, Energy)
    print("\n--- Approach 1: Physics-Based Features ---")
    X_physics = extractor.get_physics_features(X_clean)
    
    # Split
    X_train_phys, X_test_phys, y_train, y_test, temps_train, temps_test = train_test_split(
        X_physics, y, temps, test_size=0.3, random_state=42
    )
    
    # Train Logistic Regression
    model_phys = ModelTrainer('logistic')
    model_phys.train(X_train_phys, y_train)
    model_phys.save_model(os.path.join(MODEL_DIR, "model_physics.pkl"))
    acc_phys, cm_phys, report_phys = model_phys.evaluate(X_test_phys, y_test)
    print(f"Physics Model Accuracy: {acc_phys:.4f}")
    print(report_phys)
    
    # Plots
    plot_confusion_matrix(cm_phys, title="Confusion Matrix (Physics)", 
                          filename=os.path.join(REPORT_DIR, "cm_physics.png"))
    plot_accuracy_vs_temp(model_phys, X_test_phys, y_test, temps_test, 
                          title="Accuracy vs Temp (Physics)", 
                          filename=os.path.join(REPORT_DIR, "acc_temp_physics.png"))
    
    
    # Approach B: Raw Data (Deep Learning / Random Forest)
    print("\n--- Approach 2: Raw Data (Flattened) ---")
    X_flat = extractor.get_raw_features(X_clean)
    
    # Split
    X_train_raw, X_test_raw, _, _, _, _ = train_test_split(
        X_flat, y, temps, test_size=0.3, random_state=42
    )
    
    # Train Random Forest
    model_raw = ModelTrainer('rf')
    model_raw.train(X_train_raw, y_train)
    model_raw.save_model(os.path.join(MODEL_DIR, "model_raw.pkl"))
    acc_raw, cm_raw, report_raw = model_raw.evaluate(X_test_raw, y_test)
    print(f"Raw Model (RF) Accuracy: {acc_raw:.4f}")
    print(report_raw)
    
    # Plots
    plot_confusion_matrix(cm_raw, title="Confusion Matrix (Raw Data)", 
                          filename=os.path.join(REPORT_DIR, "cm_raw.png"))
    plot_accuracy_vs_temp(model_raw, X_test_raw, y_test, temps_test, 
                          title="Accuracy vs Temp (Raw - RF)", 
                          filename=os.path.join(REPORT_DIR, "acc_temp_raw.png"))
                          
    # Combined ROC Curve
    from analysis import plot_combined_roc_curve
    
    models_dict = {
        "Physics (Logistic)": model_phys,
        "Raw Data (RF)": model_raw
    }
    
    X_test_dict = {
        "Physics (Logistic)": X_test_phys,
        "Raw Data (RF)": X_test_raw
    }
    
    plot_combined_roc_curve(models_dict, X_test_dict, y_test, 
                           title="ROC Curve Comparison", 
                           filename=os.path.join(REPORT_DIR, "roc_combined.png"))

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Check the {REPORT_DIR} folder for results.")

if __name__ == "__main__":
    main()
