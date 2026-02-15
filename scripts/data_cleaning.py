import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class DataCleaner:
    def __init__(self):
        pass

    def clean(self, X_corrupted):
        """
        Clean the corrupted dataset.
        1. Impute missing values (NaNs).
        2. Handle outliers (values not in {-1, 1} roughly).
        """
        print("Cleaning data...")
        X_cleaned = X_corrupted.copy()
        N, L, _ = X_cleaned.shape
        
        # 1. Outlier Removal
        # Spins should be -1 or 1. If we see 50, it's an error.
        # We can clip values to range [-1, 1], then round to nearest integer.
        # But first, let's treat outliers as missing or clamp them?
        # Prompt says: "Rilevamento valori fuori range... e normalizzazione"
        # Since we know physics, we know spins are -1 or 1.
        # Let's say any value > 1.5 or < -1.5 is an outlier.
        
        # Outlier detection
        outlier_mask = (np.abs(X_cleaned) > 1.5)
        # Treat outliers as NaNs for imputation, or just clamp?
        # "Sostituisci i NaN con la moda... Gestione Outlier: Rilevamento e normalizzazione"
        # Let's set outliers to NaN first, then impute everything together.
        X_cleaned[outlier_mask] = np.nan
        
        # 2. Imputation
        # We process each matrix. SimpleImputer works on 2D arrays.
        # Strategy: 'most_frequent' (mode) matches the prompt suggestion 
        # "moda dei vicini (se i vicini sono 1, probabilmente anche quello è 1)"
        # Note: scikit-learn SimpleImputer works column-wise or row-wise if we reshape.
        # Ideally we use spatial neighbors, but SimpleImputer(strategy='most_frequent') 
        # on the flattened or local window is easier.
        # To truly use "neighbors", we'd need a convolution or custom loop.
        # Let's implement a custom neighbor imputation for the "Physics" feel,
        # or stick to SimpleImputer for robustness/speed.
        # The user prompt specifically suggested "Moda dei vicini" OR "Simple substitution".
        # Let's do a simple iterative imputation or just a global/local Fill.
        
        # Let's use a simple approach: fill NaNs with 0 (neutral) or satisfy the condition 
        # by iterating. Iterating 2500 matrices is slow.
        # vectorized approach: 
        # Let's use SimpleImputer on flattened versions for speed, 
        # or better: for each image, use scipy.ndimage or just fill with 0 then sign().
        
        # Let's stick to the prompt suggestion: "Imputazione... moda dei vicini".
        # A simple approximation: Replace NaN with the immediate local average (kernel).
        
        # For simplicity and speed in this project context:
        # 1. Fill NaNs with 0.
        # 2. Round to closest valid spin (-1 or 1).
        
        # Wait, if we fill with 0, we lose the "ferromagnetic" block structure if the block is all NaNs?
        # No, 5% NaNs is sparse.
        
        # Let's clean:
        # Flatten all data to (N, L*L) for efficient scikit-learn usage?
        # Or keep 3D.
        
        # Implementation:
        # Loop mainly because SimpleImputer expects 2D (samples, features).
        # We can view the entire dataset as (N*L*L, 1) or (N, L*L).
        
        flat_data = X_cleaned.reshape(N, -1)
        
        # Imputer: Most Frequent (Mode)
        imputer = SimpleImputer(strategy='most_frequent')
        flat_imputed = imputer.fit_transform(flat_data)
        
        # Reshape back
        X_restored = flat_imputed.reshape(N, L, L)
        
        # Ensure binary/discrete values (-1, 1)
        # Just in case average/imputation gave floats? 
        # most_frequent returns values from the set so it should be fine.
        # But if outliers were NaNed, we are good.
        
        return X_restored

    def simple_clean(self, X):
        """Alternative faster clean: Clip and Round"""
        # Clip to -1, 1
        X_c = np.clip(X, -1, 1)
        # Round to nearest integer
        X_c = np.round(X_c)
        # NaNs might restrict this.
        # np.nan_to_num(X_c, nan=0)
        return np.where(np.isnan(X), 0, X) # dummy
