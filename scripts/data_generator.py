import numpy as np
import random
import os
from tqdm import tqdm

class IsingSimulation:
    def __init__(self, L=16):
        self.L = L

    def _energy(self, lattice):
        """Calculate total energy of the lattice for Metropolis step"""
        # Efficient calculation using numpy rolling
        # H = -J * sum(s_i * s_j)
        # We assume J=1
        
        interaction = lattice * (
            np.roll(lattice, 1, axis=0) +
            np.roll(lattice, -1, axis=0) +
            np.roll(lattice, 1, axis=1) +
            np.roll(lattice, -1, axis=1)
        )
        return -np.sum(interaction) / 2  # Each pair counted twice

    def _metropolis_step(self, lattice, T):
        """Perform one Metropolis sweep over the lattice"""
        for _ in range(self.L * self.L):
            x = np.random.randint(0, self.L)
            y = np.random.randint(0, self.L)
            
            s = lattice[x, y]
            # Calculate energy change diff
            # neighbors sum
            nb = lattice[(x+1)%self.L, y] + \
                 lattice[(x-1)%self.L, y] + \
                 lattice[x, (y+1)%self.L] + \
                 lattice[x, (y-1)%self.L]
            
            dE = 2 * s * nb
            
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                lattice[x, y] = -s
        return lattice

    def generate_lattice(self, T, steps=None):
        """
        Generate a single equilibrated lattice at temperature T
        
        Args:
            T: Temperature
            steps: Number of thermalization steps. If None, automatically determined:
                   - 5000 steps for critical region (2.0 < T < 2.5)
                   - 1000 steps otherwise
        """
        # Adaptive thermalization: more steps near critical temperature
        if steps is None:
            if 2.0 < T < 2.5:
                steps = 5000  # Critical slowing down requires more thermalization
            else:
                steps = 1000
        
        # Start random -1 or 1
        lattice = np.random.choice([-1, 1], size=(self.L, self.L))
        
        # Thermalize
        for _ in range(steps):
            self._metropolis_step(lattice, T)
            
        return lattice

    def generate_dataset(self, n_low=1000, n_high=1000, n_crit=500):
        """
        Generate full dataset.
        Low T (Ordered): 0 < T < 2.0 -> Label 0
        High T (Disordered): 2.5 < T < 5.0 -> Label 1
        Critical T (Testing): 2.1 < T < 2.4
        """
        X = []
        y = []
        temperatures = []
        
        print("Generating Ordered Phase (Low T)...")
        for _ in tqdm(range(n_low)):
            T = np.random.uniform(0.1, 2.0)
            lat = self.generate_lattice(T)
            X.append(lat)
            y.append(0)
            temperatures.append(T)
            
        print("Generating Disordered Phase (High T)...")
        for _ in tqdm(range(n_high)):
            T = np.random.uniform(2.5, 4.0)
            lat = self.generate_lattice(T)
            X.append(lat)
            y.append(1)
            temperatures.append(T)
            
        print("Generating Critical Phase (T ~ 2.27)...")
        for _ in tqdm(range(n_crit)):
            T = np.random.uniform(2.1, 2.45) 
            lat = self.generate_lattice(T)
            X.append(lat)
            # Label based on 2.27 for consistency
            label = 0 if T < 2.27 else 1
            y.append(label)
            temperatures.append(T)
            
        return np.array(X), np.array(y), np.array(temperatures)

def corrupt_data(X, missing_prob=0.05, outlier_prob=0.02):
    """
    Simulate real-world sensor errors.
    """
    print("Corrupting data...")
    X_corr = X.astype(float).copy() 
    N, L, _ = X.shape
    
    # 1. Missing Values (NaN)
    mask_missing = np.random.random((N, L, L)) < missing_prob
    X_corr[mask_missing] = np.nan
    
    # 2. Outliers (Noise)
    mask_outliers = (np.random.random((N, L, L)) < outlier_prob) & (~mask_missing)
    X_corr[mask_outliers] = 50.0 
    
    return X_corr

def save_dataset(X, y, temperatures, filename="ising_data.npz"):
    print(f"Saving dataset to {filename}...")
    np.savez_compressed(filename, X=X, y=y, temperatures=temperatures)
    print("Dataset saved.")

def load_dataset(filename="ising_data.npz"):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return None, None, None
    print(f"Loading dataset from {filename}...")
    data = np.load(filename)
    return data['X'], data['y'], data['temperatures']
