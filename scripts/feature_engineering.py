import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    def get_magnetization(self, X):
        """
        Calculate Magnetization M = |sum(spins)| / N_spins
        Returns array of shape (N, 1)
        """
        # Sum over last two dimensions (L, L)
        M = np.abs(np.sum(X, axis=(1, 2)))
        N_spins = X.shape[1] * X.shape[2]
        return (M / N_spins).reshape(-1, 1)

    def get_energy(self, X):
        """
        Calculate Energy E = - sum(s_i s_j)
        Normalized by number of spins for comparison? Or total? 
        Let's do total energy or energy per spin.
        """
        # Vectorized calculation for the whole batch
        # X shape: (N, L, L)
        
        # Shift neighbors
        # axis 1 is rows, axis 2 is cols
        roll_up = np.roll(X, 1, axis=1)
        roll_down = np.roll(X, -1, axis=1)
        roll_left = np.roll(X, 1, axis=2)
        roll_right = np.roll(X, -1, axis=2)
        
        neighbors_sum = roll_up + roll_down + roll_left + roll_right
        
        # Element-wise multiply and sum
        E = -0.5 * np.sum(X * neighbors_sum, axis=(1, 2))
        
        return E.reshape(-1, 1)

    def get_physics_features(self, X):
        """
        Combine Magnetization and Energy into a feature matrix.
        Shape: (N, 2)
        """
        print("Extracting physics features (Magnetization, Energy)...")
        mag = self.get_magnetization(X)
        eng = self.get_energy(X)
        return np.hstack([mag, eng])

    def get_raw_features(self, X):
        """
        Flatten the lattice to a vector.
        Shape: (N, L*L)
        """
        print("Flattening data for Deep Learning/Raw approach...")
        return X.reshape(X.shape[0], -1)
    
    def calculate_thermodynamic_observables(self, sim, temps_list, n_samples=100):
        """
        Calculate thermodynamic observables (Magnetization, Energy, Susceptibility, Specific Heat)
        for a range of temperatures using Monte Carlo sampling.
        
        This is used for analysis, not for ML features.
        
        Args:
            sim: IsingSimulation instance
            temps_list: Array of temperatures to sample
            n_samples: Number of independent samples per temperature
            
        Returns:
            dict with arrays: 'temps', 'mag_mean', 'mag_std', 'energy_mean', 'energy_std',
                             'susceptibility', 'specific_heat'
        """
        print(f"Calculating thermodynamic observables for {len(temps_list)} temperatures...")
        
        mag_means = []
        mag_stds = []
        energy_means = []
        energy_stds = []
        susceptibilities = []
        specific_heats = []
        
        for T in temps_list:
            # Generate multiple independent samples at this temperature
            mags = []
            energies = []
            
            for _ in range(n_samples):
                lattice = sim.generate_lattice(T, steps=500)  # Fewer steps for speed
                
                # Calculate magnetization (per spin)
                N_spins = lattice.shape[0] * lattice.shape[1]
                m = np.abs(np.sum(lattice)) / N_spins
                mags.append(m)
                
                # Calculate energy (per spin)
                lattice_batch = np.array([lattice])
                e = self.get_energy(lattice_batch)[0, 0] / N_spins
                energies.append(e)
            
            mags = np.array(mags)
            energies = np.array(energies)
            
            # Mean values
            mag_mean = np.mean(mags)
            energy_mean = np.mean(energies)
            
            # Standard deviations
            mag_std = np.std(mags)
            energy_std = np.std(energies)
            
            # Susceptibility: χ = N * <M²> - <M>² / T = N * Var(M) / T
            # For per-spin quantities: χ = Var(m) / T
            # But we use absolute magnetization, so variance of |M|
            susceptibility = np.var(mags) / T if T > 0 else 0
            
            # Specific Heat: C = N * <E²> - <E>² / T² = N * Var(E) / T²
            # For per-spin: C = Var(e) / T²
            specific_heat = np.var(energies) / (T**2) if T > 0 else 0
            
            mag_means.append(mag_mean)
            mag_stds.append(mag_std)
            energy_means.append(energy_mean)
            energy_stds.append(energy_std)
            susceptibilities.append(susceptibility)
            specific_heats.append(specific_heat)
        
        return {
            'temps': np.array(temps_list),
            'mag_mean': np.array(mag_means),
            'mag_std': np.array(mag_stds),
            'energy_mean': np.array(energy_means),
            'energy_std': np.array(energy_stds),
            'susceptibility': np.array(susceptibilities),
            'specific_heat': np.array(specific_heats)
        }

