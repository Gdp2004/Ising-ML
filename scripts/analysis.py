import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_confusion_matrix(cm, title="Confusion Matrix", filename="confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ordered (0)', 'Disordered (1)'], 
                yticklabels=['Ordered (0)', 'Disordered (1)'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_accuracy_vs_temp(model, X, y, temperatures, title="Accuracy vs Temperature", filename="acc_vs_temp.png"):
    """
    Bin data by temperature and calculate accuracy for each bin.
    OR visualize individual predictions relative to T.
    """
    print(f"Generating binary predictions for {title}...")
    y_pred = model.predict(X)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Temperature': temperatures,
        'True_Label': y,
        'Predicted_Label': y_pred,
        'Correct': (y == y_pred)
    })
    
    # Binning the temperature
    # We want to see the curve, so let's round T to nearest 0.1 or 0.2
    df['Temp_Bin'] = df['Temperature'].round(1)
    
    # Group by Temp_Bin and calculate mean accuracy
    acc_series = df.groupby('Temp_Bin')['Correct'].mean()
    
    plt.figure(figsize=(8, 6))
    plt.plot(acc_series.index, acc_series.values, marker='o', linestyle='-', color='b')
    plt.axvline(x=2.27, color='r', linestyle='--', label='Critical Temp (Tc ~ 2.27)')
    plt.title(title)
    plt.xlabel('Temperature (T)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")
    
    # Also save the underlying data for inspection
    df.to_csv("analysis_results.csv", index=False)

def create_phase_transition_gif(sim, extractor, temps_list=None, filename="phase_transition.gif", fps=5):
    """
    Create an animated GIF showing phase transition from low to high temperature.
    
    Args:
        sim: IsingSimulation instance
        extractor: FeatureExtractor instance
        temps_list: List of temperatures (default: 50 frames from 0.1 to 5.0)
        filename: Output GIF filename
        fps: Frames per second
    """
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter
    
    print(f"Creating phase transition animation...")
    
    # Default temperature range
    if temps_list is None:
        temps_list = np.linspace(0.1, 5.0, 50)
    
    # Generate lattices for all temperatures
    lattices = []
    magnetizations = []
    
    print("Generating lattices at different temperatures...")
    for T in temps_list:
        lattice = sim.generate_lattice(T, steps=1000)
        lattices.append(lattice)
        
        # Calculate magnetization
        lattice_batch = np.array([lattice])
        mag = extractor.get_magnetization(lattice_batch)[0, 0]
        magnetizations.append(mag)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Initialize plots
    def init():
        ax1.clear()
        ax2.clear()
        return []
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        T = temps_list[frame]
        lattice = lattices[frame]
        mag = magnetizations[frame]
        
        # Left panel: Heatmap
        sns.heatmap(
            lattice,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Spin'},
            ax=ax1
        )
        ax1.set_title(f'Configurazione Spin a T = {T:.2f}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Right panel: Magnetization curve
        ax2.plot(temps_list[:frame+1], magnetizations[:frame+1], 'b-', linewidth=2, label='Magnetizzazione')
        ax2.plot(temps_list[frame], magnetizations[frame], 'ro', markersize=12, label=f'T = {T:.2f}')
        
        # Add critical temperature line
        ax2.axvline(x=2.27, color='gray', linestyle='--', alpha=0.5, label='Tc ≈ 2.27')
        
        # Add phase regions
        ax2.axvspan(0, 2.0, alpha=0.1, color='blue', label='Ordinato')
        ax2.axvspan(2.5, 5.0, alpha=0.1, color='red', label='Disordinato')
        
        ax2.set_xlim(0, 5.0)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel('Temperatura (T)', fontsize=12)
        ax2.set_ylabel('Magnetizzazione |M|', fontsize=12)
        ax2.set_title('Transizione di Fase', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        
        # Add text annotation
        phase = "Ordinata" if T < 2.27 else "Disordinata"
        ax2.text(0.02, 0.98, f'Fase: {phase}\nM = {mag:.3f}', 
                transform=ax2.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return []
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        animate, 
        init_func=init,
        frames=len(temps_list), 
        interval=1000/fps,
        blit=True
    )
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close()
    
    print(f"Saved animation to {filename}")
    print(f"Total frames: {len(temps_list)}, Duration: {len(temps_list)/fps:.1f} seconds")

def plot_thermodynamic_observables(observables, filename="thermodynamic_observables.png"):
    """
    Plot thermodynamic observables showing critical behavior at Tc.
    
    Args:
        observables: Dictionary from FeatureExtractor.calculate_thermodynamic_observables()
        filename: Output filename
    """
    print("Plotting thermodynamic observables...")
    
    temps = observables['temps']
    mag_mean = observables['mag_mean']
    energy_mean = observables['energy_mean']
    susceptibility = observables['susceptibility']
    specific_heat = observables['specific_heat']
    
    # Create 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Magnetization
    ax1.plot(temps, mag_mean, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axvline(x=2.27, color='r', linestyle='--', alpha=0.7, label='Tc ≈ 2.27')
    ax1.set_xlabel('Temperatura (T)', fontsize=12)
    ax1.set_ylabel('Magnetizzazione <|M|>', fontsize=12)
    ax1.set_title('Magnetizzazione vs Temperatura', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Energy
    ax2.plot(temps, energy_mean, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.axvline(x=2.27, color='r', linestyle='--', alpha=0.7, label='Tc ≈ 2.27')
    ax2.set_xlabel('Temperatura (T)', fontsize=12)
    ax2.set_ylabel('Energia <E>', fontsize=12)
    ax2.set_title('Energia vs Temperatura', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Magnetic Susceptibility (should peak at Tc)
    ax3.plot(temps, susceptibility, 'r-', linewidth=2, marker='^', markersize=4)
    ax3.axvline(x=2.27, color='r', linestyle='--', alpha=0.7, label='Tc ≈ 2.27')
    ax3.set_xlabel('Temperatura (T)', fontsize=12)
    ax3.set_ylabel('Suscettività Magnetica χ', fontsize=12)
    ax3.set_title('Suscettività Magnetica (Picco a Tc)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Highlight the peak region
    peak_region = (temps > 2.0) & (temps < 2.5)
    if np.any(peak_region):
        ax3.fill_between(temps[peak_region], 0, susceptibility[peak_region], 
                        alpha=0.2, color='red', label='Regione Critica')
    
    # 4. Specific Heat (should also peak at Tc)
    ax4.plot(temps, specific_heat, 'm-', linewidth=2, marker='d', markersize=4)
    ax4.axvline(x=2.27, color='r', linestyle='--', alpha=0.7, label='Tc ≈ 2.27')
    ax4.set_xlabel('Temperatura (T)', fontsize=12)
    ax4.set_ylabel('Calore Specifico C', fontsize=12)
    ax4.set_title('Calore Specifico (Picco a Tc)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Highlight the peak region
    if np.any(peak_region):
        ax4.fill_between(temps[peak_region], 0, specific_heat[peak_region], 
                        alpha=0.2, color='magenta', label='Regione Critica')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"Saved {filename}")
    
    # Print peak information
    chi_max_idx = np.argmax(susceptibility)
    c_max_idx = np.argmax(specific_heat)
    
    print(f"\n=== Critical Behavior Analysis ===")
    print(f"Susceptibility peak at T = {temps[chi_max_idx]:.2f} (χ = {susceptibility[chi_max_idx]:.4f})")
    print(f"Specific Heat peak at T = {temps[c_max_idx]:.2f} (C = {specific_heat[c_max_idx]:.4f})")
    print(f"Expected Tc ≈ 2.27")

def plot_roc_curve(model, X_test, y_test, title="ROC Curve", filename="roc_curve.png"):
    """
    Plot Receiver Operating Characteristic (ROC) curve.
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    print(f"Generating ROC curve for {title}...")
    
    # Get probabilities for the positive class (Class 1: Disordered)
    try:
        y_probs = model.predict_proba(X_test)[:, 1]
    except (AttributeError, NotImplementedError):
        print(f"Warning: Model does not support predict_proba. Skipping ROC verification.")
        return

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_combined_roc_curve(models_dict, X_test_dict, y_test, title="ROC Curve Comparison", filename="roc_combined.png"):
    """
    Plot combined ROC curves for multiple models.
    
    Args:
        models_dict: Dictionary {label: model_instance}
        X_test_dict: Dictionary {label: X_test_data}
        y_test: True labels (assumed same for all models)
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    print(f"Generating combined ROC curve...")
    plt.figure(figsize=(10, 8))
    
    for label, model in models_dict.items():
        if label not in X_test_dict:
            continue
            
        X_test = X_test_dict[label]
        
        try:
            # Class 1 is Disordered
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
        except (AttributeError, NotImplementedError):
            print(f"Warning: Model {label} does not support predict_proba. Skipping.")
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


