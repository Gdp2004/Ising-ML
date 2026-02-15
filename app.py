import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from data_generator import IsingSimulation
from feature_engineering import FeatureExtractor
from models import ModelTrainer

# Page configuration
st.set_page_config(
    page_title="Ising Model ML App",
    page_icon="🧲",
    layout="wide"
)

# Title
st.title("🧲 Ising Model Interactive ML App")
st.markdown("Esplora le transizioni di fase del modello di Ising con Machine Learning")

# Sidebar controls
st.sidebar.header("⚙️ Controlli")

# Temperature input with both slider and number input
col_slider, col_input = st.sidebar.columns([2, 1])

with col_slider:
    temperature_slider = st.slider(
        "Temperatura (T)",
        min_value=0.1,
        max_value=5.0,
        value=2.27,
        step=0.1,
        help="Tc critica ≈ 2.27",
        key="temp_slider"
    )

with col_input:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    temperature_input = st.number_input(
        "T",
        min_value=0.1,
        max_value=5.0,
        value=temperature_slider,
        step=0.1,
        format="%.2f",
        key="temp_input"
    )

# Use the input value if it differs from slider, otherwise use slider
temperature = temperature_input

st.sidebar.markdown(f"""
**Riferimenti Fisici:**
- T < 2.0: Fase Ordinata (Ferromagnetica)
- T ≈ 2.27: Temperatura Critica
- T > 2.5: Fase Disordinata (Paramagnetica)
""")

# Initialize session state
if 'lattice' not in st.session_state:
    st.session_state.lattice = None
if 'sim' not in st.session_state:
    st.session_state.sim = IsingSimulation(L=16)
if 'extractor' not in st.session_state:
    st.session_state.extractor = FeatureExtractor()

# Generate button
if st.sidebar.button("🔄 Genera Reticolo", type="primary"):
    with st.spinner("Generazione in corso..."):
        # Adaptive thermalization: 5000 steps for critical T (2.0-2.5), 1000 otherwise
        st.session_state.lattice = st.session_state.sim.generate_lattice(temperature)
    st.success("Reticolo generato!")

# Main content
if st.session_state.lattice is not None:
    lattice = st.session_state.lattice
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configurazione degli Spin")
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            lattice,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Spin'},
            ax=ax
        )
        ax.set_title(f"Reticolo 16×16 a T = {temperature:.2f}")
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📊 Proprietà Fisiche")
        
        # Calculate physics features
        lattice_batch = np.array([lattice])
        mag = st.session_state.extractor.get_magnetization(lattice_batch)[0, 0]
        eng = st.session_state.extractor.get_energy(lattice_batch)[0, 0]
        
        # Display metrics
        st.metric("Magnetizzazione", f"{mag:.4f}")
        st.metric("Energia", f"{eng:.2f}")
        
        st.markdown("---")
        st.subheader("🤖 Predizione ML")
        
        # Load model and predict
        model_path = os.path.join("models", "model_physics.pkl")
        if os.path.exists(model_path):
            try:
                model = ModelTrainer.load_model(model_path, model_type='logistic')
                
                # Prepare features
                features = np.array([[mag, eng]])
                prediction = model.predict(features)[0]
                
                # Get probability if available
                if hasattr(model.model, 'predict_proba'):
                    proba = model.model.predict_proba(features)[0]
                    confidence = max(proba) * 100
                else:
                    confidence = None
                
                # Display prediction
                if prediction == 0:
                    st.success("✅ **Fase Ordinata** (Ferromagnetica)")
                    st.markdown("Gli spin sono allineati")
                else:
                    st.warning("⚠️ **Fase Disordinata** (Paramagnetica)")
                    st.markdown("Gli spin sono randomici")
                
                if confidence:
                    st.progress(confidence / 100)
                    st.caption(f"Confidenza: {confidence:.1f}%")
                    
            except Exception as e:
                st.error(f"Errore nel caricamento del modello: {e}")
                st.info("Esegui prima `python scripts/main_pipeline.py` per addestrare i modelli.")
        else:
            st.warning("⚠️ Modello non trovato")
            st.info("Esegui `python scripts/main_pipeline.py` per addestrare i modelli.")
        
        # Additional info
        st.markdown("---")
        st.subheader("ℹ️ Info")
        st.markdown(f"""
        - **Dimensione**: 16×16 = 256 spin
        - **Temperatura**: {temperature:.2f}
        - **Passi MC**: 1000
        """)

else:
    st.info("👈 Usa il pulsante nella barra laterale per generare un reticolo")
    
    # Show example visualization
    st.markdown("### Come funziona")
    st.markdown("""
    1. **Seleziona la temperatura** usando lo slider
    2. **Genera un reticolo** con il pulsante
    3. **Visualizza** la configurazione degli spin
    4. **Osserva** le proprietà fisiche calcolate
    5. **Vedi la predizione** del modello ML addestrato
    
    Il modello classifica automaticamente lo stato come **Ordinato** o **Disordinato** 
    basandosi su Magnetizzazione ed Energia.
    """)

# Analysis Plots Section
st.markdown("---")
st.header("📈 Analisi e Risultati")

# Check for available plots
plot_files = {
    "Confusion Matrix (Physics)": os.path.join("reports", "cm_physics.png"),
    "Confusion Matrix (Raw Data)": os.path.join("reports", "cm_raw.png"),
    "Accuracy vs Temperature (Physics)": os.path.join("reports", "acc_temp_physics.png"),
    "Accuracy vs Temperature (Raw)": os.path.join("reports", "acc_temp_raw.png")
}

available_plots = {name: path for name, path in plot_files.items() if os.path.exists(path)}

if available_plots:
    st.markdown("### Grafici Generati dall'Analisi")
    st.markdown("Questi grafici sono stati generati eseguendo gli script di analisi.")
    
    st.markdown("#### Prestazioni dei Modelli ML")
    
    # Confusion matrices
    cm_cols = st.columns(2)
    if "Confusion Matrix (Physics)" in available_plots:
        with cm_cols[0]:
            st.image(available_plots["Confusion Matrix (Physics)"], 
                    caption="Confusion Matrix - Modello Physics-Based",
                    width='stretch')
    
    if "Confusion Matrix (Raw Data)" in available_plots:
        with cm_cols[1]:
            st.image(available_plots["Confusion Matrix (Raw Data)"], 
                    caption="Confusion Matrix - Modello Raw Data",
                    width='stretch')
    
    # Accuracy curves
    acc_cols = st.columns(2)
    if "Accuracy vs Temperature (Physics)" in available_plots:
        with acc_cols[0]:
            st.image(available_plots["Accuracy vs Temperature (Physics)"], 
                    caption="Accuracy vs Temperatura - Physics Model",
                    width='stretch')
            st.caption("⚠️ Nota: L'accuratezza cala vicino a Tc ≈ 2.27 (temperatura critica)")
    
    if "Accuracy vs Temperature (Raw)" in available_plots:
        with acc_cols[1]:
            st.image(available_plots["Accuracy vs Temperature (Raw)"], 
                    caption="Accuracy vs Temperatura - Raw Model",
                    width='stretch')


else:
    st.info("""
    **Nessun grafico di analisi trovato.**
    
    Per generare i grafici, esegui:
    - `python main_pipeline.py` - Addestra i modelli e genera confusion matrix + accuracy plots
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Ising Model ML Project**")
st.sidebar.markdown("Transizioni di fase con Machine Learning")

