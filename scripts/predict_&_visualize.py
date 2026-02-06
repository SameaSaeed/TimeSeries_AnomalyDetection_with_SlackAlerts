import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="EKS TFT Predictor", layout="wide")
sns.set_theme(style="whitegrid")

def run_predictions(df, steps=24):
    """
    Placeholder for your TFT model.predict() logic.
    This simulates the P10, P50, and P90 outputs of a Temporal Fusion Transformer.
    """
    last_date = df['DateTime'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=steps, freq='H')
    
    # Simulating TFT output structure
    base = df['CPU_Usage'].iloc[-1]
    noise = np.random.normal(0, 5, steps).cumsum()
    p50 = base + noise
    p10 = p50 - (np.random.rand(steps) * 10 + 5)
    p90 = p50 + (np.random.rand(steps) * 15 + 10)
    
    return pd.DataFrame({'DateTime': future_dates, 'p10': p10, 'p50': p50, 'p90': p90})

# --- 2. SIDEBAR UPLOAD ---
st.sidebar.title("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload EKS CSV", type=["csv"])

if uploaded_file:
    # Load Data
    data = pd.read_csv(uploaded_file, parse_dates=['DateTime'])
    
    # --- 3. DASHBOARD MAIN ---
    st.title("🚀 EKS Forecasting Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Historical vs Predicted CPU Usage")
        
        # Run Model
        with st.spinner('TFT Model calculating...'):
            pred_df = run_predictions(data)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot History
        hist_df = data.tail(48)
        sns.lineplot(data=hist_df, x='DateTime', y='CPU_Usage', label='History', color='black', ax=ax)
        
        # Plot Median Prediction
        sns.lineplot(data=pred_df, x='DateTime', y='p50', label='TFT Prediction (Median)', color='blue', linestyle='--', ax=ax)
        
        # Shade Uncertainty (P10 - P90)
        ax.fill_between(pred_df['DateTime'], pred_df['p10'], pred_df['p90'], color='blue', alpha=0.2, label='Confidence (P10-P90)')
        
        plt.xticks(rotation=45)
        ax.set_ylabel("CPU Usage (%)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("⚠️ Scaling Alerts")
        # Logic: If P90 > 80%, recommend scaling
        critical_points = pred_df[pred_df['p90'] > 80]
        
        if not critical_points.empty:
            st.error(f"Alert: {len(critical_points)} hours predicted to exceed 80% CPU.")
            st.dataframe(critical_points[['DateTime', 'p90']].rename(columns={'p90': 'Max Predicted %'}))
        else:
            st.success("Cluster stable. No scaling required for the next 24h.")
            
        st.divider()
        st.metric("Avg Predicted CPU", f"{pred_df['p50'].mean():.1f}%")
        st.metric("Peak Predicted Load", f"{pred_df['p90'].max():.1f}%")

else:
    st.info("Please upload a CSV file in the sidebar to begin analysis.")
    st.image("https://raw.githubusercontent.com", width=100)