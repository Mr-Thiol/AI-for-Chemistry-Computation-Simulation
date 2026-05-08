import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def run_pca_tsne_workflow(csv_filepath: str):
    """
    Executes PCA thresholding, data export, and iterative t-SNE visualization.
    """
    # ==========================================
    # 0. Data Loading & Preprocessing
    # ==========================================
    try:
        df_original = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filepath}.")
        return

    # Extract numeric features for scaling [cite: 589]
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    X_raw = df_original[numeric_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print(f"Loaded {X_scaled.shape[0]} molecules. Preprocessing complete.")

    # ==========================================
    # 1. Automatic PCA Thresholding & Transformation
    # ==========================================
    # Fit full PCA to determine cumulative variance [cite: 591]
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    optimal_n = np.argmax(cum_var >= 0.95) + 1
    print(f"Optimal components for >= 95% variance: {optimal_n}")
    
    # Re-fit and transform with optimal_n
    pca_final = PCA(n_components=optimal_n, random_state=42)
    X_pca = pca_final.fit_transform(X_scaled)

    # ==========================================
    # 2. Export Reduced Data
    # ==========================================
    # Generate column names dynamically based on optimal_n
    pc_columns = [f"PC{i+1}" for i in range(optimal_n)]
    df_pca = pd.DataFrame(X_pca, columns=pc_columns)
    
    # Concatenate metadata and export
    metadata_cols = ['Dataset_ID', 'SMILES']
    df_export = pd.concat([df_original[metadata_cols], df_pca], axis=1)
    
    export_filename = r"E:\JupyterPjs\AI4Chem\HW2\QM9_PCA_Reduced.csv"
    df_export.to_csv(export_filename, index=False)
    print(f"PCA-reduced data exported successfully to {export_filename}")

    # ==========================================
    # 3 & 4. t-SNE on PCA Features & Visualization
    # ==========================================
    perplexities = [10, 30, 50, 100]
    
    # Initialize 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten() # Flatten to iterate easily
    
    print("Running t-SNE iterations. This may take a moment depending on dataset size...")
    
    for i, p in enumerate(perplexities):
        # Initialize t-SNE with current perplexity
        tsne = TSNE(n_components=2, perplexity=p, random_state=42, init='pca', learning_rate='auto')
        
        # Fit-transform on the PCA-reduced data (X_pca)
        X_tsne = tsne.fit_transform(X_pca)
        
        # Scatter plot
        axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, edgecolors='w', linewidth=0.5, s=20)
        axes[i].set_title(f't-SNE (Perplexity = {p})', fontsize=14)
        axes[i].set_xlabel('t-SNE Dimension 1', fontsize=10)
        axes[i].set_ylabel('t-SNE Dimension 2', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        print(f"Completed t-SNE for perplexity {p}")

    plt.tight_layout()
    
    # Save high-resolution figure
    fig_filename = r"E:\JupyterPjs\AI4Chem\HW2\tSNE_perplexities_comparison.png"
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved successfully as {fig_filename}")
    
    # Display plot
    plt.show()

# Execute the pipeline
if __name__ == "__main__":
    # Ensure this matches the CSV generated in the previous step
    csv_path = r"E:\JupyterPjs\AI4Chem\HW2\QM9_global_properties.csv" 
    run_pca_tsne_workflow(csv_path)