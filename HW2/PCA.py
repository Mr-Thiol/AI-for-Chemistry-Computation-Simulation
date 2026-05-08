import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_variance_threshold_pca(csv_filepath: str, variance_threshold: float = 0.95):
    """
    Executes PCA with automatic variance thresholding and visualizes the results.
    """
    # ==========================================
    # 0. Data Loading & Preprocessing
    # ==========================================
    df = pd.read_csv(csv_filepath)
    
    # Dynamically select all numeric columns (excluding ID, SMILES, InChI)
    # Based on your CSV, this should capture the 14 continuous chemical properties.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X_raw = df[numeric_cols].values
    
    # Zero-mean normalization [cite: 589] and variance scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    total_features = X_scaled.shape[1]
    print(f"Loaded {X_scaled.shape[0]} molecules with {total_features} features.")

    # ==========================================
    # 1. Cumulative Variance Analysis
    # ==========================================
    # Fit PCA on all dimensions to compute the full eigenspectrum [cite: 591]
    pca_full = PCA(n_components=total_features, random_state=42)
    pca_full.fit(X_scaled)
    
    # Calculate cumulative explained variance ratio
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    # ==========================================
    # 2. Automatic Thresholding
    # ==========================================
    # Find the first index where cumulative variance >= threshold
    # np.argmax returns the first index evaluating to True. Add 1 for the count.
    optimal_n = np.argmax(cum_var >= variance_threshold) + 1
    print(f"Optimal number of components to explain >= {variance_threshold*100}% variance: {optimal_n}")

    # ==========================================
    # 3. Final PCA Transformation
    # ==========================================
    # Re-fit and transform using the optimal low-rank approximation [cite: 521, 596]
    pca_final = PCA(n_components=optimal_n, random_state=42)
    X_pca = pca_final.fit_transform(X_scaled)

    # ==========================================
    # 4. Visualization
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Scree Plot (Cumulative Variance)
    axes[0].plot(range(1, total_features + 1), cum_var, marker='o', linestyle='-', color='b', label='Cumulative Variance')
    axes[0].axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold*100}% Threshold')
    axes[0].axvline(x=optimal_n, color='g', linestyle='--', label=f'Optimal n = {optimal_n}')
    
    axes[0].set_title('PCA Cumulative Explained Variance', fontsize=14)
    axes[0].set_xlabel('Number of Principal Components', fontsize=12)
    axes[0].set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    axes[0].set_xticks(range(1, total_features + 1))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Subplot 2: Chemical Space Projection (PC1 vs PC2)
    # Even if optimal_n > 2, we project onto the first two PCs for 2D visualization
    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, edgecolors='w', linewidth=0.5, s=30)
    
    # Annotate variance explained by PC1 and PC2
    pc1_var = pca_full.explained_variance_ratio_[0] * 100
    pc2_var = pca_full.explained_variance_ratio_[1] * 100
    
    axes[1].set_title('Chemical Space Projection (PC1 vs PC2)', fontsize=14)
    axes[1].set_xlabel(f'Principal Component 1 ({pc1_var:.1f}%)', fontsize=12)
    axes[1].set_ylabel(f'Principal Component 2 ({pc2_var:.1f}%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return X_pca, optimal_n

# Execute the pipeline
if __name__ == "__main__":
    # Replace with your actual CSV path if different
    csv_path = r"E:\JupyterPjs\AI4Chem\HW2\QM9_global_properties.csv" 
    try:
        X_reduced, n_components = perform_variance_threshold_pca(csv_path, variance_threshold=0.95)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please ensure the parsing script has been run.")