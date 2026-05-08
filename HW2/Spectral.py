import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering

def run_spectral_clustering_pipeline(csv_filepath: str):
    """
    Executes Spectral Clustering using the Eigengap heuristic to automatically 
    determine the optimal number of clusters, followed by visualization.
    """
    # ==========================================
    # 0. Data Loading
    # ==========================================
    try:
        df_pca = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filepath}. Run the PCA script first.")
        return

    pc_cols = [f'PC{i}' for i in range(1, 8)]
    if not all(col in df_pca.columns for col in pc_cols):
        print("Warning: 7 PCs not found. Using available PCs.")
        pc_cols = [col for col in df_pca.columns if col.startswith('PC')]
        
    X_pca = df_pca[pc_cols].values
    print(f"Loaded {X_pca.shape[0]} molecules with {X_pca.shape[1]} Principal Components.")

    # ==========================================
    # 1. Similarity Graph Construction (W)
    # ==========================================
    gamma = 1.0 
    W = rbf_kernel(X_pca, gamma=gamma)
    np.fill_diagonal(W, 0) # Remove self-loops [cite: 988]

    # ==========================================
    # 2. Laplacian Matrix (L)
    # ==========================================
    D = np.diag(np.sum(W, axis=1))
    L = D - W # Unnormalized Laplacian [cite: 1007]

    # ==========================================
    # 3. Eigenvalue Analysis
    # ==========================================
    eigenvalues, eigenvectors = la.eigh(L)
    eigenvalues = eigenvalues[:15]

    # ==========================================
    # 4. Automatic K Selection via Eigengap
    # ==========================================
    gaps = np.diff(eigenvalues)
    
    # gaps[0] corresponds to K=1. We search for K >= 2, so we slice from index 1.
    best_gap_idx = np.argmax(gaps[1:]) + 1 
    optimal_k = best_gap_idx + 1
    
    print(f"Maximum Eigengap found after eigenvalue {optimal_k}.")
    print(f"Optimal number of clusters (K) = {optimal_k}")

    # ==========================================
    # 5. Spectral Clustering
    # ==========================================
    spectral = SpectralClustering(
        n_clusters=optimal_k, 
        affinity='rbf', 
        gamma=gamma, 
        random_state=42,
        assign_labels='kmeans'
    )
    df_pca['Spectral_Cluster'] = spectral.fit_predict(X_pca)
    
    output_csv = r"E:\JupyterPjs\AI4Chem\HW2\QM9_Spectral_Clustered.csv"
    df_pca.to_csv(output_csv, index=False)
    print(f"Spectral Clustering results saved to {output_csv}")

    # ==========================================
    # 6. Visualization
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Subplot 1: Laplacian Eigenvalues
    k_range = np.arange(1, 16)
    axes[0].plot(k_range, eigenvalues, marker='o', color='b', linestyle='-')
    axes[0].set_title('First 15 Laplacian Eigenvalues', fontsize=12)
    axes[0].set_xlabel('Eigenvalue Index (i)')
    axes[0].set_ylabel(r'Eigenvalue ($\lambda_i$)')
    axes[0].set_xticks(k_range)
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Eigengaps
    # gap_x aligns with the mathematical index i in delta_i = lambda_{i+1} - lambda_i
    gap_x = np.arange(1, 15) 
    axes[1].plot(gap_x, gaps, marker='s', color='g', linestyle='-')
    
    # FIXED: The vertical line is now plotted exactly at optimal_k to align with the peak
    axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Max Gap (K={optimal_k})')
    
    axes[1].set_title('Eigengap Heuristic', fontsize=12)
    axes[1].set_xlabel('Index (i)')
    axes[1].set_ylabel(r'Eigengap ($\lambda_{i+1} - \lambda_i$)')
    axes[1].set_xticks(gap_x)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Chemical Space Clustering (PC1 vs PC2)
    scatter = axes[2].scatter(
        df_pca['PC1'], 
        df_pca['PC2'], 
        c=df_pca['Spectral_Cluster'], 
        cmap='tab10', 
        s=20, 
        alpha=0.8, 
        edgecolors='w', 
        linewidth=0.5
    )
    axes[2].set_title(f'Spectral Clustering (K={optimal_k})', fontsize=12)
    axes[2].set_xlabel('Principal Component 1')
    axes[2].set_ylabel('Principal Component 2')
    
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.8)
    axes[2].legend(handles, [f"Cluster {i}" for i in range(optimal_k)], title="Clusters")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_filename = r"E:\JupyterPjs\AI4Chem\HW2\Spectral_Clustering_Evaluation.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Figure saved successfully as {output_filename}")
    plt.show()

if __name__ == "__main__":
    csv_path = r"E:\JupyterPjs\AI4Chem\HW2\QM9_Spectral_Clustered.csv" 
    run_spectral_clustering_pipeline(csv_path)