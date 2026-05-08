import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans_pipeline(csv_filepath: str):
    """
    Executes K-Means clustering on PCA-reduced data, automatically determines 
    the optimal K, and visualizes the evaluation metrics and final clusters.
    """
    # ==========================================
    # 0. Data Loading
    # ==========================================
    try:
        df_pca = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filepath}. Run the PCA script first.")
        return

    # Extract only the Principal Component columns for clustering
    pc_cols = [col for col in df_pca.columns if col.startswith('PC')]
    X_pca = df_pca[pc_cols].values
    
    print(f"Loaded {X_pca.shape[0]} molecules with {X_pca.shape[1]} Principal Components.")

    # ==========================================
    # 1. Iterative Evaluation (K = 2 to 12)
    # ==========================================
    K_range = range(2, 13)
    sse = []
    sil_scores = []
    
    print("Evaluating K values from 2 to 12...")
    for k in K_range:
        # n_init=10 mitigates sensitivity to initial points [cite: 967]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        
        # Record Sum of Squared Errors (Inertia) [cite: 921]
        sse.append(kmeans.inertia_)
        # Record Silhouette Score
        sil_scores.append(silhouette_score(X_pca, labels))

    # ==========================================
    # 2. Automatic K Selection
    # ==========================================
    # Find the index of the maximum Silhouette Score
    best_idx = np.argmax(sil_scores)
    optimal_k = K_range[best_idx]
    max_sil_score = sil_scores[best_idx]
    
    print(f"Optimal K determined as {optimal_k} (Silhouette Score: {max_sil_score:.4f})")

    # ==========================================
    # 3. Final Clustering
    # ==========================================
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_pca['Cluster'] = final_kmeans.fit_predict(X_pca)
    
    # Export the final clustered dataframe
    output_csv = r"E:\JupyterPjs\AI4Chem\HW2\QM9_PCA_Clustered.csv"
    df_pca.to_csv(output_csv, index=False)
    print(f"Clustered data saved to {output_csv}")

    # ==========================================
    # 4. Comprehensive Visualization
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Subplot 1: Elbow Method (SSE vs K)
    axes[0].plot(K_range, sse, marker='o', linestyle='-', color='b')
    axes[0].set_title('Elbow Method (Inertia)', fontsize=12)
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Sum of Squared Errors (SSE)')
    axes[0].set_xticks(K_range)
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Silhouette Analysis
    axes[1].plot(K_range, sil_scores, marker='s', linestyle='-', color='g')
    axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
    axes[1].set_title('Silhouette Analysis', fontsize=12)
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_xticks(K_range)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Chemical Space Clustering (PC1 vs PC2)
    # Using 'tab10' for discrete qualitative colors
    scatter = axes[2].scatter(
        df_pca['PC1'], 
        df_pca['PC2'], 
        c=df_pca['Cluster'], 
        cmap='tab10', 
        s=20, 
        alpha=0.8, 
        edgecolors='w', 
        linewidth=0.5
    )
    axes[2].set_title(f'K-Means Clustering (K={optimal_k})', fontsize=12)
    axes[2].set_xlabel('Principal Component 1')
    axes[2].set_ylabel('Principal Component 2')
    
    # Create a discrete legend for the clusters
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.8)
    axes[2].legend(handles, [f"Cluster {i}" for i in range(optimal_k)], title="Clusters")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save and show figure
    fig_filename = r"E:\JupyterPjs\AI4Chem\HW2\KMeans_Evaluation.png"
    plt.savefig(fig_filename, bbox_inches='tight')
    print(f"Figure saved successfully as {fig_filename}")
    
    plt.show()

# Execute the pipeline
if __name__ == "__main__":
    # Ensure this matches the CSV generated in the PCA step
    csv_path = r"E:\JupyterPjs\AI4Chem\HW2\QM9_PCA_Reduced.csv" 
    run_kmeans_pipeline(csv_path)