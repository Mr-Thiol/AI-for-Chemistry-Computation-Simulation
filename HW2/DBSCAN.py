import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def run_dbscan_pipeline(csv_filepath: str):
    """
    Executes an automated DBSCAN clustering pipeline, exports the labeled data,
    prints noise points to the terminal, and visualizes the results.
    """
    # ==========================================
    # 1. Setup & Data Loading
    # ==========================================
    try:
        df_pca = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filepath}. Run the PCA script first.")
        return

    # Extract the PC columns
    pc_cols = [f'PC{i}' for i in range(1, 8)]
    if not all(col in df_pca.columns for col in pc_cols):
        print("Warning: 7 PCs not found. Using all available PC columns.")
        pc_cols = [col for col in df_pca.columns if col.startswith('PC')]
        
    X_pca = df_pca[pc_cols].values
    
    # Rule of thumb: min_samples = 2 * dimensions
    min_samples = 2 * X_pca.shape[1] 
    print(f"Loaded {X_pca.shape[0]} molecules. min_samples set to {min_samples}.")

    # ==========================================
    # 2. K-Distance Computation
    # ==========================================
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_pca)
    distances, indices = nn.kneighbors(X_pca)
    
    k_distances = distances[:, -1]
    k_distances_sorted = np.sort(k_distances)

    # ==========================================
    # 3. Automatic 'eps' Detection (Geometric Method)
    # ==========================================
    x = np.arange(len(k_distances_sorted))
    x_norm = x / np.max(x)
    y_norm = k_distances_sorted / np.max(k_distances_sorted)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    p0 = np.column_stack((x_norm, y_norm))
    numerator = np.abs((p2[0] - p1[0]) * (p1[1] - p0[:, 1]) - (p1[0] - p0[:, 0]) * (p2[1] - p1[1]))
    denominator = np.linalg.norm(p2 - p1)
    distances_to_line = numerator / denominator

    elbow_idx = np.argmax(distances_to_line)
    optimal_eps = k_distances_sorted[elbow_idx]
    
    print(f"Geometric elbow detected at index {elbow_idx}. Optimal eps = {optimal_eps:.4f}")

    # ==========================================
    # 4. DBSCAN Clustering & Export
    # ==========================================
    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_pca)
    
    df_pca['DBSCAN_Cluster'] = labels
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")

    # Export to CSV
    output_csv = r"E:\JupyterPjs\AI4Chem\HW2\QM9_DBSCAN_Clustered.csv"
    df_pca.to_csv(output_csv, index=False)
    print(f"DBSCAN results successfully saved to {output_csv}")

    # Print noise points to terminal
    print(f"\n{'-'*20} NOISE POINTS DETECTED {'-'*20}")
    noise_df = df_pca[df_pca['DBSCAN_Cluster'] == -1]
    if not noise_df.empty:
        # Print ID and SMILES to identify the outlier molecules
        cols_to_print = ['Dataset_ID', 'SMILES'] if 'Dataset_ID' in noise_df.columns else noise_df.columns.tolist()
        print(noise_df[cols_to_print].to_string(index=False))
    else:
        print("No noise points detected.")
    print(f"{'-'*65}\n")

    # ==========================================
    # 5. Visualization
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: K-Distance Curve
    axes[0].plot(x, k_distances_sorted, color='b', linestyle='-', linewidth=2)
    axes[0].plot(elbow_idx, optimal_eps, marker='o', color='r', markersize=8)
    
    axes[0].annotate(f'Optimal eps: {optimal_eps:.3f}', 
                     xy=(elbow_idx, optimal_eps), 
                     xytext=(elbow_idx - len(x)*0.3, optimal_eps + np.max(k_distances_sorted)*0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                     fontsize=10, fontweight='bold')
                     
    axes[0].set_title(f'{min_samples}-Distance Curve', fontsize=12)
    axes[0].set_xlabel('Points sorted by distance')
    axes[0].set_ylabel(f'{min_samples}-th Nearest Neighbor Distance')
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Clustering Result (PC1 vs PC2)
    noise_mask = labels == -1
    cluster_mask = labels != -1

    axes[1].scatter(
        df_pca.loc[noise_mask, 'PC1'], 
        df_pca.loc[noise_mask, 'PC2'], 
        c='grey', 
        alpha=0.4, 
        s=15, 
        label='Noise (-1)',
        edgecolors='none'
    )

    scatter = axes[1].scatter(
        df_pca.loc[cluster_mask, 'PC1'], 
        df_pca.loc[cluster_mask, 'PC2'], 
        c=df_pca.loc[cluster_mask, 'DBSCAN_Cluster'], 
        cmap='tab10', 
        alpha=0.8, 
        s=20, 
        edgecolors='w', 
        linewidth=0.5
    )

    axes[1].set_title(f'DBSCAN Clustering (eps={optimal_eps:.2f})', fontsize=12)
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    
    handles, legend_labels = scatter.legend_elements(prop="colors", alpha=0.8)
    cluster_names = [f"Cluster {i}" for i in set(labels) if i != -1]
    
    if n_noise > 0:
        import matplotlib.patches as mpatches
        noise_patch = mpatches.Circle((0,0), radius=2, color='grey', alpha=0.4)
        handles.insert(0, noise_patch)
        cluster_names.insert(0, "Noise (-1)")
        
    axes[1].legend(handles, cluster_names, title="Clusters", loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_filename = r"E:\JupyterPjs\AI4Chem\HW2\DBSCAN_Evaluation.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Figure saved successfully as {output_filename}")
    plt.show()

if __name__ == "__main__":
    csv_path = r"E:\JupyterPjs\AI4Chem\HW2\QM9_PCA_Reduced.csv" 
    run_dbscan_pipeline(csv_path)