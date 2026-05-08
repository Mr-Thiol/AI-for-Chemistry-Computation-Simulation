import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_tsne_property_mapping(csv_filepath: str):
    """
    Computes t-SNE (perplexity=30) and maps original chemical properties 
    onto the 2D layout in a multi-subplot grid.
    """
    # ==========================================
    # 1. Data Loading & Preparation
    # ==========================================
    try:
        df_original = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filepath}.")
        return

    # Extract continuous numeric properties dynamically
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
    X_raw = df_original[numeric_cols].values
    
    print(f"Loaded {len(df_original)} molecules with {len(numeric_cols)} numeric properties.")

    # ==========================================
    # 2. Compute t-SNE Coordinates (Perplexity = 30)
    # ==========================================
    # Preprocess and PCA (Best practice for t-SNE stability)
    X_scaled = StandardScaler().fit_transform(X_raw)
    
    pca = PCA(random_state=42)
    pca.fit(X_scaled)
    optimal_n = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
    X_pca = PCA(n_components=optimal_n, random_state=42).fit_transform(X_scaled)
    
    print("Computing t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    tsne_coords = tsne.fit_transform(X_pca)

    # ==========================================
    # 3. Grid Layout & Iterative Plotting
    # ==========================================
    # Create a 4x4 grid (16 subplots total, enough for up to 16 properties)
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    # Loop through the properties and plot
    for i, prop in enumerate(numeric_cols):
        ax = axes[i]
        
        # Scatter plot colored by the current property
        # Using 'viridis' as a perceptually uniform colormap
        scatter = ax.scatter(
            tsne_coords[:, 0], 
            tsne_coords[:, 1], 
            c=df_original[prop], 
            cmap='viridis', 
            s=15, 
            alpha=0.8,
            edgecolors='none'
        )
        
        # Aesthetics
        ax.set_title(prop, fontsize=12, fontweight='bold')
        ax.set_xticks([]) # Remove x ticks
        ax.set_yticks([]) # Remove y ticks
        
        # Add colorbar specific to this subplot
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

    # ==========================================
    # 4. Cleanup & Export
    # ==========================================
    # Hide any unused subplots if there are fewer than 16 properties
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    # Save the figure
    output_filename = r"E:\JupyterPjs\AI4Chem\HW2\tSNE_Property_Mapping.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved successfully as {output_filename}")
    
    plt.show()

# Execute the pipeline
if __name__ == "__main__":
    # Ensure this matches your generated CSV file
    csv_path = r"E:\JupyterPjs\AI4Chem\HW2\QM9_global_properties.csv" 
    plot_tsne_property_mapping(csv_path)