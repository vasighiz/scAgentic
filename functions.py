import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union
import os

def convert_10x_to_h5ad(
    mtx_path: str,
    barcodes_path: str,
    features_path: str,
    output_path: str
) -> None:
    """
    Convert 10x Genomics format files to AnnData h5ad format.
    
    Args:
        mtx_path: Path to matrix.mtx or matrix.mtx.gz file
        barcodes_path: Path to barcodes.tsv or barcodes.tsv.gz file
        features_path: Path to features.tsv or features.tsv.gz file
        output_path: Path where to save the h5ad file
    """
    # Ensure all input files exist
    for file_path in [mtx_path, barcodes_path, features_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Read the 10x data
        adata = sc.read_10x_mtx(
            path=os.path.dirname(mtx_path),
            var_names='gene_symbols',
            cache=True
        )
        
        # Save as h5ad
        adata.write_h5ad(output_path)
        print(f"Successfully converted and saved to {output_path}")
        
    except Exception as e:
        error_msg = f"Error during conversion: {str(e)}\n"
        error_msg += f"Matrix file: {mtx_path}\n"
        error_msg += f"Barcodes file: {barcodes_path}\n"
        error_msg += f"Features file: {features_path}"
        raise Exception(error_msg)

def load_data(file_path: str) -> sc.AnnData:
    """
    Load AnnData object from h5ad file.
    
    Args:
        file_path: Path to h5ad file
        
    Returns:
        AnnData object
    """
    return sc.read_h5ad(file_path)

def preprocess_data(
    adata: sc.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_percent_mt: float = 20.0,
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15
) -> sc.AnnData:
    """
    Preprocess single-cell data using Scanpy pipeline.
    
    Args:
        adata: AnnData object
        min_genes: Minimum number of genes expressed in a cell
        min_cells: Minimum number of cells expressing a gene
        max_percent_mt: Maximum percentage of mitochondrial genes
        n_top_genes: Number of highly variable genes to select
        n_pcs: Number of principal components for PCA
        n_neighbors: Number of neighbors for UMAP
        
    Returns:
        Preprocessed AnnData object
    """
    try:
        # First calculate basic QC metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        # Detect mitochondrial genes if not already present
        if 'mito' not in adata.var.columns:
            # Convert var_names to string if they're not already
            var_names = adata.var_names.astype(str)
            adata.var['mito'] = var_names.str.startswith(('MT-', 'mt-', 'M-', 'm-'))
        
        # Calculate mitochondrial percentage
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mito'], inplace=True)
        
        # Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        # Filter cells based on mitochondrial percentage
        if 'pct_counts_mt' in adata.obs.columns:
            adata = adata[adata.obs['pct_counts_mt'] < max_percent_mt, :]
        else:
            print("Warning: Could not find mitochondrial percentage column. Skipping mitochondrial filtering.")
        
        # Normalize data
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable]
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
        
        # Run PCA
        sc.tl.pca(adata, n_comps=n_pcs, mask_var="highly_variable")
        
        # Compute neighborhood graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        # Run UMAP
        sc.tl.umap(adata)
        
        # Save preprocessed data
        input_path = adata.uns.get('input_path', '')
        if input_path:
            output_dir = os.path.dirname(input_path)
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            preprocessed_path = os.path.join(output_dir, f"{base_name}_preprocessed.h5ad")
            adata.write(preprocessed_path)
            print(f"Preprocessed data saved to: {preprocessed_path}")
        
        return adata
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        print("\nAvailable columns in adata.obs:")
        print(adata.obs.columns.tolist())
        print("\nAvailable columns in adata.var:")
        print(adata.var.columns.tolist())
        raise

def plot_umap(
    adata: sc.AnnData,
    color: Optional[str] = None,
    size: float = 10,
    alpha: float = 0.6,
    frameon: bool = True,
    legend_loc: str = 'on data',
    legend_fontsize: int = 10,
    legend_fontweight: str = 'bold',
    title: Optional[str] = None,
    palette: str = 'tab20'
) -> plt.Figure:
    """Generate publication-quality UMAP plot."""
    try:
        # Verify UMAP coordinates exist
        if 'X_umap' not in adata.obsm:
            raise ValueError("UMAP coordinates not found. Please run UMAP analysis first.")
        
        # Set up the figure with higher DPI and better style
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        # Verify color parameter if provided
        if color is not None:
            if color not in adata.obs.columns and color not in adata.var_names:
                raise ValueError(f"Color parameter '{color}' not found in observations or variables.")
        
        # Plot UMAP
        sc.pl.umap(
            adata,
            color=color,
            size=size,
            alpha=alpha,
            frameon=frameon,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight,
            title=title,
            palette=palette,
            show=False,
            ax=ax,
            return_fig=False
        )
        
        # Enhance plot aesthetics
        ax.set_xlabel('UMAP1', fontsize=14, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=14, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add grid with custom style
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        
        # Add a light box around the plot
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_umap: {str(e)}")
        print("\nAvailable observation columns:")
        print(adata.obs.columns.tolist())
        print("\nAvailable variable names:")
        print(adata.var_names.tolist())
        raise

def plot_pca(
    adata: sc.AnnData,
    color: Optional[str] = None,
    components: Optional[Union[str, List[int]]] = None,
    title: Optional[str] = None,
    size: int = 12,
    alpha: float = 0.8,
    palette: str = 'tab20',
    legend_loc: str = 'right margin',
    legend_fontsize: int = 10,
    legend_fontweight: str = 'bold',
    title_fontsize: int = 16,
    title_fontweight: str = 'bold',
    axis_label_fontsize: int = 14,
    axis_label_fontweight: str = 'bold',
    tick_label_fontsize: int = 12,
    grid_linewidth: float = 0.5,
    grid_alpha: float = 0.3,
    spine_linewidth: float = 1.5,
    dpi: int = 150
) -> plt.Figure:
    """Plot PCA visualization with enhanced aesthetics."""
    try:
        # Check if PCA exists
        if 'X_pca' not in adata.obsm:
            raise ValueError("PCA coordinates not found. Please run PCA first.")
        
        # Convert components to string format if it's a list
        if isinstance(components, list):
            components = ','.join(map(str, components))
        
        # Verify color parameter if provided
        if color is not None:
            if color not in adata.obs.columns and color not in adata.var_names:
                raise ValueError(f"Color parameter '{color}' not found in the dataset.")
        
        # Create figure with high DPI
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        
        # Plot PCA
        sc.pl.pca(
            adata,
            color=color,
            components=components,
            show=False,
            ax=ax,
            size=size,
            alpha=alpha,
            palette=palette,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_fontweight=legend_fontweight
        )
        
        # Enhance plot aesthetics
        if title is None:
            title = f"PCA visualization{f' colored by {color}' if color else ''}"
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=20)
        
        # Set axis labels
        if components is None:
            ax.set_xlabel('PC1', fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)
            ax.set_ylabel('PC2', fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)
        else:
            comps = components.split(',')
            ax.set_xlabel(f'PC{comps[0]}', fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)
            ax.set_ylabel(f'PC{comps[1]}', fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=grid_alpha, linewidth=grid_linewidth)
        
        # Enhance spine visibility
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
        
        # Add light box around plot
        ax.patch.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_pca: {str(e)}")
        if color is not None:
            print("\nAvailable observation columns:")
            print(adata.obs.columns.tolist())
            print("\nAvailable variable names:")
            print(adata.var_names.tolist())
        raise

def compare_cell_populations(
    adata: sc.AnnData,
    groupby: str,
    gene_list: List[str]
) -> pd.DataFrame:
    """
    Compare gene expression between cell populations.
    
    Args:
        adata: AnnData object
        groupby: Key for grouping cells
        gene_list: List of genes to compare
        
    Returns:
        DataFrame with statistical results
    """
    sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon')
    results = pd.DataFrame()
    
    for group in adata.obs[groupby].unique():
        group_results = pd.DataFrame(
            adata.uns['rank_genes_groups']['names'][group],
            columns=['gene']
        )
        group_results['scores'] = adata.uns['rank_genes_groups']['scores'][group]
        group_results['pvals'] = adata.uns['rank_genes_groups']['pvals'][group]
        group_results['group'] = group
        results = pd.concat([results, group_results])
    
    return results

def get_gene_expression(
    adata: sc.AnnData,
    gene: str,
    groupby: Optional[str] = None
) -> pd.DataFrame:
    """
    Get expression values for a specific gene.
    
    Args:
        adata: AnnData object
        gene: Gene name
        groupby: Optional grouping key
        
    Returns:
        DataFrame with expression values
    """
    if gene not in adata.var_names:
        raise ValueError(f"Gene {gene} not found in dataset")
    
    expression = pd.DataFrame({
        'expression': adata[:, gene].X.toarray().flatten(),
        'cell': adata.obs_names
    })
    
    if groupby:
        expression[groupby] = adata.obs[groupby]
    
    return expression 