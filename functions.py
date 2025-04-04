import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union, Dict, Any
import os
import tempfile
from matplotlib.backends.backend_pdf import PdfPages
import requests
from bs4 import BeautifulSoup
import re
import scipy.sparse
import subprocess
from datetime import datetime

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
    missing_files = []
    for file_path, file_type in [(mtx_path, 'matrix'), (barcodes_path, 'barcodes'), (features_path, 'features')]:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_type} file: {file_path}")
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files:\n" + "\n".join(missing_files))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Create a temporary directory for symbolic links
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create symbolic links with expected names
            matrix_link = os.path.join(temp_dir, 'matrix.mtx.gz' if mtx_path.endswith('.gz') else 'matrix.mtx')
            barcodes_link = os.path.join(temp_dir, 'barcodes.tsv.gz' if barcodes_path.endswith('.gz') else 'barcodes.tsv')
            features_link = os.path.join(temp_dir, 'features.tsv.gz' if features_path.endswith('.gz') else 'features.tsv')
            
            # Create symbolic links
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copy2(mtx_path, matrix_link)
                shutil.copy2(barcodes_path, barcodes_link)
                shutil.copy2(features_path, features_link)
            else:  # Unix-like systems
                os.symlink(mtx_path, matrix_link)
                os.symlink(barcodes_path, barcodes_link)
                os.symlink(features_path, features_link)
            
            # Read the 10x data
            adata = sc.read_10x_mtx(
                path=temp_dir,
                var_names='gene_symbols',
                cache=True,
                gex_only=True
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
    min_genes: int = None,
    min_cells: int = None,
    max_percent_mt: float = None,
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    resolution: float = 0.5,
    batch_key: str = None
) -> Tuple[sc.AnnData, Dict[str, plt.Figure]]:
    """
    Preprocess single-cell data using Scanpy pipeline, following the official tutorial.
    
    Args:
        adata: AnnData object containing the data
        min_genes: Minimum number of genes per cell
        min_cells: Minimum number of cells per gene
        max_percent_mt: Maximum percentage of mitochondrial genes
        n_top_genes: Number of highly variable genes
        n_pcs: Number of principal components
        n_neighbors: Number of neighbors for computing the neighborhood graph
        resolution: Resolution for Leiden clustering
        batch_key: Key in adata.obs for batch information
        
    Returns:
        Tuple containing:
        - Preprocessed AnnData object
        - Dictionary of generated figures
    """
    # Make a copy of the AnnData object
    adata = adata.copy()
    figures = {}  # Store figures separately
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Generate highest expressed genes plot
    plt.figure(figsize=(8, 6))
    sc.pl.highest_expr_genes(adata, n_top=20, show=False)
    figures['highest_expr_genes'] = plt.gcf()
    plt.close()
    
    # Store QC metrics before filtering as numpy arrays
    adata.uns['qc_metrics_before'] = {
        'n_genes_by_counts': adata.obs['n_genes_by_counts'].to_numpy(),
        'total_counts': adata.obs['total_counts'].to_numpy(),
        'pct_counts_mt': adata.obs['pct_counts_mt'].to_numpy()
    }
    
    # Auto-detect QC thresholds if not provided
    if min_genes is None:
        min_genes = int(np.percentile(adata.obs['n_genes_by_counts'], 5))
    if min_cells is None:
        min_cells = 3
    if max_percent_mt is None:
        max_percent_mt = np.percentile(adata.obs['pct_counts_mt'], 95)
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Filter cells based on mitochondrial percentage
    adata = adata[adata.obs['pct_counts_mt'] < max_percent_mt, :]
    
    # Store raw counts
    adata.layers['counts'] = adata.X.copy()
    
    # Normalize total per cell
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log transform
    sc.pp.log1p(adata)
    
    # Store normalized and log-transformed data
    adata.layers['log1p'] = adata.X.copy()
    
    # Identify highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        flavor='seurat',
        subset=True
    )
    
    # Generate HVG plot but don't store it in adata
    sc.pl.highly_variable_genes(adata, show=False)
    figures['highly_variable_genes'] = plt.gcf()
    plt.close()
    
    # Scale data
    sc.pp.scale(adata, max_value=10)
    
    # Run PCA
    sc.tl.pca(adata, svd_solver='arpack', use_highly_variable=True, n_comps=n_pcs)
    
    # Generate PCA variance ratio plot but don't store it in adata
    sc.pl.pca_variance_ratio(adata, n_pcs=50, show=False)
    figures['pca_variance'] = plt.gcf()
    plt.close()
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    
    # Run UMAP
    sc.tl.umap(adata)
    
    # Run Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)
    
    # Store preprocessing parameters
    adata.uns['preprocessing_params'] = {
        'min_genes': int(min_genes),
        'min_cells': int(min_cells),
        'max_percent_mt': float(max_percent_mt),
        'n_top_genes': int(n_top_genes),
        'n_pcs': int(n_pcs),
        'n_neighbors': int(n_neighbors),
        'resolution': float(resolution),
        'batch_key': batch_key
    }
    
    return adata, figures

def plot_umap(
    adata: sc.AnnData,
    color: Optional[str] = None,
    size: float = 10,
    alpha: float = 0.6,
    frameon: bool = True,
    legend_loc: str = 'on data',
    legend_fontsize: int = 8,
    legend_fontweight: str = 'bold',
    title: Optional[str] = None,
    palette: str = 'tab20'
) -> plt.Figure:
    """Generate publication-quality UMAP plot."""
    try:
        # Verify UMAP coordinates exist
        if 'X_umap' not in adata.obsm:
            raise ValueError("UMAP coordinates not found. Please run UMAP analysis first.")
        
        # Set up the figure with consistent size
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        
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
        ax.set_xlabel('UMAP1', fontsize=10, fontweight='bold')
        ax.set_ylabel('UMAP2', fontsize=10, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Add grid with custom style
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=8, width=1.5, length=6)
        
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
    size: int = 10,
    alpha: float = 0.8,
    palette: str = 'tab20',
    legend_loc: str = 'right margin',
    legend_fontsize: int = 8,
    legend_fontweight: str = 'bold',
    title_fontsize: int = 12,
    title_fontweight: str = 'bold',
    axis_label_fontsize: int = 10,
    axis_label_fontweight: str = 'bold',
    tick_label_fontsize: int = 8,
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
        
        # Create figure with consistent size
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        
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
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight, pad=10)
        
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

def analyze_qc_metrics(adata: sc.AnnData) -> Dict[str, Any]:
    """
    Analyze QC metrics from the dataset and suggest filtering parameters.
    
    Args:
        adata: AnnData object containing the data
        
    Returns:
        Dictionary containing QC metrics and suggested filtering parameters
    """
    try:
        # Calculate basic QC metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        # Get number of cells and genes
        n_cells = adata.n_obs
        n_genes = adata.n_vars
        
        # Get genes per cell and total counts
        genes_per_cell = adata.obs['n_genes_by_counts'].values
        total_counts = adata.obs['total_counts'].values
        
        # Calculate mitochondrial percentage if available
        mito_percent = None
        if 'pct_counts_mt' in adata.obs.columns:
            mito_percent = adata.obs['pct_counts_mt'].values
        
        # Calculate statistics
        stats = {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'genes_per_cell': genes_per_cell,
            'total_counts': total_counts,
            'mito_percent': mito_percent,
            'min_genes': int(np.percentile(genes_per_cell, 5)),  # 5th percentile
            'max_genes': int(np.percentile(genes_per_cell, 95)),  # 95th percentile
            'min_counts': int(np.percentile(total_counts, 5)),  # 5th percentile
            'max_counts': int(np.percentile(total_counts, 95)),  # 95th percentile
            'max_mito': float(np.percentile(mito_percent, 95)) if mito_percent is not None else 20.0  # 95th percentile
        }
        
        return stats
        
    except Exception as e:
        print(f"Error in analyze_qc_metrics: {str(e)}")
        print("\nAvailable columns in adata.obs:")
        print(adata.obs.columns.tolist())
        raise

def plot_qc_metrics(adata: sc.AnnData) -> plt.Figure:
    """
    Plot QC metrics on UMAP.
    
    Args:
        adata: AnnData object containing the data
        
    Returns:
        Matplotlib figure with QC metric plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    # Plot UMAP colored by QC metrics
    metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
    for i, metric in enumerate(metrics):
        if metric in adata.obs.columns:
            sc.pl.umap(
                adata,
                color=metric,
                show=False,
                ax=axes[i],
                title=f'UMAP colored by {metric}'
            )
    
    # Plot UMAP colored by clusters
    sc.pl.umap(
        adata,
        color='leiden',
        show=False,
        ax=axes[3],
        title='UMAP colored by clusters'
    )
    
    plt.tight_layout()
    return fig

def plot_pca_variance(adata: sc.AnnData) -> plt.Figure:
    """
    Plot PCA variance explained.
    
    Args:
        adata: AnnData object containing the data
        
    Returns:
        Matplotlib figure with PCA variance plot
    """
    if 'pca' not in adata.uns:
        raise ValueError("PCA not found. Please run PCA first.")
    
    var_ratio = adata.uns['pca']['variance_ratio']
    cumsum = np.cumsum(var_ratio)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(var_ratio) + 1), cumsum, 'b-', label='Cumulative')
    ax.plot(range(1, len(var_ratio) + 1), var_ratio, 'r-', label='Individual')
    ax.set_title('PCA Variance Explained', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Principal Component', fontsize=10, fontweight='bold')
    ax.set_ylabel('Variance Explained', fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    
    return fig

def generate_latex_report(
    adata: sc.AnnData,
    output_dir: str,
    study_info: Dict[str, str],
    params: Dict[str, Any],
    qc_stats: Dict[str, Any],
    figures: Optional[Dict[str, plt.Figure]] = None,
    runtime: float = None
) -> str:
    """
    Generate a LaTeX report with all analysis results.
    
    Args:
        adata: AnnData object containing the data
        output_dir: Directory to save the report
        study_info: Dictionary containing study information
        params: Dictionary containing analysis parameters
        qc_stats: Dictionary containing QC statistics
        figures: Optional dictionary of pre-generated figures from preprocessing
        runtime: Total runtime of the analysis in seconds
        
    Returns:
        Path to the generated PDF report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figures as PNG files
    figure_paths = {}
    if figures is not None:
        for name, fig in figures.items():
            png_path = os.path.join(output_dir, f"{name}.png")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            figure_paths[name] = png_path
            plt.close(fig)
    
    # Generate LaTeX content
    latex_content = f"""
\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}
\\usepackage{{geometry}}
\\usepackage{{fancyhdr}}
\\usepackage{{caption}}
\\usepackage{{subcaption}}

% Page geometry
\\geometry{{a4paper, margin=1in}}

% Header and footer
\\pagestyle{{fancy}}
\\fancyhf{{}}
\\fancyhead[L]{{scAgentic}}
\\fancyhead[R]{{Single-Cell Analysis Report}}
\\fancyfoot[C]{{Page \\thepage}}

% Document info
\\title{{Single-Cell RNA-seq Analysis Report}}
\\author{{Akram Vasighizaker}}
\\date{{{datetime.now().strftime('%B %d, %Y')}}}

% Begin document
\\begin{{document}}

% Title page
\\begin{{titlepage}}
    \\centering
    \\vspace*{{2cm}}
    
    % Logo if available
    \\IfFileExists{{scagentic_logo.png}}{{
        \\includegraphics[width=0.3\\textwidth]{{scagentic_logo.png}}
        \\vspace{{1cm}}
    }}{{}}
    
    \\Huge\\textbf{{Single-Cell RNA-seq Analysis Report}}\\\\[1cm]
    
    \\Large\\textbf{{Akram Vasighizaker}}\\\\[0.5cm]
    
    \\large{{{datetime.now().strftime('%B %d, %Y')}}}\\\\[0.5cm]
    
    \\large{{GEO Accession: {study_info.get('geo_accession', 'Not available')}}}\\\\[2cm]
    
    \\vfill
    \\large{{Generated by scAgentic - AI-Powered Single-Cell Analysis}}
\\end{{titlepage}}

% Table of Contents
\\tableofcontents
\\newpage

% Study Information
\\section{{Study Information}}
\\subsection{{Metadata}}
\\begin{{table}}[h]
    \\centering
    \\begin{{tabular}}{{ll}}
        \\toprule
        \\textbf{{Field}} & \\textbf{{Value}} \\\\
        \\midrule
        Title & {study_info.get('title', 'Not available')} \\\\
        Organism & {study_info.get('organism', 'Not available')} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Study metadata from GEO}}
\\end{{table}}

\\subsection{{Study Summary}}
{study_info.get('summary', 'Not available')}

% Analysis Summary
\\section{{Analysis Summary}}
\\begin{{table}}[h]
    \\centering
    \\begin{{tabular}}{{ll}}
        \\toprule
        \\textbf{{Metric}} & \\textbf{{Value}} \\\\
        \\midrule
        Total Cells & {adata.n_obs:,} \\\\
        Total Genes & {adata.n_vars:,} \\\\
        Number of Clusters & {len(adata.obs['leiden'].unique())} \\\\
        Runtime & {runtime:.2f} seconds \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Summary of analysis results}}
\\end{{table}}

\\subsection{{Analysis Parameters}}
\\begin{{table}}[h]
    \\centering
    \\begin{{tabular}}{{ll}}
        \\toprule
        \\textbf{{Parameter}} & \\textbf{{Value}} \\\\
        \\midrule
        Min Genes per Cell & {params.get('min_genes', 'N/A')} \\\\
        Min Cells per Gene & {params.get('min_cells', 'N/A')} \\\\
        Max \% MT & {params.get('max_percent_mt', 'N/A')} \\\\
        Top Genes & {params.get('n_top_genes', 'N/A')} \\\\
        Number of PCs & {params.get('n_pcs', 'N/A')} \\\\
        Resolution & {params.get('resolution', 'N/A')} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Analysis parameters used}}
\\end{{table}}

% Quality Control
\\section{{Quality Control}}
\\subsection{{Highest Expressed Genes}}
\\IfFileExists{{{figure_paths.get('highest_expr_genes', '')}}}{{
    \\begin{{figure}}[h]
        \\centering
        \\includegraphics[width=0.8\\textwidth]{{{figure_paths.get('highest_expr_genes', '')}}}
        \\caption{{Top 20 genes by expression level across all cells}}
    \\end{{figure}}
}}{{}}

\\subsection{{QC Distributions}}
\\IfFileExists{{{figure_paths.get('qc_distributions', '')}}}{{
    \\begin{{figure}}[h]
        \\centering
        \\includegraphics[width=0.8\\textwidth]{{{figure_paths.get('qc_distributions', '')}}}
        \\caption{{Distribution of key QC metrics including number of genes per cell, total counts, and mitochondrial percentage}}
    \\end{{figure}}
}}{{}}

% Feature Selection
\\section{{Feature Selection}}
\\subsection{{Highly Variable Genes}}
\\IfFileExists{{{figure_paths.get('highly_variable_genes', '')}}}{{
    \\begin{{figure}}[h]
        \\centering
        \\includegraphics[width=0.8\\textwidth]{{{figure_paths.get('highly_variable_genes', '')}}}
        \\caption{{Selection of highly variable genes based on normalized dispersion}}
    \\end{{figure}}
}}{{}}

% Dimensionality Reduction
\\section{{Dimensionality Reduction}}
\\subsection{{PCA Variance Ratio}}
\\IfFileExists{{{figure_paths.get('pca_variance', '')}}}{{
    \\begin{{figure}}[h]
        \\centering
        \\includegraphics[width=0.8\\textwidth]{{{figure_paths.get('pca_variance', '')}}}
        \\caption{{Variance explained by each principal component}}
    \\end{{figure}}
}}{{}}

\\subsection{{PCA Visualization}}
\\IfFileExists{{{figure_paths.get('pca', '')}}}{{
    \\begin{{figure}}[h]
        \\centering
        \\includegraphics[width=0.8\\textwidth]{{{figure_paths.get('pca', '')}}}
        \\caption{{PCA visualization showing cell clusters in reduced dimensional space}}
    \\end{{figure}}
}}{{}}

\\subsection{{UMAP Visualization}}
\\IfFileExists{{{figure_paths.get('umap', '')}}}{{
    \\begin{{figure}}[h]
        \\centering
        \\includegraphics[width=0.8\\textwidth]{{{figure_paths.get('umap', '')}}}
        \\caption{{UMAP visualization of cell clusters}}
    \\end{{figure}}
}}{{}}

% Differential Expression
\\section{{Differential Expression Analysis}}
\\IfFileExists{{{figure_paths.get('de', '')}}}{{
    \\begin{{figure}}[h]
        \\centering
        \\includegraphics[width=0.8\\textwidth]{{{figure_paths.get('de', '')}}}
        \\caption{{Top differentially expressed genes for each cluster}}
    \\end{{figure}}
}}{{}}

% Appendix
\\appendix
\\section{{Top Differentially Expressed Genes by Cluster}}
"""
    
    # Add DE genes table if available
    if 'rank_genes_groups' in adata.uns:
        for cluster in adata.obs['leiden'].unique():
            latex_content += f"""
\\subsection{{Cluster {cluster}}}
\\begin{{table}}[h]
    \\centering
    \\begin{{tabular}}{{lrr}}
        \\toprule
        \\textbf{{Gene}} & \\textbf{{Score}} & \\textbf{{P-value}} \\\\
        \\midrule
"""
            for i in range(20):
                gene = adata.uns['rank_genes_groups']['names'][cluster][i]
                score = adata.uns['rank_genes_groups']['scores'][cluster][i]
                pval = adata.uns['rank_genes_groups']['pvals'][cluster][i]
                latex_content += f"        {gene} & {score:.2f} & {pval:.2e} \\\\\n"
            
            latex_content += """
        \\bottomrule
    \\end{tabular}
    \\caption{Top 20 differentially expressed genes}
\\end{table}
"""
    
    latex_content += """
\\end{document}
"""
    
    # Save LaTeX file
    tex_path = os.path.join(output_dir, 'report.tex')
    with open(tex_path, 'w') as f:
        f.write(latex_content)
    
    # Compile LaTeX to PDF
    try:
        subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_path], 
                      cwd=output_dir, check=True)
        subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_path], 
                      cwd=output_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling LaTeX: {e}")
        raise
    
    # Return path to the generated PDF
    return os.path.join(output_dir, 'report.pdf')

def generate_consolidated_report(
    adata: sc.AnnData,
    output_dir: str,
    study_info: Dict[str, str],
    params: Dict[str, Any],
    qc_stats: Dict[str, Any],
    figures: Optional[Dict[str, plt.Figure]] = None,
    runtime: float = None
) -> str:
    """
    Generate a consolidated PDF report with all analysis results.
    
    Args:
        adata: AnnData object containing the data
        output_dir: Directory to save the report
        study_info: Dictionary containing study information
        params: Dictionary containing analysis parameters
        qc_stats: Dictionary containing QC statistics
        figures: Optional dictionary of pre-generated figures from preprocessing
        runtime: Total runtime of the analysis in seconds
        
    Returns:
        Path to the generated PDF report
    """
    from latex_report import generate_latex_report
    
    return generate_latex_report(
        adata=adata,
        output_dir=output_dir,
        study_info=study_info,
        params=params,
        qc_stats=qc_stats,
        figures=figures,
        runtime=runtime
    )

def fetch_geo_metadata(accession: str) -> Dict[str, str]:
    """
    Fetch metadata from NCBI GEO for a given accession number.
    
    Args:
        accession: GEO accession number (e.g., GSE242790)
        
    Returns:
        Dictionary containing study metadata
    """
    try:
        # Construct the GEO URL
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
        
        # Fetch the page
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize metadata dictionary with default values
        metadata = {
            'title': 'Not available',
            'organism': 'Not available',
            'tissue': 'Not available',
            'summary': 'Not available',
            'status': 'Not available',
            'source_name': 'Not available'
        }
        
        # Extract title
        title_elem = soup.find('td', string=re.compile('Title', re.IGNORECASE))
        if title_elem and title_elem.find_next('td'):
            metadata['title'] = title_elem.find_next('td').text.strip()
        
        # Extract organism
        organism_elem = soup.find('td', string=re.compile('Organism', re.IGNORECASE))
        if organism_elem and organism_elem.find_next('td'):
            metadata['organism'] = organism_elem.find_next('td').text.strip()
        
        # Extract tissue type
        tissue_elem = soup.find('td', string=re.compile('Tissue', re.IGNORECASE))
        if tissue_elem and tissue_elem.find_next('td'):
            metadata['tissue'] = tissue_elem.find_next('td').text.strip()
        
        # Extract summary
        summary_elem = soup.find('td', string=re.compile('Summary', re.IGNORECASE))
        if summary_elem and summary_elem.find_next('td'):
            metadata['summary'] = summary_elem.find_next('td').text.strip()
        
        # Extract status
        status_elem = soup.find('td', string=re.compile('Status', re.IGNORECASE))
        if status_elem and status_elem.find_next('td'):
            metadata['status'] = status_elem.find_next('td').text.strip()
        
        # Extract source name
        source_elem = soup.find('td', string=re.compile('Source name', re.IGNORECASE))
        if source_elem and source_elem.find_next('td'):
            metadata['source_name'] = source_elem.find_next('td').text.strip()
        
        return metadata
        
    except Exception as e:
        print(f"Error fetching GEO metadata: {str(e)}")
        return {
            'title': 'Not available',
            'organism': 'Not available',
            'tissue': 'Not available',
            'summary': 'Not available',
            'status': 'Not available',
            'source_name': 'Not available'
        } 