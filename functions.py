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
    min_genes: int = 200,
    min_cells: int = 3,
    max_percent_mt: float = 20.0,
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    resolution: float = 0.5
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
        resolution: Resolution for Leiden clustering
        
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
        
        # Run Leiden clustering
        sc.tl.leiden(adata, resolution=resolution)
        
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

def plot_qc_metrics(
    adata: sc.AnnData,
    qc_stats: Dict[str, Any],
    output_dir: str,
    dpi: int = 150
) -> None:
    """Generate comprehensive QC plots."""
    # Set style
    plt.style.use('default')
    
    # 1. Genes per cell histogram
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)  # Consistent size
    sns.histplot(qc_stats['genes_per_cell'], bins=50, ax=ax)
    ax.set_title('Distribution of Genes per Cell', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Number of Genes', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'genes_per_cell.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # 2. Total counts histogram
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)  # Consistent size
    sns.histplot(qc_stats['total_counts'], bins=50, ax=ax)
    ax.set_title('Distribution of Total Counts per Cell', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Total Counts', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'total_counts.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # 3. Mitochondrial percentage plot (if available)
    if qc_stats['mito_percent'] is not None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)  # Consistent size
        sns.histplot(qc_stats['mito_percent'], bins=50, ax=ax)
        ax.set_title('Distribution of Mitochondrial Percentage', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Mitochondrial Percentage', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'mito_percentage.pdf'), bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    # 4. Genes vs Counts scatter plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)  # Consistent size
    ax.scatter(qc_stats['total_counts'], qc_stats['genes_per_cell'], alpha=0.5)
    ax.set_title('Genes vs Total Counts', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Total Counts', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Genes', fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'genes_vs_counts.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_pca_variance(adata: sc.AnnData, output_dir: str, dpi: int = 150) -> None:
    """Plot PCA variance explained."""
    if 'pca' not in adata.uns:
        return
    
    var_ratio = adata.uns['pca']['variance_ratio']
    cumsum = np.cumsum(var_ratio)
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)  # Consistent size
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
    fig.savefig(os.path.join(output_dir, 'pca_variance.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_qc_report(
    adata: sc.AnnData,
    qc_stats: Dict[str, Any],
    output_dir: str,
    study_info: Dict[str, str],
    params: Dict[str, Any]
) -> None:
    """Generate a comprehensive QC report in PDF format."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    # Create PDF document
    doc = SimpleDocTemplate(
        os.path.join(output_dir, 'qc_report.pdf'),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Content
    story = []
    
    # Title
    story.append(Paragraph("Single-Cell RNA-seq Quality Control Report", title_style))
    story.append(Spacer(1, 20))
    
    # Study Information
    story.append(Paragraph("Study Information", heading_style))
    story.append(Spacer(1, 10))
    
    # Add GEO link if available
    geo_link = study_info.get('geo_link', '')
    geo_accession = study_info.get('geo_accession', '')
    if geo_link and geo_accession:
        story.append(Paragraph(f"GEO Study: <link href='{geo_link}'>{geo_accession}</link>", normal_style))
        story.append(Spacer(1, 10))
    
    study_data = [
        ["Study Name:", study_info.get('study_name', 'N/A')],
        ["Tissue:", study_info.get('tissue', 'N/A')],
        ["Species:", study_info.get('species', 'N/A')],
        ["Purpose:", study_info.get('purpose', 'N/A')]
    ]
    
    study_table = Table(study_data, colWidths=[150, 300])
    study_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(study_table)
    story.append(Spacer(1, 20))
    
    # QC Metrics
    story.append(Paragraph("Quality Control Metrics", heading_style))
    story.append(Spacer(1, 10))
    
    metrics_data = [
        ["Metric", "Value"],
        ["Total Cells", str(qc_stats['n_cells'])],
        ["Total Genes", str(qc_stats['n_genes'])],
        ["Min Genes per Cell", str(qc_stats['min_genes'])],
        ["Max Genes per Cell", str(qc_stats['max_genes'])],
        ["Min Total Counts", str(qc_stats['total_counts'][0])],
        ["Max Total Counts", str(qc_stats['total_counts'][-1])],
        ["Max Mitochondrial %", f"{qc_stats['max_mito']:.1f}%"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[200, 100])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Parameters Used
    story.append(Paragraph("Analysis Parameters", heading_style))
    story.append(Spacer(1, 10))
    
    param_data = [
        ["Parameter", "Value"],
        ["Min Genes", str(params.get('min_genes', 'N/A'))],
        ["Min Cells", str(params.get('min_cells', 'N/A'))],
        ["Max % MT", str(params.get('max_percent_mt', 'N/A'))],
        ["Top Genes", str(params.get('n_top_genes', 'N/A'))],
        ["Number of PCs", str(params.get('n_pcs', 'N/A'))],
        ["Neighbors", str(params.get('n_neighbors', 'N/A'))],
        ["Resolution", str(params.get('resolution', 'N/A'))]
    ]
    
    param_table = Table(param_data, colWidths=[200, 100])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(param_table)
    
    # Build PDF
    doc.build(story)

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
            'summary': 'Not available'
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
        
        return metadata
        
    except Exception as e:
        print(f"Error fetching GEO metadata: {str(e)}")
        return {
            'title': 'Not available',
            'organism': 'Not available',
            'tissue': 'Not available',
            'summary': 'Not available'
        }

def generate_consolidated_report(
    adata: sc.AnnData,
    output_dir: str,
    study_info: Dict[str, str],
    params: Dict[str, Any],
    qc_stats: Dict[str, Any]
) -> str:
    """
    Generate a consolidated PDF report with all analysis results.
    
    Args:
        adata: AnnData object with processed data
        output_dir: Directory to save the report
        study_info: Dictionary containing study information
        params: Dictionary containing analysis parameters
        qc_stats: Dictionary containing QC statistics
        
    Returns:
        Path to the generated PDF report
    """
    # Create PDF document
    pdf_path = os.path.join(output_dir, 'consolidated_report.pdf')
    pdf = PdfPages(pdf_path)
    
    # Title page
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=150)
    ax.axis('off')
    title = "Single-Cell RNA-seq Analysis Report"
    if study_info.get('geo_accession'):
        title += f"\nGEO Accession: {study_info['geo_accession']}"
    ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=16, fontweight='bold')
    pdf.savefig(fig)
    plt.close(fig)
    
    # Study metadata page
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=150)
    ax.axis('off')
    metadata_text = "Study Metadata\n\n"
    metadata_text += f"Title: {study_info.get('title', 'Not available')}\n\n"
    metadata_text += f"Organism: {study_info.get('organism', 'Not available')}\n\n"
    metadata_text += f"Tissue: {study_info.get('tissue', 'Not available')}\n\n"
    metadata_text += f"Summary:\n{study_info.get('summary', 'Not available')}\n\n"
    metadata_text += f"GEO Link: {study_info.get('geo_link', 'Not available')}"
    ax.text(0.1, 0.9, metadata_text, fontsize=12, va='top')
    pdf.savefig(fig)
    plt.close(fig)
    
    # Preprocessing parameters
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=150)
    ax.axis('off')
    param_text = "Preprocessing Parameters\n\n"
    for key, value in params.items():
        param_text += f"{key}: {value}\n"
    ax.text(0.1, 0.9, param_text, fontsize=12, va='top')
    pdf.savefig(fig)
    plt.close(fig)
    
    # QC plots
    # 1. Genes per cell
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    sns.histplot(qc_stats['genes_per_cell'], bins=50, ax=ax)
    ax.set_title('Distribution of Genes per Cell', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Number of Genes', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    
    # 2. Total counts
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    sns.histplot(qc_stats['total_counts'], bins=50, ax=ax)
    ax.set_title('Distribution of Total Counts per Cell', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Total Counts', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    
    # 3. Mitochondrial percentage
    if qc_stats['mito_percent'] is not None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        sns.histplot(qc_stats['mito_percent'], bins=50, ax=ax)
        ax.set_title('Distribution of Mitochondrial Percentage', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Mitochondrial Percentage', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    # PCA elbow plot
    if 'pca' in adata.uns:
        var_ratio = adata.uns['pca']['variance_ratio']
        cumsum = np.cumsum(var_ratio)
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
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
        pdf.savefig(fig)
        plt.close(fig)
    
    # PCA plot
    if 'X_pca' in adata.obsm:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        sc.pl.pca(adata, color='leiden', show=False, ax=ax)
        ax.set_title('PCA Plot', fontsize=12, fontweight='bold', pad=10)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    # UMAP plot
    if 'X_umap' in adata.obsm:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        sc.pl.umap(adata, color='leiden', show=False, ax=ax)
        ax.set_title('UMAP Plot', fontsize=12, fontweight='bold', pad=10)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    # DE analysis summary
    if 'rank_genes_groups' in adata.uns:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False, ax=ax)
        ax.set_title('Differential Expression Analysis', fontsize=12, fontweight='bold', pad=10)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    # Close the PDF
    pdf.close()
    
    return pdf_path 