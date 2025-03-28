import streamlit as st
import scanpy as sc
import pandas as pd
from typing import Optional, List, Dict, Any
from functions import (
    load_data,
    preprocess_data,
    plot_umap,
    compare_cell_populations,
    get_gene_expression,
    plot_pca
)
import matplotlib.pyplot as plt
import os

class ScRNAseqAgent:
    def __init__(self, file_path: str):
        """
        Initialize the ScRNAseqAgent.
        
        Args:
            file_path: Path to the h5ad file
        """
        self.file_path = file_path
        self.adata = sc.read_h5ad(file_path)
        self.adata.uns['input_path'] = file_path  # Store the input path for later use
        self.processed = False  # Initialize processed flag
        
    def load_dataset(self, file_path: str) -> bool:
        """Load a dataset from an h5ad file."""
        try:
            self.adata = sc.read_h5ad(file_path)
            # Store the input file path
            self.adata.uns['input_path'] = file_path
            st.success("Dataset loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return False
    
    def preprocess(
        self,
        min_genes: int = 200,
        min_cells: int = 3,
        max_percent_mt: float = 20.0,
        n_top_genes: int = 2000,
        n_pcs: int = 50,
        n_neighbors: int = 15,
        n_clusters: int = 10,
        resolution: float = 0.5
    ) -> bool:
        """Preprocess the data with enhanced visualization."""
        try:
            # Create output directory in the same folder as input file
            input_path = self.adata.uns.get('input_path', '')
            if input_path:
                # Get the directory of the input file
                input_dir = os.path.dirname(input_path)
                # Create analysis_results subdirectory in the same folder
                output_dir = os.path.join(input_dir, 'analysis_results')
            else:
                output_dir = 'analysis_results'
            os.makedirs(output_dir, exist_ok=True)
            
            # Show preprocessing progress
            st.write("Starting preprocessing pipeline...")
            
            # Basic filtering
            st.write("1. Filtering cells and genes...")
            sc.pp.filter_cells(self.adata, min_genes=min_genes)
            sc.pp.filter_genes(self.adata, min_cells=min_cells)
            
            # Calculate QC metrics
            st.write("2. Calculating QC metrics...")
            sc.pp.calculate_qc_metrics(self.adata, inplace=True)
            
            # Detect mitochondrial genes
            var_names = self.adata.var_names.astype(str)
            self.adata.var['mito'] = var_names.str.startswith(('MT-', 'mt-', 'M-', 'm-'))
            
            # Filter based on mitochondrial percentage
            if 'pct_counts_mt' in self.adata.obs.columns:
                sc.pp.filter_cells(self.adata, max_percent_mt=max_percent_mt)
            
            # Normalize and scale
            st.write("3. Normalizing and scaling data...")
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.scale(self.adata, max_value=10)
            
            # Find highly variable genes
            st.write("4. Finding highly variable genes...")
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes)
            
            # Show quality metrics
            st.subheader("Quality Metrics")
            metrics_df = pd.DataFrame({
                'Total cells': [self.adata.n_obs],
                'Total genes': [self.adata.n_vars],
                'Highly variable genes': [sum(self.adata.var.highly_variable)],
                'Number of PCs': [n_pcs],
                'Number of neighbors': [n_neighbors]
            })
            st.dataframe(metrics_df)
            metrics_df.to_csv(os.path.join(output_dir, 'quality_metrics.csv'), index=False)
            
            # Show highly variable genes plot
            st.subheader("Highly Variable Genes")
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            sc.pl.highly_variable_genes(self.adata, show=False)
            ax.set_title('Highly Variable Genes', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Mean expression', fontsize=14, fontweight='bold')
            ax.set_ylabel('Dispersion', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            plt.grid(True, linestyle='--', alpha=0.3, color='gray')
            plt.tight_layout()
            st.pyplot(fig)
            fig.savefig(os.path.join(output_dir, 'hvg_plot.pdf'), bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # PCA
            st.write("5. Running PCA...")
            sc.tl.pca(self.adata, use_highly_variable=True, n_comps=n_pcs)
            
            # Compute neighborhood graph
            st.write("6. Computing neighborhood graph...")
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
            
            # UMAP
            st.write("7. Computing UMAP...")
            sc.tl.umap(self.adata)
            
            # Clustering
            st.write("8. Performing clustering...")
            sc.tl.leiden(self.adata, resolution=resolution)
            
            # Generate PCA plot with clustering colors
            st.subheader("PCA Visualization")
            fig = plot_pca(
                self.adata,
                components='1,2',
                color='leiden',
                title='PCA visualization colored by cluster (PC1 vs PC2)'
            )
            st.pyplot(fig)
            fig.savefig(os.path.join(output_dir, 'pca.pdf'), bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # Generate UMAP plot
            st.subheader("UMAP Visualization")
            fig = plot_umap(
                self.adata,
                color='leiden',
                title='UMAP visualization colored by cluster'
            )
            st.pyplot(fig)
            fig.savefig(os.path.join(output_dir, 'umap.pdf'), bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # Show gene expression distribution for top genes
            st.subheader("Expression of Top Highly Variable Genes")
            top_genes = self.adata.var_names[self.adata.var.highly_variable][:5].tolist()
            for gene in top_genes:
                fig, ax = plt.subplots(figsize=(10, 4))
                sc.pl.violin(self.adata, gene, show=False, ax=ax)
                ax.set_title(f'Expression of {gene}', fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('')
                ax.set_ylabel('Expression', fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
                plt.tight_layout()
                st.pyplot(fig)
                fig.savefig(os.path.join(output_dir, f'violin_plot_{gene}.pdf'), bbox_inches='tight', dpi=300)
                plt.close(fig)
            
            # Save preprocessed data in the same directory as input file
            if input_path:
                # Get the accession number from the input file name
                accession = os.path.splitext(os.path.basename(input_path))[0]
                # Save in the same directory as input file
                output_path = os.path.join(input_dir, f'{accession}_preprocessed.h5ad')
            else:
                output_path = 'preprocessed.h5ad'
            self.adata.write(output_path)
            st.success(f"Preprocessing completed! Results saved to {output_dir}")
            
            # Set processed flag
            self.processed = True
            
            return True
            
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
            return False
    
    def generate_umap(self, color: Optional[str] = None) -> None:
        """Generate UMAP plot with optional coloring."""
        if not self.processed:
            st.error("Please preprocess the data first!")
            return
        
        try:
            # Create output directory
            input_path = self.adata.uns.get('input_path', '')
            if input_path:
                output_dir = os.path.join(os.path.dirname(input_path), 'analysis_results')
            else:
                output_dir = 'analysis_results'
            os.makedirs(output_dir, exist_ok=True)
            
            # Verify color parameter if provided
            if color is not None:
                if color not in self.adata.obs.columns and color not in self.adata.var_names:
                    st.error(f"Color parameter '{color}' not found in the dataset.")
                    st.write("Available observation columns:")
                    st.write(self.adata.obs.columns.tolist())
                    st.write("Available variable names:")
                    st.write(self.adata.var_names.tolist())
                    return
            
            # Generate and display plot
            fig = plot_umap(
                self.adata,
                color=color,
                title=f"UMAP visualization{f' colored by {color}' if color else ''}"
            )
            st.pyplot(fig)
            
            # Save plot
            if color:
                fig.savefig(os.path.join(output_dir, f'umap_{color}.pdf'), bbox_inches='tight', dpi=300)
            else:
                fig.savefig(os.path.join(output_dir, 'umap.pdf'), bbox_inches='tight', dpi=300)
            
            plt.close(fig)  # Close the figure to free memory
            
        except Exception as e:
            st.error(f"Error generating UMAP plot: {str(e)}")
            st.write("\nAvailable observation columns:")
            st.write(self.adata.obs.columns.tolist())
            st.write("\nAvailable variable names:")
            st.write(self.adata.var_names.tolist())
            raise
    
    def analyze_cell_populations(
        self,
        n_genes: int = 25,
        method: str = 'wilcoxon',
        min_fold_change: float = 0.25,
        pval_cutoff: float = 0.05
    ) -> None:
        """Analyze cell populations with enhanced visualization."""
        try:
            # Create output directory in the same folder as input file
            input_path = self.adata.uns.get('input_path', '')
            if input_path:
                # Get the directory of the input file
                input_dir = os.path.dirname(input_path)
                # Create analysis_results subdirectory in the same folder
                output_dir = os.path.join(input_dir, 'analysis_results')
            else:
                output_dir = 'analysis_results'
            os.makedirs(output_dir, exist_ok=True)
            
            # Perform differential expression analysis
            sc.tl.rank_genes_groups(
                self.adata,
                groupby='leiden',
                method=method,
                n_genes=n_genes
            )
            
            # Show heatmap of marker genes
            st.subheader("Marker Genes Heatmap")
            fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
            sc.pl.rank_genes_groups_heatmap(
                self.adata,
                n_genes=n_genes,
                show=False
            )
            ax.set_title('Marker Genes Heatmap', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            fig.savefig(os.path.join(output_dir, 'marker_genes_heatmap.pdf'), bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # Show detailed results table
            st.subheader("Differential Expression Results")
            results = pd.DataFrame()
            for cluster in self.adata.obs['leiden'].unique():
                cluster_results = sc.get.rank_genes_groups_df(
                    self.adata,
                    group=cluster,
                    key='rank_genes_groups'
                )
                cluster_results['cluster'] = cluster
                results = pd.concat([results, cluster_results])
            
            # Filter results based on fold change and p-value
            results = results[
                (abs(results['logfoldchanges']) >= min_fold_change) &
                (results['pvals_adj'] <= pval_cutoff)
            ]
            
            # Display results
            st.dataframe(results)
            
            # Save results to CSV
            results.to_csv(os.path.join(output_dir, 'differential_expression_results.csv'), index=False)
            
            # Show violin plots for top genes
            st.subheader("Expression of Top Marker Genes")
            for cluster in self.adata.obs['leiden'].unique():
                cluster_results = sc.get.rank_genes_groups_df(
                    self.adata,
                    group=cluster,
                    key='rank_genes_groups'
                )
                top_genes = cluster_results.head(3)['names'].tolist()
                
                for gene in top_genes:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sc.pl.violin(self.adata, gene, groupby='leiden', show=False, ax=ax)
                    ax.set_title(f'Expression of {gene} in Cluster {cluster}', fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('Expression', fontsize=12)
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    fig.savefig(os.path.join(output_dir, f'violin_plot_{gene}_cluster_{cluster}.pdf'), bbox_inches='tight', dpi=300)
                    plt.close(fig)
            
            st.success(f"Analysis completed! Results saved to {output_dir}")
            
        except Exception as e:
            st.error(f"Error during cell population analysis: {str(e)}")
    
    def show_gene_expression(
        self,
        gene: str
    ) -> None:
        """Show gene expression across cells."""
        if not self.processed:
            st.error("Please preprocess the data first!")
            return
        
        try:
            # Verify gene exists
            if gene not in self.adata.var_names:
                st.error(f"Gene '{gene}' not found in the dataset.")
                st.write("Available genes:")
                st.write(self.adata.var_names.tolist())
                return
            
            # Create output directory
            input_path = self.adata.uns.get('input_path', '')
            if input_path:
                output_dir = os.path.join(os.path.dirname(input_path), 'analysis_results')
            else:
                output_dir = 'analysis_results'
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate and display plot
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            sc.pl.umap(self.adata, color=gene, show=False, ax=ax)
            
            # Enhance plot aesthetics
            ax.set_title(f'Expression of {gene}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('UMAP1', fontsize=14, fontweight='bold')
            ax.set_ylabel('UMAP2', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Save plot
            fig.savefig(os.path.join(output_dir, f'expression_{gene}.pdf'), bbox_inches='tight', dpi=300)
            plt.close(fig)  # Close the figure to free memory
            
        except Exception as e:
            st.error(f"Error showing gene expression: {str(e)}")
            st.write("\nAvailable genes:")
            st.write(self.adata.var_names.tolist())
            raise
    
    def get_available_genes(self) -> List[str]:
        """Get list of available genes in the dataset."""
        if self.adata is None:
            return []
        return list(self.adata.var_names)
    
    def get_available_metadata(self) -> List[str]:
        """Get list of available metadata columns."""
        if self.adata is None:
            return []
        return list(self.adata.obs.columns) 