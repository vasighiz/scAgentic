import streamlit as st
import os
from agent import ScRNAseqAgent
from functions import (
    convert_10x_to_h5ad,
    analyze_qc_metrics,
    plot_qc_metrics,
    plot_pca_variance,
    preprocess_data,
    generate_consolidated_report,
    fetch_geo_metadata,
    plot_pca,
    plot_umap
)
from report_generator import generate_pdf_report, sanitize_latex
import gzip
import scanpy as sc
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from typing import List, Dict, Any
import re
from datetime import datetime
import subprocess

def process_query_with_llm(query: str, available_functions: List[str]) -> Dict[str, Any]:
    """
    Process user query using local LLM (Mistral via Ollama).
    
    Args:
        query: User's natural language query
        available_functions: List of available function names
        
    Returns:
        Dictionary containing interpreted query parameters
    """
    try:
        # Prepare the prompt for the LLM
        prompt = f"""
        You are a helpful assistant for single-cell RNA-seq analysis. 
        Available functions: {', '.join(available_functions)}
        
        User query: {query}
        
        Please interpret this query and return a JSON object with:
        1. function_name: The most appropriate function to call
        2. parameters: Any relevant parameters extracted from the query
        3. confidence: A score between 0 and 1 indicating how confident you are in the interpretation
        
        Example response:
        {{
            "function_name": "plot_umap",
            "parameters": {{"color": "leiden"}},
            "confidence": 0.9
        }}
        """
        
        # Call Ollama API (assuming it's running locally)
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            try:
                # Parse the LLM's response
                interpretation = json.loads(result['response'])
                return interpretation
            except json.JSONDecodeError:
                return {
                    'function_name': None,
                    'parameters': {},
                    'confidence': 0.0
                }
        else:
            return {
                'function_name': None,
                'parameters': {},
                'confidence': 0.0
            }
            
    except Exception as e:
        print(f"Error in LLM processing: {str(e)}")
        return {
            'function_name': None,
            'parameters': {},
            'confidence': 0.0
        }

def execute_interpreted_query(interpretation: Dict[str, Any], adata: sc.AnnData) -> Any:
    """
    Execute the interpreted query using the appropriate function.
    
    Args:
        interpretation: Dictionary containing interpreted query parameters
        adata: AnnData object containing the data
        
    Returns:
        Result of the function execution
    """
    if interpretation['confidence'] < 0.5:
        return None
        
    function_name = interpretation['function_name']
    parameters = interpretation['parameters']
    
    # Map function names to actual functions
    function_map = {
        'plot_umap': plot_umap,
        'plot_pca': plot_pca,
        'analyze_qc_metrics': analyze_qc_metrics,
        'plot_qc_metrics': plot_qc_metrics,
        'plot_pca_variance': plot_pca_variance
    }
    
    if function_name in function_map:
        try:
            # Add adata as the first parameter
            return function_map[function_name](adata, **parameters)
        except Exception as e:
            print(f"Error executing {function_name}: {str(e)}")
            return None
    return None

def interpret_query_text(query: str) -> Dict[str, Any]:
    """
    Interpret user query using text matching and regular expressions.
    
    Args:
        query: User's natural language query
        
    Returns:
        Dictionary containing interpreted query parameters
    """
    query = query.lower()
    
    # Define synonyms and patterns
    patterns = {
        'umap': {
            'keywords': ['umap', 'visualization', 'plot', 'show', 'display'],
            'color_patterns': [
                (r'color by (.+)', 1),
                (r'colored by (.+)', 1),
                (r'by (.+)', 1),
                (r'expression of (.+)', 1)
            ],
            'default_color': 'leiden'
        },
        'pca': {
            'keywords': ['pca', 'principal component', 'dimensionality'],
            'color_patterns': [
                (r'color by (.+)', 1),
                (r'colored by (.+)', 1),
                (r'by (.+)', 1)
            ],
            'default_color': 'leiden'
        },
        'de_genes': {
            'keywords': ['differential expression', 'de genes', 'marker genes', 'top genes'],
            'cluster_patterns': [
                (r'cluster (\d+)', 1),
                (r'group (\d+)', 1),
                (r'population (\d+)', 1)
            ]
        },
        'qc_metrics': {
            'keywords': ['qc', 'quality control', 'metrics', 'statistics', 'distribution']
        }
    }
    
    # Check for each function type
    for func_name, func_patterns in patterns.items():
        # Check if any keywords match
        if any(keyword in query for keyword in func_patterns['keywords']):
            params = {}
            
            # Extract color parameter for visualization functions
            if 'color_patterns' in func_patterns:
                for pattern, group in func_patterns['color_patterns']:
                    match = re.search(pattern, query)
                    if match:
                        color = match.group(group).strip()
                        # Map common terms to actual column names
                        color_mapping = {
                            'clusters': 'leiden',
                            'cluster': 'leiden',
                            'groups': 'leiden',
                            'group': 'leiden',
                            'cell type': 'cell_type',
                            'cell types': 'cell_type'
                        }
                        params['color'] = color_mapping.get(color, color)
                        break
                else:
                    params['color'] = func_patterns['default_color']
            
            # Extract cluster number for DE analysis
            if 'cluster_patterns' in func_patterns:
                for pattern, group in func_patterns['cluster_patterns']:
                    match = re.search(pattern, query)
                    if match:
                        params['cluster'] = int(match.group(group))
                        break
            
            return {
                'function_name': func_name,
                'parameters': params,
                'confidence': 0.8  # High confidence for exact matches
            }
    
    return {
        'function_name': None,
        'parameters': {},
        'confidence': 0.0
    }

def check_pdflatex_installed() -> bool:
    """Check if pdflatex is installed and accessible."""
    try:
        subprocess.run(['pdflatex', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def generate_pdf_report_safe(
    output_dir: str,
    study_info: dict,
    parameters: dict,
    plot_files: list[str],
    st_context: Any
) -> str:
    """
    Safely generate a PDF report with proper error handling and logging.
    
    Args:
        output_dir: Directory to save report.tex and final_report.pdf
        study_info: Dictionary containing study information
        parameters: Dictionary of preprocessing parameters
        plot_files: List of plot filenames in output_dir
        st_context: Streamlit context for displaying messages
        
    Returns:
        Path to the generated PDF file
        
    Raises:
        RuntimeError: If PDF generation fails
    """
    # Check if pdflatex is installed
    if not check_pdflatex_installed():
        st_context.error("LaTeX is not installed. Please install pdflatex to generate PDF reports.")
        raise RuntimeError("pdflatex is not installed")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the output directory
    st_context.info(f"Generating PDF report in: {os.path.abspath(output_dir)}")
    
    try:
        # Generate PDF report
        pdf_path = generate_pdf_report(
            output_dir=output_dir,
            study_info=study_info,
            parameters=parameters,
            plot_files=plot_files
        )
        
        # Verify PDF was generated
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        if os.path.getsize(pdf_path) == 0:
            raise RuntimeError("Generated PDF is empty")
        
        st_context.success(f"PDF report generated successfully: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        # Log the error
        st_context.error(f"Error generating PDF report: {str(e)}")
        
        # Print LaTeX logs if they exist
        log_files = ['pdflatex_stdout.log', 'pdflatex_stderr.log', 'report.tex']
        for log_file in log_files:
            log_path = os.path.join(output_dir, log_file)
            if os.path.exists(log_path):
                st_context.error(f"\nContents of {log_file}:")
                with open(log_path, 'r', encoding='utf-8') as f:
                    st_context.code(f.read())
        
        raise RuntimeError(f"PDF generation failed: {str(e)}")

def main():
    st.set_page_config(
        page_title="scAgentic - Single-Cell Analysis",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .main .block-container {
            padding-top: 2rem;
        }
        .stPlotlyChart {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .step-message {
            color: #1f77b4;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        /* Custom sidebar width */
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 200px !important;
            max-width: 250px !important;
        }
        /* Adjust sidebar content padding */
        .css-1d391kg {
            padding-top: 1rem;
        }
        /* Adjust logo container */
        .css-1d391kg > div:first-child {
            padding: 0.5rem;
            text-align: center;
        }
        /* Adjust logo image */
        .css-1d391kg > div:first-child img {
            width: 120px !important;
            height: auto !important;
        }
        /* Upload section styling */
        .upload-section {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .file-info {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
        .upload-warning {
            color: #856404;
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("scagentic_logo.png", width=120)
        st.markdown("### scAgentic")
        st.markdown("AI-Powered Single-Cell Analysis")
        
        # Data info box
        if 'study_info' in st.session_state:
            st.markdown("### Dataset Information")
            info_box = f"""
            <div class="info-box">
                <p><strong>GEO ID:</strong> {st.session_state.study_info.get('geo_accession', 'N/A')}</p>
                <p><strong>Tissue:</strong> {st.session_state.study_info.get('tissue', 'N/A')}</p>
                <p><strong>Organism:</strong> {st.session_state.study_info.get('organism', 'N/A')}</p>
            </div>
            """
            st.markdown(info_box, unsafe_allow_html=True)
        
        # Analysis parameters
        st.markdown("### Analysis Parameters")
        with st.form("analysis_params"):
            min_genes = st.number_input("Min genes per cell", min_value=1, value=200)
            min_cells = st.number_input("Min cells per gene", min_value=1, value=3)
            max_percent_mt = st.number_input("Max % MT", min_value=0, max_value=100, value=20)
            n_top_genes = st.number_input("Number of HVGs", min_value=100, value=2000)
            n_pcs = st.number_input("Number of PCs", min_value=1, value=50)
            resolution = st.number_input("Clustering resolution", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            
            submitted = st.form_submit_button("Update Parameters")
            if submitted:
                st.session_state.params = {
                    'min_genes': min_genes,
                    'min_cells': min_cells,
                    'max_percent_mt': max_percent_mt,
                    'n_top_genes': n_top_genes,
                    'n_pcs': n_pcs,
                    'resolution': resolution
                }
        
        # Download buttons
        if 'output_dir' in st.session_state:
            st.markdown("### Download Results")
            
            # PDF Report
            if os.path.exists(os.path.join(st.session_state.output_dir, 'report.pdf')):
                with open(os.path.join(st.session_state.output_dir, 'report.pdf'), 'rb') as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name="scagentic_report.pdf",
                        mime="application/pdf"
                    )
            
            # Processed Data
            if os.path.exists(os.path.join(st.session_state.output_dir, 'processed_data.h5ad')):
                with open(os.path.join(st.session_state.output_dir, 'processed_data.h5ad'), 'rb') as f:
                    st.download_button(
                        label="Download Processed Data",
                        data=f,
                        file_name="processed_data.h5ad",
                        mime="application/octet-stream"
                    )
    
    # Main content
    st.title("Single-Cell RNA-seq Analysis")
    
    # Data upload section
    st.markdown("### Upload Data")
    st.markdown("""
        <div class="upload-section">
            <p>Upload your single-cell RNA-seq data in either format:</p>
            <ul>
                <li><strong>AnnData (.h5ad) file:</strong> A pre-processed single-cell dataset</li>
                <li><strong>10X Genomics files:</strong> Three files from 10X Genomics output (matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Unified file uploader with no type restrictions
    uploaded_files = st.file_uploader(
        "Upload your data files",
        type=None,  # Accept any file type
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if 'adata' not in st.session_state:
            with st.spinner("Loading data..."):
                try:
                    # Create data directory if it doesn't exist
                    os.makedirs("data", exist_ok=True)
                    
                    # Check if it's a single h5ad file
                    if len(uploaded_files) == 1 and uploaded_files[0].name.endswith('.h5ad'):
                        # Handle h5ad file
                        file_path = os.path.join("data", uploaded_files[0].name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_files[0].getbuffer())
                        
                        # Load data
                        adata = sc.read_h5ad(file_path)
                        
                        # Extract study info from filename
                        geo_id = uploaded_files[0].name.split('.')[0]
                        
                    # Check if it's 10X Genomics files
                    elif len(uploaded_files) >= 3:
                        # Check if all required 10X files are present
                        matrix_file = next((f for f in uploaded_files if any(x in f.name.lower() for x in ['matrix.mtx', 'matrix.mtx.gz'])), None)
                        features_file = next((f for f in uploaded_files if any(x in f.name.lower() for x in ['features.tsv', 'features.tsv.gz'])), None)
                        barcodes_file = next((f for f in uploaded_files if any(x in f.name.lower() for x in ['barcodes.tsv', 'barcodes.tsv.gz'])), None)
                        
                        if not all([matrix_file, features_file, barcodes_file]):
                            missing_files = []
                            if not matrix_file: missing_files.append("matrix.mtx or matrix.mtx.gz")
                            if not features_file: missing_files.append("features.tsv or features.tsv.gz")
                            if not barcodes_file: missing_files.append("barcodes.tsv or barcodes.tsv.gz")
                            raise ValueError(f"Missing required 10X files: {', '.join(missing_files)}")
                        
                        # Create temporary directory for 10X files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save uploaded files
                            matrix_path = os.path.join(temp_dir, "matrix.mtx.gz")
                            features_path = os.path.join(temp_dir, "features.tsv.gz")
                            barcodes_path = os.path.join(temp_dir, "barcodes.tsv.gz")
                            
                            # Handle compressed and uncompressed files
                            def save_file(file_obj, target_path):
                                if file_obj.name.endswith('.gz'):
                                    with open(target_path, "wb") as f:
                                        f.write(file_obj.getbuffer())
                                else:
                                    # Compress the file if it's not already compressed
                                    with open(target_path, "wb") as f:
                                        with gzip.GzipFile(fileobj=f, mode='wb') as gz:
                                            gz.write(file_obj.getbuffer())
                            
                            save_file(matrix_file, matrix_path)
                            save_file(features_file, features_path)
                            save_file(barcodes_file, barcodes_path)
                            
                            # Load 10X data
                            adata = sc.read_10x_mtx(temp_dir)
                        
                        # Extract study info from the first file name
                        geo_id = matrix_file.name.split('_')[0] if '_' in matrix_file.name else "dataset"
                        
                    else:
                        st.markdown("""
                            <div class="upload-warning">
                                <p><strong>Invalid file combination:</strong></p>
                                <p>Please upload either:</p>
                                <ul>
                                    <li>A single .h5ad file, or</li>
                                    <li>Three 10X Genomics files (matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz)</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                        raise ValueError("Invalid file combination")
                    
                    # Store data in session state
                    st.session_state.adata = adata
                    
                    # Store study info
                    st.session_state.study_info = {
                        'geo_accession': geo_id,
                        'title': f"Single-cell RNA-seq analysis of {geo_id}",
                        'organism': "Homo sapiens",  # Default, can be updated
                        'tissue': "Not specified",  # Default, can be updated
                        'summary': "Single-cell RNA-seq analysis performed using scAgentic"
                    }
                    
                    # Display data summary
                    st.success("Data loaded successfully!")
                    st.markdown(f"""
                        <div class="file-info">
                            <p><strong>Dataset Summary:</strong></p>
                            <ul>
                                <li>Number of cells: {adata.n_obs:,}</li>
                                <li>Number of genes: {adata.n_vars:,}</li>
                                <li>GEO ID: {geo_id}</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    st.session_state.adata = None
                    st.session_state.study_info = None
    
    # Analysis steps
    if 'adata' in st.session_state and st.session_state.adata is not None:
        adata = st.session_state.adata
        
        # Initialize output directory if not exists
        if 'output_dir' not in st.session_state:
            st.session_state.output_dir = f"analysis_results_{st.session_state.study_info['geo_accession']}"
            os.makedirs(st.session_state.output_dir, exist_ok=True)
        
        # Initialize parameters if not exists
        if 'params' not in st.session_state:
            st.session_state.params = {
                'min_genes': 200,
                'min_cells': 3,
                'max_percent_mt': 20,
                'n_top_genes': 2000,
                'n_pcs': 50,
                'resolution': 0.5
            }
        
        # Initialize figures dictionary if not exists
        if 'figures' not in st.session_state:
            st.session_state.figures = {}
        
        # Initialize preprocessing flag if not exists
        if 'preprocessing_done' not in st.session_state:
            st.session_state.preprocessing_done = False
        
        # Run automatic preprocessing if data is loaded but preprocessing hasn't been done
        if not st.session_state.preprocessing_done:
            with st.spinner("Running automatic preprocessing pipeline..."):
                try:
                    # Quality Control
                    st.markdown('<p class="step-message">Running quality control...</p>', unsafe_allow_html=True)
                    adata.var['mt'] = adata.var_names.str.startswith('MT-')
                    sc.pp.calculate_qc_metrics(
                        adata,
                        qc_vars=['mt'],
                        percent_top=None,
                        log1p=False,
                        inplace=True
                    )
                    
                    # Create QC plots
                    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                               jitter=0.4, multi_panel=True, show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'qc_distributions.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['qc_distributions'] = fig
                    plt.close(fig)
                    
                    # Filter cells
                    st.markdown('<p class="step-message">Filtering cells...</p>', unsafe_allow_html=True)
                    sc.pp.filter_cells(adata, min_genes=st.session_state.params['min_genes'])
                    sc.pp.filter_genes(adata, min_cells=st.session_state.params['min_cells'])
                    adata = adata[adata.obs['pct_counts_mt'] < st.session_state.params['max_percent_mt'], :]
                    
                    # Normalize and scale
                    st.markdown('<p class="step-message">Normalizing and scaling data...</p>', unsafe_allow_html=True)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.scale(adata, max_value=10)
                    
                    # Find HVGs
                    st.markdown('<p class="step-message">Finding highly variable genes...</p>', unsafe_allow_html=True)
                    sc.pp.highly_variable_genes(adata, n_top_genes=st.session_state.params['n_top_genes'])
                    
                    # Plot HVGs
                    sc.pl.highly_variable_genes(adata, show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'highly_variable_genes.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['highly_variable_genes'] = fig
                    plt.close(fig)
                    
                    # Run PCA
                    st.markdown('<p class="step-message">Running PCA...</p>', unsafe_allow_html=True)
                    sc.tl.pca(adata, use_highly_variable=True)
                    
                    # Plot PCA variance ratio
                    sc.pl.pca_variance_ratio(adata, n_pcs=st.session_state.params['n_pcs'], show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'pca_variance.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['pca_variance'] = fig
                    plt.close(fig)
                    
                    # Compute neighborhood graph
                    st.markdown('<p class="step-message">Computing neighborhood graph...</p>', unsafe_allow_html=True)
                    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=st.session_state.params['n_pcs'])
                    
                    # Run UMAP
                    st.markdown('<p class="step-message">Running UMAP...</p>', unsafe_allow_html=True)
                    sc.tl.umap(adata)
                    
                    # Plot UMAP
                    sc.pl.umap(adata, show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'umap.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['umap'] = fig
                    plt.close(fig)
                    
                    # Run clustering
                    st.markdown('<p class="step-message">Running clustering...</p>', unsafe_allow_html=True)
                    sc.tl.leiden(adata, resolution=st.session_state.params['resolution'])
                    
                    # Plot UMAP with clusters
                    sc.pl.umap(adata, color=['leiden'], show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'umap_clusters.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['umap_clusters'] = fig
                    plt.close(fig)
                    
                    # Run DE analysis
                    st.markdown('<p class="step-message">Running differential expression analysis...</p>', unsafe_allow_html=True)
                    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
                    
                    # Plot DE genes
                    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'de.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['de'] = fig
                    plt.close(fig)
                    
                    # Generate violin plots for top DE genes
                    st.markdown('<p class="step-message">Generating violin plots for top DE genes...</p>', unsafe_allow_html=True)
                    for cluster in adata.obs['leiden'].unique():
                        # Get top genes for this cluster
                        top_genes = adata.uns['rank_genes_groups']['names'][cluster][:5]
                        for gene in top_genes:
                            sc.pl.violin(adata, gene, groupby='leiden', show=False)
                            fig = plt.gcf()
                            if fig.get_axes():
                                fig.savefig(os.path.join(st.session_state.output_dir, f'violin_{gene}.png'),
                                          dpi=300, bbox_inches='tight')
                            plt.close(fig)
                    
                    # Save processed data
                    adata.write(os.path.join(st.session_state.output_dir, 'processed_data.h5ad'))
                    
                    # Generate PDF report
                    st.markdown('<p class="step-message">Generating PDF report...</p>', unsafe_allow_html=True)
                    try:
                        # Ensure all required plots exist
                        required_plots = [
                            'qc_distributions.png',
                            'highly_variable_genes.png',
                            'pca_variance.png',
                            'umap.png',
                            'umap_clusters.png',
                            'de.png'
                        ]
                        
                        missing_plots = []
                        for plot in required_plots:
                            plot_path = os.path.join(st.session_state.output_dir, plot)
                            if not os.path.exists(plot_path):
                                missing_plots.append(plot)
                        
                        if missing_plots:
                            raise FileNotFoundError(f"Missing required plots: {', '.join(missing_plots)}")
                        
                        # Generate PDF report with proper error handling
                        pdf_path = generate_pdf_report_safe(
                            output_dir=st.session_state.output_dir,
                            study_info=st.session_state.study_info,
                            parameters=st.session_state.params,
                            plot_files=required_plots,
                            st_context=st
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating PDF report: {str(e)}")
                        st.session_state.preprocessing_done = False
                        raise
                    
                    # Mark preprocessing as done
                    st.session_state.preprocessing_done = True
                    
                    st.success("Automatic preprocessing completed successfully!")
                    st.markdown("""
                        <div class="info-box">
                            <p><strong>Preprocessing Summary:</strong></p>
                            <ul>
                                <li>Filtered cells: {:,}</li>
                                <li>Filtered genes: {:,}</li>
                                <li>Number of clusters: {}</li>
                            </ul>
                            <p>You can now adjust parameters and re-run specific steps through the chat interface.</p>
                        </div>
                    """.format(adata.n_obs, adata.n_vars, len(adata.obs['leiden'].unique())), 
                    unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error in automatic preprocessing: {str(e)}")
                    st.session_state.preprocessing_done = False
        
        # Chat interface for parameter adjustments
        st.markdown("### Ask Questions About Your Data")
        user_question = st.text_input("Ask a question about your data or request parameter adjustments:")
        if user_question:
            with st.spinner("Processing your request..."):
                try:
                    # Process the question and generate response
                    response = process_question(user_question, adata)
                    st.write(response)
                    
                    # Check if the question was about parameter adjustments
                    if "parameter" in user_question.lower() or "redo" in user_question.lower():
                        # Reset preprocessing flag to allow re-running with new parameters
                        st.session_state.preprocessing_done = False
                        st.experimental_rerun()
                        
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main() 