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
    st_context: Any,
    analysis_steps: list[dict] = None
) -> str:
    """
    Generate a PDF report with proper error handling.
    
    Args:
        output_dir: Directory to save report.tex and final_report.pdf
        study_info: Dictionary containing study information
        parameters: Dictionary of preprocessing parameters
        plot_files: List of plot filenames in output_dir
        st_context: Streamlit context for displaying messages
        analysis_steps: List of dictionaries containing analysis steps information
        
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
            plot_files=plot_files,
            analysis_steps=analysis_steps
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
        log_path = os.path.join(output_dir, 'report.log')
        if os.path.exists(log_path):
            st_context.error("\nContents of report.log:")
            with open(log_path, 'r', encoding='utf-8') as f:
                st_context.code(f.read())
        
        raise RuntimeError(f"PDF generation failed: {str(e)}")

def main():
    st.set_page_config(
        page_title="scAgentic - Single-Cell Analysis",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for a cleaner interface
    st.markdown("""
        <style>
        .stApp {
            max-width: 1000px;
            margin: 0 auto;
            padding: 1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 2rem;
        }
        .assistant-message {
            background-color: #f5f5f5;
            margin-right: 2rem;
        }
        .message-content {
            flex-grow: 1;
        }
        .message-icon {
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
        }
        .user-icon {
            background-color: #2196f3;
            color: white;
        }
        .assistant-icon {
            background-color: #757575;
            color: white;
        }
        .upload-section {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        .results-section {
            margin-top: 2rem;
        }
        .plot-container {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .step-message {
            color: #1f77b4;
            font-weight: bold;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background-color: #e3f2fd;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main content
    st.title("🧬 scAgentic")
    st.markdown("AI-Powered Single-Cell Analysis")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Data upload section
    if 'adata' not in st.session_state:
        st.markdown("""
            <div class="upload-section">
                <h3>Upload Your Data</h3>
                <p>Upload your single-cell RNA-seq data in either format:</p>
                <ul>
                    <li><strong>AnnData (.h5ad) file:</strong> A pre-processed single-cell dataset</li>
                    <li><strong>10X Genomics files:</strong> Three files from 10X Genomics output (matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Unified file uploader
        uploaded_files = st.file_uploader(
            "Upload your data files",
            type=None,
            accept_multiple_files=True
        )
        
        if uploaded_files:
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
                        'summary': "Single-cell RNA-seq analysis performed using scAgentic"
                    }
                    
                    # Fetch additional GEO metadata
                    with st.spinner("Fetching GEO metadata..."):
                        geo_metadata = fetch_geo_metadata(geo_id)
                        # Update study info with fetched metadata
                        st.session_state.study_info.update({
                            'title': geo_metadata['title'],
                            'organism': geo_metadata['organism'],
                            'summary': geo_metadata['summary'],
                            'status': geo_metadata['status'],
                            'source_name': geo_metadata['source_name'],
                            'geo_url': f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_id}"
                        })
                    
                    # Display data summary
                    st.success("Data loaded successfully!")
                    st.markdown(f"""
                        <div class="file-info">
                            <p><strong>Dataset Summary:</strong></p>
                            <ul>
                                <li>Number of cells: {adata.n_obs:,}</li>
                                <li>Number of genes: {adata.n_vars:,}</li>
                                <li>GEO ID: <a href="{st.session_state.study_info['geo_url']}" target="_blank">{geo_id}</a></li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display GEO metadata in a two-column format
                    st.markdown("### GEO Metadata")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Title:**")
                        st.markdown("**Status:**")
                        st.markdown("**Source Name:**")
                        st.markdown("**Organism:**")
                    
                    with col2:
                        st.markdown(f"**{st.session_state.study_info['title']}**")
                        st.markdown(f"**{st.session_state.study_info.get('status', 'Not available')}**")
                        st.markdown(f"**{st.session_state.study_info.get('source_name', 'Not available')}**")
                        st.markdown(f"**{st.session_state.study_info['organism']}**")
                    
                    # Add welcome message to chat
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"Welcome! I've loaded your dataset with {adata.n_obs:,} cells and {adata.n_vars:,} genes. I'll now run the analysis pipeline automatically."
                    })
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    st.session_state.adata = None
                    st.session_state.study_info = None
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            st.markdown(f"""
                <div class="chat-message {message['role']}-message">
                    <div class="message-icon {message['role']}-icon">
                        {'👤' if message['role'] == 'user' else '🧬'}
                    </div>
                    <div class="message-content">
                        {message['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
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
        
        # Initialize analysis steps list if not exists
        if 'analysis_steps' not in st.session_state:
            st.session_state.analysis_steps = []
        
        # Run automatic preprocessing if data is loaded but preprocessing hasn't been done
        if not st.session_state.preprocessing_done:
            with st.spinner("Running analysis pipeline..."):
                try:
                    # Clear previous analysis steps
                    st.session_state.analysis_steps = []
                    
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
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Quality Control',
                            'description': 'Calculated quality control metrics and generated distribution plots for genes, counts, and mitochondrial content.',
                            'plot': 'qc_distributions.png'
                        })
                    plt.close(fig)
                    
                    # Filter cells
                    st.markdown('<p class="step-message">Filtering cells...</p>', unsafe_allow_html=True)
                    sc.pp.filter_cells(adata, min_genes=st.session_state.params['min_genes'])
                    sc.pp.filter_genes(adata, min_cells=st.session_state.params['min_cells'])
                    adata = adata[adata.obs['pct_counts_mt'] < st.session_state.params['max_percent_mt'], :]
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'Filtering Cells',
                        'description': f'Filtered cells with at least {st.session_state.params["min_genes"]} genes, genes expressed in at least {st.session_state.params["min_cells"]} cells, and cells with less than {st.session_state.params["max_percent_mt"]}% mitochondrial content.',
                        'plot': None
                    })
                    
                    # Normalize and scale
                    st.markdown('<p class="step-message">Normalizing and scaling data...</p>', unsafe_allow_html=True)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.scale(adata, max_value=10)
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'Normalization and Scaling',
                        'description': 'Normalized total counts per cell, applied log transformation, and scaled the data.',
                        'plot': None
                    })
                    
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
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Highly Variable Genes',
                            'description': f'Identified the top {st.session_state.params["n_top_genes"]} highly variable genes for downstream analysis.',
                            'plot': 'highly_variable_genes.png'
                        })
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
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Principal Component Analysis',
                            'description': f'Performed PCA on highly variable genes and visualized variance explained by the top {st.session_state.params["n_pcs"]} principal components.',
                            'plot': 'pca_variance.png'
                        })
                    plt.close(fig)
                    
                    # Compute neighborhood graph
                    st.markdown('<p class="step-message">Computing neighborhood graph...</p>', unsafe_allow_html=True)
                    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=st.session_state.params['n_pcs'])
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'Neighborhood Graph',
                        'description': f'Computed the neighborhood graph using {st.session_state.params["n_pcs"]} principal components and 10 nearest neighbors.',
                        'plot': None
                    })
                    
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
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'UMAP Visualization',
                            'description': 'Generated UMAP dimensionality reduction visualization of the dataset.',
                            'plot': 'umap.png'
                        })
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
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Clustering',
                            'description': f'Performed Leiden clustering with resolution {st.session_state.params["resolution"]} and visualized clusters on UMAP.',
                            'plot': 'umap_clusters.png'
                        })
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
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Differential Expression Analysis',
                            'description': 'Identified differentially expressed genes between clusters using the Wilcoxon rank-sum test.',
                            'plot': 'de.png'
                        })
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
                            st_context=st,
                            analysis_steps=st.session_state.analysis_steps
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating PDF report: {str(e)}")
                        st.session_state.preprocessing_done = False
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': f"I encountered an error while generating the PDF report: {str(e)}"
                        })
                    
                    # Mark preprocessing as done
                    st.session_state.preprocessing_done = True
                    
                    st.success("Analysis complete! I've generated visualizations and a PDF report. You can ask me questions about the results or request parameter adjustments.")
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
                    
                    # Add completion message to chat
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"Analysis complete! I've generated visualizations and a PDF report. You can ask me questions about the results or request parameter adjustments."
                    })
                    
                except Exception as e:
                    st.error(f"Error in analysis pipeline: {str(e)}")
                    st.session_state.preprocessing_done = False
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"I encountered an error during analysis: {str(e)}"
                    })
        
        # Display results if preprocessing is done
        if st.session_state.preprocessing_done:
            st.markdown("### Analysis Results")
            
            # Display analysis steps
            st.markdown("#### Analysis Pipeline")
            for i, step in enumerate(st.session_state.analysis_steps, 1):
                with st.expander(f"{i}. {step['step']}", expanded=True):
                    st.markdown(step['description'])
                    if step['plot'] and os.path.exists(os.path.join(st.session_state.output_dir, step['plot'])):
                        st.image(os.path.join(st.session_state.output_dir, step['plot']))
            
            # Display plots in a grid
            st.markdown("#### Visualization Gallery")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Quality Control")
                if os.path.exists(os.path.join(st.session_state.output_dir, 'qc_distributions.png')):
                    st.image(os.path.join(st.session_state.output_dir, 'qc_distributions.png'))
                
                st.markdown("#### Highly Variable Genes")
                if os.path.exists(os.path.join(st.session_state.output_dir, 'highly_variable_genes.png')):
                    st.image(os.path.join(st.session_state.output_dir, 'highly_variable_genes.png'))
            
            with col2:
                st.markdown("#### PCA")
                if os.path.exists(os.path.join(st.session_state.output_dir, 'pca_variance.png')):
                    st.image(os.path.join(st.session_state.output_dir, 'pca_variance.png'))
                
                st.markdown("#### UMAP")
                if os.path.exists(os.path.join(st.session_state.output_dir, 'umap.png')):
                    st.image(os.path.join(st.session_state.output_dir, 'umap.png'))
            
            # Full-width plots
            st.markdown("#### Clustering")
            if os.path.exists(os.path.join(st.session_state.output_dir, 'umap_clusters.png')):
                st.image(os.path.join(st.session_state.output_dir, 'umap_clusters.png'))
            
            st.markdown("#### Differential Expression")
            if os.path.exists(os.path.join(st.session_state.output_dir, 'de.png')):
                st.image(os.path.join(st.session_state.output_dir, 'de.png'))
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(os.path.join(st.session_state.output_dir, 'final_report.pdf')):
                    with open(os.path.join(st.session_state.output_dir, 'final_report.pdf'), 'rb') as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name="scagentic_report.pdf",
                            mime="application/pdf"
                        )
            
            with col2:
                if os.path.exists(os.path.join(st.session_state.output_dir, 'processed_data.h5ad')):
                    with open(os.path.join(st.session_state.output_dir, 'processed_data.h5ad'), 'rb') as f:
                        st.download_button(
                            label="📥 Download Processed Data",
                            data=f,
                            file_name="processed_data.h5ad",
                            mime="application/octet-stream"
                        )
        
        # Chat interface
        st.markdown("### Ask Questions About Your Data")
        user_question = st.text_input("Type your question here...", key="user_input")
        if user_question:
            # Add user message to chat
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
            
            with st.spinner("Processing your request..."):
                try:
                    # Process the question and generate response
                    response = process_question(user_question, adata)
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    # Check if the question was about parameter adjustments
                    if "parameter" in user_question.lower() or "redo" in user_question.lower():
                        # Reset preprocessing flag to allow re-running with new parameters
                        st.session_state.preprocessing_done = False
                        st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"I encountered an error while processing your question: {str(e)}"
                    })
            
            # Clear the input
            st.session_state.user_input = ""

if __name__ == "__main__":
    main() 