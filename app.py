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
import shutil
import scipy.sparse as sp

# Configure scanpy settings
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

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

def process_question(question: str, adata: sc.AnnData) -> str:
    """
    Process a user question about the data and generate a response.
    
    Args:
        question: The user's question
        adata: The AnnData object containing the data
        
    Returns:
        A string response to the user's question
    """
    # Convert question to lowercase for easier matching
    question = question.lower()
    
    try:
        # Handle questions about data dimensions
        if any(x in question for x in ['how many cells', 'number of cells']):
            return f"The dataset contains {adata.n_obs:,} cells."
        
        if any(x in question for x in ['how many genes', 'number of genes']):
            return f"The dataset contains {adata.n_vars:,} genes."
            
        # Handle questions about clusters
        if 'cluster' in question and 'how many' in question:
            if 'leiden' in adata.obs.columns:
                n_clusters = len(adata.obs['leiden'].unique())
                return f"The analysis identified {n_clusters} clusters using the Leiden algorithm."
            else:
                return "Clustering has not been performed yet. Please run the preprocessing pipeline first."
        
        # Handle questions about specific genes
        if 'expression' in question:
            # Try to find a gene name in the question
            genes = [gene for gene in adata.var_names if gene.upper() in question.upper()]
            if genes:
                gene = genes[0]
                if gene in adata.var_names:
                    mean_expr = adata[:, gene].X.mean()
                    return f"The mean expression of {gene} across all cells is {mean_expr:.2f}."
                else:
                    return f"Gene {gene} was not found in the dataset."
            else:
                return "I couldn't identify which gene you're asking about. Please specify the gene name."
        
        # Handle questions about metadata
        if any(x in question for x in ['metadata', 'what information', 'what data']):
            metadata = list(adata.obs.columns)
            if metadata:
                return f"The dataset contains the following metadata: {', '.join(metadata)}"
            else:
                return "No metadata is available for this dataset."
        
        # Handle questions about highly variable genes
        if 'highly variable' in question:
            if 'highly_variable' in adata.var.columns:
                n_hvg = adata.var['highly_variable'].sum()
                return f"There are {n_hvg:,} highly variable genes in the dataset."
            else:
                return "Highly variable genes have not been computed yet. Please run the preprocessing pipeline first."
        
        # Handle questions about mitochondrial content
        if any(x in question for x in ['mitochondrial', 'mt content']):
            if 'pct_counts_mt' in adata.obs.columns:
                mean_mt = adata.obs['pct_counts_mt'].mean()
                return f"The mean mitochondrial content across all cells is {mean_mt:.2f}%."
            else:
                return "Mitochondrial content has not been calculated yet. Please run the preprocessing pipeline first."
        
        # Handle questions about rerunning analysis
        if any(x in question for x in ['rerun', 'redo', 'run again']):
            return "To rerun the analysis, I'll reset the preprocessing flag. You can adjust parameters through the chat interface."
        
        # Default response for unrecognized questions
        return ("I'm not sure how to answer that question. You can ask me about:\n"
                "- Number of cells or genes\n"
                "- Cluster information\n"
                "- Gene expression\n"
                "- Available metadata\n"
                "- Highly variable genes\n"
                "- Mitochondrial content\n"
                "- Rerunning the analysis")
                
    except Exception as e:
        return f"I encountered an error while processing your question: {str(e)}"

def main():
    st.set_page_config(
        page_title="scAgentic",
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
    st.markdown("AI-Powered Single-Cell RNA-seq Data Analysis")
    
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
                        features_file = next((f for f in uploaded_files if any(x in f.name.lower() for x in ['features.tsv', 'features.tsv.gz', 'genes.tsv', 'genes.tsv.gz'])), None)
                        barcodes_file = next((f for f in uploaded_files if any(x in f.name.lower() for x in ['barcodes.tsv', 'barcodes.tsv.gz'])), None)
                        
                        if not all([matrix_file, features_file, barcodes_file]):
                            missing_files = []
                            if not matrix_file: missing_files.append("matrix.mtx or matrix.mtx.gz")
                            if not features_file: missing_files.append("features.tsv or features.tsv.gz or genes.tsv or genes.tsv.gz")
                            if not barcodes_file: missing_files.append("barcodes.tsv or barcodes.tsv.gz")
                            raise ValueError(f"Missing required 10X files: {', '.join(missing_files)}")
                        
                        # Create temporary directory for 10X files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save uploaded files with proper handling of compressed and uncompressed files
                            matrix_path = os.path.join(temp_dir, "matrix.mtx")
                            features_path = os.path.join(temp_dir, "features.tsv")
                            barcodes_path = os.path.join(temp_dir, "barcodes.tsv")
                            
                            # Handle compressed and uncompressed files
                            def save_file(file_obj, target_path):
                                if file_obj.name.endswith('.gz'):
                                    # If file is already compressed, decompress it
                                    with gzip.open(file_obj, 'rb') as f_in:
                                        with open(target_path, 'wb') as f_out:
                                            f_out.write(f_in.read())
                                else:
                                    # If file is not compressed, save it as is
                                    with open(target_path, 'wb') as f:
                                        f.write(file_obj.getbuffer())
                            
                            # Save the files
                            save_file(matrix_file, matrix_path)
                            save_file(features_file, features_path)
                            save_file(barcodes_file, barcodes_path)
                            
                            # Create alternative filenames for different 10X formats
                            alt_matrix_path = os.path.join(temp_dir, "matrix.mtx.gz")
                            alt_features_path = os.path.join(temp_dir, "genes.tsv")
                            alt_barcodes_path = os.path.join(temp_dir, "barcodes.tsv.gz")
                            
                            # Create copies for alternative filenames
                            if os.path.exists(matrix_path):
                                shutil.copy2(matrix_path, alt_matrix_path)
                            if os.path.exists(features_path):
                                shutil.copy2(features_path, alt_features_path)
                            if os.path.exists(barcodes_path):
                                shutil.copy2(barcodes_path, alt_barcodes_path)
                            
                            # Verify files were saved correctly
                            if not all(os.path.exists(path) and os.path.getsize(path) > 0 for path in [matrix_path, features_path, barcodes_path]):
                                missing_files = []
                                if not os.path.exists(matrix_path) or os.path.getsize(matrix_path) == 0:
                                    missing_files.append("matrix.mtx")
                                if not os.path.exists(features_path) or os.path.getsize(features_path) == 0:
                                    missing_files.append("features.tsv")
                                if not os.path.exists(barcodes_path) or os.path.getsize(barcodes_path) == 0:
                                    missing_files.append("barcodes.tsv")
                                raise ValueError(f"Failed to save required 10X files: {', '.join(missing_files)}")
                            
                            # Load 10X data
                            try:
                                st.info(f"Loading 10X data from temporary directory: {temp_dir}")
                                st.info(f"Files being used: matrix.mtx, features.tsv, barcodes.tsv")
                                adata = sc.read_10x_mtx(temp_dir)
                                st.success(f"Successfully loaded data with {adata.n_obs} cells and {adata.n_vars} genes")
                            except Exception as e:
                                st.warning(f"First attempt to load 10X data failed: {str(e)}")
                                st.info("Trying alternative file naming convention...")
                                
                                # Try with alternative file naming convention
                                try:
                                    # Rename files to match alternative convention
                                    if os.path.exists(matrix_path):
                                        os.rename(matrix_path, os.path.join(temp_dir, "matrix.mtx.gz"))
                                    if os.path.exists(features_path):
                                        os.rename(features_path, os.path.join(temp_dir, "genes.tsv"))
                                    if os.path.exists(barcodes_path):
                                        os.rename(barcodes_path, os.path.join(temp_dir, "barcodes.tsv.gz"))
                                    
                                    # Try loading again
                                    adata = sc.read_10x_mtx(temp_dir)
                                    st.success(f"Successfully loaded data with {adata.n_obs} cells and {adata.n_vars} genes")
                                except Exception as e2:
                                    st.warning(f"Error loading 10X data with alternative naming: {str(e2)}")
                                    st.info("Trying direct file reading as a last resort...")
                                    
                                    # Try direct file reading as a last resort
                                    try:
                                        # Read matrix file
                                        import pandas as pd
                                        import scipy.sparse as sp
                                        
                                        # Read barcodes
                                        with open(barcodes_path, 'r') as f:
                                            barcodes = [line.strip() for line in f]
                                        
                                        # Read features
                                        features_df = pd.read_csv(features_path, sep='\t', header=None)
                                        if features_df.shape[1] >= 2:
                                            gene_names = features_df.iloc[:, 1].values
                                        else:
                                            gene_names = features_df.iloc[:, 0].values
                                        
                                        # Read matrix
                                        with open(matrix_path, 'r') as f:
                                            # Skip header lines
                                            for _ in range(3):
                                                f.readline()
                                            
                                            # Read matrix dimensions
                                            dims = f.readline().strip().split()
                                            n_rows, n_cols, n_entries = map(int, dims)
                                            
                                            # Read matrix entries
                                            rows, cols, data = [], [], []
                                            for line in f:
                                                row, col, val = map(int, line.strip().split())
                                                rows.append(row-1)  # Convert to 0-based indexing
                                                cols.append(col-1)
                                                data.append(val)
                                        
                                        # Create sparse matrix
                                        matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
                                        
                                        # Create AnnData object
                                        adata = sc.AnnData(matrix.T)
                                        adata.var_names = gene_names
                                        adata.obs_names = barcodes
                                        
                                        st.success(f"Successfully loaded data with {adata.n_obs} cells and {adata.n_vars} genes using direct file reading")
                                    except Exception as e3:
                                        st.error(f"Error loading 10X data with direct file reading: {str(e3)}")
                                        # Log the contents of the temporary directory for debugging
                                        st.error("Contents of temporary directory:")
                                        for file in os.listdir(temp_dir):
                                            file_path = os.path.join(temp_dir, file)
                                            file_size = os.path.getsize(file_path)
                                            st.error(f"- {file}: {file_size} bytes")
                                        raise ValueError(f"Failed to load 10X data: {str(e3)}")
                        
                        # Extract study info from the first file name
                        geo_id = matrix_file.name.split('_')[0] if '_' in matrix_file.name else "dataset"
                        
                    else:
                        st.markdown("""
                            <div class="upload-warning">
                                <p><strong>Invalid file combination:</strong></p>
                                <p>Please upload either:</p>
                                <ul>
                                    <li>A single .h5ad file, or</li>
                                    <li>Three 10X Genomics files (matrix.mtx/matrix.mtx.gz, features.tsv/features.tsv.gz/genes.tsv/genes.tsv.gz, barcodes.tsv/barcodes.tsv.gz)</li>
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
                    
                    # Plot highest expressed genes (FIRST STEP)
                    st.markdown('<p class="step-message">Plotting highest expressed genes...</p>', unsafe_allow_html=True)
                    sc.pl.highest_expr_genes(adata, n_top=20, show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'highest_expr_genes.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['highest_expr_genes'] = fig
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Highest Expressed Genes',
                            'description': 'Shows the genes that yield the highest fraction of counts in each single cell, across all cells. This plot helps identify potential technical artifacts or highly expressed genes that might dominate the dataset.',
                            'plot': 'highest_expr_genes.png'
                        })
                    plt.close(fig)
                    
                    # Basic filtering
                    st.markdown('<p class="step-message">Performing basic filtering...</p>', unsafe_allow_html=True)
                    
                    # Store original cell and gene counts
                    original_cells = adata.n_obs
                    original_genes = adata.n_vars
                    
                    # Filter cells with at least 200 genes
                    sc.pp.filter_cells(adata, min_genes=200)
                    cells_after_min_genes = adata.n_obs
                    
                    # Filter genes expressed in at least 3 cells
                    sc.pp.filter_genes(adata, min_cells=3)
                    genes_after_min_cells = adata.n_vars
                    
                    # Basic filtering message
                    filtering_message = f"Filtered out {original_genes - genes_after_min_cells} genes that are detected in less than 3 cells"
                    
                    # Add step to analysis steps (for PDF report)
                    st.session_state.analysis_steps.append({
                        'step': 'Basic filtering',
                        'description': filtering_message,
                        'plot': None
                    })
                    
                    # Display filtering results (for screen)
                    st.markdown("### Basic Filtering Results")
                    st.markdown(f"**{filtering_message}**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Original cells", original_cells)
                    with col2:
                        st.metric("Cells after filtering", cells_after_min_genes)
                    with col3:
                        st.metric("Original genes", original_genes)
                    with col4:
                        st.metric("Genes after filtering", genes_after_min_cells)
                    
                    # Calculate QC metrics
                    st.markdown('<p class="step-message">Calculating QC metrics...</p>', unsafe_allow_html=True)
                    adata.var['mt'] = adata.var_names.str.startswith('MT-')
                    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'QC Metrics Calculation',
                        'description': 'Calculated quality control metrics including:\n- Number of genes expressed in each cell\n- Total counts per cell\n- Percentage of counts in mitochondrial genes',
                        'plot': None
                    })
                    
                    # Create QC violin plots
                    st.markdown('<p class="step-message">Generating QC violin plots...</p>', unsafe_allow_html=True)
                    sc.pl.violin(
                        adata,
                        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                        jitter=0.4,
                        multi_panel=True,
                        show=False
                    )
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'qc_distributions.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['qc_distributions'] = fig
                        
                        # Display the violin plots on the screen
                        st.pyplot(fig)
                        
                        # Add explanation about filtering thresholds
                        st.markdown("### Filtering Thresholds")
                        st.markdown("Based on these plots, we choose the following thresholds for filtering:")
                        st.markdown("- Maximum number of genes per cell: 2500")
                        st.markdown("- Maximum percentage of mitochondrial genes: 5%")
                        
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'QC Violin Plots',
                            'description': 'Generated violin plots showing the distribution of QC metrics:\n- Number of genes expressed in each cell\n- Total counts per cell\n- Percentage of counts in mitochondrial genes\n\nThese plots help identify potential filtering thresholds for removing low-quality cells.',
                            'plot': 'qc_distributions.png'
                        })
                    plt.close(fig)
                    
                    # Additional filtering based on violin plots
                    st.markdown('<p class="step-message">Performing additional filtering based on QC metrics...</p>', unsafe_allow_html=True)
                    
                    # Store cell count before additional filtering
                    cells_before_additional_filtering = adata.n_obs
                    
                    # Filter cells based on QC metrics
                    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
                    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
                    
                    # Calculate cells removed
                    cells_removed = cells_before_additional_filtering - adata.n_obs
                    
                    # Display filtering results with the requested message
                    st.markdown(f"**Removed {cells_removed} cells that have too many mitochondrial genes expressed or too many total counts. You can later ask via chat to set another thresholds for filtering.**")
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'Additional QC Filtering',
                        'description': f'Performed additional filtering based on QC metrics:\n- Removed cells with more than 2500 genes\n- Removed cells with more than 5% mitochondrial genes\n\nTotal cells removed: {cells_removed}',
                        'plot': None
                    })
                    
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
                            'description': 'Calculated quality control metrics and generated distribution plots for genes, counts, and mitochondrial content. The violin plots show: (1) The number of genes expressed in the count matrix, (2) The total counts per cell, and (3) The percentage of counts in mitochondrial genes. It is useful to consider QC metrics jointly by inspecting a scatter plot colored by pct_counts_mt.',
                            'plot': 'qc_distributions.png'
                        })
                    plt.close(fig)
                    
                    # Create QC scatter plots
                    st.markdown('<p class="step-message">Analyzing QC metrics...</p>', unsafe_allow_html=True)
                    
                    # Create scatter plots for threshold selection
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt", show=False)
                        fig1 = plt.gcf()
                        if fig1.get_axes():
                            fig1.savefig(os.path.join(st.session_state.output_dir, 'qc_scatter_mt.png'),
                                      dpi=300, bbox_inches='tight')
                            st.session_state.figures['qc_scatter_mt'] = fig1
                            st.pyplot(fig1)
                        plt.close(fig1)
                    
                    with col2:
                        sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", show=False)
                        fig2 = plt.gcf()
                        if fig2.get_axes():
                            fig2.savefig(os.path.join(st.session_state.output_dir, 'qc_scatter_genes.png'),
                                      dpi=300, bbox_inches='tight')
                            st.session_state.figures['qc_scatter_genes'] = fig2
                            st.pyplot(fig2)
                        plt.close(fig2)
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'QC Scatter Plots',
                        'description': 'Scatter plots showing the relationship between total counts and (1) percentage of mitochondrial genes, (2) number of genes per cell. These plots help identify potential filtering thresholds for removing low-quality cells.',
                        'plot': 'qc_scatter_mt.png, qc_scatter_genes.png'
                    })
                    
                    # Calculate thresholds based on data distribution
                    suggested_genes = int(np.percentile(adata.obs.n_genes_by_counts, 95))
                    suggested_mt = float(np.percentile(adata.obs.pct_counts_mt, 95))
                    
                    # Apply filtering with calculated thresholds
                    filtered_adata = adata[adata.obs.n_genes_by_counts < suggested_genes, :].copy()
                    filtered_adata = filtered_adata[filtered_adata.obs.pct_counts_mt < suggested_mt, :].copy()
                    
                    # Show filtering results
                    st.markdown("### Filtering Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Original cells", adata.n_obs)
                    with col2:
                        st.metric("Filtered cells", filtered_adata.n_obs)
                    with col3:
                        st.metric("Cells removed", adata.n_obs - filtered_adata.n_obs)
                    
                    # Update adata with filtered data
                    adata = filtered_adata
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'Cell Filtering',
                        'description': f'Filtered out cells based on QC metrics:\n- Removed cells with more than {suggested_genes} genes (95th percentile)\n- Removed cells with more than {suggested_mt:.1f}% mitochondrial genes (95th percentile)\n\nTotal cells removed: {adata.n_obs - filtered_adata.n_obs}',
                        'plot': None
                    })
                    
                    # Doublet detection
                    st.markdown('<p class="step-message">Running doublet detection...</p>', unsafe_allow_html=True)
                    
                    # Check if 'sample' column exists in adata.obs
                    batch_key = "sample" if "sample" in adata.obs.columns else None
                    
                    # Run scrublet for doublet detection
                    try:
                        sc.pp.scrublet(adata, batch_key=batch_key)
                        
                        # Filter out predicted doublets
                        adata = adata[~adata.obs['predicted_doublet'], :]
                        
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Doublet Detection',
                            'description': 'As a next step, we run a doublet detection algorithm. Identifying doublets is crucial as they can lead to misclassifications or distortions in downstream analysis steps. Scanpy contains the doublet detection method Scrublet [Wolock2019]. Scrublet predicts cell doublets using a nearest-neighbor classifier of observed transcriptomes and simulated doublets. One can either filter directly on predicted_doublet or use the doublet_score later during clustering to filter clusters with high doublet scores. In this step, we filter directly on predicted_doublet. You can change it later by interacting in the chat.',
                            'plot': None
                        })
                    except Exception as e:
                        error_message = str(e)
                        if "No module named 'skimage'" in error_message:
                            st.warning("""
                                **Doublet Detection Failed: Missing Dependency**
                                
                                The doublet detection step requires the scikit-image package which is not installed.
                                
                                To install the missing dependency, run:
                                ```
                                pip install scikit-image
                                ```
                                
                                After installing, restart the application and the doublet detection will work properly.
                                
                                Continuing with the analysis pipeline without doublet detection.
                            """)
                            # Add step to analysis steps with error information
                            st.session_state.analysis_steps.append({
                                'step': 'Doublet Detection',
                                'description': '**Doublet Detection Failed: Missing Dependency**\n\nThe doublet detection step requires the scikit-image package which is not installed. To install the missing dependency, run: `pip install scikit-image`. After installing, restart the application and the doublet detection will work properly. The analysis pipeline continued without doublet detection.',
                                'plot': None
                            })
                        else:
                            st.warning(f"Doublet detection failed: {error_message}. Continuing with the analysis pipeline.")
                            # Add step to analysis steps with error information
                            st.session_state.analysis_steps.append({
                                'step': 'Doublet Detection',
                                'description': f'**Doublet Detection Failed**\n\nThe doublet detection step encountered an error: {error_message}. The analysis pipeline continued without doublet detection.',
                                'plot': None
                            })
                    
                    # Normalize and scale
                    st.markdown('<p class="step-message">Normalizing and scaling data...</p>', unsafe_allow_html=True)
                    
                    # Save count data
                    adata.layers["counts"] = adata.X.copy()
                    
                    # Total-count normalize (library-size correct) the data matrix to 10,000 reads per cell
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    st.markdown("**Total-count normalized to 10,000 reads per cell**")
                    
                    # Logarithmize the data
                    sc.pp.log1p(adata)
                    st.markdown("**Log-transformed the data**")
                    
                    # Scale the data
                    sc.pp.scale(adata, max_value=10)
                    st.markdown("**Scaled the data to have zero mean and unit variance**")
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'Normalization and Scaling',
                        'description': 'Total-count normalized the data matrix to 10,000 reads per cell, so that counts become comparable among cells. Then logarithmized the data with log1p transformation and scaled the data to have zero mean and unit variance.',
                        'plot': None
                    })
                    
                    # Find HVGs
                    st.markdown('<p class="step-message">Finding highly variable genes...</p>', unsafe_allow_html=True)
                    
                    # Check if 'sample' column exists in adata.obs
                    batch_key = "sample" if "sample" in adata.obs.columns else None
                    
                    # Run highly variable genes selection with specified parameters
                    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key=batch_key)
                    st.markdown("**Identified highly variable genes with parameters: min_mean=0.0125, max_mean=3, min_disp=0.5**")
                    
                    # Plot HVGs
                    sc.pl.highly_variable_genes(adata, show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'highly_variable_genes.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['highly_variable_genes'] = fig
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Feature Selection',
                            'description': 'Identified highly variable genes with parameters: min_mean=0.0125, max_mean=3, min_disp=0.5. These genes are used for downstream analysis to reduce dimensionality and focus on the most informative features.',
                            'plot': 'highly_variable_genes.png'
                        })
                    plt.close(fig)
                    
                    # Set the .raw attribute of the AnnData object
                    st.markdown('<p class="step-message">Setting raw attribute...</p>', unsafe_allow_html=True)
                    adata.raw = adata.copy()
                    st.markdown("**Set the .raw attribute to the normalized and logarithmized raw gene expression for later use in differential testing and visualizations**")
                    
                    # Filter to only include highly variable genes
                    st.markdown('<p class="step-message">Filtering to highly variable genes...</p>', unsafe_allow_html=True)
                    adata = adata[:, adata.var.highly_variable]
                    st.markdown(f"**Filtered to {adata.n_vars} highly variable genes**")
                    
                    # Regress out effects of total counts and mitochondrial genes
                    st.markdown('<p class="step-message">Regressing out effects of total counts and mitochondrial genes...</p>', unsafe_allow_html=True)
                    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
                    st.markdown("**Regressed out effects of total counts per cell and the percentage of mitochondrial genes expressed**")
                    
                    # Scale the data
                    st.markdown('<p class="step-message">Scaling data to unit variance...</p>', unsafe_allow_html=True)
                    sc.pp.scale(adata, max_value=10)
                    st.markdown("**Scaled each gene to unit variance and clipped values exceeding standard deviation 10**")
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'Data Preprocessing',
                        'description': 'Filtered to highly variable genes, regressed out effects of total counts and mitochondrial genes, and scaled the data to unit variance. This preprocessing step helps reduce technical noise and focuses the analysis on the most informative features.',
                        'plot': None
                    })
                    
                    # Run PCA
                    st.markdown('<p class="step-message">Running PCA...</p>', unsafe_allow_html=True)
                    sc.tl.pca(adata, svd_solver="arpack")
                    st.markdown("**Reduced the dimensionality of the data by running principal component analysis (PCA), which reveals the main axes of variation and denoises the data**")
                    
                    # Plot PCA variance ratio
                    sc.pl.pca_variance_ratio(adata, n_pcs=st.session_state.params['n_pcs'], log=True, show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'pca_variance.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['pca_variance'] = fig
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Dimensionality Reduction',
                            'description': 'Reduced the dimensionality of the data by running principal component analysis (PCA), which reveals the main axes of variation and denoises the data. We inspect the contribution of single PCs to the total variance in the data. This gives us information about how many PCs we should consider in order to compute the neighborhood relations of cells, e.g. used in the clustering function Leiden or tSNE. In experience, there does not seem to be signifigant downside to overestimating the numer of principal components.',
                            'plot': 'pca_variance.png'
                        })
                    plt.close(fig)
                    
                    # Plot PCA with sample and QC metrics
                    if 'sample' in adata.obs.columns:
                        # Set a more vibrant color palette for better visibility
                        sc.settings.palette = 'viridis'  # Using viridis palette which is more vibrant and colorblind-friendly
                        sc.pl.pca(
                            adata,
                            color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
                            dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
                            ncols=2,
                            size=8,
                            show=False
                        )
                        fig = plt.gcf()
                        if fig.get_axes():
                            fig.savefig(os.path.join(st.session_state.output_dir, 'pca_metrics.png'),
                                      dpi=300, bbox_inches='tight')
                            st.session_state.figures['pca_metrics'] = fig
                            # Update the analysis step with additional information
                            st.session_state.analysis_steps[-1]['description'] += ' We can also plot the principal components to see if there are any potentially undesired features (e.g. batch, QC metrics) driving signifigant variation in this dataset. In the case there isn\'t anything too alarming, but it\'s still a good idea to explore this.'
                            st.session_state.analysis_steps[-1]['plot'] = 'pca_metrics.png'
                        plt.close(fig)
                    else:
                        # If 'sample' column doesn't exist, create a simplified PCA plot with just QC metrics
                        # Set a more vibrant color palette for better visibility
                        sc.settings.palette = 'viridis'  # Using viridis palette which is more vibrant and colorblind-friendly
                        sc.pl.pca(
                            adata,
                            color=["pct_counts_mt", "pct_counts_mt"],
                            dimensions=[(0, 1), (2, 3)],
                            ncols=2,
                            size=8,
                            show=False
                        )
                        fig = plt.gcf()
                        if fig.get_axes():
                            fig.savefig(os.path.join(st.session_state.output_dir, 'pca_metrics.png'),
                                      dpi=300, bbox_inches='tight')
                            st.session_state.figures['pca_metrics'] = fig
                            # Update the analysis step with additional information
                            st.session_state.analysis_steps[-1]['description'] += ' We can also plot the principal components to see if there are any potentially undesired features (e.g. QC metrics) driving signifigant variation in this dataset. In the case there isn\'t anything too alarming, but it\'s still a good idea to explore this.'
                            st.session_state.analysis_steps[-1]['plot'] = 'pca_metrics.png'
                        plt.close(fig)
                    
                    # Compute neighborhood graph
                    st.markdown('<p class="step-message">Computing neighborhood graph...</p>', unsafe_allow_html=True)
                    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=st.session_state.params['n_pcs'])
                    
                    # Add step to analysis steps
                    st.session_state.analysis_steps.append({
                        'step': 'Nearest Neighbor Graph Construction',
                        'description': 'We compute the neighborhood graph of cells using the PCA representation of the data. This graph can then be embedded in two dimensions for visualization with UMAP (McInnes et al., 2018). If you inspect batch effects in your UMAP it can be beneficial to integrate across samples and perform batch correction/integration. We use scanorama and scvi-tools for batch integration.',
                        'plot': None
                    })
                    
                    # Run UMAP
                    st.markdown('<p class="step-message">Running UMAP...</p>', unsafe_allow_html=True)
                    sc.tl.umap(adata)
                    
                    # Plot UMAP with sample information if available
                    if 'sample' in adata.obs.columns:
                        sc.pl.umap(
                            adata,
                            color="sample",
                            size=2,  # Setting a smaller point size to prevent overlap
                            show=False
                        )
                        fig = plt.gcf()
                        if fig.get_axes():
                            fig.savefig(os.path.join(st.session_state.output_dir, 'umap_sample.png'),
                                      dpi=300, bbox_inches='tight')
                            st.session_state.figures['umap_sample'] = fig
                            # Add UMAP visualization step
                            st.session_state.analysis_steps.append({
                                'step': 'UMAP Visualization',
                                'description': 'UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that provides a non-linear embedding of the data. We visualize the UMAP embedding colored by sample to identify potential batch effects. If batch effects are observed, batch correction methods like scanorama or scvi-tools can be applied.',
                                'plot': 'umap_sample.png'
                            })
                        plt.close(fig)
                    else:
                        # If 'sample' column doesn't exist, skip UMAP visualization and move directly to clustering
                        st.info("No 'sample' column found in the dataset. Skipping UMAP visualization and proceeding to clustering.")
                    
                    # Run clustering
                    st.markdown('<p class="step-message">Running clustering...</p>', unsafe_allow_html=True)
                    # Using the igraph implementation and a fixed number of iterations for faster clustering
                    sc.tl.leiden(adata, resolution=st.session_state.params['resolution'], flavor="igraph", n_iterations=2)
                    
                    # Plot UMAP with clusters
                    sc.pl.umap(adata, color=['leiden'], show=False)
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'umap_clusters.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['umap_clusters'] = fig
                        # Add step to analysis steps with the requested description
                        st.session_state.analysis_steps.append({
                            'step': 'Clustering',
                            'description': 'As with Seurat and many other frameworks, we recommend the Leiden graph-clustering method (community detection based on optimizing modularity) [Traag2019]. Note that Leiden clustering directly clusters the neighborhood graph of cells, which we already computed in the previous section.',
                            'plot': 'umap_clusters.png'
                        })
                    plt.close(fig)
                    
                    # Re-assess quality control and cell filtering
                    st.markdown('<p class="step-message">Re-assessing quality control and cell filtering...</p>', unsafe_allow_html=True)
                    
                    # Create log-transformed QC metrics if they don't exist
                    if 'log1p_total_counts' not in adata.obs.columns:
                        adata.obs['log1p_total_counts'] = np.log1p(adata.obs['total_counts'])
                    if 'log1p_n_genes_by_counts' not in adata.obs.columns:
                        adata.obs['log1p_n_genes_by_counts'] = np.log1p(adata.obs['n_genes_by_counts'])
                    
                    # Plot UMAP with QC metrics
                    sc.pl.umap(
                        adata,
                        color=["leiden", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
                        wspace=0.5,
                        ncols=2,
                        show=False
                    )
                    fig = plt.gcf()
                    if fig.get_axes():
                        fig.savefig(os.path.join(st.session_state.output_dir, 'umap_qc.png'),
                                  dpi=300, bbox_inches='tight')
                        st.session_state.figures['umap_qc'] = fig
                        # Add step to analysis steps
                        st.session_state.analysis_steps.append({
                            'step': 'Re-assess Quality Control and Cell Filtering',
                            'description': 'As indicated before, we will now re-assess our filtering strategy by visualizing different QC metrics using UMAP. Additionally, we visualize the distribution of predicted doublets and doublet scores across clusters to assess the quality of our doublet detection. This helps identify if certain clusters are enriched for doublets, which might indicate technical artifacts rather than biological cell types.',
                            'plot': 'umap_qc.png'
                        })
                    plt.close(fig)
                    
                    # Add UMAP plot with doublet information if available
                    if 'predicted_doublet' in adata.obs.columns and 'doublet_score' in adata.obs.columns:
                        sc.pl.umap(
                            adata,
                            color=["leiden", "predicted_doublet", "doublet_score"],
                            # increase horizontal space between panels
                            wspace=0.5,
                            ncols=2,
                            show=False
                        )
                        fig = plt.gcf()
                        if fig.get_axes():
                            fig.savefig(os.path.join(st.session_state.output_dir, 'umap_doublets_qc.png'),
                                      dpi=300, bbox_inches='tight')
                            st.session_state.figures['umap_doublets_qc'] = fig
                            # Update the analysis step to include the new plot
                            st.session_state.analysis_steps[-1]['plot'] = 'umap_qc.png, umap_doublets_qc.png'
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
                            'qc_scatter_mt.png',
                            'qc_scatter_genes.png',
                            'highly_variable_genes.png',
                            'pca_variance.png',
                            'umap_clusters.png',
                            'de.png'
                        ]
                        
                        # Add optional plots that may not exist
                        optional_plots = ['pca_metrics.png', 'umap_sample.png']
                        
                        # Check for missing required plots
                        missing_plots = []
                        for plot in required_plots:
                            plot_path = os.path.join(st.session_state.output_dir, plot)
                            if not os.path.exists(plot_path):
                                missing_plots.append(plot)
                        
                        if missing_plots:
                            raise FileNotFoundError(f"Missing required plots: {', '.join(missing_plots)}")
                        
                        # Check for optional plots and add them if they exist
                        available_plots = required_plots.copy()
                        for plot in optional_plots:
                            plot_path = os.path.join(st.session_state.output_dir, plot)
                            if os.path.exists(plot_path):
                                available_plots.append(plot)
                            # Don't show warning for umap_sample.png as it's expected to be missing for datasets without sample column
                            elif plot != 'umap_sample.png':
                                st.warning(f"Optional plot '{plot}' not found. This plot will not be included in the PDF report.")
                        
                        # Generate PDF report with proper error handling
                        pdf_path = generate_pdf_report_safe(
                            output_dir=st.session_state.output_dir,
                            study_info=st.session_state.study_info,
                            parameters=st.session_state.params,
                            plot_files=available_plots,
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
                                <li>Doublets removed: {:,}</li>
                                <li>Number of clusters: {}</li>
                            </ul>
                            <p>You can now adjust parameters and re-run specific steps through the chat interface.</p>
                        </div>
                    """.format(adata.n_obs, adata.n_vars, 
                               int(adata.obs['predicted_doublet'].sum()) if 'predicted_doublet' in adata.obs.columns else 0, 
                               len(adata.obs['leiden'].unique())), 
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
            # Remove the "Analysis Results" section from UI
            # Keep only the download buttons
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