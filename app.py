import streamlit as st
import os
from agent import ScRNAseqAgent
from functions import (
    convert_10x_to_h5ad,
    analyze_qc_metrics,
    plot_qc_metrics,
    plot_pca_variance,
    generate_qc_report,
    preprocess_data,
    generate_consolidated_report,
    fetch_geo_metadata,
    plot_pca,
    plot_umap
)
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

# Set page config
st.set_page_config(
    page_title="Single-Cell RNA-seq Analysis",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stTooltip {
        background-color: #f0f2f6;
        border: 1px solid #e1e4e8;
        border-radius: 4px;
        padding: 8px;
        font-size: 14px;
        color: #24292e;
    }
    .stTooltip:hover {
        background-color: #e1e4e8;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Single-Cell RNA-seq Analysis")
st.markdown("""
    This application provides a comprehensive pipeline for analyzing single-cell RNA sequencing data.
    Upload your 10x Genomics data and follow the steps below to analyze and visualize your data.
    """)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'study_info' not in st.session_state:
    st.session_state.study_info = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Data Upload & Preprocessing", "Analysis", "Chat"])

with tab1:
    # File upload section
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader(
        "Upload your 10x Genomics data files",
        type=['mtx', 'h5', 'h5ad', 'gz', 'mtx.gz', 'tsv.gz', 'csv.gz'],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            try:
                # Reset preprocessing state when new files are uploaded
                st.session_state.preprocessed = False
                
                # Extract accession number and subfolder from the first file name
                file_name = uploaded_files[0].name
                # Try to detect GEO accession from folder name or file name
                accession = None
                if '_' in file_name:
                    accession = file_name.split('_')[0]
                elif '/' in file_name:
                    accession = file_name.split('/')[0]
                
                # Validate if it's a GEO accession (GSE, GSM, or GDS format)
                if accession and (accession.startswith(('GSE', 'GSM', 'GDS'))):
                    st.session_state.geo_accession = accession
                    st.session_state.geo_link = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
                    
                    # Fetch GEO metadata
                    with st.spinner("Fetching study metadata from GEO..."):
                        geo_metadata = fetch_geo_metadata(accession)
                        st.session_state.study_info.update(geo_metadata)
                        
                        # Display metadata in an info box
                        st.info("""
                            **Study Metadata from GEO:**
                            
                            **Title:** {title}
                            **Organism:** {organism}
                            **Tissue:** {tissue}
                            **Summary:** {summary}
                            """.format(**geo_metadata))
                else:
                    st.warning("Could not detect GEO accession number from file names.")
                    st.session_state.geo_accession = None
                    st.session_state.geo_link = None
                
                # Get the subfolder path from the file name
                subfolder = os.path.dirname(file_name)
                if not subfolder:  # If no subfolder in the name, use 'data' as default
                    subfolder = 'data'
                
                # Create the full path for the subfolder
                dataset_dir = os.path.join(os.getcwd(), subfolder)
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Convert to h5ad if needed
                if any(f.name.endswith(('.mtx', '.h5', '.gz', '.mtx.gz', '.tsv.gz', '.csv.gz')) for f in uploaded_files):
                    # Create a temporary directory for processing
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded files temporarily
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getvalue())
                            file_paths.append(file_path)
                        
                        # Find the matrix file
                        matrix_file = next(f for f in file_paths if f.endswith(('.mtx', '.mtx.gz')))
                        # Find the barcodes file
                        barcodes_file = next(f for f in file_paths if f.endswith(('barcodes.tsv', 'barcodes.tsv.gz')))
                        # Find the features file
                        features_file = next(f for f in file_paths if f.endswith(('features.tsv', 'features.tsv.gz')))
                        
                        # Create output path in the original subfolder
                        output_path = os.path.join(dataset_dir, f'{accession}.h5ad')
                        
                        # Convert to h5ad
                        convert_10x_to_h5ad(
                            mtx_path=matrix_file,
                            barcodes_path=barcodes_file,
                            features_path=features_file,
                            output_path=output_path
                        )
                        h5ad_path = output_path
                else:
                    # For h5ad files, save directly in the original subfolder
                    h5ad_path = os.path.join(dataset_dir, f'{accession}.h5ad')
                    with open(h5ad_path, 'wb') as f:
                        f.write(uploaded_files[0].getvalue())
                
                # Initialize agent with the h5ad file
                st.session_state.agent = ScRNAseqAgent(h5ad_path)
                
                # Display data summary
                st.success("Data loaded successfully!")
                st.write(f"Number of cells: {st.session_state.agent.adata.n_obs}")
                st.write(f"Number of genes: {st.session_state.agent.adata.n_vars}")
                
                # Analyze QC metrics
                qc_stats = analyze_qc_metrics(st.session_state.agent.adata)
                
                # Display QC plots
                st.subheader("Quality Control Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Genes per cell distribution")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(qc_stats['genes_per_cell'], bins=50, ax=ax)
                    ax.set_title('Genes per Cell Distribution')
                    ax.set_xlabel('Number of Genes')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                
                with col2:
                    st.write("Total counts distribution")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(qc_stats['total_counts'], bins=50, ax=ax)
                    ax.set_title('Total Counts Distribution')
                    ax.set_xlabel('Total Counts')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                
                if qc_stats['mito_percent'] is not None:
                    st.write("Mitochondrial percentage distribution")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(qc_stats['mito_percent'], bins=50, ax=ax)
                    ax.set_title('Mitochondrial Percentage Distribution')
                    ax.set_xlabel('Mitochondrial Percentage')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                
                # Store QC stats in session state
                st.session_state.qc_stats = qc_stats
                
                # Study information
                st.header("2. Study Information")
                
                # Display GEO link if available
                if hasattr(st.session_state, 'geo_link') and st.session_state.geo_link:
                    st.markdown(f"**GEO Study Link:** [{st.session_state.geo_accession}]({st.session_state.geo_link})")
                
                st.session_state.study_info['study_name'] = st.text_input(
                    "Study Name",
                    help="Enter a descriptive name for your study"
                )
                st.session_state.study_info['tissue'] = st.text_input(
                    "Tissue",
                    help="Specify the tissue type or cell type being analyzed"
                )
                st.session_state.study_info['species'] = st.text_input(
                    "Species",
                    help="Enter the species name (e.g., Homo sapiens, Mus musculus)"
                )
                st.session_state.study_info['purpose'] = st.text_area(
                    "Study Purpose",
                    help="Briefly describe the purpose and goals of your study"
                )
                
                # Preprocessing parameters with tooltips
                st.header("3. Preprocessing Parameters")
                st.markdown("""
                    The following parameters have been automatically calculated based on your data.
                    You can adjust them if needed.
                    """)
                
                col1, col2 = st.columns(2)
                with col1:
                    min_genes = st.number_input(
                        "Minimum genes per cell",
                        min_value=1,
                        max_value=qc_stats['max_genes'],
                        value=qc_stats['min_genes'],
                        help="Cells with fewer genes will be filtered out. Calculated as 5th percentile."
                    )
                    min_cells = st.number_input(
                        "Minimum cells per gene",
                        min_value=1,
                        max_value=qc_stats['n_cells'],
                        value=3,
                        help="Genes expressed in fewer cells will be filtered out."
                    )
                    max_percent_mt = st.number_input(
                        "Maximum percentage of mitochondrial genes",
                        min_value=0.0,
                        max_value=100.0,
                        value=qc_stats['max_mito'],
                        help="Cells with higher mitochondrial percentage will be filtered out."
                    )
                
                with col2:
                    n_top_genes = st.number_input(
                        "Number of highly variable genes",
                        min_value=100,
                        max_value=qc_stats['n_genes'],
                        value=2000,
                        help="Number of most variable genes to use for PCA."
                    )
                    n_pcs = st.number_input(
                        "Number of principal components",
                        min_value=1,
                        max_value=50,
                        value=50,
                        help="Number of PCs to compute for dimensionality reduction."
                    )
                    n_neighbors = st.number_input(
                        "Number of neighbors",
                        min_value=1,
                        max_value=50,
                        value=15,
                        help="Number of neighbors for computing the neighborhood graph."
                    )
                    resolution = st.number_input(
                        "Clustering resolution",
                        min_value=0.1,
                        max_value=2.0,
                        value=0.5,
                        help="Resolution for Leiden clustering. Higher values give more clusters."
                    )
                
                # Preprocess button
                if st.button("Run Preprocessing"):
                    with st.spinner("Preprocessing data..."):
                        try:
                            # Create output directory with GEO accession
                            output_dir = os.path.join(
                                os.path.dirname(h5ad_path),
                                f'analysis_results_{st.session_state.geo_accession}' if hasattr(st.session_state, 'geo_accession') and st.session_state.geo_accession else 'analysis_results'
                            )
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Preprocess data
                            st.session_state.agent.adata = preprocess_data(
                                st.session_state.agent.adata,
                                min_genes=min_genes,
                                min_cells=min_cells,
                                max_percent_mt=max_percent_mt,
                                n_top_genes=n_top_genes,
                                n_pcs=n_pcs,
                                n_neighbors=n_neighbors,
                                resolution=resolution
                            )
                            
                            # Analyze QC metrics
                            qc_stats = analyze_qc_metrics(st.session_state.agent.adata)
                            
                            # Generate consolidated report
                            pdf_path = generate_consolidated_report(
                                st.session_state.agent.adata,
                                output_dir,
                                {
                                    **st.session_state.study_info,
                                    'geo_link': st.session_state.geo_link if hasattr(st.session_state, 'geo_link') else '',
                                    'geo_accession': st.session_state.geo_accession if hasattr(st.session_state, 'geo_accession') else ''
                                },
                                {
                                    'min_genes': min_genes,
                                    'min_cells': min_cells,
                                    'max_percent_mt': max_percent_mt,
                                    'n_top_genes': n_top_genes,
                                    'n_pcs': n_pcs,
                                    'n_neighbors': n_neighbors,
                                    'resolution': resolution
                                },
                                qc_stats
                            )
                            
                            # Save preprocessed data
                            preprocessed_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(h5ad_path))[0]}_preprocessed.h5ad")
                            st.session_state.agent.adata.write(preprocessed_path)
                            
                            st.success("Preprocessing completed successfully!")
                            st.session_state.preprocessed = True
                            
                            # Add download button for the report
                            with open(pdf_path, "rb") as file:
                                st.download_button(
                                    label="Download Analysis Report",
                                    data=file,
                                    file_name="consolidated_report.pdf",
                                    mime="application/pdf"
                                )
                            
                        except Exception as e:
                            st.error(f"Error during preprocessing: {str(e)}")
                            st.session_state.preprocessed = False
            
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

with tab2:
    # Analysis section
    if st.session_state.agent is None:
        st.error("Please load data first!")
    else:
        # Check for preprocessed data file
        input_path = st.session_state.agent.adata.uns.get('input_path', '')
        if input_path:
            input_dir = os.path.dirname(input_path)
            accession = os.path.splitext(os.path.basename(input_path))[0]
            preprocessed_file = os.path.join(input_dir, f'{accession}_preprocessed.h5ad')
            
            if not os.path.exists(preprocessed_file):
                st.error("Please preprocess the data first!")
            else:
                # Load the preprocessed data
                try:
                    st.session_state.agent.adata = sc.read_h5ad(preprocessed_file)
                    st.session_state.agent.processed = True
                    st.session_state.preprocessed = True
                except Exception as e:
                    st.error(f"Error loading preprocessed data: {str(e)}")
                    st.stop()
                
                st.header("Analysis")
                
                # Create subtabs for different analysis types
                subtab1, subtab2, subtab3 = st.tabs(["UMAP Visualization", "Cell Population Analysis", "Gene Expression"])
                
                with subtab1:
                    # UMAP visualization
                    st.subheader("UMAP Visualization")
                    color_by = st.selectbox(
                        "Color by",
                        options=['leiden', 'cell_type'] + list(st.session_state.agent.adata.obs.columns),
                        help="Select a variable to color the UMAP plot"
                    )
                    
                    if st.button("Generate UMAP"):
                        with st.spinner("Generating UMAP plot..."):
                            try:
                                fig = plot_umap(
                                    st.session_state.agent.adata,
                                    color=color_by,
                                    title=f"UMAP colored by {color_by}"
                                )
                                st.pyplot(fig)
                                
                                # Save UMAP plot
                                output_dir = os.path.join(
                                    os.path.dirname(h5ad_path),
                                    f'analysis_results_{st.session_state.geo_accession}' if hasattr(st.session_state, 'geo_accession') and st.session_state.geo_accession else 'analysis_results'
                                )
                                fig.savefig(os.path.join(output_dir, 'umap.pdf'), bbox_inches='tight', dpi=300)
                                plt.close(fig)
                                
                            except Exception as e:
                                st.error(f"Error generating UMAP: {str(e)}")
                
                with subtab2:
                    # Cell population analysis
                    st.subheader("Cell Population Analysis")
                    
                    # Analysis parameters
                    n_genes = st.slider(
                        "Number of marker genes per cluster",
                        min_value=1,
                        max_value=50,
                        value=25,
                        help="Number of marker genes to identify for each cluster"
                    )
                    
                    method = st.selectbox(
                        "Statistical method",
                        options=['wilcoxon', 't-test', 'logreg'],
                        help="Statistical method for differential expression analysis"
                    )
                    
                    min_fold_change = st.slider(
                        "Minimum log fold change",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.25,
                        step=0.1,
                        help="Minimum log fold change for marker genes"
                    )
                    
                    pval_cutoff = st.slider(
                        "P-value cutoff",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.05,
                        step=0.01,
                        help="P-value cutoff for statistical significance"
                    )
                    
                    if st.button("Analyze Populations"):
                        with st.spinner("Analyzing cell populations..."):
                            try:
                                # Create output directory with GEO accession
                                output_dir = os.path.join(
                                    os.path.dirname(h5ad_path),
                                    f'analysis_results_{st.session_state.geo_accession}' if hasattr(st.session_state, 'geo_accession') and st.session_state.geo_accession else 'analysis_results'
                                )
                                os.makedirs(output_dir, exist_ok=True)
                                
                                # Perform differential expression analysis
                                sc.tl.rank_genes_groups(
                                    st.session_state.agent.adata,
                                    'leiden',
                                    method=method,
                                    n_genes=n_genes
                                )
                                
                                # Display results
                                fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
                                sc.pl.rank_genes_groups(
                                    st.session_state.agent.adata,
                                    n_genes=n_genes,
                                    sharey=False,
                                    show=False,
                                    ax=ax
                                )
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Save heatmap
                                fig.savefig(os.path.join(output_dir, 'marker_genes_heatmap.pdf'), bbox_inches='tight', dpi=300)
                                plt.close(fig)
                                
                                # Display detailed results
                                results = pd.DataFrame()
                                for group in st.session_state.agent.adata.obs['leiden'].unique():
                                    group_results = pd.DataFrame({
                                        'gene': st.session_state.agent.adata.uns['rank_genes_groups']['names'][group],
                                        'scores': st.session_state.agent.adata.uns['rank_genes_groups']['scores'][group],
                                        'pvals': st.session_state.agent.adata.uns['rank_genes_groups']['pvals'][group],
                                        'logfoldchanges': st.session_state.agent.adata.uns['rank_genes_groups']['logfoldchanges'][group],
                                        'group': group
                                    })
                                    results = pd.concat([results, group_results])
                                
                                # Filter results
                                results = results[
                                    (results['logfoldchanges'].abs() >= min_fold_change) &
                                    (results['pvals'] <= pval_cutoff)
                                ]
                                
                                # Save results to CSV
                                results.to_csv(os.path.join(output_dir, 'marker_genes.csv'), index=False)
                                
                                # Display results in the app
                                st.dataframe(results)
                                
                                # Generate violin plots for top genes
                                top_genes = results.groupby('group').head(5)['gene'].unique()
                                for gene in top_genes:
                                    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
                                    sc.pl.violin(
                                        st.session_state.agent.adata,
                                        gene,
                                        groupby='leiden',
                                        show=False,
                                        ax=ax
                                    )
                                    plt.tight_layout()
                                    fig.savefig(os.path.join(output_dir, f'violin_{gene}.pdf'), bbox_inches='tight', dpi=300)
                                    plt.close(fig)
                                
                            except Exception as e:
                                st.error(f"Error analyzing cell populations: {str(e)}")
                
                with subtab3:
                    # Gene expression visualization
                    st.subheader("Gene Expression Visualization")
                    gene = st.text_input(
                        "Enter gene name",
                        help="Enter a gene name to visualize its expression"
                    )
                    if gene:
                        st.session_state.agent.show_gene_expression(gene)

with tab3:
    # Chat interface
    st.title("Chat Interface")
    
    # Display example prompts
    st.info("""
        **Try asking questions like:**
        
        - "Show me the UMAP colored by clusters"
        - "What are the top differentially expressed genes in cluster 1?"
        - "Plot PDCD1 expression on UMAP"
        - "Summarize cell type distribution"
        - "Show me the PCA plot"
        - "What's the distribution of genes per cell?"
    """)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your single-cell data..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process the query
        with st.chat_message("assistant"):
            try:
                # Load data if not already loaded
                if st.session_state.agent is None:
                    st.error("Please upload a dataset first!")
                    st.stop()
                
                # List of available functions
                available_functions = [
                    'plot_umap',
                    'plot_pca',
                    'analyze_qc_metrics',
                    'plot_qc_metrics',
                    'plot_pca_variance'
                ]
                
                # Process query with LLM
                interpretation = process_query_with_llm(prompt, available_functions)
                
                if interpretation['confidence'] >= 0.5 and interpretation['function_name']:
                    # Execute the interpreted query
                    result = execute_interpreted_query(interpretation, st.session_state.agent.adata)
                    
                    if result is not None:
                        # Display the result
                        if isinstance(result, plt.Figure):
                            st.pyplot(result)
                        elif isinstance(result, pd.DataFrame):
                            st.dataframe(result)
                        else:
                            st.write(result)
                            
                        # Add success message to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "I've processed your request. Here's what I found."
                        })
                    else:
                        st.error("I couldn't execute that query. Please try rephrasing or use one of the example prompts above.")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "I couldn't execute that query. Please try rephrasing or use one of the example prompts above."
                        })
                else:
                    st.error("I couldn't match your question to a known function. Try rephrasing or use one of the example prompts above.")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "I couldn't match your question to a known function. Try rephrasing or use one of the example prompts above."
                    })
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}"
                })

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