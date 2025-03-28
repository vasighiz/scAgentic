import streamlit as st
import os
from agent import ScRNAseqAgent
from functions import (
    convert_10x_to_h5ad,
    analyze_qc_metrics,
    plot_qc_metrics,
    plot_pca_variance,
    generate_qc_report
)
import gzip
import scanpy as sc
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

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
                accession = file_name.split('_')[0]  # Assuming accession is the first part of the filename
                
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
                            # Create output directory
                            output_dir = os.path.join(os.path.dirname(h5ad_path), 'analysis_results')
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Run preprocessing
                            success = st.session_state.agent.preprocess(
                                min_genes=min_genes,
                                min_cells=min_cells,
                                max_percent_mt=max_percent_mt,
                                n_top_genes=n_top_genes,
                                n_pcs=n_pcs,
                                n_neighbors=n_neighbors,
                                resolution=resolution
                            )
                            
                            if success:
                                # Generate PCA variance plot
                                plot_pca_variance(st.session_state.agent.adata, output_dir)
                                
                                # Generate QC report
                                generate_qc_report(
                                    st.session_state.agent.adata,
                                    qc_stats,
                                    output_dir,
                                    st.session_state.study_info,
                                    {
                                        'min_genes': min_genes,
                                        'min_cells': min_cells,
                                        'max_percent_mt': max_percent_mt,
                                        'n_top_genes': n_top_genes,
                                        'n_pcs': n_pcs,
                                        'n_neighbors': n_neighbors,
                                        'resolution': resolution
                                    }
                                )
                                
                                st.success(f"Preprocessing completed! Results saved to {output_dir}")
                                st.session_state.preprocessed = True  # Set preprocessing state in session
                                st.session_state.agent.processed = True  # Set preprocessing state in agent
                            else:
                                st.error("Preprocessing failed. Please check the parameters and try again.")
                            
                        except Exception as e:
                            st.error(f"Error during preprocessing: {str(e)}")
            
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
                        options=['leiden'] + list(st.session_state.agent.adata.obs.columns),
                        help="Select a variable to color the UMAP plot by"
                    )
                    st.session_state.agent.generate_umap(color=color_by)
                
                with subtab2:
                    # Cell population analysis
                    st.subheader("Cell Population Analysis")
                    
                    # Analysis parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        n_genes = st.number_input(
                            "Number of marker genes per cluster",
                            min_value=1,
                            max_value=50,
                            value=25,
                            help="Number of marker genes to show per cluster"
                        )
                        method = st.selectbox(
                            "Statistical method",
                            options=['wilcoxon', 't-test', 'logreg'],
                            help="Statistical method for differential expression analysis"
                        )
                    
                    with col2:
                        min_fold_change = st.number_input(
                            "Minimum log fold change",
                            min_value=0.0,
                            max_value=5.0,
                            value=0.25,
                            help="Minimum log fold change for marker genes"
                        )
                        pval_cutoff = st.number_input(
                            "P-value cutoff",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.05,
                            help="P-value cutoff for marker genes"
                        )
                    
                    if st.button("Analyze Populations"):
                        with st.spinner("Analyzing cell populations..."):
                            st.session_state.agent.analyze_cell_populations(
                                n_genes=n_genes,
                                method=method,
                                min_fold_change=min_fold_change,
                                pval_cutoff=pval_cutoff
                            )
                
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
                
                # Process different types of queries
                if "umap" in prompt.lower():
                    # Extract color parameter if specified
                    color = None
                    if "color by" in prompt.lower():
                        color = prompt.lower().split("color by")[-1].strip()
                    st.session_state.agent.generate_umap(color=color)
                
                elif "compare" in prompt.lower() and "populations" in prompt.lower():
                    st.session_state.agent.analyze_cell_populations()
                
                elif "expression" in prompt.lower():
                    # Extract gene name if specified
                    gene = st.session_state.agent.get_available_genes()[0]  # Default to first gene
                    st.session_state.agent.show_gene_expression(gene=gene)
                
                else:
                    st.write("I can help you with:")
                    st.write("- Generating UMAP plots (try asking about UMAP)")
                    st.write("- Comparing cell populations (try asking about population comparison)")
                    st.write("- Analyzing gene expression (try asking about gene expression)")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "I've processed your request. Let me know if you need anything else!"
            }) 