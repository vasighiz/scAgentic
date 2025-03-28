import streamlit as st
import os
from agent import ScRNAseqAgent
from functions import convert_10x_to_h5ad
import gzip

# Set page config
st.set_page_config(
    page_title="scRNA-seq Analysis Chat",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = ScRNAseqAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'converted_file_path' not in st.session_state:
    st.session_state.converted_file_path = None

# Sidebar
with st.sidebar:
    st.title("🧬 scRNA-seq Analysis")
    st.markdown("---")
    
    # File upload section
    st.subheader("Data Input")
    uploaded_file = st.file_uploader(
        "Upload h5ad file",
        type=['h5ad'],
        key='h5ad_upload'
    )
    
    # 10x Genomics format conversion
    st.subheader("Convert 10x Format")
    st.write("Upload all three 10x Genomics files at once:")
    uploaded_files = st.file_uploader(
        "Upload matrix.mtx(.gz), barcodes.tsv(.gz), and features.tsv(.gz)",
        type=['mtx', 'gz', 'tsv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Organize uploaded files
        mtx_file = None
        barcodes_file = None
        features_file = None
        
        for file in uploaded_files:
            if 'matrix' in file.name.lower():
                mtx_file = file
            elif 'barcodes' in file.name.lower():
                barcodes_file = file
            elif 'features' in file.name.lower():
                features_file = file
        
        if mtx_file and barcodes_file and features_file:
            if st.button("Convert to h5ad"):
                with st.spinner("Converting files..."):
                    try:
                        # Get the original folder name from the first file
                        original_folder = os.path.dirname(mtx_file.name)
                        if original_folder:
                            output_dir = os.path.join("data", original_folder)
                        else:
                            # If no folder, use the matrix file name without extension and "_matrix.mtx"
                            base_name = os.path.splitext(os.path.basename(mtx_file.name))[0]
                            if base_name.endswith('.gz'):
                                base_name = os.path.splitext(base_name)[0]
                            # Remove "_matrix.mtx" if present
                            base_name = base_name.replace('_matrix.mtx', '')
                            output_dir = os.path.join("data", base_name)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save uploaded files temporarily
                        temp_dir = os.path.join(output_dir, "temp_10x")
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        # Handle matrix.mtx file
                        matrix_path = os.path.join(temp_dir, "matrix.mtx.gz" if mtx_file.name.endswith('.gz') else "matrix.mtx")
                        if mtx_file.name.endswith('.gz'):
                            with open(matrix_path, 'wb') as f:
                                f.write(mtx_file.getvalue())
                        else:
                            with open(matrix_path, 'wb') as f:
                                f.write(mtx_file.getvalue())
                        
                        # Handle barcodes file
                        barcodes_path = os.path.join(temp_dir, "barcodes.tsv.gz" if barcodes_file.name.endswith('.gz') else "barcodes.tsv")
                        if barcodes_file.name.endswith('.gz'):
                            with open(barcodes_path, 'wb') as f:
                                f.write(barcodes_file.getvalue())
                        else:
                            with open(barcodes_path, 'wb') as f:
                                f.write(barcodes_file.getvalue())
                        
                        # Handle features file
                        features_path = os.path.join(temp_dir, "features.tsv.gz" if features_file.name.endswith('.gz') else "features.tsv")
                        if features_file.name.endswith('.gz'):
                            with open(features_path, 'wb') as f:
                                f.write(features_file.getvalue())
                        else:
                            with open(features_path, 'wb') as f:
                                f.write(features_file.getvalue())
                        
                        # Convert to h5ad with original folder name
                        output_path = os.path.join(output_dir, f"{os.path.basename(output_dir)}.h5ad")
                        convert_10x_to_h5ad(
                            matrix_path,
                            barcodes_path,
                            features_path,
                            output_path
                        )
                        st.session_state.converted_file_path = output_path
                        st.success(f"Conversion completed! Data has been saved to {output_path}")
                        
                        # Automatically load the converted data
                        st.session_state.agent.load_dataset(output_path)
                        
                    except Exception as e:
                        st.error(f"Error during conversion: {str(e)}")
        else:
            st.warning("Please upload all three required files: matrix.mtx(.gz), barcodes.tsv(.gz), and features.tsv(.gz)")
    
    # Preprocessing parameters
    st.subheader("Preprocessing Parameters")
    min_genes = st.number_input("Min genes per cell", value=200)
    min_cells = st.number_input("Min cells per gene", value=3)
    max_percent_mt = st.number_input("Max % mitochondrial", value=20.0)
    n_top_genes = st.number_input("Number of HVGs", value=2000)
    n_pcs = st.number_input("Number of PCs", value=50)
    n_neighbors = st.number_input("Number of neighbors", value=15)
    
    if st.button("Preprocess Data"):
        if st.session_state.agent.adata is None:
            if uploaded_file:
                st.session_state.agent.load_dataset(uploaded_file)
            elif st.session_state.converted_file_path:
                st.session_state.agent.load_dataset(st.session_state.converted_file_path)
            else:
                st.error("Please upload a dataset or convert 10x data first!")
                st.stop()
        
        st.session_state.agent.preprocess(
            min_genes=min_genes,
            min_cells=min_cells,
            max_percent_mt=max_percent_mt,
            n_top_genes=n_top_genes,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors
        )

# Main content
st.title("Single-cell RNA-seq Analysis Chat")

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
            if st.session_state.agent.adata is None:
                if uploaded_file:
                    st.session_state.agent.load_dataset(uploaded_file)
                elif st.session_state.converted_file_path:
                    st.session_state.agent.load_dataset(st.session_state.converted_file_path)
                else:
                    st.error("Please upload a dataset or convert 10x data first!")
                    st.stop()
            
            # Process different types of queries
            if "umap" in prompt.lower():
                # Extract color parameter if specified
                color = None
                if "color by" in prompt.lower():
                    color = prompt.lower().split("color by")[-1].strip()
                st.session_state.agent.generate_umap(color=color)
            
            elif "compare" in prompt.lower() and "populations" in prompt.lower():
                # Extract groupby and genes if specified
                groupby = st.session_state.agent.get_available_metadata()[0]  # Default to first metadata column
                genes = st.session_state.agent.get_available_genes()[:5]  # Default to first 5 genes
                st.session_state.agent.analyze_cell_populations(groupby=groupby, gene_list=genes)
            
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