# scAgentic: Genomics Data (Single-cell RNA-seq) Analysis Comprehensive Platform

A Streamlit-based application for interactive single-cell RNA-seq analysis with a chat-like interface. This tool allows users to analyze single-cell data, generate UMAP plots, compare cell populations, and explore gene expression patterns through natural language queries.

## Features

- Chat interface for natural language queries about single-cell data
- Support for both h5ad and 10x Genomics format data
- Interactive UMAP visualization
- Cell population comparison analysis
- Gene expression analysis
- Configurable preprocessing parameters
- Docker support for easy deployment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vasighiz/scAgentic.git
cd scAgentic
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your data:
   - Either upload an h5ad file directly
   - Or convert 10x Genomics format files (matrix.mtx, barcodes.tsv, features.tsv)

4. Configure preprocessing parameters in the sidebar

5. Start chatting! Try queries like:
   - "Show me a UMAP plot"
   - "Generate UMAP colored by cell type"
   - "Compare cell populations"
   - "Show expression of gene X"

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t scagentic .
```

2. Run the container:
```bash
docker run -p 8501:8501 scagentic
```

## Project Structure

```
scAgentic/
├── app.py              # Main Streamlit application
├── agent.py            # Chat agent implementation
├── functions.py        # Utility functions
├── requirements.txt    # Python dependencies
├── data/              # Directory for input data
├── outputs/           # Directory for analysis outputs
└── README.md          # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
