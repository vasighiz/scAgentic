# scAgentic

🚧 This project is under active development (Day 1 of 90). I’m documenting my daily progress here as I build scAgentic — an AI copilot for single-cell data analysis.

Day 1- The app currently uses a local LLM (Mistral via Ollama) for processing natural language queries about single-cell data, with functions like process_query_with_llm() that interpret user questions and map them to specific analysis functions, but this feature is not yet fully implemented as indicated by the ☐ AI Copilot status in the README.

## Project Description
scAgentic is an AI-powered single-cell RNA-seq data analysis platform that combines automated preprocessing pipelines with an intelligent copilot interface, enabling researchers to perform comprehensive single-cell analysis through natural language interactions.

## Features
- ✅ **Preprocessing Pipeline**
  - Quality control and filtering
  - Normalization and scaling
  - Highly variable gene selection
  - Dimensionality reduction (PCA)
  - Clustering (Leiden)
  - Differential expression analysis
  - PDF report generation

- ☐ **Literature Integration**
  - Gene set enrichment analysis
  - Pathway analysis
  - Literature-based cell type annotation

- ☐ **AI Copilot**
  - Natural language query processing
  - Interactive parameter adjustment
  - Automated analysis suggestions
  - Context-aware responses

- ☐ **Visualization**
  - Interactive UMAP plots
  - Gene expression heatmaps
  - Violin plots
  - QC metric visualizations

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/vasighiz/scAgentic.git
cd scAgentic
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

### Usage
1. Upload your single-cell RNA-seq data (either .h5ad file or 10X Genomics files)
2. The preprocessing pipeline will run automatically
3. Use the chat interface to ask questions about your data or request specific analyses
4. Download the PDF report and processed data for further analysis

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
