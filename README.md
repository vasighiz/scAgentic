# 📦 scAgent v0.1 Release

Welcome to the first public release of scAgent, an AI-powered assistant for single-cell RNA-seq analysis. This version includes the core functionality for automatic preprocessing, QC visualization, and report generation — designed to be a starting point for researchers who want fast insights from their 10x or Anndata files.

________________________________________

## ✅ Features in v0.1
- Upload support for raw 10x files or .h5ad objects
- Automatic preprocessing pipeline (no parameter tuning required)
- Best-practice defaults + intelligent data-based thresholds (optional)
- UMAP, PCA, and QC visualizations (Violin, Scatter, HVG)
- Clean UI with auto-pilot mode and side info panel
- Publication-ready PDF report generation (LaTeX-powered)
- GEO accession auto-detection with study metadata linking
- Docker + Streamlit app ready

________________________________________

## 🔜 Planned in v0.2
- LLM integration for natural language querying
- Chat-driven exploration of cell populations, marker genes, and more
- Real-time explanation of plots and QC steps
- RAG with GEO descriptions for study summaries

________________________________________

## 🚀 Get Started

### Option 1: Run via Docker
```bash
git clone https://github.com/vasighiz/scagentic.git
cd scagentic
docker build -t scagent .
docker run -p 8501:8501 scagent
```

### Option 2: Run Locally
```bash
git clone https://github.com/vasighiz/scagentic.git
cd scagentic
conda env create -f environment.yml
conda activate scagentic
streamlit run app.py
```

## 📂 Input Formats
- Raw 10x folder: matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz
- Processed: .h5ad

________________________________________

## 📄 License
MIT License

________________________________________

## 👩‍💻 Author
Akram Vasighizaker | [LinkedIn](https://www.linkedin.com/in/akram-vasighizaker/) | [Website](https://vasighiz.github.io/)

If you find this tool helpful, give it a ⭐ on GitHub and share with other bioinformaticians!

________________________________________

## ✉️ Feedback / Feature Requests
Please open an issue or email me if you'd like to suggest a feature, report a bug, or collaborate. 
