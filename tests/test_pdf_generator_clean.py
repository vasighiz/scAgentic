import os
import shutil
import subprocess
import sys
import pytest
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from report_generator import generate_pdf_report, sanitize_latex

# Constants
BASE_DIR = Path(__file__).parent.parent
TEST_OUTPUT_DIR = BASE_DIR / 'test_output'

def print_latex_logs(test_dir: Path) -> None:
    """Print contents of LaTeX log files for debugging."""
    print("\nLaTeX Log Files:")
    print("=" * 80)
    
    # Print report.tex if it exists
    if (test_dir / 'report.tex').exists():
        print("\nreport.tex:")
        print("-" * 80)
        with open(test_dir / 'report.tex', 'r', encoding='utf-8') as f:
            print(f.read())
    
    # Print stdout log if it exists
    if (test_dir / 'pdflatex_stdout.log').exists():
        print("\npdflatex_stdout.log:")
        print("-" * 80)
        with open(test_dir / 'pdflatex_stdout.log', 'r', encoding='utf-8') as f:
            print(f.read())
    
    # Print stderr log if it exists
    if (test_dir / 'pdflatex_stderr.log').exists():
        print("\npdflatex_stderr.log:")
        print("-" * 80)
        with open(test_dir / 'pdflatex_stderr.log', 'r', encoding='utf-8') as f:
            print(f.read())
    
    print("=" * 80)

def write_debug_file(path: Path, content: str) -> None:
    """Write debug information to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_dummy_plot(output_dir: Path, filename: str = 'dummy_plot.png') -> Path:
    """Create a dummy plot for testing."""
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title('Dummy Plot for Testing')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    
    plot_path = output_dir / filename
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def cleanup_test_dir(test_dir: Path) -> None:
    """Clean up the test directory before each test."""
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

def check_pdflatex_installed() -> bool:
    """Check if pdflatex is installed and accessible."""
    try:
        subprocess.run(['pdflatex', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def test_generate_pdf_report_success():
    """Test successful PDF generation with valid inputs."""
    # Skip if pdflatex is not installed
    if not check_pdflatex_installed():
        pytest.skip("pdflatex is not installed")
    
    # Setup
    test_dir = TEST_OUTPUT_DIR / 'success_test'
    cleanup_test_dir(test_dir)
    
    # Create test data
    study_info = {
        'title': 'Test Study',
        'geo_accession': 'GSE12345',
        'organism': 'Homo sapiens',
        'tissue': 'Brain'
    }
    
    parameters = {
        'min_genes': 200,
        'min_cells': 3,
        'max_percent_mt': 20,
        'n_top_genes': 2000,
        'n_pcs': 50
    }
    
    # Create dummy plot
    plot_path = create_dummy_plot(test_dir)
    
    try:
        # Generate PDF
        pdf_path = generate_pdf_report(
            output_dir=str(test_dir),
            study_info=study_info,
            parameters=parameters,
            plot_files=['dummy_plot.png']
        )
        
        # Verify PDF was generated
        assert os.path.exists(pdf_path), f"PDF not found at {pdf_path}"
        assert os.path.getsize(pdf_path) > 0, "Generated PDF is empty"
        
        # Verify log files exist
        assert (test_dir / 'pdflatex_stdout.log').exists(), "stdout log not generated"
        assert (test_dir / 'pdflatex_stderr.log').exists(), "stderr log not generated"
        
        print(f'✅ PDF generated successfully: {pdf_path}')
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        print_latex_logs(test_dir)
        raise

def test_generate_pdf_report_missing_plot():
    """Test PDF generation with missing plot file."""
    # Skip if pdflatex is not installed
    if not check_pdflatex_installed():
        pytest.skip("pdflatex is not installed")
    
    # Setup
    test_dir = TEST_OUTPUT_DIR / 'missing_plot_test'
    cleanup_test_dir(test_dir)
    
    # Create test data
    study_info = {
        'title': 'Test Study',
        'geo_accession': 'GSE12345',
        'organism': 'Homo sapiens',
        'tissue': 'Brain'
    }
    
    parameters = {
        'min_genes': 200,
        'min_cells': 3,
        'max_percent_mt': 20
    }
    
    try:
        # Generate PDF with non-existent plot
        pdf_path = generate_pdf_report(
            output_dir=str(test_dir),
            study_info=study_info,
            parameters=parameters,
            plot_files=['nonexistent_plot.png']
        )
        
        # Verify PDF was generated (should still work, just without the plot)
        assert os.path.exists(pdf_path), f"PDF not found at {pdf_path}"
        assert os.path.getsize(pdf_path) > 0, "Generated PDF is empty"
        
        # Verify log files exist
        assert (test_dir / 'pdflatex_stdout.log').exists(), "stdout log not generated"
        assert (test_dir / 'pdflatex_stderr.log').exists(), "stderr log not generated"
        
        print(f'✅ PDF generated successfully without plot: {pdf_path}')
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        print_latex_logs(test_dir)
        raise

def test_generate_pdf_report_with_special_chars():
    """Test PDF generation with special characters in inputs."""
    # Skip if pdflatex is not installed
    if not check_pdflatex_installed():
        pytest.skip("pdflatex is not installed")
    
    # Setup
    test_dir = TEST_OUTPUT_DIR / 'special_chars_test'
    cleanup_test_dir(test_dir)
    
    # Create test data with special characters
    study_info = {
        'title': 'Test Study with % & _ # { } ~ ^ \\',
        'geo_accession': 'GSE12345_%',
        'organism': 'Homo sapiens',
        'tissue': 'Brain & Tissue'
    }
    
    parameters = {
        'min_genes': 200,
        'min_cells': 3,
        'max_percent_mt': 20,
        'special_param': 'value with % & _ # { } ~ ^ \\'
    }
    
    # Create dummy plot
    plot_path = create_dummy_plot(test_dir)
    
    try:
        # Generate PDF
        pdf_path = generate_pdf_report(
            output_dir=str(test_dir),
            study_info=study_info,
            parameters=parameters,
            plot_files=['dummy_plot.png']
        )
        
        # Verify PDF was generated
        assert os.path.exists(pdf_path), f"PDF not found at {pdf_path}"
        assert os.path.getsize(pdf_path) > 0, "Generated PDF is empty"
        
        # Verify log files exist
        assert (test_dir / 'pdflatex_stdout.log').exists(), "stdout log not generated"
        assert (test_dir / 'pdflatex_stderr.log').exists(), "stderr log not generated"
        
        print(f'✅ PDF generated successfully with special characters: {pdf_path}')
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        print_latex_logs(test_dir)
        raise 