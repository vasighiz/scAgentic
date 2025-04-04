import os
import subprocess
import shutil
from datetime import datetime

def sanitize_latex(text: str) -> str:
    """
    Escape special LaTeX characters in text.
    """
    # First escape backslashes to avoid double escaping
    text = text.replace('\\', '\\textbackslash{}')
    
    # Then escape other special characters
    special_chars = {
        '_': '\\_',
        '%': '\\%',
        '&': '\\&',
        '#': '\\#',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}'
    }
    
    for char, escape in special_chars.items():
        text = text.replace(char, escape)
    
    return text

def check_file_exists(file_path: str) -> bool:
    try:
        return os.path.exists(file_path) and os.access(file_path, os.R_OK)
    except Exception:
        return False

def generate_pdf_report(
    output_dir: str,
    study_info: dict,
    parameters: dict,
    plot_files: list[str]
) -> str:
    """
    Generate a scientific-style LaTeX PDF report.
    
    Args:
        output_dir: Directory to save report.tex and final_report.pdf
        study_info: Dictionary containing study information
        parameters: Dictionary of preprocessing parameters
        plot_files: List of plot filenames in output_dir
        
    Returns:
        Path to the generated PDF file
        
    Raises:
        FileNotFoundError: If required files are missing
        RuntimeError: If LaTeX compilation fails
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy logo if it exists
    logo_src = "scagentic_logo.png"
    logo_dst = os.path.join(output_dir, "scagentic_logo.png")
    if os.path.exists(logo_src):
        shutil.copy2(logo_src, logo_dst)
    
    # Sanitize text inputs
    study_info = {k: sanitize_latex(str(v)) for k, v in study_info.items()}
    
    # Generate LaTeX content
    latex_content = f"""
\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\usepackage{{booktabs}}
\\usepackage{{caption}}
\\usepackage{{subcaption}}
\\usepackage{{geometry}}
\\usepackage{{hyperref}}
\\usepackage{{float}}
\\usepackage{{enumitem}}
\\usepackage{{xcolor}}

% Page geometry
\\geometry{{a4paper, margin=1in}}

% Hyperref settings
\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan
}}

% Custom commands
\\newcommand{{\\datasetname}}[1]{{\\textbf{{#1}}}}
\\newcommand{{\\paramname}}[1]{{\\texttt{{#1}}}}

% Title page
\\begin{{document}}
\\begin{{titlepage}}
    \\centering
    \\vspace*{{2cm}}

    % Logo
    \\includegraphics[width=0.4\\textwidth]{{scagentic_logo.png}}
    \\vspace{{1cm}}

    % Title
    \\Huge\\textbf{{Single-Cell RNA-seq Analysis Report}}
    \\vspace{{1cm}}

    % Study information
    \\Large\\textbf{{{study_info['title']}}}
    \\vspace{{0.5cm}}

    % GEO accession
    \\Large\\textbf{{GEO Accession: {study_info['geo_accession']}}}
    \\vspace{{0.5cm}}

    % Species and tissue
    \\Large\\textbf{{Species: {study_info['organism']}}}
    \\vspace{{0.5cm}}
    \\vspace{{1cm}}

    % Date
    \\large\\today
\\end{{titlepage}}

% Table of contents
\\tableofcontents
\\newpage

% Study information
\\section{{Study Information}}
\\begin{{itemize}}
    \\item \\textbf{{GEO Accession:}} {study_info['geo_accession']}
    \\item \\textbf{{Status:}} {study_info.get('status', 'Not available')}
    \\item \\textbf{{Title:}} {study_info['title']}
    \\item \\textbf{{Source Name:}} {study_info.get('source_name', 'Not available')}
    \\item \\textbf{{Organism:}} {study_info['organism']}
    \\item \\textbf{{Analysis Date:}} {{{datetime.now().strftime('%B %d, %Y')}}}
\\end{{itemize}}

% Add a link to the GEO page
\\begin{{quote}}
    \\textbf{{GEO Link:}} \\url{{{study_info.get('geo_url', f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={study_info["geo_accession"]}')}}}
\\end{{quote}}

% Analysis parameters
\\section{{Analysis Parameters}}
\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{ll}}
        \\toprule
        \\textbf{{Parameter}} & \\textbf{{Value}} \\\\
        \\midrule
"""
    
    # Add parameters to table
    for param, value in parameters.items():
        # Convert underscores to spaces and capitalize for display
        param_display = param.replace('_', ' ').title()
        # Sanitize the value
        value_sanitized = sanitize_latex(str(value))
        latex_content += f"        {param_display} & {value_sanitized} \\\\\n"
    
    latex_content += """
        \\bottomrule
    \\end{tabular}
    \\caption{Analysis parameters used in preprocessing}
    \\label{tab:parameters}
\\end{table}

% Results
\\section{{Results}}
"""
    
    # Add plots
    for i, plot_file in enumerate(plot_files, 1):
        if os.path.exists(os.path.join(output_dir, plot_file)):
            latex_content += f"""
    \\begin{{figure}}[H]
        \\centering
        \\includegraphics[width=0.8\\textwidth]{{{plot_file}}}
        \\caption{{{sanitize_latex(plot_file.replace('.png', '').replace('_', ' ').title())}}}
        \\label{{fig:{plot_file.replace('.png', '')}}}
    \\end{{figure}}
    \\newpage
"""
    
    latex_content += """
\\end{document}
"""
    
    # Save LaTeX file
    tex_path = os.path.join(output_dir, 'report.tex')
    with open(tex_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(latex_content)
    
    # Store original directory
    original_dir = os.getcwd()
    
    try:
        # Change to the output directory
        os.chdir(output_dir)
        
        # Run pdflatex twice to ensure references are updated
        for _ in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'report.tex'],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            # Save logs
            with open('report.log', 'w', encoding='utf-8') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write('\n\nSTDERR:\n')
                    f.write(result.stderr)
            
            if result.returncode != 0:
                raise RuntimeError(
                    "LaTeX compilation failed. See report.log for details."
                )
        
        # Rename the output to final_report.pdf
        if os.path.exists('report.pdf'):
            if os.path.exists('final_report.pdf'):
                os.remove('final_report.pdf')
            os.rename('report.pdf', 'final_report.pdf')
        else:
            raise FileNotFoundError("PDF file was not generated")
        
        # Clean up auxiliary files
        for ext in ['.aux', '.out']:
            if os.path.exists(f'report{ext}'):
                os.remove(f'report{ext}')
        
        return os.path.join(output_dir, 'final_report.pdf')
        
    except Exception as e:
        print(f"Error during PDF generation: {str(e)}")
        raise
    finally:
        # Always restore the original working directory
        try:
            os.chdir(original_dir)
        except Exception as e:
            print(f"Warning: Failed to restore original directory: {str(e)}")
