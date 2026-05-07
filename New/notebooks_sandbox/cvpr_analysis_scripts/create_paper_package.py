#!/usr/bin/env python3
"""
Create complete CVPR paper package with all figures and data
"""

import shutil
import subprocess
from pathlib import Path
import json

# Paths
BASE_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox")
CVPR_DIR = BASE_DIR / "CVPR 2026" / "CVPR_2026_Submission_Template"
FIGURES_DIR = BASE_DIR / "CVPR 2026" / "figures"
PACKAGE_DIR = BASE_DIR / "CVPR_2026_PACKAGE"
DATA_DIR = BASE_DIR / "cvpr_analysis_scripts" / "data"

print("="*80)
print("CREATING CVPR 2026 PAPER PACKAGE")
print("="*80)

# 1. Create package directory
print("\n1. Creating package directory...")
if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)
PACKAGE_DIR.mkdir(parents=True)

# 2. Copy LaTeX source files
print("\n2. Copying LaTeX source files...")
latex_dir = PACKAGE_DIR / "latex"
latex_dir.mkdir()

# Copy main files
for file in ['main.tex', 'cvpr.sty', 'preamble.tex']:
    src = CVPR_DIR / file
    if src.exists():
        shutil.copy2(src, latex_dir / file)
        print(f"   Copied: {file}")

# Copy section files
sec_dir = latex_dir / "sec"
sec_dir.mkdir()
src_sec_dir = CVPR_DIR / "sec"
if src_sec_dir.exists():
    for sec_file in src_sec_dir.glob("*.tex"):
        shutil.copy2(sec_file, sec_dir / sec_file.name)
        print(f"   Copied: sec/{sec_file.name}")

# 3. Copy bibliography
print("\n3. Copying bibliography...")
bib_src = BASE_DIR / "My Library.bib"
if bib_src.exists():
    shutil.copy2(bib_src, latex_dir / "My_Library.bib")
    print(f"   Copied: My Library.bib")
    
    # Update main.tex to use correct bib path
    main_tex = latex_dir / "main.tex"
    if main_tex.exists():
        content = main_tex.read_text()
        content = content.replace("../../My Library", "My_Library")
        main_tex.write_text(content)
        print("   Updated bibliography path in main.tex")

# 4. Copy figures
print("\n4. Copying figures...")
figures_pkg_dir = latex_dir / "figures"
figures_pkg_dir.mkdir()

if FIGURES_DIR.exists():
    for fig in FIGURES_DIR.glob("*.png"):
        shutil.copy2(fig, figures_pkg_dir / fig.name)
        print(f"   Copied: {fig.name}")
    for fig in FIGURES_DIR.glob("*.pdf"):
        shutil.copy2(fig, figures_pkg_dir / fig.name)
        print(f"   Copied: {fig.name}")
    for fig in FIGURES_DIR.glob("*.csv"):
        shutil.copy2(fig, figures_pkg_dir / fig.name)
        print(f"   Copied: {fig.name}")

# Update figure paths in section files
print("\n5. Updating figure paths in LaTeX files...")
for sec_file in sec_dir.glob("*.tex"):
    content = sec_file.read_text()
    content = content.replace("../../figures/", "figures/")
    sec_file.write_text(content)
    print(f"   Updated: {sec_file.name}")

# 6. Copy analysis data
print("\n6. Copying analysis data...")
data_pkg_dir = PACKAGE_DIR / "data"
data_pkg_dir.mkdir()

if DATA_DIR.exists():
    for data_file in DATA_DIR.glob("*.csv"):
        shutil.copy2(data_file, data_pkg_dir / data_file.name)
        print(f"   Copied: {data_file.name}")
    for data_file in DATA_DIR.glob("*.pkl"):
        shutil.copy2(data_file, data_pkg_dir / data_file.name)
        print(f"   Copied: {data_file.name}")

# 7. Create README
print("\n7. Creating README...")
readme_content = """# CVPR 2026 Paper Package

## Hyper-Representations via Multiparameter Persistent Homology

This package contains all materials for the CVPR 2026 submission.

### Contents

- `latex/` - Complete LaTeX source files
  - `main.tex` - Main document
  - `sec/` - All section files (abstract, intro, related work, methodology, experiments, results, conclusion)
  - `figures/` - All generated figures
  - `My_Library.bib` - Bibliography from Zotero
  
- `data/` - Analysis data
  - `experiment_metrics.csv` - Comprehensive metrics from 59 experiments (86 columns)
  - Other analysis outputs

### Compilation Instructions

#### On Linux/Mac:
```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

#### On Windows:
Use TeXworks, TeXstudio, or Overleaf

#### Using Overleaf:
1. Zip the `latex/` directory
2. Upload to Overleaf as a new project
3. Set compiler to pdfLaTeX
4. Compile

### Paper Statistics

- **Experiments analyzed:** 59
- **Loss functions:** 22
- **Data overlaps:** 3 (overlap 0, 1, 2)
- **Metrics extracted:** 86 columns including:
  - Training history (train/val loss, convergence)
  - Detailed distance metrics (euclidean, manhattan, cosine, frobenius, wasserstein, etc.)
  - Layerwise metrics for each CNN layer
  - Topology data (GW distances over time)
  - CNN validation samples

### Key Figures

1. **C0_correlation_matrix.png** - Correlation analysis for Overlap 0
2. **D1_correlation_matrix.png** - Correlation analysis for Overlap 1
3. **E2_correlation_matrix.png** - Correlation analysis for Overlap 2
4. **loss_comparison_analysis.png** - Comprehensive loss function comparison
5. **statistical_summary.csv** - Summary statistics table

### Citations

The paper uses references from the Zotero library, focusing on:
- Multiparameter persistent homology (Carlsson, Botnan, etc.)
- Topological data analysis for neural networks (Ballester, Birdal, etc.)
- Neural network theory and analysis

### Contact

For questions about compilation or content, please contact the authors.

---
Generated: April 2, 2026
"""

readme_file = PACKAGE_DIR / "README.md"
readme_file.write_text(readme_content)
print("   Created: README.md")

# 8. Create compilation script
print("\n8. Creating compilation script...")
compile_script = """#!/bin/bash
# Compile CVPR 2026 Paper

cd latex

echo "First pass..."
pdflatex -interaction=nonstopmode main.tex

echo "Running bibtex..."
bibtex main

echo "Second pass..."
pdflatex -interaction=nonstopmode main.tex

echo "Third pass..."
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "Compilation complete!"
echo "Output: main.pdf"

# Clean up auxiliary files
rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot
cd sec && rm -f *.aux && cd ..

echo "Cleaned auxiliary files"
"""

compile_sh = PACKAGE_DIR / "compile.sh"
compile_sh.write_text(compile_script)
compile_sh.chmod(0o755)
print("   Created: compile.sh")

# 9. Create package info
print("\n9. Creating package info...")
info = {
    "package_name": "CVPR_2026_Hyper_Representations",
    "created": "2026-04-02",
    "experiments": 59,
    "loss_functions": 22,
    "overlaps": 3,
    "metrics_columns": 86,
    "figures_count": len(list(figures_pkg_dir.glob("*"))) if figures_pkg_dir.exists() else 0,
    "sections": ["abstract", "intro", "related_work", "methodology", "experiments", "results", "conclusion"]
}

info_file = PACKAGE_DIR / "package_info.json"
with open(info_file, 'w') as f:
    json.dump(info, f, indent=2)
print("   Created: package_info.json")

# 10. Create zip archive
print("\n10. Creating zip archive...")
zip_file = BASE_DIR / "CVPR_2026_Paper_Package.zip"
if zip_file.exists():
    zip_file.unlink()

shutil.make_archive(
    str(BASE_DIR / "CVPR_2026_Paper_Package"),
    'zip',
    PACKAGE_DIR
)
print(f"   Created: {zip_file}")
print(f"   Size: {zip_file.stat().st_size / 1024 / 1024:.2f} MB")

print("\n" + "="*80)
print("PACKAGE CREATION COMPLETE")
print("="*80)
print(f"\nPackage directory: {PACKAGE_DIR}")
print(f"Zip archive: {zip_file}")
print("\nTo compile the paper:")
print("  1. Extract the zip file")
print("  2. cd into the latex/ directory")
print("  3. Run: pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex")
print("\nOr use the provided compile.sh script")
print("="*80)
