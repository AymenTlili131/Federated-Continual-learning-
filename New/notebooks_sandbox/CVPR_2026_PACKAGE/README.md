# CVPR 2026 Paper Package

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
