#!/bin/bash
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
