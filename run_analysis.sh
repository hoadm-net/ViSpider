#!/bin/bash
# Quick start script for semantic alignment analysis

echo "=========================================="
echo "ViSpider Semantic Alignment Analysis"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -f "README.md" ] || [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   cd /Users/hoadinh/Desktop/Code/ViSpider"
    exit 1
fi

# Check if data file exists
if [ ! -f "data/manual_translations/vispider_train_2000.json" ]; then
    echo "❌ Error: Data file not found"
    echo "   Expected: data/manual_translations/vispider_train_2000.json"
    exit 1
fi

echo "✓ Project setup OK"
echo ""
echo "=========================================="
echo "Running Semantic Alignment Analysis"
echo "=========================================="
echo ""

# Run analysis
python3 semantic_alignment_analysis.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Analysis completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - similarity_analysis.json"
    echo "  - bottom_100_for_review.json"
    echo "  - embeddings_cache.json (cached for future runs)"
    echo ""
    echo "Next step:"
    echo "  python review_bottom_samples.py"
    echo ""
else
    echo ""
    echo "❌ Analysis failed. Please check the error messages above."
    echo ""
fi
