#!/bin/bash
# DEEM PyPI Publication Script
# Run this after setting up accounts and tokens

set -e  # Exit on error

echo "════════════════════════════════════════════════"
echo "  DEEM Package Publication Script"
echo "════════════════════════════════════════════════"
echo ""

# Check we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "deem" ]; then
    echo "❌ Error: Must run from /home/dsi/maymona3/rbm_python/"
    exit 1
fi

# Check .pypirc exists
if [ ! -f "$HOME/.pypirc" ]; then
    echo "❌ Error: ~/.pypirc not found!"
    echo "   Create it with your API tokens first."
    echo "   See PUBLISHING_GUIDE.md for template."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check tools installed
if ! command -v twine &> /dev/null; then
    echo "📦 Installing twine..."
    pip install twine --quiet
fi

echo ""
echo "📋 Pre-flight checklist:"
echo "   ✓ pyproject.toml exists"
echo "   ✓ deem/ package exists"
echo "   ✓ ~/.pypirc configured"
echo "   ✓ Virtual environment active"
echo "   ✓ twine installed"
echo ""

# Ask which repository
echo "Select upload destination:"
echo "  1) TestPyPI (recommended first)"
echo "  2) Production PyPI (FINAL - cannot be undone!)"
echo "  3) Both (TestPyPI first, then PyPI)"
echo "  4) Cancel"
echo ""
read -p "Choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "🧪 Uploading to TestPyPI..."
        twine upload --repository testpypi dist/*
        echo ""
        echo "✅ Uploaded to TestPyPI!"
        echo ""
        echo "📦 Test installation:"
        echo "   pip install --index-url https://test.pypi.org/simple/ \\"
        echo "               --extra-index-url https://pypi.org/simple/ deem"
        echo ""
        echo "   python -c \"from deem import DEEM; print('Works!')\""
        echo ""
        echo "🌐 View at: https://test.pypi.org/project/deem/"
        ;;
    2)
        echo ""
        read -p "⚠️  Upload to PRODUCTION PyPI? This CANNOT be undone! [yes/NO]: " confirm
        if [ "$confirm" = "yes" ]; then
            echo ""
            echo "🚀 Uploading to PyPI..."
            twine upload dist/*
            echo ""
            echo "✅ Published to PyPI!"
            echo ""
            echo "🎉 Package is now public!"
            echo "   pip install deem"
            echo ""
            echo "🌐 View at: https://pypi.org/project/deem/"
        else
            echo "Cancelled."
        fi
        ;;
    3)
        echo ""
        echo "🧪 Step 1: Uploading to TestPyPI..."
        twine upload --repository testpypi dist/*
        echo ""
        echo "✅ Uploaded to TestPyPI!"
        echo ""
        read -p "Test the installation, then press Enter to continue to PyPI..."
        echo ""
        read -p "⚠️  Upload to PRODUCTION PyPI? This CANNOT be undone! [yes/NO]: " confirm
        if [ "$confirm" = "yes" ]; then
            echo ""
            echo "🚀 Step 2: Uploading to PyPI..."
            twine upload dist/*
            echo ""
            echo "✅ Published to PyPI!"
            echo ""
            echo "🎉 Package is now public!"
            echo "   pip install deem"
            echo ""
            echo "🌐 View at: https://pypi.org/project/deem/"
        else
            echo "PyPI upload cancelled."
        fi
        ;;
    4)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════"
echo "  Done!"
echo "════════════════════════════════════════════════"
