#!/bin/sh
# Generate call / caller / include dependency graphs from the CUDA/C++ source.
# Needs: doxygen + graphviz (dot).
#   macOS:  brew install doxygen graphviz
#   ubuntu: sudo apt-get install -y doxygen graphviz
# Output: build/codegraph/html/index.html
set -e
cd "$(dirname "$0")/.."
if ! command -v doxygen >/dev/null 2>&1; then
    echo "error: doxygen not found (brew install doxygen graphviz)" >&2
    exit 1
fi
if ! command -v dot >/dev/null 2>&1; then
    echo "error: graphviz 'dot' not found (brew install graphviz)" >&2
    exit 1
fi
mkdir -p src
doxygen Doxyfile
echo "codegraph: build/codegraph/html/index.html"
