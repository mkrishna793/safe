#!/usr/bin/env bash
# Run the symbolic verification suite under SageMath.
set -euo pipefail
cd "$(dirname "$0")/../.."

if command -v sage >/dev/null 2>&1; then
  exec sage archmath/sage/derive.sage "$@"
fi

if command -v micromamba >/dev/null 2>&1; then
  exec micromamba run -n sage sage archmath/sage/derive.sage "$@"
fi

if [ -x "$HOME/.local/bin/micromamba" ]; then
  exec "$HOME/.local/bin/micromamba" run -n sage sage archmath/sage/derive.sage "$@"
fi

echo "SageMath not found. Install with: micromamba create -n sage -c conda-forge sage" >&2
exit 1
