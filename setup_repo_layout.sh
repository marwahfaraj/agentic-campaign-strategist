#!/usr/bin/env bash
# Reshape the repo into the agreed capstone layout.
# Run from the repo root:  bash setup_repo_layout.sh
# Optional:                bash setup_repo_layout.sh --remove-legacy   (also deletes old scripts/)
set -euo pipefail

echo "==> 1. Making sure we are in a git repo root"
test -d .git || { echo "ERROR: run this from the repo root"; exit 1; }

echo "==> 2. Creating notebooks/ folder"
mkdir -p notebooks

echo "==> 3. Bringing Rebecca's notebook from the rcloe branch into notebooks/"
git fetch origin rcloe
if [ -f notebooks/01_data_cleaning_eda.ipynb ]; then
  echo "    Already exists, skipping."
else
  git show origin/rcloe:01_data_cleaning_eda.ipynb > notebooks/01_data_cleaning_eda.ipynb
  echo "    Copied to notebooks/01_data_cleaning_eda.ipynb"
fi

echo "==> 4. Installing the src/ package in editable mode (for notebook imports)"
pip install -e . --quiet || echo "    WARNING: pip install failed; notebooks will fall back to sys.path"

if [ "${1:-}" == "--remove-legacy" ]; then
  echo "==> 5. Removing legacy scripts/ folder (kept in git history)"
  git rm -r --quiet scripts/
else
  echo "==> 5. Keeping scripts/ for now. Re-run with --remove-legacy to delete it."
fi

echo "==> 6. Staging changes"
git add notebooks/ pyproject.toml .gitignore setup_repo_layout.sh

echo ""
echo "Done. Review with 'git status' and 'git diff --staged', then commit, e.g.:"
echo "  git checkout -b matt/repo-layout"
echo "  git commit -m 'Add notebooks folder, notebook skeletons, and packaging'"
echo "  git push -u origin matt/repo-layout"
echo ""
echo "NOTE: Tell Rebecca before merging. Her open branch still has the notebook"
echo "in the root; easiest is she runs 'git mv 01_data_cleaning_eda.ipynb notebooks/'"
echo "on her branch, or merges her PR first and then this branch."
