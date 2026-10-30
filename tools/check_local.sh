#!/usr/bin/env bash
# Checks that need neither torch nor a GPU. Run before every push; CI runs the same script.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m py_compile solution/*.py tools/*.py
python3 tools/lint_rules.py solution/
git fetch -q origin main 2>/dev/null || true
status=0
for ref in origin/main c7c2fcd; do   # c7c2fcd = the organiser's starter kit
  if git rev-parse -q --verify "$ref^{commit}" >/dev/null; then
    if ! git diff --quiet "$ref" -- harness tests task templates/kaggle_notebook.ipynb; then
      echo "harness/tests/task/templates/kaggle_notebook.ipynb differ from $ref (must not be changed):"
      git diff --stat "$ref" -- harness tests task templates/kaggle_notebook.ipynb
      status=1
    fi
  fi
done
[ "$status" = 0 ] && echo "check_local: OK"
exit "$status"
