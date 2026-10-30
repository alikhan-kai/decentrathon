#!/usr/bin/env bash
# Run the debug notebook on YOUR Kaggle account and bring the results back.
#   tools/kaggle_run.sh <quick|full|full_profile> [BRANCH]
# Pushes the current commit to origin/BRANCH (default: the current branch; no --force), builds the
# notebook for that exact commit, pushes it to Kaggle on T4, waits (50 min max), downloads the
# output to kaggle/out_<MODE>_<time>/ and runs tools/summarize.py on it.
# The Kaggle token is read by the kaggle CLI from your own environment or ~/.kaggle; this script
# only checks that it exists and never prints it.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: tools/kaggle_run.sh <quick|full|full_profile> [BRANCH]}"
case "$MODE" in quick|full|full_profile) ;; *) echo "MODE must be quick, full or full_profile"; exit 2;; esac
BRANCH="${2:-$(git rev-parse --abbrev-ref HEAD)}"
SLUG="soup-kernel-debug"
REPO_URL="https://github.com/alikhan-kai/decentrathon.git"

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is not clean: commit or stash first (Kaggle clones the pushed commit)."; exit 1
fi
if ! command -v kaggle >/dev/null 2>&1; then
  [ -f .venv-kaggle/bin/activate ] || { python3 -m venv .venv-kaggle && .venv-kaggle/bin/pip install -q kaggle; }
  # shellcheck disable=SC1091
  source .venv-kaggle/bin/activate
fi
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ ! -f "$HOME/.kaggle/access_token" ] && [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "No Kaggle token: create one on kaggle.com (Settings, API) and save it as ~/.kaggle/access_token (chmod 600) or export KAGGLE_API_TOKEN."; exit 1
fi
USERNAME="$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')"
[ -n "$USERNAME" ] || { echo "Cannot get the Kaggle username: log in first."; exit 1; }

git push origin "HEAD:refs/heads/$BRANCH"
COMMIT="$(git rev-parse HEAD)"
echo "branch $BRANCH, commit $COMMIT, mode $MODE, kaggle user $USERNAME"

PUSH_DIR=kaggle/_push
rm -rf "$PUSH_DIR" && mkdir -p "$PUSH_DIR"
python3 tools/build_notebook.py --out "$PUSH_DIR/soup_kernel_debug.ipynb" --repo-url "$REPO_URL" \
  --branch "$BRANCH" --commit "$COMMIT" --mode "$MODE" >/dev/null
python3 - "$USERNAME/$SLUG" <<'PY'
import json, sys
meta = json.load(open("kaggle/kernel-metadata.json"))
meta["id"] = sys.argv[1]
json.dump(meta, open("kaggle/_push/kernel-metadata.json", "w"), indent=2)
PY

kaggle kernels push -p "$PUSH_DIR" --accelerator NvidiaTeslaT4

OUT="kaggle/out_${MODE}_$(date +%Y%m%d_%H%M%S)"
status=""
for _ in $(seq 1 100); do   # 100 x 30 s = 50 min
  sleep 30
  status="$(kaggle kernels status "$USERNAME/$SLUG" 2>&1 | tail -1)"
  echo "$(date +%H:%M:%S) $status"
  case "$status" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac
done
kaggle kernels output "$USERNAME/$SLUG" -p "$OUT" >/dev/null 2>&1 || true
echo "output: $OUT"
python3 tools/summarize.py "$OUT" || true
case "$status" in *COMPLETE*) ;; *) echo "Kernel did not complete: $status"; exit 1;; esac
