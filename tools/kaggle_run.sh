#!/usr/bin/env bash
# Run the notebook on YOUR Kaggle account (T4) and bring the results back.
#   KAGGLE_USERNAME=<your kaggle login> tools/kaggle_run.sh <MODE> [BRANCH] [MODULE_TEST]
#   MODE: unit (pytest -q -rs of MODULE_TEST, 1-2 min) | quick | full | full_profile | check_and_ops | pack_check | quick_ab | pack_and_idle | robust
#   BRANCH: default = the current branch. MODULE_TEST: for unit, a file or a directory (default solution/)
# Pushes the current branch to origin (never --force; main only with ALLOW_MAIN=1), builds the
# notebook for the COMMIT OF THIS BRANCH, pushes it to Kaggle, waits (50 min max), downloads the
# output to kaggle/out_<MODE>_<time>/ and runs tools/summarize.py on it.
# The username comes from KAGGLE_USERNAME (or the 4th argument). The token is read by the kaggle CLI
# from your own environment / ~/.kaggle; this script only checks that it exists, never prints it.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: KAGGLE_USERNAME=<login> tools/kaggle_run.sh <unit|quick|full|full_profile|check_and_ops|pack_check|quick_ab|ab_overhead|ab_commits|pack_and_idle|robust> [BRANCH] [MODULE_TEST]}"
case "$MODE" in unit|quick|full|full_profile|check_and_ops|pack_check|quick_ab|ab_overhead|ab_commits|pack_and_idle|robust) ;; *) echo "bad MODE: $MODE"; exit 2;; esac
BRANCH="${2:-$(git rev-parse --abbrev-ref HEAD)}"
MODULE_TEST="${3:-}"
[ "$MODE" != unit ] || MODULE_TEST="${MODULE_TEST:-solution/}"   # a file or a directory
USERNAME="${KAGGLE_USERNAME:-${4:-}}"
SLUG="${KAGGLE_SLUG:-soup-kernel-debug}"
REPO_URL="https://github.com/alikhan-kai/decentrathon.git"

[ -n "$USERNAME" ] || { echo "Set KAGGLE_USERNAME=<your kaggle login> (or pass it as the 4th argument)."; exit 1; }
[ "$MODE" != unit ] || [ -n "$MODULE_TEST" ] || { echo "unit needs MODULE_TEST, e.g. solution/test_nf4.py"; exit 2; }
if [ "$BRANCH" = main ] && [ "${ALLOW_MAIN:-0}" != 1 ]; then
  echo "Refusing to push main directly: work on feat/<branch> and open a PR (integrator: ALLOW_MAIN=1)."; exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is not clean: commit or stash first (Kaggle clones the pushed commit)."; exit 1
fi
if ! command -v kaggle >/dev/null 2>&1; then
  [ -f .venv-kaggle/bin/activate ] || { python3 -m venv .venv-kaggle && .venv-kaggle/bin/pip install -q kaggle; }
  # shellcheck disable=SC1091
  source .venv-kaggle/bin/activate
fi
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ ! -f "$HOME/.kaggle/access_token" ] && [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "No Kaggle token: kaggle.com, Settings, API, Create New Token; save as ~/.kaggle/access_token (chmod 600) or export KAGGLE_API_TOKEN."; exit 1
fi

# NO_PUSH=1: do not push; the notebook clones BRANCH (any ref already in origin) at COMMIT_OVERRIDE.
# ab_commits: COMMIT_A and COMMIT_B (both already in origin) are compared in one session; nothing is pushed.
if [ "${NO_PUSH:-0}" != 1 ] && [ "$MODE" != ab_commits ]; then
  git push origin "HEAD:refs/heads/$BRANCH"
fi
if [ "$MODE" = ab_commits ]; then   # the clone is made at A; the local HEAD is not in origin
  COMMIT="${COMMIT_OVERRIDE:-${COMMIT_A:?ab_commits needs COMMIT_A and COMMIT_B}}"
else
  COMMIT="${COMMIT_OVERRIDE:-$(git rev-parse HEAD)}"
fi
echo "branch $BRANCH, commit $COMMIT, mode $MODE, kaggle user $USERNAME"

PUSH_DIR="$(python3 tools/build_notebook.py --username "$USERNAME" --repo "$REPO_URL" --branch "$BRANCH" \
  --commit "$COMMIT" --commit-a "${COMMIT_A:-}" --commit-b "${COMMIT_B:-}" --mode "$MODE" --module-test "$MODULE_TEST" --slug "$SLUG")"
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
