#!/usr/bin/env bash
#
# publish-site.sh — push gh-pages to deploy the live landing page.
#
# Effect: pushes the current `gh-pages` branch to origin, which fires
# `.github/workflows/jekyll-gh-pages.yml` and builds the live site at
# https://lookdeep.github.io/chair-falls-analysis/.
#
# Usage:
#   bash scripts/publish-site.sh [--dry-run]
#
# Requires: git, gh (authenticated to lookdeep/chair-falls-analysis).

set -euo pipefail

REPO="lookdeep/chair-falls-analysis"
LIVE="gh-pages"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0
            ;;
    esac
done

run() {
    if $DRY_RUN; then
        echo "DRY-RUN $ $*"
    else
        echo "$ $*"
        "$@"
    fi
}

echo "==> Verifying clean working tree on $LIVE"
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "$LIVE" ]]; then
    echo "ERROR: must be on $LIVE (currently on $current_branch)" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree has uncommitted changes" >&2
    git status --short
    exit 1
fi

echo "==> Pushing $LIVE to origin (triggers Jekyll build workflow)"
run git push origin "$LIVE"

echo "==> Done. Watch the deployment at:"
echo "    https://github.com/$REPO/actions"
echo "    https://lookdeep.github.io/chair-falls-analysis/"
