#!/usr/bin/env bash
#
# publish-site.sh — promote gh-pages-draft to the live deployment branch.
#
# Run when the J-BHI manuscript is accepted and the public landing page
# should go live at https://lookdeep.github.io/chair-falls-analysis/.
#
# Effects:
#   1. Fast-forwards / hard-resets local `gh-pages` to current `gh-pages-draft`.
#   2. Pushes `gh-pages` to origin, which fires
#      `.github/workflows/jekyll-gh-pages.yml` and builds the live site.
#   3. (One-time) Flips the GitHub Pages source from `main/docs` to
#      "GitHub Actions" so the deployed workflow output is what's served.
#
# Re-run safe: idempotent on the merge / push, no-op on the settings flip
# once Pages already points at the Actions build.
#
# Usage:
#   bash scripts/publish-site.sh [--dry-run]
#
# Requires: git, gh (authenticated to lookdeep/chair-falls-analysis).

set -euo pipefail

REPO="lookdeep/chair-falls-analysis"
DRAFT="gh-pages-draft"
LIVE="gh-pages"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            sed -n '2,22p' "$0"
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

echo "==> Verifying clean working tree on $DRAFT"
git fetch origin "$DRAFT" "$LIVE" 2>/dev/null || true
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "$DRAFT" ]]; then
    echo "ERROR: must be on $DRAFT (currently on $current_branch)" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree has uncommitted changes" >&2
    git status --short
    exit 1
fi

echo "==> Pointing $LIVE at current $DRAFT tip"
draft_sha=$(git rev-parse "$DRAFT")
echo "    $DRAFT @ $draft_sha"

if git show-ref --verify --quiet "refs/heads/$LIVE"; then
    run git branch -f "$LIVE" "$DRAFT"
else
    run git branch "$LIVE" "$DRAFT"
fi

echo "==> Pushing $LIVE to origin (triggers Jekyll build workflow)"
run git push origin "$LIVE"

echo "==> Ensuring GitHub Pages serves the Actions build (not main/docs)"
current_source=$(gh api "repos/$REPO/pages" --jq '.source.branch + ":" + .source.path' 2>/dev/null || echo "unknown")
echo "    Current Pages source: $current_source"
if [[ "$current_source" != "$LIVE:/" && "$current_source" != "unknown" ]]; then
    run gh api -X POST "repos/$REPO/pages" -f "build_type=workflow" 2>/dev/null \
        || run gh api -X PUT "repos/$REPO/pages" -f "build_type=workflow"
fi

echo "==> Done. Watch the deployment at:"
echo "    https://github.com/$REPO/actions"
echo "    https://lookdeep.github.io/chair-falls-analysis/"
