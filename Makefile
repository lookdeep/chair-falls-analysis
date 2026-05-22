.PHONY: serve test publish publish-dry-run

# Serve site locally with live reload at http://localhost:4000.
serve test:
	bundle exec jekyll serve --watch

# Push gh-pages to origin and trigger the Jekyll deploy workflow.
# Run only from the gh-pages branch when you want the site live.
publish:
	bash scripts/publish-site.sh

# Preview the publish steps without pushing or mutating settings.
publish-dry-run:
	bash scripts/publish-site.sh --dry-run
