# Makefile
.PHONY: build swag clean version all

# Default image name - can be overridden with make IMAGE=your-image-name
IMAGE ?= krasaee/alethic-ism-api:latest

# Build the Docker image directly
build:
	docker build -t $(IMAGE) .

# Version bump (patch version)
version:
	@echo "Bumping patch version..."
	@git fetch --tags
	@LATEST_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo ""); \
	if [[ -z "$$LATEST_TAG" ]]; then \
		MAJOR=0; MINOR=1; PATCH=0; \
		OLD_TAG="<none>"; \
	else \
		OLD_TAG="$$LATEST_TAG"; \
		VERSION="$${LATEST_TAG#v}"; \
		IFS='.' read -r MAJOR MINOR PATCH <<< "$$VERSION"; \
		PATCH=$$((PATCH + 1)); \
	fi; \
	NEW_TAG="v$${MAJOR}.$${MINOR}.$${PATCH}"; \
	git tag -a "$$NEW_TAG" -m "Release $$NEW_TAG"; \
	git push origin "$$NEW_TAG"; \
	echo "➜ bumped $${OLD_TAG} → $${NEW_TAG}"

# Clean up old images and containers
clean:
	docker system prune -f

# Show help
help:
	@echo "Available targets:"
	@echo "  build    - Build Docker image"
	@echo "  version  - Bump patch version and create git tag"
	@echo "  clean    - Clean up old Docker images and containers"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  IMAGE    - Docker image name (default: krasaee/alethic-ism-api:latest)"