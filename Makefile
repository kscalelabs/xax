# Makefile

all: format static-checks test
.PHONY: all

format:
	@black xax tests examples
	@ruff format xax tests examples
	@ruff check --fix xax tests examples
.PHONY: format

static-checks:
	@black --diff --check xax tests examples
	@ruff check xax tests examples
	@mypy --install-types --non-interactive xax tests examples
.PHONY: static-checks

test:
	python -m pytest
.PHONY: test
