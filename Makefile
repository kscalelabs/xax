# Makefile

py-files := $(shell git ls-files '*.py')

all: format static-checks test
.PHONY: all

format:
	@black $(py-files)
	@ruff format $(py-files)
.PHONY: format

static-checks:
	@black --diff --check $(py-files)
	@ruff check $(py-files)
	@mypy --install-types --non-interactive $(py-files)
.PHONY: static-checks

test:
	python -m pytest
.PHONY: test
