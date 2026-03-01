.PHONY: help install sync lint format test test-all cov check build clean
.DEFAULT_GOAL = help

# ANSI Color Codes for pretty terminal output
BLUE   := \033[36m
YELLOW := \033[33m
GREEN  := \033[32m
RED    := \033[31m
RESET  := \033[0m

PKGROOT = gauss_flows
TESTS = tests

help:	## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation

.PHONY: install
install: ## Install all project dependencies
	@printf "$(YELLOW)>>> Initiating environment synchronization and dependency installation...$(RESET)\n"
	@uv sync --all-extras
	@uv run pre-commit install
	@printf "$(GREEN)>>> Environment is ready and pre-commit hooks are active.$(RESET)\n"

.PHONY: sync
sync: ## Update lock file and sync dependencies using uv
	@printf "$(YELLOW)>>> Updating and syncing dependencies with uv...$(RESET)\n"
	@uv lock --upgrade
	@uv sync --all-extras
	@printf "$(GREEN)>>> uv environment synchronized.$(RESET)\n"

##@ Formatting and Linting

.PHONY: lint
lint: ## Run ruff check
	@printf "$(YELLOW)>>> Executing static analysis...$(RESET)\n"
	@uv run ruff check $(PKGROOT)/ $(TESTS)/
	@printf "$(GREEN)>>> Linting checks passed.$(RESET)\n"

.PHONY: format
format: ## Run ruff formatter
	@printf "$(YELLOW)>>> Formatting code with ruff...$(RESET)\n"
	@uv run ruff format $(PKGROOT)/ $(TESTS)/
	@uv run ruff check --fix $(PKGROOT)/ $(TESTS)/
	@printf "$(GREEN)>>> Codebase formatted successfully.$(RESET)\n"

##@ Testing

.PHONY: test
test: ## Run pytest (excluding slow tests)
	@printf "$(YELLOW)>>> Launching test suite...$(RESET)\n"
	@uv run pytest $(TESTS)/ -v -m "not slow"
	@printf "$(GREEN)>>> Tests passed.$(RESET)\n"

.PHONY: test-all
test-all: ## Run all tests including slow tests
	@printf "$(YELLOW)>>> Launching full test suite...$(RESET)\n"
	@uv run pytest $(TESTS)/ -v
	@printf "$(GREEN)>>> All tests passed.$(RESET)\n"

.PHONY: cov
cov: ## Run tests with coverage
	@printf "$(YELLOW)>>> Running tests with coverage...$(RESET)\n"
	@uv run pytest --cov=$(PKGROOT) --cov-report=xml --cov-report=term-missing $(TESTS)/
	@printf "$(GREEN)>>> Coverage report generated.$(RESET)\n"

##@ Quality

.PHONY: check
check: lint test ## Run all checks (lint + test)
	@printf "$(GREEN)>>> All checks passed.$(RESET)\n"

##@ Build

.PHONY: build
build: ## Build the package
	@printf "$(YELLOW)>>> Building package...$(RESET)\n"
	@uv run python -m build
	@printf "$(GREEN)>>> Package built successfully.$(RESET)\n"

.PHONY: clean
clean: ## Clean build artifacts
	@printf "$(YELLOW)>>> Cleaning build artifacts...$(RESET)\n"
	@rm -rf dist/ build/ *.egg-info/ .pytest_cache/ htmlcov/ reports/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@printf "$(GREEN)>>> Cleanup complete.$(RESET)\n"
