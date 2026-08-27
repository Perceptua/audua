# audua — shortcuts for the pipeline, dev tasks, and the background UI.
#
# Usage:
#   make plan FILE=recording.m4a
#   make run FILE=recording.m4a ARGS="--merge-gap 8"
#   make batch
#   make ui-up
#   make ui-down

.DEFAULT_GOAL := help

UI_HOST ?= 127.0.0.1
UI_PORT ?= 8765
UI_PID_FILE := processing/output/.ui.pid

.PHONY: help sync test lint format \
        plan run batch verify \
        ui ui-up ui-down ui-status ui-restart

help: ## Show this help.
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

## --- setup & dev -----------------------------------------------------------

sync: ## Install dependencies, including dev tools.
	uv sync

test: ## Run the test suite.
	uv run pytest

lint: ## Lint with ruff.
	uv run ruff check .

format: ## Format with ruff.
	uv run ruff format .

## --- pipeline ----------------------------------------------------------------

plan: ## Preview the clips that would be cut from FILE. Writes only plan.json.
	uv run audua plan $(FILE) $(ARGS)

run: ## Segment and transcribe FILE (a file or a directory).
	uv run audua run $(FILE) $(ARGS)

batch: ## Work the processing/raw inbox end to end: run, digest, file, report.
	uv run audua batch $(ARGS)

verify: ## Re-check clip/transcript pairing for a finished run at DIR.
	uv run audua verify $(DIR)

## --- ui -----------------------------------------------------------------

ui: ## Serve the UI in the foreground. Ctrl-C to stop.
	uv run audua ui --host $(UI_HOST) --port $(UI_PORT) $(ARGS)

ui-up: ## Start the UI detached in the background.
	@mkdir -p $(dir $(UI_PID_FILE))
	@if [ -f $(UI_PID_FILE) ] && kill -0 "$$(cat $(UI_PID_FILE))" 2>/dev/null; then \
		echo "UI already running (pid $$(cat $(UI_PID_FILE)))."; \
	else \
		out=$$(uv run audua ui --host $(UI_HOST) --port $(UI_PORT) --background $(ARGS)); \
		echo "$$out"; \
		echo "$$out" | sed -n 's/.*pid \([0-9]*\).*/\1/p' > $(UI_PID_FILE); \
	fi

ui-down: ## Stop the background UI started with ui-up.
	@if [ -f $(UI_PID_FILE) ]; then \
		pid=$$(cat $(UI_PID_FILE)); \
		if kill -0 "$$pid" 2>/dev/null; then \
			kill "$$pid" && echo "Stopped UI (pid $$pid)."; \
		else \
			echo "No process at pid $$pid — already stopped."; \
		fi; \
		rm -f $(UI_PID_FILE); \
	else \
		echo "$(UI_PID_FILE) not found — UI wasn't started with ui-up."; \
	fi

ui-status: ## Check whether the background UI is running.
	@if [ -f $(UI_PID_FILE) ] && kill -0 "$$(cat $(UI_PID_FILE))" 2>/dev/null; then \
		echo "UI running (pid $$(cat $(UI_PID_FILE))) — http://$(UI_HOST):$(UI_PORT)/"; \
	else \
		echo "UI not running."; \
	fi

ui-restart: ui-down ui-up ## Restart the background UI.
