# Usage: make <target>
# Run `make help` to list all targets.

.PHONY: help dev down logs shell test test-unit test-integration test-e2e \
        test-cov lint typecheck format migrate migrate-new migrate-down \
        seed costs models-download clean reset

# ── Colours ───────────────────────────────────────────────────────────────────
CYAN  := \033[0;36m
RESET := \033[0m

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "$(CYAN)Kinesio-Track developer commands$(RESET)"
	@echo ""
	@echo "  $(CYAN)make dev$(RESET)               Start all services with hot-reload"
	@echo "  $(CYAN)make down$(RESET)              Stop and remove all containers"
	@echo "  $(CYAN)make logs$(RESET)              Follow API logs"
	@echo "  $(CYAN)make shell$(RESET)             Open a shell in the API container"
	@echo ""
	@echo "  $(CYAN)make test$(RESET)              Run the full test suite"
	@echo "  $(CYAN)make test-unit$(RESET)         Run unit tests only (fast, no DB)"
	@echo "  $(CYAN)make test-integration$(RESET)  Run integration tests (needs DB)"
	@echo "  $(CYAN)make test-e2e$(RESET)          Run end-to-end tests"
	@echo "  $(CYAN)make test-cov$(RESET)          Run tests with HTML coverage report"
	@echo ""
	@echo "  $(CYAN)make lint$(RESET)              Run ruff linter"
	@echo "  $(CYAN)make format$(RESET)            Auto-fix lint issues with ruff"
	@echo "  $(CYAN)make typecheck$(RESET)         Run mypy type checks"
	@echo ""
	@echo "  $(CYAN)make migrate$(RESET)           Apply all pending Alembic migrations"
	@echo "  $(CYAN)make migrate-new msg=...$(RESET) Generate a new migration"
	@echo "  $(CYAN)make migrate-down$(RESET)      Roll back one migration step"
	@echo ""
	@echo "  $(CYAN)make seed$(RESET)              Seed dev DB with sample data"
	@echo "  $(CYAN)make costs$(RESET)             Print Claude API spend this month"
	@echo "  $(CYAN)make models-download$(RESET)   Download MediaPipe model weights"
	@echo "  $(CYAN)make clean$(RESET)             Remove Python cache files"
	@echo "  $(CYAN)make reset$(RESET)             Wipe DB volume and re-seed (destructive)"
	@echo ""

# ── Docker Compose shortcuts ──────────────────────────────────────────────────

dev:
	docker compose up --build

dev-bg:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

logs-worker:
	docker compose logs -f worker

shell:
	docker compose exec api bash

shell-db:
	docker compose exec postgres psql -U kt_user -d kinesiotrack

# ── Testing ───────────────────────────────────────────────────────────────────
# Tests run against a live postgres+redis started via docker compose.
# The test DB uses a separate schema to avoid clobbering dev data.

TEST_DEPS_UP := docker compose up postgres redis -d --wait

test: _test-deps-up
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration: _test-deps-up
	pytest tests/integration/ -v --tb=short

test-e2e: _test-deps-up
	pytest tests/e2e/ -v --tb=short -s

test-cov: _test-deps-up
	pytest tests/ \
	  --cov=app \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  --tb=short
	@echo "$(CYAN)Coverage report: open htmlcov/index.html$(RESET)"

test-ws:
	@echo "$(CYAN)Opening wscat to local WS endpoint$(RESET)"
	@echo "Usage: type JSON messages, e.g. {\"type\":\"PING\",\"timestamp_ms\":1}"
	npx wscat -c "ws://localhost:80/ws/session/$(SESSION_ID)?token=$(TOKEN)"

_test-deps-up:
	$(TEST_DEPS_UP)

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check app/ tests/

format:
	ruff check --fix app/ tests/
	ruff format app/ tests/

typecheck:
	mypy app/ --ignore-missing-imports

check: lint typecheck
	@echo "$(CYAN)All checks passed$(RESET)"

# ── Database migrations ───────────────────────────────────────────────────────

migrate: _test-deps-up
	alembic upgrade head

migrate-new: _test-deps-up
ifndef msg
	$(error Usage: make migrate-new msg="describe the change")
endif
	alembic revision --autogenerate -m "$(msg)"

migrate-down: _test-deps-up
	alembic downgrade -1

migrate-history:
	alembic history --verbose

migrate-current:
	alembic current

# ── Dev utilities ─────────────────────────────────────────────────────────────

seed: _test-deps-up
	docker compose exec api python scripts/seed_dev_data.py

costs:
	docker compose exec api python scripts/check_claude_costs.py

models-download:
	bash scripts/download_models.sh

check-env:
	docker compose exec api python scripts/check_env.py

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(CYAN)Cache files removed$(RESET)"

reset:
	@echo "$(CYAN)WARNING: This will wipe the postgres volume and re-seed. Ctrl-C to cancel.$(RESET)"
	@sleep 3
	docker compose down -v
	docker compose up postgres redis -d --wait
	docker compose run --rm api alembic upgrade head
	docker compose run --rm api python scripts/seed_dev_data.py
	@echo "$(CYAN)DB reset and seeded$(RESET)"