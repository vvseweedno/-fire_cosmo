.PHONY: install run test lint clean

install:
	uv pip install -e ".[dev]"

run:
	python -m src.nexus.main

test:
	pytest

lint:
	ruff check src/nexus tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov data/cache

dev:
	uvicorn src.nexus.api.app:app --reload --host 0.0.0.0 --port 8000

seed:
	python scripts/seed_settlements.py
