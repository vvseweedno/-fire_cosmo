.PHONY: install test lint check serve docker

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

check: lint test

serve:
	uvicorn app.main:app --reload

docker:
	docker build -t ready-prototype .
