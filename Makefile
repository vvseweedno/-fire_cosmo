.PHONY: install test lint check serve docker finalize

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

finalize:
	@test -n "$(TRAIN_DIR)" || (echo "TRAIN_DIR is required" && exit 2)
	@test -n "$(TEST_DIR)" || (echo "TEST_DIR is required" && exit 2)
	python scripts/finalize_competition.py --train-dir "$(TRAIN_DIR)" --test-dir "$(TEST_DIR)" --work-dir "$${WORK_DIR:-final_run}"
