.PHONY: install test lint check serve docker preflight-train preflight-test release-gate smoke-inference benchmark scorecard manifest finalize

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

preflight-train:
	@test -n "$(TRAIN_DIR)" || (echo "TRAIN_DIR is required" && exit 2)
	python scripts/preflight_dataset.py --data-dir "$(TRAIN_DIR)" --mode train --deep --output "$${WORK_DIR:-final_run}/preflight_train.json"

preflight-test:
	@test -n "$(TEST_DIR)" || (echo "TEST_DIR is required" && exit 2)
	python scripts/preflight_dataset.py --data-dir "$(TEST_DIR)" --mode test --deep --output "$${WORK_DIR:-final_run}/preflight_test.json"

release-gate:
	@test -n "$(EVIDENCE)" || (echo "EVIDENCE is required" && exit 2)
	python scripts/verify_proven_release.py --evidence "$(EVIDENCE)" --output "$${WORK_DIR:-final_run}/final_validation.json"

smoke-inference:
	@test -n "$(TEST_DIR)" || (echo "TEST_DIR is required" && exit 2)
	python inference.py --data-dir "$(TEST_DIR)" --output "$${WORK_DIR:-final_run}/submission_smoke.csv" $${MODEL_CONFIG:+--model-config "$$MODEL_CONFIG"}

benchmark:
	@test -n "$(TEST_DIR)" || (echo "TEST_DIR is required" && exit 2)
	python scripts/benchmark_inference.py --data-dir "$(TEST_DIR)" --output-dir "$${WORK_DIR:-final_run}/artifacts/benchmark_inference" --output "$${WORK_DIR:-final_run}/artifacts/benchmark_inference.json" $${MODEL_CONFIG:+--model-config "$$MODEL_CONFIG"}

scorecard:
	python scripts/readiness_scorecard.py --work-dir "$${WORK_DIR:-final_run}" --output "$${WORK_DIR:-final_run}/artifacts/readiness_scorecard.json"

manifest:
	python scripts/reproduce_final.py --work-dir "$${WORK_DIR:-final_run}" --output release/final_manifest.json

finalize:
	@test -n "$(TRAIN_DIR)" || (echo "TRAIN_DIR is required" && exit 2)
	@test -n "$(TEST_DIR)" || (echo "TEST_DIR is required" && exit 2)
	python scripts/finalize_competition.py --train-dir "$(TRAIN_DIR)" --test-dir "$(TEST_DIR)" --work-dir "$${WORK_DIR:-final_run}"
