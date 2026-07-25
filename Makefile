.PHONY: install install-all test lint format-check smoke clean

install:
	python -m pip install -e ".[dev]"

install-all:
	python -m pip install -e ".[dev,all-cpu]"

test:
	python -m pytest

lint:
	python -m ruff check src tests
	python -m mypy src/ml_runtime_bench

format-check:
	python -m ruff format --check src tests

smoke:
	python -m ml_runtime_bench suite --config configs/smoke.json --output results/smoke

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info results/smoke
