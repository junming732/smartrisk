.PHONY: bootstrap lint format test dev docker-build docker-run

bootstrap:
	python -m pip install --upgrade pip
	pip install -e .[llm] pre-commit pytest ruff black uvicorn
	pre-commit install

lint:
	ruff check .

format:
	ruff format .
	black .

test:
	pytest -q || true

dev:
	uvicorn smartrisk.serve.api:app --reload

docker-build:
	docker build -t smartrisk:latest .

docker-run:
	docker run --rm -p 8000:8000 smartrisk:latest
